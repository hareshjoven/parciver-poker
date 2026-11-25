/* Headless DOM/Canvas harness for the §35.A experiment dashboard client.
 *
 * We verify
 * the REAL static/experiment.js the way app.js is verified: stub a minimal DOM +
 * 2D canvas + EventSource, load the client, feed it an actual events.jsonl from a
 * decider run, drive a few animation frames, and assert every draw path executed
 * without throwing and that the panels were populated. Exits non-zero on any error.
 *
 *   node scripts/check_experiment_dom.mjs <path-to-events.jsonl>
 */
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const logPath = process.argv[2] || path.join(ROOT, "runs");
const jsPath = path.join(ROOT, "src", "parciver", "dashboard", "static", "experiment.js");
const htmlPath = path.join(ROOT, "src", "parciver", "dashboard", "static", "experiment.html");

// -- collect the element ids experiment.html declares (so getElementById is faithful)
const html = fs.readFileSync(htmlPath, "utf8");
const htmlIds = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));

// -- minimal 2D canvas context: every method a no-op, every getter benign
function makeCtx() {
  const noop = () => {};
  return new Proxy({
    canvas: { width: 600, height: 300 },
    measureText: () => ({ width: 20 }),
    createLinearGradient: () => ({ addColorStop: noop }),
    getContext: () => makeCtx(),
    setTransform: noop, clearRect: noop, beginPath: noop, moveTo: noop, lineTo: noop,
    arc: noop, arcTo: noop, rect: noop, closePath: noop, fill: noop, stroke: noop,
    fillText: noop, save: noop, restore: noop, translate: noop, rotate: noop, clip: noop,
    setLineDash: noop, fillRect: noop, strokeRect: noop,
  }, { get(t, p) { return p in t ? t[p] : (typeof p === "string" ? 0 : undefined); }, set() { return true; } });
}

const listeners = {};
function makeEl(id) {
  const style = {};
  const el = {
    id, _text: "", _html: "", className: "", hidden: false, children: [],
    style, width: 600, height: 300,
    getContext: () => makeCtx(),
    getBoundingClientRect: () => ({ width: 600, height: 300, left: 0, top: 0 }),
    addEventListener: (t, f) => { (listeners[id] ||= {})[t] = f; },
    appendChild(c) { this.children.push(c); return c; },
    removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); },
    querySelector() { return makeEl(id + "-q"); },
    set textContent(v) { this._text = String(v); }, get textContent() { return this._text; },
    set innerHTML(v) { this._html = String(v); this.children = []; }, get innerHTML() { return this._html; },
  };
  return el;
}

const elements = new Map();
function getEl(id) { if (!elements.has(id)) elements.set(id, makeEl(id)); return elements.get(id); }

let esInstance = null;
class EventSource { constructor() { esInstance = this; this.onmessage = null; this.onopen = null; this.onerror = null; } }

const rafQueue = [];
const sandbox = {
  console,
  document: {
    documentElement: getEl("html"),
    getElementById: (id) => getEl(id),
    createElement: (tag) => makeEl("new-" + tag),
    addEventListener: (t, f) => { (listeners.document ||= {})[t] = f; },
  },
  window: { devicePixelRatio: 2, addEventListener: () => {} },
  getComputedStyle: () => ({ getPropertyValue: () => "#38bdf8" }),
  performance: { now: () => Date.now() },
  requestAnimationFrame: (fn) => { rafQueue.push(fn); return rafQueue.length; },
  cancelAnimationFrame: () => {},
  EventSource,
  Date,
  Math,
  isFinite,
  parseInt,
  JSON,
};
sandbox.globalThis = sandbox;

// -- load + run the client in the sandbox
const code = fs.readFileSync(jsPath, "utf8");
try {
  vm.runInNewContext(code, sandbox, { filename: "experiment.js" });
} catch (e) { console.error("LOAD ERROR:", e); process.exit(1); }

if (!esInstance || typeof esInstance.onmessage !== "function") {
  console.error("client did not open an EventSource"); process.exit(1);
}

// -- feed the real event log as one SSE batch (the server coalesces; the client
//    handles a big batch), then drive a few frames.
const lines = fs.readFileSync(logPath, "utf8").split("\n").filter((l) => l.trim());
const events = lines.map((l) => JSON.parse(l));
let applied = 0;
try {
  // deliver in a few batches to exercise incremental application + dirty flags
  for (let i = 0; i < events.length; i += 7) {
    const batch = events.slice(i, i + 7);
    esInstance.onmessage({ data: JSON.stringify(batch) });
    applied += batch.length;
    // run any queued frames after each batch
    let n = rafQueue.length; while (n-- > 0) { const fn = rafQueue.shift(); fn(Date.now()); }
  }
  // a few extra frames for good measure
  for (let k = 0; k < 5; k++) { let n = rafQueue.length; while (n-- > 0) { const fn = rafQueue.shift(); fn(Date.now()); } }
} catch (e) { console.error("RENDER ERROR after", applied, "events:", e); process.exit(1); }

// -- assertions: the panels were populated
const verdict = getEl("stat-verdict")._text;
const gameName = getEl("game-name")._text;
const logChildren = getEl("event-log").children.length;
const tierKey = getEl("tier-key")._html;
const problems = [];
if (!verdict || verdict === "—") problems.push("stat-verdict never set");
if (!gameName.includes("kuhn") && !gameName.includes("leduc")) problems.push("game-name not set: " + gameName);
if (logChildren === 0) problems.push("event-log empty");
if (!tierKey || !tierKey.includes("myopic")) problems.push("tier-key not built");
// referenced ids must exist in the HTML (catch typos between js and html)
const missing = [];
for (const id of ["stat-verdict", "stat-headroom", "c-frontier", "c-headroom", "c-ablation",
  "c-screen", "c-progress", "c-decomp", "event-log", "frontier-fam", "tier-key",
  "kpi-best", "kpi-succ", "kpi-nfam", "kpi-a5a4", "decomp-note", "done-overlay"]) {
  if (!htmlIds.has(id)) missing.push(id);
}
if (missing.length) problems.push("ids referenced by js but absent in html: " + missing.join(", "));

if (problems.length) { console.error("FAIL:\n  " + problems.join("\n  ")); process.exit(1); }
console.log(`OK — experiment.js rendered ${applied} events clean; verdict="${verdict}", game="${gameName}", log rows=${logChildren}`);
