/* Headless DOM/Canvas harness for the S3.1 HU-exploitation-spine experiment client.
 *
 * Verifies the REAL static/experiment3.js: stub a minimal DOM + 2D canvas +
 * EventSource, load the client, feed it an actual events.jsonl from a step3 run,
 * drive a few animation frames, and assert every draw path executed without throwing
 * and that the panels/table/gates were populated. Exits non-zero on any error.
 *
 *   node scripts/check_experiment3_dom.mjs <path-to-events.jsonl>
 */
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
let logPath = process.argv[2];
if (!logPath) {
  const runs = path.join(ROOT, "runs");
  const dirs = fs.readdirSync(runs).filter((d) => d.startsWith("rmapss-step3-"))
    .map((d) => path.join(runs, d, "events.jsonl")).filter((p) => fs.existsSync(p));
  dirs.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  logPath = dirs[0];
}
const jsPath = path.join(ROOT, "src", "parciver", "dashboard", "static", "experiment3.js");
const htmlPath = path.join(ROOT, "src", "parciver", "dashboard", "static", "experiment3.html");
if (!logPath || !fs.existsSync(logPath)) { console.error("no events.jsonl found:", logPath); process.exit(1); }

const html = fs.readFileSync(htmlPath, "utf8");
const htmlIds = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));

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
    id, _text: "", _html: "", className: "", hidden: false, children: [], dataset: {},
    style, width: 600, height: 300,
    getContext: () => makeCtx(),
    getBoundingClientRect: () => ({ width: 600, height: 300, left: 0, top: 0 }),
    addEventListener: (t, f) => { (listeners[id] ||= {})[t] = f; },
    appendChild(c) { this.children.push(c); return c; },
    removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); },
    prepend(c) { this.children.unshift(c); return c; },
    querySelector() { return makeEl(id + "-q"); },
    querySelectorAll() { const a = []; a.forEach = Array.prototype.forEach.bind(a); return a; },
    set textContent(v) { this._text = String(v); }, get textContent() { return this._text; },
    set innerHTML(v) { this._html = String(v); this.children = []; }, get innerHTML() { return this._html; },
    set onclick(f) { (listeners[id] ||= {}).click = f; }, get onclick() { return (listeners[id] || {}).click; },
    get lastChild() { return this.children[this.children.length - 1]; },
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
    querySelector: (sel) => getEl("qs:" + sel),
    querySelectorAll: () => { const a = []; a.forEach = Array.prototype.forEach.bind(a); return a; },
    addEventListener: (t, f) => { (listeners.document ||= {})[t] = f; },
  },
  window: { devicePixelRatio: 2, addEventListener: () => {} },
  getComputedStyle: () => ({ getPropertyValue: () => "#38bdf8" }),
  performance: { now: () => Date.now() },
  requestAnimationFrame: (fn) => { rafQueue.push(fn); return rafQueue.length; },
  cancelAnimationFrame: () => {},
  EventSource, Date, Math, isFinite, parseInt, JSON, Array, Object,
};
sandbox.globalThis = sandbox;

const code = fs.readFileSync(jsPath, "utf8");
try {
  vm.runInNewContext(code, sandbox, { filename: "experiment3.js" });
} catch (e) { console.error("LOAD ERROR:", e); process.exit(1); }

if (!esInstance || typeof esInstance.onmessage !== "function") {
  console.error("client did not open an EventSource"); process.exit(1);
}

const lines = fs.readFileSync(logPath, "utf8").split("\n").filter((l) => l.trim());
const events = lines.map((l) => JSON.parse(l));
let applied = 0;
try {
  for (let i = 0; i < events.length; i += 5) {
    const batch = events.slice(i, i + 5);
    esInstance.onmessage({ data: JSON.stringify(batch) });
    applied += batch.length;
    let n = rafQueue.length; while (n-- > 0) { const fn = rafQueue.shift(); fn(Date.now()); }
  }
  for (let k = 0; k < 5; k++) { let n = rafQueue.length; while (n-- > 0) { const fn = rafQueue.shift(); fn(Date.now()); } }
} catch (e) { console.error("RENDER ERROR after", applied, "events:", e); process.exit(1); }

const verdict = getEl("stat-verdict")._text;
const gameName = getEl("game-name")._text;
const logChildren = getEl("event-log").children.length;
const tbodyHtml = getEl("qs:#qtab tbody")._html;
const gatesHtml = getEl("gates-list")._html;
const problems = [];
if (!verdict || verdict === "—") problems.push("stat-verdict never set");
if (!gameName.includes("nlhe_small")) problems.push("game-name not set: " + gameName);
if (logChildren === 0) problems.push("event-log empty");
if (!tbodyHtml || !tbodyHtml.includes("<tr")) problems.push("per-opponent table not populated");
if (!gatesHtml || !gatesHtml.includes("<li")) problems.push("gates list not populated");
const missing = [];
for (const id of ["stat-verdict", "stat-gain", "stat-S", "stat-thr", "c-frontier", "c-alpha",
  "c-safety", "gates-list", "qtab", "event-log", "kpi-safety", "done-overlay"]) {
  if (!htmlIds.has(id)) missing.push(id);
}
if (missing.length) problems.push("ids referenced by js but absent in html: " + missing.join(", "));

if (problems.length) { console.error("FAIL:\n  " + problems.join("\n  ")); process.exit(1); }
console.log(`OK — experiment3.js rendered ${applied} events clean; verdict="${verdict}", game="${gameName}", log rows=${logChildren}`);
