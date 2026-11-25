// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
/* Headless DOM/canvas smoke test for the real S3.2 dashboard client.
 * Usage: node scripts/check_experiment32_dom.mjs <events.jsonl> */
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const logPath = process.argv[2];
if (!logPath || !fs.existsSync(logPath)) { console.error("missing events.jsonl"); process.exit(2); }
const jsPath = path.join(ROOT, "src", "parciver", "dashboard", "static", "experiment32.js");
const htmlPath = path.join(ROOT, "src", "parciver", "dashboard", "static", "experiment32.html");
const html = fs.readFileSync(htmlPath, "utf8");
const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map((match) => match[1]));

function context() {
  const noop = () => {};
  return new Proxy({ setTransform:noop, clearRect:noop, beginPath:noop, moveTo:noop,
    lineTo:noop, stroke:noop, fillRect:noop, arc:noop, fill:noop, fillText:noop,
    setLineDash:noop }, { get:(target, key) => key in target ? target[key] : 0, set:() => true });
}
const elements = new Map();
function element(id) {
  if (elements.has(id)) return elements.get(id);
  const el = { id, children:[], className:"", hidden:false, _text:"", _html:"", style:{}, width:600, height:210,
    getBoundingClientRect:() => ({ width:600, height:210 }), getContext:() => context(),
    prepend(child) { this.children.unshift(child); }, appendChild(child) { this.children.push(child); },
    removeChild(child) { this.children.splice(this.children.indexOf(child), 1); },
    querySelector(selector) { return element(`${id}:${selector}`); },
    set textContent(value) { this._text = String(value); }, get textContent() { return this._text; },
    set innerHTML(value) { this._html = String(value); }, get innerHTML() { return this._html; },
    get lastChild() { return this.children[this.children.length - 1]; } };
  elements.set(id, el); return el;
}
let source;
class EventSource { constructor() { source = this; } }
const sandbox = { console, Date, Math, Number, JSON, EventSource,
  document:{ documentElement:element("html"), getElementById:element, createElement:(tag) => element(`new:${tag}`) },
  getComputedStyle:() => ({ getPropertyValue:(name) => ({
    "--border":"#292d32", "--ink-3":"#747a83", "--accent":"#8ab4f8",
    "--accent-2":"#c58af9", "--green":"#81c995", "--amber":"#f9ab00", "--red":"#f28b82"
  })[name] || "#888888" }),
  window:{ devicePixelRatio:1, addEventListener:() => {} } };
sandbox.globalThis = sandbox;

try { vm.runInNewContext(fs.readFileSync(jsPath, "utf8"), sandbox, { filename:"experiment32.js" }); }
catch (error) { console.error("LOAD ERROR", error); process.exit(1); }
let logEvents;
try {
  logEvents = fs.readFileSync(logPath, "utf8").split("\n").filter(Boolean).map(JSON.parse);
  source.onmessage({ data:JSON.stringify(logEvents) });
} catch (error) { console.error("RENDER ERROR", error); process.exit(1); }

const problems = [];
for (const id of ["game-name","stat-verdict","stat-progress","stat-gain","stat-coverage",
  "run-stage","stat-elapsed","stat-eta","stat-eta-note","stat-workload","stat-throughput","stat-last-event",
  "run-progress-label","run-progress-percent","run-progress-bar","phase-strip","calibration-cards",
  "c-product","c-alpha","c-proxy","c-model","case-table","gates-list","event-log","done-overlay"]) {
  if (!ids.has(id)) problems.push(`missing HTML id ${id}`);
}
if (!element("game-name")._text.includes("S3.2")) problems.push("game name not rendered");
const complete = logEvents.some((event) => event.type === "run_end" || event.step32);
if (!element("stat-workload")._text.includes("/")) problems.push("live workload not rendered");
if (!element("stat-elapsed")._text || element("stat-elapsed")._text === "—") problems.push("elapsed telemetry not rendered");
if (!element("stat-eta")._text) problems.push("ETA telemetry not rendered");
if (complete) {
  if (!element("case-table:tbody")._html.includes("<tr>")) problems.push("case table empty");
  if (!element("gates-list")._html.includes("artifact/completeness")) problems.push("gates not rendered");
  if (element("done-overlay").hidden) problems.push("completion overlay hidden");
} else if (!element("event-log").children.length) {
  problems.push("live event log empty");
}
if (problems.length) { console.error("FAIL\n" + problems.join("\n")); process.exit(1); }
const rows = (element("case-table:tbody")._html.match(/<tr>/g) || []).length;
console.log(`OK — S3.2 dashboard rendered ${rows} case row(s), live telemetry, and ${element("event-log").children.length} event-log row(s)`);
