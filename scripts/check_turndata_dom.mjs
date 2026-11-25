// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
/* Headless smoke test for the turn-data (D1) watcher page: replays an
   events.jsonl through turndata.js against a stand-in DOM and canvas, and
   checks that every panel rendered without throwing.
   usage: node scripts/check_turndata_dom.mjs runs/watch-turn-data-<time>/events.jsonl */
import fs from "node:fs"; import path from "node:path"; import vm from "node:vm"; import { fileURLToPath } from "node:url";
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const STATIC = path.join(ROOT, "src", "parciver", "dashboard", "static");
const logPath = process.argv[2];
if (!logPath || !fs.existsSync(logPath)) { console.error("missing events.jsonl"); process.exit(2); }
const html = fs.readFileSync(path.join(STATIC, "turndata.html"), "utf8");
const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
const elements = new Map();
let calls = 0, nonFinite = 0;
function context() {  // a 2D context that accepts everything and flags non-finite coordinates
  return new Proxy({}, {
    get(t, k) { return k in t ? t[k] : (...args) => { calls += 1; if (args.some((a) => typeof a === "number" && !Number.isFinite(a))) nonFinite += 1; }; },
    set(t, k, v) { t[k] = v; return true; },
  });
}
function element(id) {
  if (elements.has(id)) return elements.get(id);
  const e = { id, tagName: "DIV", children: [], className: "", hidden: false, title: "", _text: "", _html: "", style: {},
    prepend(x) { this.children.unshift(x); }, querySelector(s) { return element(`${id}:${s}`); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 900, height: 300 }; }, getContext() { return context(); },
    set textContent(v) { this._text = String(v); }, get textContent() { return this._text; },
    set innerHTML(v) { this._html = String(v); }, get innerHTML() { return this._html; } };
  elements.set(id, e); return e;
}
let source;
class EventSource { constructor() { source = this; } close() {} }
const sandbox = { console, Date, Math, Number, String, JSON, Map, Set, Array, Object, Infinity, NaN, EventSource,
  document: { getElementById: element, createElement: (t) => element(`new:${t}:${Math.random()}`), documentElement: {}, addEventListener: () => {} },
  window: { addEventListener: () => {}, devicePixelRatio: 1 }, getComputedStyle: () => ({ getPropertyValue: () => "#888" }),
  requestAnimationFrame: (f) => f(), setInterval: () => 0 };
sandbox.globalThis = sandbox;
try {
  vm.runInNewContext(fs.readFileSync(path.join(STATIC, "turndata.js"), "utf8"), sandbox, { filename: "turndata.js" });
  source.onmessage({ data: JSON.stringify(fs.readFileSync(logPath, "utf8").split("\n").filter(Boolean).map(JSON.parse)) });
} catch (error) { console.error(error); process.exit(1); }
const required = ["sub", "stat-rows", "stat-expl", "stat-eta", "stat-pace", "stat-res", "stat-status", "run-label", "run-bar", "strip",
  "tiles", "chart-shards", "chart-hist", "hist-note", "chart-spr", "chart-prec", "verdicts", "chart-conv", "chart-pace", "pace-kpi",
  "worst", "event-log", "done-overlay"];
const problems = required.filter((id) => !ids.has(id)).map((id) => `missing #${id} in turndata.html`);
const text = (id) => element(id)._text, inner = (id) => element(id)._html;
if (!text("stat-rows-sub").includes("shards")) problems.push("progress not rendered");
if (!/%/.test(inner("stat-expl"))) problems.push("exploitability not rendered");
if (!(inner("tiles").match(/class="tile/g) || []).length) problems.push("tiles not rendered");
if (!(inner("worst:tbody").match(/<tr>/g) || []).length) problems.push("worst rows not rendered");
if (!text("stat-eta")) problems.push("clock not rendered");
if (nonFinite) problems.push(`${nonFinite} canvas call(s) with a non-finite number`);
// Data reaches the page escaped: tags the page never writes (a check file's `"j": "<b>7</b>"`) must not appear.
for (const [id, e] of elements) if (/<(b|i|x|\d+)>/.test(e._html)) problems.push(`unescaped data in #${id}`);
if (problems.length) { console.error(problems.join("\n")); process.exit(1); }
if (process.env.SHOW) {  // the rendered text, tags stripped
  const strip = (s) => s.replace(/<\/(div|tr|li)>/g, "\n").replace(/<[^>]+>/g, " ").replace(/[ \t]+/g, " ").trim();
  for (const id of ["sub", "stat-rows-sub", "stat-expl", "stat-expl-sub", "stat-eta", "stat-pace", "stat-res", "stat-status", "tiles", "hist-note", "pace-kpi", "verdicts", "worst:tbody"]) {
    console.log(`--- ${id}\n${strip(inner(id) || text(id))}`);
  }
}
console.log(`OK — turn-data page: ${(inner("worst:tbody").match(/<tr>/g) || []).length} worst rows, ` +
  `${(inner("verdicts").match(/class="verdict"/g) || []).length} verdicts, ${calls} canvas calls; ${text("stat-rows-sub")}`);
