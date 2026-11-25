// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
/* parciver · R-MAPSS §38 H-Q fragility→EV predictor — experiment dashboard client.
 *
 * Self-contained (does not share app.js). Connects to the same decoupled SSE transport
 * (/events) and renders §38-specific panels from the `stepq` / `stepq_final` checkpoint
 * payloads: the fragility→extractable-EV scatter (held highlighted), held-out |ρ| bars vs
 * the dist_eq baseline + the ρ* line, the ceiling-vs-realized-capture gap, the
 * cheap-vs-expensive-reference comparison, and the per-opponent table. Append-only store +
 * one rAF render loop with dirty flags.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function fmt(x, d = 2) { return (x == null || !isFinite(x)) ? "—" : (x >= 0 ? "+" : "") + x.toFixed(d); }
function fmtDur(s) { if (s == null || !isFinite(s)) return "—"; s = Math.max(0, Math.round(s)); const m = Math.floor(s / 60), sec = s % 60; return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`; }

function fitCanvas(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const r = cv.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
  if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}
const CSSVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = {};
for (const k of ["accent", "accent-2", "amber", "green", "raise", "call", "fold", "ink", "ink-2", "ink-3", "border", "border-2", "red", "panel"]) COL[k] = CSSVar("--" + k);
function hexToRgb(h) { h = (h || "#888").replace("#", ""); if (h.length === 3) h = h.split("").map(c => c + c).join(""); const n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
function rgba(rgb, a) { return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`; }
function roundRect(ctx, x, y, w, h, r) { r = Math.min(r, w / 2, h / 2); ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }

const SPLIT_COL = { fit: () => COL.accent, held: () => COL.amber, transfer: () => COL.green };

// ── store ─────────────────────────────────────────────────────────────────
const store = {
  config: null, rho_star: 0.70, delta_rho: 0.15, status: "connecting",
  startTs: null, lastElapsed: 0, lastWall: 0, ended: null,
  games: {},                 // game -> stepq payload
  order: [],                 // game arrival order
  view: null,                // currently displayed game
  final: null,               // combined verdict payload
  dirty: { header: true, scatter: true, rho: true, gap: true, cheap: true, table: true },

  apply(e) {
    switch (e.type) {
      case "run_start": {
        this.config = e.config || {};
        if (this.config.rho_star != null) this.rho_star = this.config.rho_star;
        if (this.config.delta_rho != null) this.delta_rho = this.config.delta_rho;
        this.status = "live"; this.startTs = e.ts;
        renderChips(); this.dirty.header = true;
        addLog(e.ts, "phase_change", `${e.game} · fragility→EV predictor started`);
        break;
      }
      case "metrics": {
        if (e.elapsed_s != null) { this.lastElapsed = e.elapsed_s; this.lastWall = performance.now(); this.dirty.header = true; }
        break;
      }
      case "checkpoint": {
        if (e.stepq) {
          const g = e.stepq.game;
          if (!(g in this.games)) this.order.push(g);
          this.games[g] = e.stepq;
          this.view = g;                 // show the freshest game
          buildGameSel();
          for (const k in this.dirty) this.dirty[k] = true;
        }
        if (e.stepq_final) { this.final = e.stepq_final; this.dirty.header = true; }
        break;
      }
      case "milestone": {
        addLog(e.ts, e.kind || "note", e.message || "");
        break;
      }
      case "run_end": {
        this.status = "done"; this.ended = e; this.dirty.header = true;
        showDone(e); addLog(e.ts, "phase_change", `run ended — ${e.reason}`);
        break;
      }
    }
  },
};

// ── header + chrome ─────────────────────────────────────────────────────────
function renderChips() {
  const c = $("config-chips"); if (!c || !store.config) return;
  const cfg = store.config;
  const chips = [`ρ*=${(cfg.rho_star ?? store.rho_star)}`, `Δρ=${(cfg.delta_rho ?? store.delta_rho)}`,
    `base=${cfg.baseline || "dist_eq"}`];
  c.innerHTML = chips.map(t => `<span class="chip-mini mono">${t}</span>`).join(" ");
}
function buildGameSel() {
  const el = $("game-sel"); if (!el) return;
  el.innerHTML = store.order.map(g =>
    `<button data-g="${g}" class="${g === store.view ? "on" : ""}">${g}</button>`).join("");
  el.querySelectorAll("button").forEach(b => b.onclick = () => {
    store.view = b.dataset.g; buildGameSel();
    for (const k in store.dirty) store.dirty[k] = true;
  });
}

function cur() { return store.view ? store.games[store.view] : null; }

function metricPasses(m, baseAbs) {
  if (m.is_baseline || m.is_ref) return false;
  const a = Math.abs(m.rho_held);
  const sign = Math.sign(m.rho_held || 1);
  const sstable = m.rho_fit === 0 || m.rho_held === 0 || Math.sign(m.rho_fit) === sign;
  const tstable = m.rho_transfer === 0 || m.rho_held === 0 || Math.sign(m.rho_transfer) === sign;
  return a >= store.rho_star && a - baseAbs >= store.delta_rho && sstable && tstable;
}
function featured(s) {
  if (!s) return null;
  const base = s.baseline_abs_rho_held ?? 0;
  const cheap = s.metrics.filter(m => !m.is_baseline && !m.is_ref);
  const passing = cheap.filter(m => metricPasses(m, base));
  const pool = passing.length ? passing : cheap;
  return pool.reduce((a, b) => (Math.abs(b.rho_held) > Math.abs((a || {}).rho_held ?? -9) ? b : a), null);
}

function renderHeader() {
  const s = cur();
  const D = store.dirty; if (!D.header) return; D.header = false;
  $("game-name").textContent = s ? `§38 fragility → EV predictor — ${s.game}` : "§38 fragility → EV predictor — waiting…";
  const nowEl = store.lastWall ? store.lastElapsed + (performance.now() - store.lastWall) / 1000 : store.lastElapsed;
  $("stat-elapsed").textContent = fmtDur(nowEl);
  const st = $("stat-status");
  st.innerHTML = store.status === "done" ? '<span class="dot ok"></span>complete'
    : store.status === "live" ? '<span class="dot live"></span>live' : '<span class="dot idle"></span>connecting';
  if (s) {
    $("stat-nc").textContent = (s.eq_nashconv ?? 0).toFixed(4);
    $("stat-nheld").textContent = s.n_held ?? "—";
    $("stat-baserho").textContent = fmt(s.baseline_abs_rho_held, 2);
    const f = featured(s);
    $("stat-bestrho").textContent = f ? fmt(Math.abs(f.rho_held), 2) : "—";
    $("stat-bestname").textContent = f ? f.name : "—";
  }
  // verdict: prefer the combined final; else the current game's
  const vEl = $("stat-verdict");
  let v = store.final ? store.final.verdict : (s ? s.verdict : "running");
  vEl.textContent = v || "—";
  vEl.className = /GO|PASS/.test(v) ? "v-go" : /KILL/.test(v) ? "v-kill" : /FLAT/.test(v) ? "v-flat" : "v-run";
  if (store.final && store.final.winners && store.final.winners.length)
    $("stat-role").textContent = "winners: " + store.final.winners.join(", ");
}

// ── panel 1: fragility → EV scatter ─────────────────────────────────────────
function axes(ctx, w, h, pad) {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = rgba(hexToRgb(COL.border), 0.9); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
}
function drawScatter() {
  const { ctx, w, h } = fitCanvas($("c-scatter"));
  const s = cur(); const f = featured(s);
  $("scatter-metric").textContent = f ? `${f.name}  (ρ_held ${fmt(f.rho_held, 2)})` : "—";
  const pad = { l: 52, r: 14, t: 14, b: 30 };
  axes(ctx, w, h, pad);
  if (!s || !f) return;
  const name = f.name;
  const rows = s.rows;
  const xs = rows.map(r => r.m[name]).filter(v => v != null);
  const ys = rows.map(r => r.y_ceil);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(0, ...ys), ymax = Math.max(...ys, 1e-6);
  const X = v => pad.l + (xmax > xmin ? (v - xmin) / (xmax - xmin) : 0.5) * (w - pad.l - pad.r);
  const Y = v => (h - pad.b) - (v - ymin) / (ymax - ymin) * (h - pad.t - pad.b);
  // y gridlines
  ctx.fillStyle = COL["ink-3"]; ctx.font = "10px ui-monospace,monospace"; ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) { const yv = ymin + (ymax - ymin) * i / 4; const py = Y(yv); ctx.strokeStyle = rgba(hexToRgb(COL.border), 0.4); ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(w - pad.r, py); ctx.stroke(); ctx.fillText(yv.toFixed(2), pad.l - 6, py + 3); }
  ctx.save(); ctx.translate(12, (h) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillStyle = COL["ink-2"]; ctx.fillText("Y_ceil (exact extractable EV)", 0, 0); ctx.restore();
  ctx.textAlign = "center"; ctx.fillStyle = COL["ink-2"]; ctx.fillText(name + " (cheap, BR-free)", (pad.l + w - pad.r) / 2, h - 8);
  // points
  for (const r of rows) {
    const v = r.m[name]; if (v == null) continue;
    const px = X(v), py = Y(r.y_ceil);
    const isEq = r.klass === "eq";
    const col = hexToRgb((SPLIT_COL[r.split] || (() => COL["ink-3"]))());
    ctx.beginPath(); ctx.arc(px, py, isEq ? 4 : 5, 0, 2 * Math.PI);
    if (isEq) { ctx.strokeStyle = rgba(hexToRgb(COL["ink-3"]), 0.9); ctx.lineWidth = 1.5; ctx.stroke(); }
    else { ctx.fillStyle = rgba(col, 0.85); ctx.fill(); }
  }
}

// ── panel 2: |ρ_held| bars per metric ───────────────────────────────────────
function drawRho() {
  const { ctx, w, h } = fitCanvas($("c-rho"));
  const s = cur();
  const pad = { l: 40, r: 12, t: 14, b: 54 };
  ctx.clearRect(0, 0, w, h);
  if (!s) return;
  const base = s.baseline_abs_rho_held ?? 0;
  const ms = s.metrics.slice();     // include baseline + ref
  const n = ms.length;
  const bw = (w - pad.l - pad.r) / n * 0.62;
  const gap = (w - pad.l - pad.r) / n;
  const ymax = 1.0;
  const Y = v => (h - pad.b) - v / ymax * (h - pad.t - pad.b);
  // Δρ band above baseline, and ρ* line
  ctx.fillStyle = rgba(hexToRgb(COL.green), 0.08);
  ctx.fillRect(pad.l, Y(Math.min(1, base + store.delta_rho)), w - pad.l - pad.r, Y(base) - Y(Math.min(1, base + store.delta_rho)));
  for (const [val, col, lab] of [[store.rho_star, COL.green, "ρ*"], [base, COL["ink-2"], "dist_eq"]]) {
    ctx.strokeStyle = rgba(hexToRgb(col), 0.8); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(pad.l, Y(val)); ctx.lineTo(w - pad.r, Y(val)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = rgba(hexToRgb(col), 0.9); ctx.font = "9px ui-monospace,monospace"; ctx.textAlign = "left"; ctx.fillText(lab, pad.l + 2, Y(val) - 3);
  }
  ctx.font = "10px ui-monospace,monospace";
  ms.forEach((m, i) => {
    const a = Math.abs(m.rho_held);
    const x = pad.l + i * gap + (gap - bw) / 2;
    let col = COL["ink-3"];
    if (m.is_ref) col = COL["accent-2"];
    else if (metricPasses(m, base)) col = COL.green;
    else if (!m.is_baseline) col = COL.amber;
    const y = Y(a), bh = (h - pad.b) - y;
    ctx.fillStyle = rgba(hexToRgb(col), m.is_baseline ? 0.4 : 0.85);
    roundRect(ctx, x, y, bw, Math.max(1, bh), 3); ctx.fill();
    ctx.fillStyle = COL.ink; ctx.textAlign = "center"; ctx.fillText(fmt(m.rho_held, 2), x + bw / 2, y - 4);
    // label (rotated)
    ctx.save(); ctx.translate(x + bw / 2, h - pad.b + 6); ctx.rotate(-Math.PI / 5); ctx.textAlign = "right";
    ctx.fillStyle = m.is_baseline ? COL["ink-3"] : (m.is_ref ? COL["accent-2"] : COL["ink-2"]);
    ctx.fillText(m.name.replace("*", "") + (m.is_ref ? "*" : ""), 0, 0); ctx.restore();
  });
}

// ── panel 3: ceiling vs realized capture (held) ─────────────────────────────
function drawGap() {
  const { ctx, w, h } = fitCanvas($("c-gap"));
  const s = cur();
  ctx.clearRect(0, 0, w, h);
  if (!s) return;
  const held = s.rows.filter(r => r.split === "held" && r.klass !== "eq");
  const pad = { l: 34, r: 10, t: 12, b: 46 };
  const n = Math.max(1, held.length);
  const gap = (w - pad.l - pad.r) / n; const bw = gap * 0.36;
  const ymax = Math.max(1e-6, ...held.map(r => Math.max(r.y_ceil, r.y_cap)));
  const Y = v => (h - pad.b) - clamp(v, 0, ymax) / ymax * (h - pad.t - pad.b);
  ctx.strokeStyle = rgba(hexToRgb(COL.border), 0.6); ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
  let capSum = 0, ceilSum = 0;
  held.forEach((r, i) => {
    capSum += Math.max(0, r.y_cap); ceilSum += r.y_ceil;
    const x0 = pad.l + i * gap + (gap - 2 * bw - 3) / 2;
    ctx.fillStyle = rgba(hexToRgb(COL["accent-2"]), 0.85); roundRect(ctx, x0, Y(r.y_ceil), bw, (h - pad.b) - Y(r.y_ceil), 2); ctx.fill();
    ctx.fillStyle = rgba(hexToRgb(COL.green), 0.85); roundRect(ctx, x0 + bw + 3, Y(Math.max(0, r.y_cap)), bw, (h - pad.b) - Y(Math.max(0, r.y_cap)), 2); ctx.fill();
    ctx.save(); ctx.translate(x0 + bw, h - pad.b + 5); ctx.rotate(-Math.PI / 5); ctx.textAlign = "right"; ctx.fillStyle = COL["ink-3"]; ctx.font = "9px ui-monospace,monospace"; ctx.fillText(r.label.slice(0, 12), 0, 0); ctx.restore();
  });
  $("kpi-gap").textContent = ceilSum > 0 ? `captured ${(100 * capSum / ceilSum).toFixed(0)}% of ceiling` : "—";
  ctx.fillStyle = COL["accent-2"]; ctx.font = "10px ui-monospace,monospace"; ctx.textAlign = "left"; ctx.fillText("■ Y_ceil", pad.l, pad.t + 4);
  ctx.fillStyle = COL.green; ctx.fillText("■ Y_cap", pad.l + 60, pad.t + 4);
}

// ── panel 4: cheap vs expensive vs baseline ─────────────────────────────────
function drawCheap() {
  const { ctx, w, h } = fitCanvas($("c-cheap"));
  const s = cur();
  ctx.clearRect(0, 0, w, h);
  if (!s) return;
  const f = featured(s);
  const ref = s.metrics.find(m => m.is_ref);
  const items = [
    { l: f ? f.name : "cheap", v: f ? Math.abs(f.rho_held) : 0, c: COL.green },
    { l: ref ? ref.name.replace("*", "*") : "nashconv_opp*", v: ref ? Math.abs(ref.rho_held) : 0, c: COL["accent-2"] },
    { l: "dist_eq", v: s.baseline_abs_rho_held ?? 0, c: COL["ink-3"] },
  ];
  const pad = { l: 40, r: 12, t: 14, b: 40 };
  const gap = (w - pad.l - pad.r) / items.length; const bw = gap * 0.5;
  const Y = v => (h - pad.b) - v * (h - pad.t - pad.b);
  ctx.strokeStyle = rgba(hexToRgb(COL.green), 0.7); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(pad.l, Y(store.rho_star)); ctx.lineTo(w - pad.r, Y(store.rho_star)); ctx.stroke(); ctx.setLineDash([]);
  items.forEach((it, i) => {
    const x = pad.l + i * gap + (gap - bw) / 2;
    ctx.fillStyle = rgba(hexToRgb(it.c), 0.85); roundRect(ctx, x, Y(it.v), bw, (h - pad.b) - Y(it.v), 3); ctx.fill();
    ctx.fillStyle = COL.ink; ctx.font = "11px ui-monospace,monospace"; ctx.textAlign = "center"; ctx.fillText(it.v.toFixed(2), x + bw / 2, Y(it.v) - 4);
    ctx.fillStyle = COL["ink-2"]; ctx.font = "10px ui-monospace,monospace"; ctx.fillText(it.l, x + bw / 2, h - pad.b + 16);
  });
}

// ── panel 5: per-opponent table ─────────────────────────────────────────────
function renderTable() {
  const s = cur(); if (!s) return;
  const metricNames = s.metrics.map(m => m.name);
  const thead = document.querySelector("#qtab thead");
  const tbody = document.querySelector("#qtab tbody");
  thead.innerHTML = "<tr><th>opponent</th><th>split</th>" +
    metricNames.map(n => `<th>${n.replace("*", "*")}</th>`).join("") +
    "<th>Y_ceil</th><th>Y_cap</th></tr>";
  tbody.innerHTML = s.rows.map(r => {
    const cls = r.klass === "eq" ? "eqrow" : r.split;
    const mcells = metricNames.map(n => `<td>${(r.m[n] ?? 0).toFixed(3)}</td>`).join("");
    return `<tr class="${cls}"><td>${r.label}</td><td>${r.split}</td>${mcells}<td>${fmt(r.y_ceil, 3)}</td><td>${fmt(r.y_cap, 3)}</td></tr>`;
  }).join("");
}

// ── event log + done overlay ────────────────────────────────────────────────
const LOG_KIND = { gate: "ok", warning: "warn", phase_change: "phase", milestone: "note", note: "note" };
function addLog(ts, kind, msg) {
  const ul = $("event-log"); if (!ul) return;
  const li = document.createElement("li"); li.className = "log-row " + (LOG_KIND[kind] || "note");
  li.innerHTML = `<span class="lk">${kind}</span><span class="lm">${msg}</span>`;
  ul.prepend(li);
  while (ul.children.length > 140) ul.removeChild(ul.lastChild);
}
function showDone(e) {
  const ov = $("done-overlay"); const sum = (e.summary || {});
  const v = sum.verdict || "—";
  $("done-title").textContent = /GO/.test(v) ? "H-Q GO — fragility predicts EV" : /KILL/.test(v) ? "H-Q KILL" : "H-Q complete";
  $("done-badge").textContent = /GO/.test(v) ? "✓" : /KILL/.test(v) ? "✕" : "✓";
  $("done-badge").style.color = /GO/.test(v) ? COL.green : /KILL/.test(v) ? COL.red : COL.accent;
  $("done-sub").textContent = (sum.winners && sum.winners.length) ? ("winning metric(s): " + sum.winners.join(", ")) : v;
  const stats = $("done-stats"); stats.innerHTML = "";
  for (const g of store.order) {
    const s = store.games[g]; if (!s) continue;
    const f = featured(s);
    const box = document.createElement("div"); box.className = "done-stat";
    box.innerHTML = `<div class="ds-label">${g}</div><div class="ds-value mono">${s.verdict}</div>` +
      `<div class="ds-sub mono">${f ? f.name + " " + fmt(f.rho_held, 2) : ""} · base ${fmt(s.baseline_abs_rho_held, 2)}</div>`;
    stats.appendChild(box);
  }
  ov.hidden = false;
}
$("done-dismiss").addEventListener("click", () => { $("done-overlay").hidden = true; });
$("done-overlay").addEventListener("click", (e) => { if (e.target === $("done-overlay")) $("done-overlay").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("done-overlay").hidden = true; });

// ── render loop ─────────────────────────────────────────────────────────────
function frame() {
  const D = store.dirty;
  renderHeader();
  if (D.scatter) { drawScatter(); D.scatter = false; }
  if (D.rho) { drawRho(); D.rho = false; }
  if (D.gap) { drawGap(); D.gap = false; }
  if (D.cheap) { drawCheap(); D.cheap = false; }
  if (D.table) { renderTable(); D.table = false; }
  requestAnimationFrame(frame);
}
let resizeRAF = 0;
window.addEventListener("resize", () => { cancelAnimationFrame(resizeRAF); resizeRAF = requestAnimationFrame(() => { for (const k in store.dirty) store.dirty[k] = true; }); });

function connect() {
  const es = new EventSource("/events");
  es.onmessage = (ev) => { let b; try { b = JSON.parse(ev.data); } catch (_) { return; } for (const e of b) { try { store.apply(e); } catch (err) { console.error("apply", err, e); } } };
  es.onopen = () => { if (store.status === "connecting") $("stat-status").innerHTML = '<span class="dot idle"></span>waiting for events'; };
}
connect();
requestAnimationFrame(frame);
