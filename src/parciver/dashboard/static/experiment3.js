// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
/* parciver · R-MAPSS S3.1 HU exploitation spine (nlhe_small) — experiment dashboard client.
 *
 * Self-contained (does not share app.js). Connects to the same decoupled SSE transport
 * (/events) and renders S3.1-specific panels from the `step3_meta` / `step3_row` / `step3`
 * checkpoint payloads: the captured-vs-available-EV exact-BR frontier (held highlighted, the
 * hidden board-blind boundary case marked), the α schedule (α_conf vs α_safe), the
 * safety-budget headroom (exact exploitability increase vs the budget line), the pre-registered
 * gates, and the per-opponent table. Append-only store + one rAF render loop with dirty flags.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function fmt(x, d = 3) { return (x == null || !isFinite(x)) ? "—" : (x >= 0 ? "+" : "") + x.toFixed(d); }
function fmtu(x, d = 3) { return (x == null || !isFinite(x)) ? "—" : x.toFixed(d); }
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
function roundRect(ctx, x, y, w, h, r) { r = Math.min(r, w / 2, Math.abs(h) / 2); ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }

// color per opponent by class/split
function colFor(r) {
  if (r.klass === "equilibrium") return COL["ink-3"];
  if (r.klass === "hidden") return COL["accent-2"];
  return r.split === "fit" ? COL.accent : COL.amber;
}
function isEq(r) { return r.klass === "equilibrium"; }

// ── store ─────────────────────────────────────────────────────────────────
const store = {
  config: null, status: "connecting",
  startTs: null, lastElapsed: 0, lastWall: 0, ended: null,
  meta: null,                // thresholds payload
  rows: {},                  // label -> row
  order: [],                 // label arrival order
  gates: null, verdict: "running", detail: null, allPass: null,
  dirty: { header: true, frontier: true, alpha: true, safety: true, gates: true, table: true },

  markAll() { for (const k in this.dirty) this.dirty[k] = true; },
  putRow(r) { if (!(r.label in this.rows)) this.order.push(r.label); this.rows[r.label] = r; },

  apply(e) {
    switch (e.type) {
      case "run_start":
        this.config = e.config || {}; this.status = "live"; this.startTs = e.ts;
        renderChips(); this.dirty.header = true;
        addLog(e.ts, "phase_change", `${e.game} · HU exploitation spine started`);
        break;
      case "metrics":
        if (e.elapsed_s != null) { this.lastElapsed = e.elapsed_s; this.lastWall = performance.now(); this.dirty.header = true; }
        break;
      case "checkpoint":
        if (e.step3_meta) { this.meta = e.step3_meta; this.markAll(); }
        if (e.step3_row) { this.putRow(e.step3_row); this.markAll(); }
        if (e.step3) {
          const s = e.step3;
          this.rows = {}; this.order = [];
          for (const r of (s.rows || [])) this.putRow(r);
          this.gates = s.gates || null; this.verdict = s.verdict || this.verdict;
          this.detail = s.detail || null; this.allPass = s.all_pass;
          this.meta = { scale_S: s.scale_S, base_expl: s.base_expl, delta_sub: s.delta_sub, tol_eq: s.tol_eq, budget: s.budget, kappa: s.kappa };
          this.markAll();
        }
        break;
      case "milestone":
        addLog(e.ts, e.kind || "note", e.message || "");
        break;
      case "run_end":
        this.status = "done"; this.ended = e; this.markAll();
        showDone(e); addLog(e.ts, "phase_change", `run ended — ${e.reason}`);
        break;
    }
  },
};

function rowsArr() { return store.order.map(l => store.rows[l]); }
function heldSub() { return rowsArr().filter(r => r.klass === "suboptimal" && r.split === "held"); }
function meanHeldGain() {
  if (store.detail && store.detail["mean exploit gain (suboptimal HELD-OUT)"] != null)
    return store.detail["mean exploit gain (suboptimal HELD-OUT)"];
  const h = heldSub(); return h.length ? h.reduce((a, r) => a + r.exploit_gain, 0) / h.length : null;
}

// ── header + chrome ─────────────────────────────────────────────────────────
function renderChips() {
  const c = $("config-chips"); if (!c || !store.config) return;
  const cfg = store.config;
  const chips = [cfg.protocol || "historical exploratory", `seed=${cfg.anchor_seed ?? 0}`,
    `κ=${cfg.kappa ?? 0.35}`, `anchor=${((cfg.anchor_iters ?? 0) / 1e6).toFixed(1)}M`,
    `c_sub=${cfg.c_sub ?? "—"}`];
  c.innerHTML = chips.map(t => `<span class="chip-mini mono">${t}</span>`).join(" ");
}
function renderHeader() {
  if (!store.dirty.header) return; store.dirty.header = false;
  $("game-name").textContent = "S3.1 · HU exploitation spine — nlhe_small";
  const nowEl = store.lastWall ? store.lastElapsed + (performance.now() - store.lastWall) / 1000 : store.lastElapsed;
  $("stat-elapsed").textContent = fmtDur(nowEl);
  const st = $("stat-status");
  st.innerHTML = store.status === "done" ? '<span class="dot ok"></span>complete'
    : store.status === "live" ? '<span class="dot live"></span>live' : '<span class="dot idle"></span>connecting';
  const m = store.meta;
  if (m) {
    $("stat-S").textContent = fmtu(m.scale_S, 3);
    $("stat-thr").textContent = `${fmtu(m.delta_sub, 3)} / ${fmtu(m.tol_eq, 3)} / ${fmtu(m.budget, 3)}`;
    $("stat-expl").textContent = fmtu(m.base_expl, 4);
  }
  const g = meanHeldGain();
  $("stat-gain").textContent = fmt(g, 3);
  const h = heldSub();
  const ceil = h.length ? h.reduce((a, r) => a + r.ceiling_gain, 0) / h.length : null;
  $("stat-ceil").textContent = fmt(ceil, 3);
  const vEl = $("stat-verdict");
  const v = store.verdict || "running";
  const passed = /SHIP|^PASS/.test(v);
  const confirmation = store.config && store.config.protocol === "confirmation-v1";
  vEl.textContent = passed ? (confirmation ? "SEED PASS" : "EXPLORATORY") : /FAIL/.test(v) ? "FAIL" : "running";
  vEl.className = passed ? "v-ship" : /FAIL/.test(v) ? "v-fail" : "v-run";
  $("stat-role").textContent = passed
    ? (confirmation ? "confirmation seed passed — aggregate verdict requires all three seeds"
                    : "exploratory gates passed — historical criteria were revised; confirmation pending")
    : "confidence-scheduled, safety-gated single-opponent exploitation · exact controls";
}

// ── panel 1: captured vs available EV (exact-BR frontier) ────────────────────
function axes(ctx, w, h, pad) {
  ctx.strokeStyle = rgba(hexToRgb(COL.border), 0.9); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
}
function drawFrontier() {
  const { ctx, w, h } = fitCanvas($("c-frontier"));
  ctx.clearRect(0, 0, w, h);
  const rows = rowsArr(); const pad = { l: 54, r: 16, t: 16, b: 34 };
  axes(ctx, w, h, pad);
  if (!rows.length) return;
  const xmax = Math.max(0.1, ...rows.map(r => r.ceiling_gain));
  const ymin = Math.min(0, ...rows.map(r => r.exploit_gain));
  const ymax = Math.max(0.05, ...rows.map(r => r.exploit_gain), xmax * 0.25);
  const X = v => pad.l + clamp(v / xmax, 0, 1) * (w - pad.l - pad.r);
  const Y = v => (h - pad.b) - (v - ymin) / (ymax - ymin) * (h - pad.t - pad.b);
  // gridlines
  ctx.fillStyle = COL["ink-3"]; ctx.font = "10px ui-monospace,monospace";
  for (let i = 0; i <= 4; i++) { const yv = ymin + (ymax - ymin) * i / 4; const py = Y(yv); ctx.strokeStyle = rgba(hexToRgb(COL.border), 0.35); ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(w - pad.r, py); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(yv.toFixed(2), pad.l - 6, py + 3); }
  // y=x diagonal (capture everything)
  ctx.strokeStyle = rgba(hexToRgb(COL["ink-3"]), 0.5); ctx.setLineDash([4, 4]); ctx.beginPath();
  ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(Math.min(xmax, ymax)), Y(Math.min(xmax, ymax))); ctx.stroke(); ctx.setLineDash([]);
  // delta_sub line
  if (store.meta) {
    const yl = Y(store.meta.delta_sub);
    ctx.strokeStyle = rgba(hexToRgb(COL.green), 0.8); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(pad.l, yl); ctx.lineTo(w - pad.r, yl); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = rgba(hexToRgb(COL.green), 0.9); ctx.textAlign = "left"; ctx.fillText("δ_sub", pad.l + 3, yl - 3);
  }
  // axis labels
  ctx.save(); ctx.translate(13, h / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillStyle = COL["ink-2"]; ctx.fillText("captured gain (deployable)", 0, 0); ctx.restore();
  ctx.textAlign = "center"; ctx.fillStyle = COL["ink-2"]; ctx.fillText("available exact-BR ceiling gain", (pad.l + w - pad.r) / 2, h - 6);
  // points
  for (const r of rows) {
    const px = X(r.ceiling_gain), py = Y(r.exploit_gain);
    const col = hexToRgb(colFor(r));
    ctx.beginPath(); ctx.arc(px, py, isEq(r) ? 4 : 5.5, 0, 2 * Math.PI);
    if (isEq(r)) { ctx.strokeStyle = rgba(hexToRgb(COL["ink-3"]), 0.9); ctx.lineWidth = 1.5; ctx.stroke(); }
    else { ctx.fillStyle = rgba(col, 0.85); ctx.fill(); }
    if (r.klass === "hidden") { ctx.strokeStyle = rgba(col, 0.9); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(px, py, 9, 0, 2 * Math.PI); ctx.stroke(); }
  }
}

// ── panel 2: the α schedule (alpha used vs conf vs safe) ─────────────────────
function drawAlpha() {
  const { ctx, w, h } = fitCanvas($("c-alpha"));
  ctx.clearRect(0, 0, w, h);
  const rows = rowsArr(); const pad = { l: 34, r: 12, t: 14, b: 60 };
  if (!rows.length) return;
  const n = rows.length; const gap = (w - pad.l - pad.r) / n; const bw = gap * 0.5;
  const Y = v => (h - pad.b) - clamp(v, 0, 1) * (h - pad.t - pad.b);
  // 0..1 gridlines
  ctx.fillStyle = COL["ink-3"]; ctx.font = "9px ui-monospace,monospace"; ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) { const yv = i / 4; const py = Y(yv); ctx.strokeStyle = rgba(hexToRgb(COL.border), 0.3); ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(w - pad.r, py); ctx.stroke(); ctx.fillText(yv.toFixed(1), pad.l - 4, py + 3); }
  rows.forEach((r, i) => {
    const cx = pad.l + i * gap + gap / 2;
    const x = cx - bw / 2;
    // alpha used bar
    ctx.fillStyle = rgba(hexToRgb(COL.accent), 0.85); roundRect(ctx, x, Y(r.alpha), bw, (h - pad.b) - Y(r.alpha), 3); ctx.fill();
    // alpha_conf marker (amber) and alpha_safe marker (red)
    for (const [val, c] of [[r.alpha_conf, COL.amber], [r.alpha_safe, COL.red]]) {
      ctx.strokeStyle = rgba(hexToRgb(c), 0.95); ctx.lineWidth = 2; ctx.beginPath();
      ctx.moveTo(cx - bw / 2 - 2, Y(val)); ctx.lineTo(cx + bw / 2 + 2, Y(val)); ctx.stroke();
    }
    // label
    ctx.save(); ctx.translate(cx, h - pad.b + 6); ctx.rotate(-Math.PI / 4); ctx.textAlign = "right";
    ctx.fillStyle = colFor(r); ctx.font = "9px ui-monospace,monospace"; ctx.fillText(r.label.slice(0, 13), 0, 0); ctx.restore();
  });
}

// ── panel 3: safety-budget headroom (Δ exploitability vs budget) ─────────────
function drawSafety() {
  const { ctx, w, h } = fitCanvas($("c-safety"));
  ctx.clearRect(0, 0, w, h);
  const rows = rowsArr(); const pad = { l: 44, r: 12, t: 12, b: 56 };
  if (!rows.length || !store.meta) return;
  const budget = store.meta.budget || 1e-6;
  const ymax = Math.max(budget * 1.25, ...rows.map(r => r.dexpl || 0));
  const n = rows.length; const gap = (w - pad.l - pad.r) / n; const bw = gap * 0.55;
  const Y = v => (h - pad.b) - clamp(v, 0, ymax) / ymax * (h - pad.t - pad.b);
  ctx.strokeStyle = rgba(hexToRgb(COL.border), 0.6); ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
  let over = 0, maxd = 0;
  rows.forEach((r, i) => {
    const d = Math.max(0, r.dexpl || 0); maxd = Math.max(maxd, d);
    const bad = r.dexpl > budget + 1e-9; if (bad) over++;
    const x = pad.l + i * gap + (gap - bw) / 2;
    ctx.fillStyle = rgba(hexToRgb(bad ? COL.red : COL.green), 0.8); roundRect(ctx, x, Y(d), bw, (h - pad.b) - Y(d), 2); ctx.fill();
    ctx.save(); ctx.translate(x + bw / 2, h - pad.b + 5); ctx.rotate(-Math.PI / 4); ctx.textAlign = "right"; ctx.fillStyle = COL["ink-3"]; ctx.font = "9px ui-monospace,monospace"; ctx.fillText(r.label.slice(0, 12), 0, 0); ctx.restore();
  });
  // budget line
  const yb = Y(budget);
  ctx.strokeStyle = rgba(hexToRgb(COL.amber), 0.9); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(pad.l, yb); ctx.lineTo(w - pad.r, yb); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = rgba(hexToRgb(COL.amber), 0.95); ctx.textAlign = "left"; ctx.font = "9px ui-monospace,monospace"; ctx.fillText("budget " + budget.toFixed(3), pad.l + 3, yb - 3);
  // y ticks
  ctx.fillStyle = COL["ink-3"]; ctx.textAlign = "right";
  for (let i = 0; i <= 2; i++) { const yv = ymax * i / 2; ctx.fillText(yv.toFixed(3), pad.l - 4, Y(yv) + 3); }
  $("kpi-safety").textContent = over === 0 ? `all within budget (max ${maxd.toFixed(4)})` : `${over} OVER budget`;
  $("kpi-safety").style.color = over === 0 ? COL.green : COL.red;
}

// ── panel 4: gates + KPIs ────────────────────────────────────────────────────
function renderGates() {
  const g = meanHeldGain();
  $("k-gain").textContent = fmt(g, 3);
  const eqs = rowsArr().filter(r => r.klass === "equilibrium");
  $("k-eq").textContent = eqs.length ? fmtu(Math.max(...eqs.map(r => Math.abs(r.exploit_gain))), 3) : "—";
  const md = rowsArr().length ? Math.max(...rowsArr().map(r => r.dexpl || 0)) : null;
  $("k-dexpl").textContent = md == null ? "—" : fmtu(md, 4);
  const ul = $("gates-list"); if (!ul) return;
  const entries = store.gates ? Object.entries(store.gates)
    : ["safety: exact exploitability within budget (all opponents)",
       "beats anchor vs suboptimal HELD-OUT (> delta_sub)",
       "matches anchor vs near-eq (|gain| <= tol)",
       "matches hole-blind model boundary (|gain| <= tol)",
       "observed-model BR <= exact BR ceiling (no hallucination, §28)"].map(k => [k, null]);
  ul.innerHTML = entries.map(([k, v]) => {
    const cls = v === true ? "ok" : v === false ? "no" : "wait";
    const sym = v === true ? "✓" : v === false ? "✕" : "·";
    return `<li><span class="b ${cls}">${sym}</span><span>${k}</span></li>`;
  }).join("");
}

// ── panel 5: per-opponent table ──────────────────────────────────────────────
function renderTable() {
  const rows = rowsArr();
  const thead = document.querySelector("#qtab thead");
  const tbody = document.querySelector("#qtab tbody");
  thead.innerHTML = "<tr><th>opponent</th><th>class</th><th>split</th><th>ev_a0</th><th>ceil</th>" +
    "<th>ev_a2</th><th>final</th><th>gain</th><th>cgain</th><th>α</th><th>α_conf</th><th>α_safe</th><th>signal</th><th>Δexpl</th></tr>";
  tbody.innerHTML = rows.map(r => {
    const cls = r.klass === "equilibrium" ? "eqrow" : r.klass === "hidden" ? "hiddenrow" : r.split;
    const dbad = (store.meta && r.dexpl > store.meta.budget + 1e-9) ? " bad" : "";
    return `<tr class="${cls}"><td>${r.label}</td><td>${r.klass}</td><td>${r.split}</td>` +
      `<td>${fmtu(r.ev_a0, 3)}</td><td>${fmtu(r.ev_ceiling, 3)}</td><td>${fmtu(r.ev_a2, 3)}</td>` +
      `<td>${fmtu(r.ev_final, 3)}</td><td>${fmt(r.exploit_gain, 3)}</td><td>${fmtu(r.ceiling_gain, 3)}</td>` +
      `<td>${fmtu(r.alpha, 2)}</td><td>${fmtu(r.alpha_conf, 2)}</td><td>${fmtu(r.alpha_safe, 2)}</td>` +
      `<td>${fmtu(r.signal, 3)}</td><td class="${dbad.trim()}">${fmt(r.dexpl, 4)}</td></tr>`;
  }).join("");
}

// ── event log + done overlay ────────────────────────────────────────────────
const LOG_KIND = { gate: "ok", warning: "warn", phase_change: "phase", milestone: "note", note: "note", opponent: "note" };
function addLog(ts, kind, msg) {
  const ul = $("event-log"); if (!ul) return;
  const li = document.createElement("li"); li.className = "log-row " + (LOG_KIND[kind] || "note");
  li.innerHTML = `<span class="lk">${kind}</span><span class="lm">${msg}</span>`;
  ul.prepend(li);
  while (ul.children.length > 160) ul.removeChild(ul.lastChild);
}
function showDone(e) {
  const ov = $("done-overlay"); const sum = (e.summary || {});
  const ship = sum.all_pass === true || /SHIP/.test(sum.verdict || "");
  const confirmation = store.config && store.config.protocol === "confirmation-v1";
  $("done-title").textContent = ship
    ? (confirmation ? "S3.1 confirmation seed PASS" : "S3.1 exploratory PASS — confirmation pending")
    : "S3.1 FAIL — gate breach";
  $("done-badge").textContent = ship ? "✓" : "✕";
  $("done-badge").style.color = ship ? COL.green : COL.red;
  $("done-sub").textContent = ship && !confirmation
    ? "Historical SHIP labels are exploratory evidence. A frozen protocol and fresh opponents are required before scaling."
    : (sum.verdict || "");
  const stats = $("done-stats"); stats.innerHTML = "";
  const items = [
    ["mean held-out gain", fmt(meanHeldGain(), 3)],
    ["scale S", fmtu(store.meta ? store.meta.scale_S : null, 3)],
    ["anchor exploitability", fmtu(store.meta ? store.meta.base_expl : null, 4)],
    ["max Δexpl", fmtu(rowsArr().length ? Math.max(...rowsArr().map(r => r.dexpl || 0)) : null, 4)],
  ];
  for (const [l, v] of items) {
    const box = document.createElement("div"); box.className = "done-stat";
    box.innerHTML = `<div class="ds-label">${l}</div><div class="ds-value mono">${v}</div>`;
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
  if (D.frontier) { drawFrontier(); D.frontier = false; }
  if (D.alpha) { drawAlpha(); D.alpha = false; }
  if (D.safety) { drawSafety(); D.safety = false; }
  if (D.gates) { renderGates(); D.gates = false; }
  if (D.table) { renderTable(); D.table = false; }
  requestAnimationFrame(frame);
}
let resizeRAF = 0;
window.addEventListener("resize", () => { cancelAnimationFrame(resizeRAF); resizeRAF = requestAnimationFrame(() => store.markAll()); });

function connect() {
  const es = new EventSource("/events");
  es.onmessage = (ev) => { let b; try { b = JSON.parse(ev.data); } catch (_) { return; } for (const e of b) { try { store.apply(e); } catch (err) { console.error("apply", err, e); } } };
  es.onopen = () => { if (store.status === "connecting") $("stat-status").innerHTML = '<span class="dot idle"></span>waiting for events'; };
}
connect();
requestAnimationFrame(frame);
