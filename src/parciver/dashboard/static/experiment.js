// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
/* parciver · R-MAPSS §35.A rich-bait decider — experiment dashboard client.
 *
 * Self-contained (does not share app.js). Connects to the same decoupled SSE
 * transport (/events) and renders §35.A-specific panels from the experiment
 * telemetry: the EV-vs-exact-BR frontier scatter, 1-step headroom vs δ per family,
 * the search-tier ablation, the future-exploitability screen, live search progress,
 * and — when headroom appears and the run escalates — the multiplayer A5−A4
 * decomposition. Append-only store + one rAF render loop with dirty flags, so a
 * fast producer coalesces into a few frames and nothing here can lag the run.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function fmt(x, d = 4) { return (x == null || !isFinite(x)) ? "—" : (x >= 0 ? "+" : "") + x.toFixed(d); }
function fmtInt(n) { return n == null ? "—" : Math.round(n).toLocaleString("en-US"); }
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

// tier → colour + label for the frontier scatter
const TIER = {
  b: { c: () => COL.accent, l: "myopic BR" },
  a: { c: () => COL.amber, l: "archetype" },
  p: { c: () => COL["ink-3"], l: "pure sample" },
  m: { c: () => COL["accent-2"], l: "mixed sample" },
  c: { c: () => COL.green, l: "coord ascent" },
};
// stable per-family colour palette
const FAM_PALETTE = ["#8ab4f8", "#c58af9", "#f9ab00", "#81c995", "#f28b82", "#d7aefb", "#78d9ec", "#b8c88a"];
const famColor = {};
function colorFor(label) { if (!(label in famColor)) famColor[label] = FAM_PALETTE[Object.keys(famColor).length % FAM_PALETTE.length]; return famColor[label]; }

// ── store ─────────────────────────────────────────────────────────────────
const store = {
  config: null, game: "", delta: null, role: "—", verdict: "running", status: "connecting",
  startTs: null, lastElapsed: 0, lastWall: 0, ended: null,
  families: new Map(),            // label -> best-headroom state summary
  order: [],                      // family label insertion order
  currentFrontier: null,          // latest experiment checkpoint payload
  escalation: null,
  traj: null,                     // learning-trajectory payload (§35.B)
  progress: { xs: [], ys: [], max: -Infinity },
  lastState: "—", lastCands: 0, lastSucc: null,
  dirty: { frontier: true, headroom: true, ablation: true, screen: true, progress: true, decomp: true, traj: true, header: true },

  apply(e) {
    switch (e.type) {
      case "run_start": {
        this.config = e.config || {}; this.game = e.game; this.delta = this.config.delta;
        this.status = "live"; this.startTs = e.ts; this.verdict = "running";
        renderChips(); this.dirty.header = true;
        addLog(e.ts, "phase_change", `${e.game} · rich-bait decider started`);
        break;
      }
      case "metrics": {
        if (e.elapsed_s != null) { this.lastElapsed = e.elapsed_s; this.lastWall = performance.now(); }
        if (e.progress === 1) {
          this.lastCands = e.candidates_evaluated || this.lastCands;
          if (e.search_best_total != null) { this.progress.xs.push(this.lastCands); this.progress.ys.push(e.search_best_total); this.progress.max = Math.max(this.progress.max, e.search_best_total); }
          this.dirty.progress = true; this.dirty.header = true;
          break;
        }
        // per-state summary
        const fam = e.family;
        if (fam) {
          const prev = this.families.get(fam);
          const rec = {
            split: e.split, headroom: e.headroom, screen_gap: e.screen_gap,
            lb: [e.lb_pure, e.lb_mixed, e.lb_coord],
            myopic: [e.myopic_immediate, e.myopic_future], myopic_val: e.myopic_val,
            best: [e.best_immediate, e.best_future], best_total: e.best_total,
            n_distinct_succ: e.n_distinct_succ, state_ix: e.state_ix,
          };
          if (!prev) this.order.push(fam);
          if (!prev || e.headroom > prev.headroom) this.families.set(fam, rec);
          this.lastState = `${fam} @${e.state_ix}`;
          this.lastSucc = e.n_distinct_succ;
          this.dirty.headroom = this.dirty.ablation = this.dirty.screen = this.dirty.header = true;
        }
        break;
      }
      case "checkpoint": {
        if (e.experiment) { this.currentFrontier = e.experiment; this.dirty.frontier = true; }
        if (e.escalation) { this.escalation = e.escalation; this.dirty.decomp = true; addLog(e.ts, "phase_change", `A5−A4 decomposition: mean A5−A4(held) ${fmt(e.escalation.mean_a5_minus_a4_held)}`); }
        if (e.trajectory) { this.traj = e.trajectory; this.dirty.traj = true; const p = $("p-traj"); if (p) p.hidden = false; }
        break;
      }
      case "milestone": {
        addLog(e.ts, logKind(e.kind), e.message);
        const mm = /VERDICT\[(\w+)\]\s*=\s*([A-Z-]+)/.exec(e.message || "");
        if (mm) { this.role = mm[1]; this.verdict = mm[2]; this.dirty.header = true; }
        break;
      }
      case "run_end": {
        this.ended = e; this.status = "done";
        if (e.summary && e.summary.verdict) { this.verdict = e.summary.verdict; this.role = e.summary.role || this.role; }
        this.dirty.header = true; showCompletion(e);
        break;
      }
    }
  },
  elapsedNow() { return this.status === "live" ? this.lastElapsed + (performance.now() - this.lastWall) / 1000 : this.lastElapsed; },
  heldMaxHeadroom() { let mx = -Infinity; for (const l of this.order) { const r = this.families.get(l); if (r && r.split === "held") mx = Math.max(mx, r.headroom); } return isFinite(mx) ? mx : null; },
  heldNGtDelta() { let n = 0; for (const l of this.order) { const r = this.families.get(l); if (r && r.split === "held" && this.delta != null && r.headroom > this.delta) n++; } return n; },
  sortedFamilies() { return [...this.order].map(l => [l, this.families.get(l)]).sort((a, b) => (a[1].split === b[1].split) ? 0 : (a[1].split === "held" ? -1 : 1)); },
};

function logKind(k) { return ({ gate: "phase_change", opponent: "note", milestone: "phase_change", warning: "warning", phase_change: "phase_change" })[k] || "note"; }

// ── header ──────────────────────────────────────────────────────────────
let chipsDone = false;
function renderChips() {
  if (chipsDone || !store.config) return; chipsDone = true;
  const c = store.config, chips = $("config-chips"); chips.innerHTML = "";
  const push = (t) => { const s = document.createElement("span"); s.className = "tag"; s.textContent = t; chips.appendChild(s); };
  if (c.m != null) push("m=" + c.m);
  if (c.n_states != null) push(c.n_states + " states");
  if (c.n_pure != null) push(c.n_pure + " pure");
  if (c.n_mixed != null) push(c.n_mixed + " mixed");
  if (c.coord) push("coord-ascent");
}
function expName() {
  return (store.config && store.config.experiment === "no-regret-learner-decider")
    ? "§35.B no-regret-learner decider" : "§35.A rich-bait decider";
}
function renderHeader() {
  $("game-name").textContent = store.game ? `${store.game} · ${expName()}` : `${expName()} — waiting…`;
  const v = $("stat-verdict");
  v.textContent = store.verdict || "—";
  // GO / NULL-REAL = green (edge / clean null); KILL / MENU-ARTIFACT = amber (attention);
  // FAIL / CONTROLS-FAILED = red; running = accent.
  const V = store.verdict || "";
  v.className = (V === "NULL-REAL" || V === "GO" || V === "FILTER-GO") ? "v-null"
    : (V === "MENU-ARTIFACT" || V === "KILL" || V === "FILTER-FLAT") ? "v-menu"
      : (V.indexOf("FAIL") >= 0) ? "v-fail" : "v-run";
  $("stat-role").textContent = store.role !== "—" ? `role: ${store.role}` : "—";
  const mh = store.heldMaxHeadroom();
  $("stat-headroom").textContent = mh == null ? "—" : fmt(mh);
  $("stat-headroom").style.color = (mh != null && store.delta != null) ? (mh > store.delta ? COL.amber : COL.green) : "#fff";
  $("stat-delta").textContent = store.delta != null ? store.delta.toFixed(3) : "—";
  const st = $("stat-status"), dc = { live: "live", done: "done", connecting: "idle" }[store.status] || "idle";
  st.innerHTML = `<span class="dot ${dc}"></span>${store.status}`;
  $("stat-state").textContent = store.lastState;
  $("stat-cands").textContent = fmtInt(store.lastCands);
  $("stat-elapsed").textContent = fmtDur(store.elapsedNow());
}

// ── frontier scatter ──────────────────────────────────────────────────────
function drawFrontier() {
  const { ctx, w, h } = fitCanvas($("c-frontier"));
  ctx.clearRect(0, 0, w, h);
  const F = store.currentFrontier;
  if (!F || !F.frontier || !F.frontier.length) return placeholder(ctx, w, h);
  $("frontier-fam").textContent = `${F.family} [${F.split}] @state ${F.state_ix} · headroom ${fmt(F.headroom)}`;
  const P = { l: 52, r: 14, t: 12, b: 30 }, x0 = P.l, x1 = w - P.r, y0 = P.t, y1 = h - P.b;
  const pts = F.frontier, my = F.myopic;
  let xmin = my[0], xmax = my[0], ymin = my[1], ymax = my[1];
  for (const p of pts) { xmin = Math.min(xmin, p[0]); xmax = Math.max(xmax, p[0]); ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]); }
  const padx = (xmax - xmin) * 0.08 || 0.5, pady = (ymax - ymin) * 0.08 || 0.5;
  xmin -= padx; xmax += padx; ymin -= pady; ymax += pady;
  const X = (v) => x0 + (v - xmin) / (xmax - xmin) * (x1 - x0);
  const Y = (v) => y1 - (v - ymin) / (ymax - ymin) * (y1 - y0);
  // gridlines + axis labels
  ctx.strokeStyle = rgba(hexToRgb(COL.border), 0.6); ctx.fillStyle = COL["ink-3"]; ctx.lineWidth = 1;
  ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) { const yv = ymin + (ymax - ymin) * i / 4, py = Y(yv); ctx.globalAlpha = .5; ctx.beginPath(); ctx.moveTo(x0, py); ctx.lineTo(x1, py); ctx.stroke(); ctx.globalAlpha = 1; ctx.fillText(yv.toFixed(2), x0 - 6, py); }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) { const xv = xmin + (xmax - xmin) * i / 4; ctx.fillText(xv.toFixed(2), X(xv), y1 + 6); }
  ctx.save(); ctx.textAlign = "center"; ctx.fillStyle = COL["ink-2"]; ctx.font = "11px var(--sans)";
  ctx.fillText("immediate EV  u₀(h, P)", (x0 + x1) / 2, h - 14);
  ctx.translate(13, (y0 + y1) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("future BR-value  BR(ρ(h,P))", 0, 0); ctx.restore();
  // iso-total lines: immediate+future = C  ->  future = C - immediate  (slope -1)
  const C0 = my[0] + my[1], Cd = C0 + (F.delta || 0);
  const isoLine = (C, style, dash) => {
    ctx.strokeStyle = style; ctx.lineWidth = 1.5; ctx.setLineDash(dash);
    // clip the line to the plot box
    const pa = [xmin, C - xmin], pb = [xmax, C - xmax];
    ctx.beginPath(); ctx.moveTo(X(pa[0]), Y(pa[1])); ctx.lineTo(X(pb[0]), Y(pb[1])); ctx.stroke(); ctx.setLineDash([]);
  };
  ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip();
  isoLine(C0, rgba(hexToRgb(COL["ink-2"]), 0.8), [5, 4]);
  isoLine(Cd, rgba(hexToRgb(COL.amber), 0.9), []);
  // points
  for (const p of pts) {
    const above = (p[0] + p[1]) > Cd + 1e-9;
    const col = TIER[p[2]] ? TIER[p[2]].c() : COL["ink-3"];
    ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), above ? 3.4 : 2.3, 0, 7);
    ctx.fillStyle = rgba(hexToRgb(col), above ? 0.95 : 0.6); ctx.fill();
    if (above) { ctx.strokeStyle = COL.raise; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 5.5, 0, 7); ctx.stroke(); }
  }
  // myopic BR star
  star(ctx, X(my[0]), Y(my[1]), 7, COL.accent);
  ctx.restore();
}
function star(ctx, cx, cy, r, color) {
  ctx.save(); ctx.beginPath();
  for (let i = 0; i < 10; i++) { const ang = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r; const x = cx + Math.cos(ang) * rr, y = cy + Math.sin(ang) * rr; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
  ctx.closePath(); ctx.fillStyle = color; ctx.fill(); ctx.restore();
}

// ── horizontal headroom bars ───────────────────────────────────────────────
function drawHeadroom() {
  const { ctx, w, h } = fitCanvas($("c-headroom"));
  ctx.clearRect(0, 0, w, h);
  const fams = store.sortedFamilies();
  if (!fams.length) return placeholder(ctx, w, h);
  const P = { l: 118, r: 44, t: 8, b: 22 }, x0 = P.l, x1 = w - P.r, y0 = P.t, y1 = h - P.b;
  const d = store.delta || 0;
  let mx = d * 1.5; for (const [, r] of fams) mx = Math.max(mx, r.headroom); mx = mx * 1.12 || 1;
  const X = (v) => x0 + clamp(v / mx, 0, 1) * (x1 - x0);
  const bh = Math.min(30, (y1 - y0) / fams.length - 6), gap = ((y1 - y0) - bh * fams.length) / Math.max(1, fams.length);
  ctx.font = "11px ui-monospace, monospace";
  fams.forEach(([label, r], i) => {
    const y = y0 + i * (bh + gap);
    ctx.fillStyle = COL["ink-2"]; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(label.length > 15 ? label.slice(0, 14) + "…" : label, x0 - 8, y + bh / 2);
    ctx.fillStyle = rgba(hexToRgb(COL.border), 0.5); roundRect(ctx, x0, y, x1 - x0, bh, 5); ctx.fill();
    const good = r.headroom <= d + 1e-9, col = good ? COL.green : COL.raise;
    roundRect(ctx, x0, y, Math.max(2, X(r.headroom) - x0), bh, 5); ctx.fillStyle = rgba(hexToRgb(col), .85); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.fillText(fmt(r.headroom), X(r.headroom) + 6, y + bh / 2);
    // split tag
    ctx.fillStyle = r.split === "held" ? COL.accent : COL["ink-3"]; ctx.textAlign = "right"; ctx.font = "9px ui-monospace, monospace";
    ctx.fillText(r.split, x0 - 8, y + bh / 2 + 12); ctx.font = "11px ui-monospace, monospace";
  });
  // δ line
  const dx = X(d); ctx.strokeStyle = rgba(hexToRgb(COL.amber), .9); ctx.setLineDash([4, 3]); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(dx, y0); ctx.lineTo(dx, y1); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = COL.amber; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("δ", dx, y1 + 4);
}

// ── grouped ablation bars ──────────────────────────────────────────────────
function drawGroupedBars(cvId, valuesByFam, labels, colors, opt = {}) {
  const { ctx, w, h } = fitCanvas($(cvId));
  ctx.clearRect(0, 0, w, h);
  const fams = store.sortedFamilies();
  if (!fams.length) return placeholder(ctx, w, h);
  const P = { l: 40, r: 10, t: 10, b: 40 }, x0 = P.l, x1 = w - P.r, y0 = P.t, y1 = h - P.b;
  let mn = 0, mx = 1e-6;
  for (const [label, r] of fams) for (const v of valuesByFam(label, r)) { if (v != null) { mn = Math.min(mn, v); mx = Math.max(mx, v); } }
  if (opt.delta != null) mx = Math.max(mx, opt.delta);
  const pad = (mx - mn) * 0.1 || 0.1; mx += pad; if (mn < 0) mn -= pad;
  const Y = (v) => y1 - (v - mn) / (mx - mn) * (y1 - y0);
  // zero / delta guides
  ctx.strokeStyle = rgba(hexToRgb(COL.border), .8); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0, Y(0)); ctx.lineTo(x1, Y(0)); ctx.stroke();
  ctx.fillStyle = COL["ink-3"]; ctx.font = "9px ui-monospace, monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  ctx.fillText("0", x0 - 4, Y(0));
  if (opt.delta != null) { const dy = Y(opt.delta); ctx.strokeStyle = rgba(hexToRgb(COL.amber), .8); ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(x0, dy); ctx.lineTo(x1, dy); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = COL.amber; ctx.fillText("δ", x0 - 4, dy); }
  const gw = (x1 - x0) / fams.length, nb = labels.length, bw = Math.min(16, (gw - 8) / nb);
  fams.forEach(([label, r], i) => {
    const vals = valuesByFam(label, r), gx = x0 + i * gw + (gw - bw * nb) / 2;
    for (let j = 0; j < nb; j++) {
      const v = vals[j]; if (v == null) continue;
      const bx = gx + j * bw, top = Y(Math.max(0, v)), bot = Y(Math.min(0, v));
      ctx.fillStyle = rgba(hexToRgb(colors[j]), .85); roundRect(ctx, bx + 1, top, bw - 2, Math.max(1, bot - top), 2); ctx.fill();
    }
    ctx.save(); ctx.fillStyle = r.split === "held" ? COL["ink-2"] : COL["ink-3"]; ctx.font = "9px ui-monospace, monospace";
    ctx.translate(x0 + i * gw + gw / 2, y1 + 6); ctx.rotate(Math.PI / 5); ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(label.length > 12 ? label.slice(0, 11) + "…" : label, 0, 0); ctx.restore();
  });
  // legend
  ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.font = "9.5px var(--sans)";
  let lx = x0;
  for (let j = 0; j < nb; j++) { ctx.fillStyle = colors[j]; roundRect(ctx, lx, y0, 9, 9, 2); ctx.fill(); ctx.fillStyle = COL["ink-2"]; ctx.fillText(labels[j], lx + 12, y0 + 4); lx += 12 + ctx.measureText(labels[j]).width + 14; }
}

// ── search-progress line ───────────────────────────────────────────────────
function drawProgress() {
  const { ctx, w, h } = fitCanvas($("c-progress"));
  ctx.clearRect(0, 0, w, h);
  const xs = store.progress.xs, ys = store.progress.ys;
  $("kpi-best").textContent = ys.length ? fmt(ys[ys.length - 1]) : "—";
  $("kpi-succ").textContent = store.lastSucc != null ? store.lastSucc : "—";
  $("kpi-nfam").textContent = store.order.length ? store.heldNGtDelta() : "—";
  if (xs.length < 2) return placeholder(ctx, w, h);
  const P = { l: 46, r: 10, t: 10, b: 20 }, x0 = P.l, x1 = w - P.r, y0 = P.t, y1 = h - P.b;
  let xmn = xs[0], xmx = xs[xs.length - 1], ymn = Math.min(...ys), ymx = Math.max(...ys);
  if (xmx - xmn < 1) xmx = xmn + 1; const pad = (ymx - ymn) * 0.1 || 0.5; ymn -= pad; ymx += pad;
  const X = (v) => x0 + (v - xmn) / (xmx - xmn) * (x1 - x0), Y = (v) => y1 - (v - ymn) / (ymx - ymn) * (y1 - y0);
  ctx.strokeStyle = rgba(hexToRgb(COL.border), .6); ctx.fillStyle = COL["ink-3"]; ctx.font = "9px ui-monospace, monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 3; i++) { const yv = ymn + (ymx - ymn) * i / 3, py = Y(yv); ctx.globalAlpha = .5; ctx.beginPath(); ctx.moveTo(x0, py); ctx.lineTo(x1, py); ctx.stroke(); ctx.globalAlpha = 1; ctx.fillText(yv.toFixed(2), x0 - 5, py); }
  ctx.beginPath(); for (let i = 0; i < xs.length; i++) { const px = X(xs[i]), py = Y(ys[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
  ctx.strokeStyle = COL.green; ctx.lineWidth = 1.5; ctx.lineJoin = "round"; ctx.stroke();
  ctx.fillStyle = COL["ink-3"]; ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText("candidates evaluated", (x0 + x1) / 2, y1 + 5);
}

// ── decomposition (A5 − A4) ────────────────────────────────────────────────
function drawDecomp() {
  const note = $("decomp-note"), kpi = $("kpi-a5a4");
  if (!store.escalation) { kpi.textContent = "—"; const { ctx, w, h } = fitCanvas($("c-decomp")); ctx.clearRect(0, 0, w, h); placeholder(ctx, w, h, "no headroom yet — escalation runs only on MENU-ARTIFACT"); return; }
  const e = store.escalation;
  kpi.textContent = `A5−A4(held) ${fmt(e.mean_a5_minus_a4_held)}`;
  kpi.style.color = Math.abs(e.mean_a5_minus_a4_held) > 1e-6 ? COL.raise : COL.green;
  note.textContent = `Rich-menu ceiling (depth ${e.depth}): mean headroom(held) ${fmt(e.mean_headroom_held)}. The multiplayer edge H2 bets on is A5−A4>0 on held-out; here mean A5−A4(held) = ${fmt(e.mean_a5_minus_a4_held)}.`;
  // grouped bars per family: headroom (ceiling−A2.5) and A5−A4
  const { ctx, w, h } = fitCanvas($("c-decomp"));
  ctx.clearRect(0, 0, w, h);
  const rows = e.per_family || []; if (!rows.length) return placeholder(ctx, w, h);
  const sorted = [...rows].sort((a, b) => (a.split === b.split) ? 0 : (a.split === "held" ? -1 : 1));
  const P = { l: 40, r: 10, t: 10, b: 40 }, x0 = P.l, x1 = w - P.r, y0 = P.t, y1 = h - P.b;
  let mn = 0, mx = 1e-6; for (const r of sorted) { const hr = r.ceiling - r["A2.5"]; mn = Math.min(mn, r.a5_minus_a4, 0); mx = Math.max(mx, hr, r.a5_minus_a4); }
  const pad = (mx - mn) * 0.12 || 0.1; mx += pad; if (mn < 0) mn -= pad;
  const Y = (v) => y1 - (v - mn) / (mx - mn) * (y1 - y0);
  ctx.strokeStyle = rgba(hexToRgb(COL.border), .8); ctx.beginPath(); ctx.moveTo(x0, Y(0)); ctx.lineTo(x1, Y(0)); ctx.stroke();
  ctx.fillStyle = COL["ink-3"]; ctx.font = "9px ui-monospace, monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText("0", x0 - 4, Y(0));
  const gw = (x1 - x0) / sorted.length, bw = Math.min(18, (gw - 8) / 2), cols = [COL["accent-2"], COL.raise];
  sorted.forEach((r, i) => {
    const vals = [r.ceiling - r["A2.5"], r.a5_minus_a4], gx = x0 + i * gw + (gw - bw * 2) / 2;
    for (let j = 0; j < 2; j++) { const v = vals[j], top = Y(Math.max(0, v)), bot = Y(Math.min(0, v)); ctx.fillStyle = rgba(hexToRgb(cols[j]), .85); roundRect(ctx, gx + j * bw + 1, top, bw - 2, Math.max(1, bot - top), 2); ctx.fill(); }
    ctx.save(); ctx.fillStyle = r.split === "held" ? COL["ink-2"] : COL["ink-3"]; ctx.font = "9px ui-monospace, monospace";
    ctx.translate(x0 + i * gw + gw / 2, y1 + 6); ctx.rotate(Math.PI / 5); ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(r.label.length > 12 ? r.label.slice(0, 11) + "…" : r.label, 0, 0); ctx.restore();
  });
  ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.font = "9.5px var(--sans)"; let lx = x0;
  for (const [lab, c] of [["ceiling−A2.5", COL["accent-2"]], ["A5−A4 (H2)", COL.raise]]) { ctx.fillStyle = c; roundRect(ctx, lx, y0, 9, 9, 2); ctx.fill(); ctx.fillStyle = COL["ink-2"]; ctx.fillText(lab, lx + 12, y0 + 4); lx += 12 + ctx.measureText(lab).width + 16; }
}

// ── learning trajectory (§35.B: opponents learning over time) ───────────────
function drawTraj() {
  const cv = $("c-traj"); if (!cv) return;
  const { ctx, w, h } = fitCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const Tr = store.traj;
  if (!Tr || !Tr.blocks || !Tr.blocks.length) return placeholder(ctx, w, h, "no learning trajectory yet");
  $("traj-fam").textContent = Tr.family || "—";
  const B = Tr.blocks, n = B.length;
  const P = { l: 48, r: 44, t: 14, b: 26 }, x0 = P.l, x1 = w - P.r, y0 = P.t, y1 = h - P.b;
  const X = (t) => n <= 1 ? (x0 + x1) / 2 : x0 + t / (n - 1) * (x1 - x0);
  // left axis: EV scale (hero_ev + br_ceiling); right axis: [0,1] (aggression + coupling)
  let emn = Infinity, emx = -Infinity;
  for (const b of B) { emn = Math.min(emn, b.hero_ev, b.br_ceiling); emx = Math.max(emx, b.hero_ev, b.br_ceiling); }
  const epad = (emx - emn) * 0.12 || 0.5; emn -= epad; emx += epad;
  const YL = (v) => y1 - (v - emn) / (emx - emn) * (y1 - y0);
  const YR = (v) => y1 - clamp(v, 0, 1) * (y1 - y0);
  // gridlines + dual axis labels
  ctx.strokeStyle = rgba(hexToRgb(COL.border), .5); ctx.fillStyle = COL["ink-3"];
  ctx.font = "9px ui-monospace, monospace"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const py = y0 + (y1 - y0) * i / 4;
    ctx.globalAlpha = .5; ctx.beginPath(); ctx.moveTo(x0, py); ctx.lineTo(x1, py); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillStyle = COL["ink-3"]; ctx.fillText((emx - (emx - emn) * i / 4).toFixed(2), x0 - 5, py);
    ctx.textAlign = "left"; ctx.fillStyle = rgba(hexToRgb(COL.amber), .8);
    ctx.fillText((1 - i / 4).toFixed(2), x1 + 5, py);
  }
  ctx.fillStyle = COL["ink-3"]; ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let t = 0; t < n; t++) ctx.fillText("blk " + t, X(t), y1 + 6);
  const line = (get, YFN, color, dash, wdt) => {
    ctx.strokeStyle = color; ctx.lineWidth = wdt || 2; ctx.setLineDash(dash || []); ctx.lineJoin = "round";
    ctx.beginPath();
    B.forEach((b, t) => { const v = get(b); if (v == null) return; const px = X(t), py = YFN(v); t ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.stroke(); ctx.setLineDash([]);
    B.forEach((b, t) => { const v = get(b); if (v == null) return; ctx.beginPath(); ctx.arc(X(t), YFN(v), 2.6, 0, 7); ctx.fillStyle = color; ctx.fill(); });
  };
  const nseat = (B[0].aggression || []).length;
  const seatCols = [COL.accent, COL["accent-2"], COL.call];
  line((b) => b.hero_ev, YL, COL.green, [], 2.4);                 // Hero realised EV
  line((b) => b.br_ceiling, YL, COL.amber, [5, 4], 1.8);         // exact-BR ceiling
  for (let s = 0; s < nseat; s++) line((b) => b.aggression[s], YR, seatCols[s % seatCols.length], [2, 2], 1.6);
  line((b) => b.coupling, YR, COL.raise, [], 2.2);               // opp→opp coupling
  // legend
  ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.font = "9.5px var(--sans)"; let lx = x0 + 2;
  const items = [["Hero EV", COL.green], ["BR ceiling", COL.amber]];
  const seats = Tr.seats && Tr.seats.length ? Tr.seats : Array.from({ length: nseat }, (_, i) => "opp" + (i + 1));
  for (let s = 0; s < nseat; s++) items.push([seats[s] + " aggr", seatCols[s % seatCols.length]]);
  items.push(["coupling", COL.raise]);
  for (const [lab, c] of items) { ctx.fillStyle = c; roundRect(ctx, lx, y0 - 2, 9, 9, 2); ctx.fill(); ctx.fillStyle = COL["ink-2"]; ctx.fillText(lab, lx + 12, y0 + 3); lx += 12 + ctx.measureText(lab).width + 12; }
}

function placeholder(ctx, w, h, msg) { ctx.fillStyle = COL["ink-3"]; ctx.font = "12px var(--sans)"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(msg || "waiting for data…", w / 2, h / 2); }

// ── tier key ──────────────────────────────────────────────────────────────
(function buildTierKey() {
  const el = $("tier-key"); if (!el) return;
  el.innerHTML = ["b", "a", "p", "m", "c"].map(t => `<span class="li"><i style="background:${TIER[t].c()}"></i>${TIER[t].l}</span>`).join("") + `<span class="li"><i style="background:${COL.accent};clip-path:polygon(50% 0,61% 35%,98% 35%,68% 57%,79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%)"></i>myopic ★</span>`;
})();

// ── event log ───────────────────────────────────────────────────────────────
function addLog(ts, kind, msg) {
  const ul = $("event-log"); const li = document.createElement("li");
  const t = new Date(ts * 1000).toLocaleTimeString("en-US", { hour12: false });
  li.innerHTML = `<span class="t">${t}</span><span class="kind ${kind}">${kind.replace("_", " ")}</span><span class="msg"></span>`;
  li.querySelector(".msg").textContent = msg; ul.appendChild(li);
  while (ul.children.length > 200) ul.removeChild(ul.firstChild);
}

// ── completion banner ───────────────────────────────────────────────────────
let doneShown = false;
function showCompletion(e) {
  if (doneShown) return; doneShown = true;
  const S = e.summary || {}, v = S.verdict || store.verdict;
  const attn = v === "MENU-ARTIFACT" || v === "KILL" || v === "FILTER-FLAT";
  $("done-badge").textContent = attn ? "!" : "✓";
  $("done-badge").className = "done-badge" + (attn ? " err" : "");
  $("done-title").textContent = `Verdict: ${v || "—"}`;
  const SUB = {
    "NULL-REAL": "Rich-bait ceiling still equals myopic within δ — the §34.5 null is genuine (not a menu artifact).",
    "MENU-ARTIFACT": "Rich baits beat myopic by > δ — the coarse menu hid headroom; see the A5−A4 decomposition.",
    "GO": "A5−A4 > δ, held-out, exact-controlled, and transfers to an un-fit learner — the multiplayer edge H2 is real. GREENLIGHT.",
    "FILTER-GO": "Kuhn filter: A5−A4 clears δ and transfers. Leduc is decisive.",
    "KILL": "A5−A4 stays sub-δ or does not transfer to an un-fit learner — H2 is a dead end at toy scale. Ship the H1 spine.",
    "FILTER-FLAT": "Kuhn filter: the multiplayer edge A5−A4 is sub-δ. A NO here is not a kill (R3); Leduc decides.",
    "CONTROLS-FAILED": "An exact control failed — the run is not trustworthy; do not read the verdict.",
  };
  $("done-sub").textContent = SUB[v] || "Run finished.";
  const stats = []; const add = (l, val) => stats.push(`<div class="d"><div class="dl">${l}</div><div class="dv">${val}</div></div>`);
  add("role", S.role || store.role);
  const a5a4 = S["mean A5-A4 (held)"]; if (a5a4 != null) add("mean A5−A4 (held)", fmt(a5a4));
  const a5a4mx = S["max A5-A4 (held)"]; if (a5a4mx != null) add("max A5−A4 (held)", fmt(a5a4mx));
  const unfit = S["un-fit-learner families A5-A4 > delta"]; if (unfit != null) add("un-fit learners > δ", Math.round(unfit));
  const mh = S["max headroom (held)"]; if (mh != null) add("max headroom (held)", fmt(mh));
  const scr = S["max future-exploitability screen gap (held)"]; if (scr != null) add("max screen gap (held)", fmt(scr));
  const nf = S["held families with headroom > delta"]; if (nf != null) add("families > δ (held)", Math.round(nf));
  if (store.escalation && a5a4 == null) add("A5−A4 (held)", fmt(store.escalation.mean_a5_minus_a4_held));
  $("done-stats").innerHTML = stats.join("");
  $("done-overlay").hidden = false;
}
$("done-dismiss").addEventListener("click", () => { $("done-overlay").hidden = true; });
$("done-overlay").addEventListener("click", (e) => { if (e.target === $("done-overlay")) $("done-overlay").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("done-overlay").hidden = true; });

// ── render loop ─────────────────────────────────────────────────────────────
function frame() {
  const D = store.dirty;
  renderHeader();
  if (D.frontier) { drawFrontier(); D.frontier = false; }
  if (D.headroom) { drawHeadroom(); D.headroom = false; }
  if (D.ablation) { drawGroupedBars("c-ablation", (l, r) => r.lb, ["pure", "+mixed", "+coord"], [COL["ink-3"], COL["accent-2"], COL.green], { delta: store.delta }); D.ablation = false; }
  if (D.screen) { drawGroupedBars("c-screen", (l, r) => [r.screen_gap], ["screen gap"], [COL.amber], { delta: store.delta }); D.screen = false; }
  if (D.progress) { drawProgress(); D.progress = false; }
  if (D.decomp) { drawDecomp(); D.decomp = false; }
  if (D.traj) { drawTraj(); D.traj = false; }
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
