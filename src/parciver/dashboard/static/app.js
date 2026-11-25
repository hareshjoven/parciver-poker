/* parciver live training dashboard — client.
 *
 * Decoupled + lag-free by construction: the server streams batched telemetry
 * events over SSE; we only ever append to in-memory ring/series buffers and set
 * dirty flags. ALL drawing happens in one requestAnimationFrame loop that redraws
 * a panel only when its data changed, decimating long series to the pixel width.
 * So a fast run (thousands of events) coalesces into a few big frames instead of a
 * flood, and rendering stays pinned to the display refresh. Nothing here can slow
 * or lag training — it never talks back to the trainer.
 */
"use strict";

// ── tiny utils ──────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;

function fmtSI(n) {
  if (n == null) return "—";
  const a = Math.abs(n);
  if (a >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (a >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(Math.round(n));
}
function fmtInt(n) { return n == null ? "—" : Math.round(n).toLocaleString("en-US"); }
function fmtDur(s) {
  if (s == null || !isFinite(s)) return "—";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const p = (x) => String(x).padStart(2, "0");
  return h > 0 ? `${h}:${p(m)}:${p(sec)}` : `${p(m)}:${p(sec)}`;
}
function fmtSci(x) {
  if (x == null) return "—";
  if (x === 0) return "0";
  if (x >= 0.01 && x < 1000) return x.toFixed(x < 1 ? 4 : 2);
  return x.toExponential(2);
}

// DPR-aware canvas; returns {ctx, w, h} in CSS pixels (ctx pre-scaled).
function fitCanvas(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const r = cv.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
  if (cv.width !== w * dpr || cv.height !== h * dpr) {
    cv.width = w * dpr; cv.height = h * dpr;
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

// ── growable, self-decimating time series ───────────────────────────────
class Series {
  constructor(cap = 120000) {
    this.cap = cap; this.xs = new Float64Array(cap); this.ys = new Float64Array(cap);
    this.n = 0; this.min = Infinity; this.max = -Infinity;
  }
  push(x, y) {
    if (y == null || !isFinite(y)) return;
    if (this.n >= this.cap) this._halve();
    this.xs[this.n] = x; this.ys[this.n] = y; this.n++;
    if (y < this.min) this.min = y; if (y > this.max) this.max = y;
  }
  _halve() { // keep every other point — preserves shape, bounds memory over long runs
    let j = 0;
    for (let i = 0; i < this.n; i += 2) { this.xs[j] = this.xs[i]; this.ys[j] = this.ys[i]; j++; }
    this.n = j;
  }
  last() { return this.n ? this.ys[this.n - 1] : null; }
}

// ── the store: apply events, hold state, raise dirty flags ──────────────
const store = {
  started: false, config: null, target: 0, game: "", variant: "",
  latest: {}, startTs: null, lastElapsed: 0, lastWall: 0, status: "connecting",
  ended: null,
  series: {
    expl: new Series(), regret: new Series(), itps: new Series(),
    infosets: new Series(), hh: new Series(), stability: new Series(),
  },
  checkpoint: null,     // {range_grid, action_freq, bet_size_freq, infosets_by_street}
  dirty: { header: true, conv: true, itps: true, cov: true, hh: true, stab: true,
           grid: false, bars: true, gauges: true },
  milestones: [],
  applied: 0,

  apply(e) {
    this.applied++;
    switch (e.type) {
      case "run_start": {
        this.started = true; this.config = e.config || {}; this.game = e.game;
        this.variant = e.variant; this.target = e.target_iters || 0;
        this.startTs = e.ts; this.status = "live";
        if (this.config.hand_names) grid.setNames(this.config.hand_names);
        this.dirty.header = true;
        addLog(e.ts, "phase_change", `run ${e.run_id} started · ${e.game}`);
        break;
      }
      case "metrics": {
        const L = this.latest;
        L.iteration = e.iteration;
        this.iterWall = performance.now();  // for smooth between-update extrapolation
        if (e.it_per_s != null) { L.it_per_s = e.it_per_s; this.series.itps.push(e.iteration, e.it_per_s); this.dirty.itps = true; }
        if (e.exploitability != null) { L.exploitability = e.exploitability; this.series.expl.push(e.iteration, e.exploitability); this.dirty.conv = true; }
        if (e.mean_abs_regret != null) { L.mean_abs_regret = e.mean_abs_regret; this.series.regret.push(e.iteration, e.mean_abs_regret); this.dirty.conv = true; }
        if (e.strategy_l1_delta != null) { L.strategy_l1_delta = e.strategy_l1_delta; this.series.stability.push(e.iteration, e.strategy_l1_delta); this.dirty.stab = true; }
        if (e.infosets != null) { L.infosets = e.infosets; this.series.infosets.push(e.iteration, e.infosets); this.dirty.cov = true; }
        if (e.hh_bb_per_100 != null) { L.hh = e.hh_bb_per_100; this.series.hh.push(e.iteration, e.hh_bb_per_100); this.dirty.hh = true; }
        if (e.resources) { L.resources = e.resources; this.dirty.gauges = true; }
        if (e.elapsed_s != null) { this.lastElapsed = e.elapsed_s; this.lastWall = performance.now(); }
        this.dirty.header = true;
        break;
      }
      case "checkpoint": {
        // MERGE non-null fields rather than replace: the range grid + infosets come
        // from the trainer's events.jsonl, while action_freq/bet_size_freq come from
        // the sidecar's panels.jsonl. Two separate producers, one merged view — so
        // neither checkpoint clobbers the other's fields.
        const c = this.checkpoint || {};
        for (const k in e) { if (e[k] != null) c[k] = e[k]; }
        this.checkpoint = c;
        if (e.range_grid) { grid.setTarget(e.range_grid); this.dirty.grid = true; }
        if (e.action_freq || e.bet_size_freq) this.dirty.bars = true;
        if (e.infosets_by_street) this.dirty.cov = true;
        break;
      }
      case "milestone":
        addLog(e.ts, e.kind, e.message);
        break;
      case "run_end":
        this.ended = e; this.status = e.reason === "completed" ? "done" : (e.reason === "error" ? "err" : "done");
        this.dirty.header = true;
        addLog(e.ts, e.reason === "error" ? "warning" : "phase_change", `run ${e.reason} @ ${fmtInt(e.iteration)} iterations`);
        showCompletion(e);
        break;
    }
  },
  elapsedNow() {
    if (this.status === "live") return this.lastElapsed + (performance.now() - this.lastWall) / 1000;
    return this.lastElapsed;
  },
  // Smoothly-extrapolated iteration: telemetry lands every few seconds, so between
  // updates we advance the displayed count by the live throughput, then snap to the
  // real value on the next update. This makes the counter FLY UP continuously at
  // 60fps instead of jumping in chunks.
  displayIter() {
    const L = this.latest;
    if (L.iteration == null) return 0;
    if (this.status === "live" && L.it_per_s > 0 && this.iterWall) {
      const dts = (performance.now() - this.iterWall) / 1000;
      return L.iteration + L.it_per_s * Math.max(0, dts);
    }
    return L.iteration;
  },
};

// Eased display values (glide toward the latest reading each frame) so no gauge or
// KPI jumps discretely when a telemetry batch lands.
const anim = { itps: 0, regret: null, infosets: 0, hh: null, cpu: 0, gpu: 0, ram: 0 };
function updateAnim(dt) {
  const L = store.latest;
  const k = Math.min(1, 9 * dt);
  const ez = (cur, tgt) => (cur == null ? tgt : cur + (tgt - cur) * k);
  if (L.it_per_s != null) anim.itps = ez(anim.itps, L.it_per_s);
  if (L.infosets != null) anim.infosets = ez(anim.infosets, L.infosets);
  const conv = L.exploitability != null ? L.exploitability : L.mean_abs_regret;
  if (conv != null) anim.regret = ez(anim.regret, conv);
  if (L.hh != null) anim.hh = ez(anim.hh, L.hh);
  const r = L.resources || {};
  if (r.cpu_pct != null) anim.cpu = ez(anim.cpu, r.cpu_pct);
  if (r.gpu_pct != null) anim.gpu = ez(anim.gpu, r.gpu_pct);
  if (r.ram_used_gb != null && r.ram_total_gb) anim.ram = ez(anim.ram, r.ram_used_gb / r.ram_total_gb * 100);
}

// ── colour helpers ──────────────────────────────────────────────────────
const CSSVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COL = {};
function loadColors() {
  for (const k of ["accent", "accent-2", "amber", "green", "raise", "call", "fold", "ink", "ink-2", "ink-3", "panel-2", "border", "red"])
    COL[k] = CSSVar("--" + k);
}
function hexToRgb(h) { h = h.replace("#", ""); if (h.length === 3) h = h.split("").map(c => c + c).join(""); const n = parseInt(h, 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
function rgba(rgb, a) { return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`; }

// ── line chart engine ───────────────────────────────────────────────────
class Chart {
  constructor(cv, opts) { this.cv = cv; this.o = Object.assign({ logX: false, logY: false, pad: { l: 46, r: 12, t: 10, b: 20 } }, opts); this.hover = null;
    cv.addEventListener("mousemove", (ev) => { const r = cv.getBoundingClientRect(); this.hover = { x: ev.clientX - r.left, y: ev.clientY - r.top }; this._dirty = true; });
    cv.addEventListener("mouseleave", () => { this.hover = null; this._dirty = true; });
  }
  draw(series) {
    const { ctx, w, h } = fitCanvas(this.cv);
    ctx.clearRect(0, 0, w, h);
    const P = this.o.pad, x0 = P.l, x1 = w - P.r, y0 = P.t, y1 = h - P.b;
    const active = series.filter(s => s.s.n > 0);
    if (!active.length) { this._placeholder(ctx, w, h); return; }

    // domain
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of active) {
      xmin = Math.min(xmin, s.s.xs[0]); xmax = Math.max(xmax, s.s.xs[s.s.n - 1]);
      ymin = Math.min(ymin, s.s.min); ymax = Math.max(ymax, s.s.max);
    }
    if (this.o.y0 != null) ymin = Math.min(ymin, this.o.y0);
    if (this.o.y1 != null) ymax = Math.max(ymax, this.o.y1);
    const lx = this.o.logX, ly = this.o.logY;
    const fx = (v) => lx ? Math.log10(Math.max(v, 1)) : v;
    const fy = (v) => ly ? Math.log10(Math.max(v, 1e-9)) : v;
    let a = fx(xmin), b = fx(xmax); if (b - a < 1e-9) b = a + 1;
    let c = fy(ymin), d = fy(ymax);
    if (ly) { c = Math.floor(c); d = Math.ceil(d); } else { const pad = (d - c) * 0.08 || 1; c -= pad; d += pad; }
    if (d - c < 1e-9) d = c + 1;
    const X = (v) => x0 + (fx(v) - a) / (b - a) * (x1 - x0);
    const Y = (v) => y1 - (fy(v) - c) / (d - c) * (y1 - y0);

    this._gridlines(ctx, x0, x1, y0, y1, c, d, ly, Y);

    // series
    for (const s of active) {
      const rgb = hexToRgb(s.color);
      const pts = decimate(s.s, X);
      // area fill
      if (s.fill) {
        ctx.beginPath(); ctx.moveTo(pts[0].px, y1);
        for (const p of pts) ctx.lineTo(p.px, Y(p.y));
        ctx.lineTo(pts[pts.length - 1].px, y1); ctx.closePath();
        ctx.fillStyle = rgba(rgb, 0.08); ctx.fill();
      }
      // restrained data line
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < pts.length; i++) { const py = Y(pts[i].y); i ? ctx.lineTo(pts[i].px, py) : ctx.moveTo(pts[i].px, py); }
      ctx.lineWidth = 1.5; ctx.strokeStyle = s.color; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.stroke();
      ctx.restore();
      // last dot
      const last = pts[pts.length - 1];
      ctx.beginPath(); ctx.arc(last.px, Y(last.y), 3.2, 0, 7); ctx.fillStyle = s.color; ctx.fill();
    }
    this._crosshair(ctx, active, X, Y, x0, x1, y0, y1, lx);
  }
  _gridlines(ctx, x0, x1, y0, y1, c, d, ly, Y) {
    ctx.strokeStyle = rgba(hexToRgb(COL["border"]), 0.7); ctx.fillStyle = COL["ink-3"];
    ctx.lineWidth = 1; ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const ticks = ly ? logTicks(c, d) : linTicks(c, d, 4);
    for (const t of ticks) {
      const py = Y(ly ? Math.pow(10, t.v) : t.v);
      if (py < y0 - 1 || py > y1 + 1) continue;
      ctx.beginPath(); ctx.moveTo(x0, py); ctx.lineTo(x1, py); ctx.globalAlpha = 0.5; ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(t.label, x0 - 6, py);
    }
  }
  _crosshair(ctx, active, X, Y, x0, x1, y0, y1, lx) {
    if (!this.hover || this.hover.x < x0 || this.hover.x > x1) return;
    const s = active[0].s;
    // nearest by pixel x
    let bi = 0, bd = Infinity;
    for (let i = 0; i < s.n; i++) { const d = Math.abs(X(s.xs[i]) - this.hover.x); if (d < bd) { bd = d; bi = i; } }
    const px = X(s.xs[bi]);
    ctx.strokeStyle = rgba(hexToRgb(COL["ink-3"]), 0.5); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(px, y0); ctx.lineTo(px, y1); ctx.stroke(); ctx.setLineDash([]);
    // readout box
    const lines = [`it ${fmtSI(s.xs[bi])}`];
    for (const a of active) { const yi = a.s.ys[Math.min(bi, a.s.n - 1)]; lines.push(`${a.label}: ${fmtSci(yi)}`); ctx.beginPath(); ctx.arc(px, Y(yi), 3, 0, 7); ctx.fillStyle = a.color; ctx.fill(); }
    ctx.font = "11px ui-monospace, monospace"; ctx.textBaseline = "top";
    const bw = Math.max(...lines.map(l => ctx.measureText(l).width)) + 14;
    let bx = px + 8; if (bx + bw > x1) bx = px - bw - 8;
    ctx.fillStyle = COL["panel-2"]; ctx.strokeStyle = COL["border"];
    roundRect(ctx, bx, y0 + 4, bw, lines.length * 14 + 8, 3); ctx.fill(); ctx.stroke();
    ctx.textAlign = "left"; ctx.fillStyle = COL["ink-2"];
    lines.forEach((l, i) => ctx.fillText(l, bx + 7, y0 + 10 + i * 14));
  }
  _placeholder(ctx, w, h) { ctx.fillStyle = COL["ink-3"]; ctx.font = "12px var(--sans)"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("waiting for data…", w / 2, h / 2); }
}

function decimate(s, X) {
  // Exact for normal sizes; min/max-per-column when a run gets huge.
  if (s.n <= 4000) { const out = []; for (let i = 0; i < s.n; i++) out.push({ px: X(s.xs[i]), y: s.ys[i] }); return out; }
  const out = []; let col = -1, lo = 0, hi = 0, loY = 0, hiY = 0;
  for (let i = 0; i < s.n; i++) {
    const px = Math.round(X(s.xs[i]));
    if (px !== col) {
      if (col >= 0) { out.push({ px: col, y: loY }); if (hiY !== loY) out.push({ px: col, y: hiY }); }
      col = px; lo = hi = i; loY = hiY = s.ys[i];
    } else { if (s.ys[i] < loY) { loY = s.ys[i]; } if (s.ys[i] > hiY) { hiY = s.ys[i]; } }
  }
  out.push({ px: col, y: s.ys[s.n - 1] });
  return out;
}
function linTicks(a, b, n) { const out = []; const step = niceStep((b - a) / n); const start = Math.ceil(a / step) * step; for (let v = start; v <= b + 1e-9; v += step) out.push({ v, label: fmtSci(v) }); return out; }
function logTicks(a, b) { const out = []; for (let e = Math.floor(a); e <= Math.ceil(b); e++) out.push({ v: e, label: "1e" + e }); return out; }
function niceStep(x) { const p = Math.pow(10, Math.floor(Math.log10(Math.abs(x) || 1))); const f = x / p; return (f >= 5 ? 5 : f >= 2 ? 2 : 1) * p; }
function roundRect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }

// ── 13x13 preflop range grid ────────────────────────────────────────────
const RANKS = "AKQJT98765432";
class RangeGrid {
  constructor(cv) {
    this.cv = cv; this.names = null; this.cellByRC = null; this.rcByName = {};
    this.cur = null; this.target = null; this.actions = null; this.hover = null;
    cv.addEventListener("mousemove", (ev) => { const r = cv.getBoundingClientRect(); this.hover = { x: ev.clientX - r.left, y: ev.clientY - r.top }; });
    cv.addEventListener("mouseleave", () => { this.hover = null; $("range-tip").hidden = true; });
  }
  setNames(names) {
    this.names = names; this.rcByName = {};
    for (const nm of names) { const [r, c] = this._rc(nm); this.rcByName[nm] = [r, c]; }
  }
  _rc(name) {
    if (name.length === 2 && name[0] === name[1]) { const i = RANKS.indexOf(name[0]); return [i, i]; }
    const hi = RANKS.indexOf(name[0]), lo = RANKS.indexOf(name[1]);
    return name.endsWith("s") ? [hi, lo] : [lo, hi];   // suited upper-right, offsuit lower-left
  }
  setTarget(rg) {
    this.actions = rg.actions;
    const cells = rg.cells;
    const grid = Array.from({ length: 13 }, () => Array.from({ length: 13 }, () => null));
    for (let i = 0; i < this.names.length; i++) {
      const [r, c] = this.rcByName[this.names[i]];
      grid[r][c] = { name: this.names[i], mix: cells[i] };
    }
    this.target = grid;
    if (!this.cur) this.cur = grid.map(row => row.map(cell => cell ? { name: cell.name, mix: cell.mix.slice() } : null));
  }
  _bucket(mix) {
    // Fold / call(+check) / raise(+bet+allin) buckets for colouring.
    let f = 0, c = 0, r = 0;
    for (let i = 0; i < this.actions.length; i++) {
      const a = this.actions[i].toLowerCase(); const v = mix[i];
      if (a.startsWith("fold")) f += v;
      else if (a.startsWith("call") || a.startsWith("check")) c += v;
      else r += v;
    }
    return [f, c, r];
  }
  animate() {
    if (!this.cur || !this.target) return false;
    let moving = false;
    for (let r = 0; r < 13; r++) for (let c = 0; c < 13; c++) {
      const cu = this.cur[r][c], tg = this.target[r][c];
      if (!cu || !tg) continue;
      for (let i = 0; i < cu.mix.length; i++) {
        const nv = lerp(cu.mix[i], tg.mix[i], 0.14);
        if (Math.abs(nv - tg.mix[i]) > 1e-4) moving = true;
        cu.mix[i] = nv;
      }
    }
    return moving;
  }
  draw() {
    const { ctx, w, h } = fitCanvas(this.cv);
    ctx.clearRect(0, 0, w, h);
    if (!this.cur) { ctx.fillStyle = COL["ink-3"]; ctx.font = "12px sans-serif"; ctx.textAlign = "center"; ctx.fillText("waiting for checkpoint…", w / 2, h / 2); return; }
    const S = Math.min(w, h), ox = (w - S) / 2, oy = (h - S) / 2, cell = S / 13, gap = Math.max(1, cell * 0.06);
    const fold = hexToRgb(COL["fold"]), call = hexToRgb(COL["call"]), raise = hexToRgb(COL["raise"]);
    ctx.font = `${Math.max(7, cell * 0.30).toFixed(0)}px ui-monospace, monospace`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    let hoverCell = null;
    for (let r = 0; r < 13; r++) for (let c = 0; c < 13; c++) {
      const cellData = this.cur[r][c]; if (!cellData) continue;
      const [bf, bc, br] = this._bucket(cellData.mix);
      const R = Math.round(fold[0] * bf + call[0] * bc + raise[0] * br);
      const G = Math.round(fold[1] * bf + call[1] * bc + raise[1] * br);
      const B = Math.round(fold[2] * bf + call[2] * bc + raise[2] * br);
      const x = ox + c * cell + gap / 2, y = oy + r * cell + gap / 2, sz = cell - gap;
      roundRect(ctx, x, y, sz, sz, Math.max(2, cell * 0.12));
      ctx.fillStyle = `rgb(${R},${G},${B})`; ctx.fill();
      if (r === c) { ctx.strokeStyle = "rgba(255,255,255,.22)"; ctx.lineWidth = 1; ctx.stroke(); }
      // label
      ctx.fillStyle = "rgba(255,255,255,.9)";
      if (cell > 22) ctx.fillText(cellData.name, x + sz / 2, y + sz / 2);
      if (this.hover && this.hover.x >= x && this.hover.x < x + sz && this.hover.y >= y && this.hover.y < y + sz) hoverCell = { cellData, x, y, sz };
    }
    if (hoverCell) this._tip(hoverCell); else $("range-tip").hidden = true;
  }
  _tip(hc) {
    const tip = $("range-tip"); const cd = hc.cellData;
    let rows = "";
    for (let i = 0; i < this.actions.length; i++) rows += `<div class="tr"><span>${this.actions[i]}</span><span>${(cd.mix[i] * 100).toFixed(0)}%</span></div>`;
    tip.innerHTML = `<div class="th">${cd.name}</div>${rows}`;
    tip.style.left = (hc.x + hc.sz / 2) + "px"; tip.style.top = hc.y + "px"; tip.hidden = false;
  }
}

// ── radial gauge ────────────────────────────────────────────────────────
function drawGauge(cv, frac, color, label) {
  const { ctx, w, h } = fitCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 8;
  const a0 = Math.PI * 0.75, a1 = Math.PI * 2.25;
  ctx.lineWidth = 8; ctx.lineCap = "round";
  ctx.beginPath(); ctx.arc(cx, cy, r, a0, a1); ctx.strokeStyle = rgba(hexToRgb(COL["border"]), 1); ctx.stroke();
  if (frac != null) {
    ctx.beginPath(); ctx.arc(cx, cy, r, a0, a0 + (a1 - a0) * clamp(frac, 0, 1));
    ctx.strokeStyle = color; ctx.stroke();
  }
  ctx.fillStyle = COL.ink; ctx.font = "560 16px ui-monospace, monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(frac == null ? "—" : Math.round(frac * 100) + "%", cx, cy - 2);
}

// ── event log ───────────────────────────────────────────────────────────
function addLog(ts, kind, msg) {
  const ul = $("event-log");
  const li = document.createElement("li");
  const t = new Date(ts * 1000).toLocaleTimeString("en-US", { hour12: false });
  li.innerHTML = `<span class="t">${t}</span><span class="kind ${kind}">${kind.replace("_", " ")}</span><span class="msg"></span>`;
  li.querySelector(".msg").textContent = msg;
  ul.appendChild(li);
  while (ul.children.length > 200) ul.removeChild(ul.firstChild);
}

// ── stacked bars (action / bet distributions) ───────────────────────────
const ACTION_COL = { f: "--fold", x: "--accent", c: "--call", r: "--raise", A: "--red" };
const ACTION_LBL = { f: "fold", x: "check", c: "call", r: "raise/bet", A: "all-in" };
const BET_COL = { "0.5pot": "--call", "1.0pot": "--amber", allin: "--red" };
function renderBars(containerId, freqByStreet, colorMap, labelMap) {
  const el = $(containerId); if (!freqByStreet) { return; }
  const streets = ["P", "F", "T", "R"].filter(s => freqByStreet[s]);
  el.innerHTML = "";
  for (const s of streets) {
    const row = document.createElement("div"); row.className = "bar-row";
    const label = document.createElement("div"); label.className = "st"; label.textContent = s;
    const track = document.createElement("div"); track.className = "bar-track";
    const mix = freqByStreet[s];
    const keys = Object.keys(mix);
    for (const k of keys) {
      const seg = document.createElement("div"); seg.className = "bar-seg";
      const pct = clamp(mix[k], 0, 1) * 100;
      seg.style.width = pct + "%";
      seg.style.background = CSSVar(colorMap[k] || "--ink-3") || "#556";
      seg.title = `${(labelMap && labelMap[k]) || k}: ${pct.toFixed(0)}%`;
      if (pct > 9) { const sp = document.createElement("span"); sp.textContent = Math.round(pct) + (pct > 22 ? "% " + ((labelMap && labelMap[k]) || k) : "%"); seg.appendChild(sp); }
      track.appendChild(seg);
    }
    row.appendChild(label); row.appendChild(track); el.appendChild(row);
  }
}

// ── completion banner + chime + notification ────────────────────────────
let completionShown = false;
function showCompletion(e) {
  if (completionShown) return; completionShown = true;
  const ok = e.reason === "completed";
  $("done-badge").textContent = ok ? "✓" : "!"; $("done-badge").className = "done-badge" + (ok ? "" : " err");
  $("done-title").textContent = ok ? "Training complete" : (e.reason === "error" ? "Training errored" : "Training stopped");
  $("done-sub").textContent = ok ? `${store.game} · ${fmtInt(e.iteration)} iterations solved.` : (e.summary && e.summary.note || "The run did not finish cleanly.");
  const S = e.summary || {}; const stats = [];
  const add = (l, v) => stats.push(`<div class="d"><div class="dl">${l}</div><div class="dv">${v}</div></div>`);
  add("iterations", fmtSI(e.iteration));
  add("wall clock", fmtDur(S.wall_seconds != null ? S.wall_seconds : store.elapsedNow()));
  if (S.final_exploitability != null) add("exploitability", fmtSci(S.final_exploitability));
  if (S.reduction_factor != null) add("expl. reduction", fmtSI(S.reduction_factor) + "×");
  if (S.final_mean_abs_regret != null) add("mean |regret|", fmtSci(S.final_mean_abs_regret));
  if (S.final_hh_bb_per_100 != null) add("head-to-head", (S.final_hh_bb_per_100 >= 0 ? "+" : "") + S.final_hh_bb_per_100.toFixed(1) + " bb/100");
  if (S.infosets != null) add("infosets", fmtSI(S.infosets));
  $("done-stats").innerHTML = stats.join("");
  $("done-overlay").hidden = false;
  chime(ok); notify(ok, e);
}
function chime(ok) {
  try {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const notes = ok ? [523.25, 659.25, 783.99, 1046.5] : [440, 330];
    notes.forEach((f, i) => {
      const o = ac.createOscillator(), g = ac.createGain();
      o.type = "sine"; o.frequency.value = f;
      const t = ac.currentTime + i * 0.12;
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.22, t + 0.03); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.5);
      o.connect(g); g.connect(ac.destination); o.start(t); o.stop(t + 0.55);
    });
  } catch (_) { /* audio needs a gesture in some browsers; the banner is the primary signal */ }
}
function notify(ok, e) {
  try {
    if (!("Notification" in window)) return;
    if (Notification.permission === "granted")
      new Notification(ok ? "parciver — training complete" : "parciver — run stopped", { body: `${store.game}: ${fmtInt(e.iteration)} iterations.`, silent: false });
  } catch (_) {}
}

// ── charts + grid instances ─────────────────────────────────────────────
loadColors();
const grid = new RangeGrid($("c-range"));
const charts = {
  conv: new Chart($("c-convergence"), { logX: true, logY: true }),
  itps: new Chart($("c-throughput"), { pad: { l: 44, r: 10, t: 8, b: 16 } }),
  cov: new Chart($("c-coverage"), { pad: { l: 44, r: 10, t: 8, b: 16 } }),
  hh: new Chart($("c-hh"), { pad: { l: 40, r: 10, t: 8, b: 16 }, y0: 0 }),
  stab: new Chart($("c-stability"), { logX: true, logY: true, pad: { l: 44, r: 10, t: 8, b: 16 } }),
};

// ── header rendering ────────────────────────────────────────────────────
function renderHeader() {
  const L = store.latest;
  const iter = store.displayIter();  // smoothly extrapolated between updates
  $("game-name").textContent = store.game ? `${store.game} · ${store.variant}` : "— waiting for run —";
  $("stat-iter").textContent = fmtSI(iter);
  $("stat-target").textContent = store.target ? fmtSI(store.target) : "—";
  $("stat-progress").textContent = store.target ? (iter / store.target * 100).toFixed(2) + "%" : "—";
  $("stat-itps").innerHTML = anim.itps ? fmtSI(anim.itps) + '<span class="unit"> it/s</span>' : "—";
  // convergence readout: exact exploitability if we have it, else the regret proxy.
  if (L.exploitability != null) { $("stat-conv-label").textContent = "exploitability"; $("stat-conv").textContent = fmtSci(anim.regret); }
  else if (L.mean_abs_regret != null) { $("stat-conv-label").textContent = "mean |regret|"; $("stat-conv").textContent = fmtSci(anim.regret); }
  const st = $("stat-status");
  const dotClass = { live: "live", done: "done", err: "err", connecting: "idle" }[store.status] || "idle";
  st.innerHTML = `<span class="dot ${dotClass}"></span>${store.status}`;
  $("stat-elapsed").textContent = fmtDur(store.elapsedNow());
  // ETA from current throughput
  let eta = "—";
  if (store.status === "live" && store.target && L.it_per_s > 0)
    eta = fmtDur((store.target - iter) / L.it_per_s);
  else if (store.status === "done") eta = "done";
  $("stat-eta").textContent = eta;

  // config chips (once)
  if (store.config && !renderHeader._chips) {
    renderHeader._chips = true;
    const c = store.config, chips = $("config-chips"); chips.innerHTML = "";
    const push = (t) => { const s = document.createElement("span"); s.className = "tag"; s.textContent = t; chips.appendChild(s); };
    if (c.stack_bb) push(c.stack_bb + "bb");
    if (c.bet_grid) push(c.bet_grid);
    if (c.abstraction) push(`abs ${fmtSI(c.abstraction.preflop)}/${fmtSI(c.abstraction.flop)}/${fmtSI(c.abstraction.turn)}/${fmtSI(c.abstraction.river)}`);
    if (c.solver) push(c.solver);
  }
}

// convergence legend reflects which series exist
function renderConvLegend() {
  const has = [];
  if (store.series.expl.n) has.push(["exploitability (exact)", COL["accent"]]);
  if (store.series.regret.n) has.push(["mean |regret| (proxy)", COL["amber"]]);
  const el = $("conv-legend");
  el.innerHTML = has.map(([l, col]) => `<span class="li"><i style="background:${col}"></i>${l}</span>`).join("");
}

// ── the single render loop ──────────────────────────────────────────────
function frame() {
  const D = store.dirty;
  const now = performance.now();
  const dt = frame._t ? Math.min(0.1, (now - frame._t) / 1000) : 0.016;
  frame._t = now;
  updateAnim(dt);
  // header + live clock/eta every frame (cheap text)
  renderHeader();

  if (D.conv) {
    renderConvLegend();
    charts.conv.draw([
      { s: store.series.expl, color: COL["accent"], fill: true, label: "expl" },
      { s: store.series.regret, color: COL["amber"], fill: true, label: "regret" },
    ]);
    D.conv = false;
  }
  if (D.itps) { charts.itps.draw([{ s: store.series.itps, color: COL["accent-2"], fill: true, label: "it/s" }]); D.itps = false; }
  if (D.cov) { charts.cov.draw([{ s: store.series.infosets, color: COL["amber"], fill: true, label: "infosets" }]); D.cov = false; }
  if (D.hh) { charts.hh.draw([{ s: store.series.hh, color: COL["green"], fill: true, label: "bb/100" }]); D.hh = false; }
  if (D.stab) { charts.stab.draw([{ s: store.series.stability, color: COL["accent"], fill: true, label: "L1 Δ" }]); $("kpi-stability").textContent = store.latest.strategy_l1_delta != null ? fmtSci(store.latest.strategy_l1_delta) : "—"; D.stab = false; }
  // KPIs glide every frame off the eased anim values (not just on data ticks).
  $("kpi-itps").textContent = store.series.itps.n ? fmtSI(anim.itps) + " it/s" : "—";
  $("kpi-infosets").textContent = store.series.infosets.n ? fmtSI(anim.infosets) : "—";
  $("kpi-hh").textContent = anim.hh != null ? (anim.hh >= 0 ? "+" : "") + anim.hh.toFixed(1) + " bb/100" : "—";

  // range grid animates toward the latest checkpoint
  const moving = grid.animate();
  if (D.grid || moving || grid.hover) { grid.draw(); D.grid = false; }

  if (D.bars && store.checkpoint) {
    renderBars("action-bars", store.checkpoint.action_freq, ACTION_COL, ACTION_LBL);
    renderBars("bet-bars", store.checkpoint.bet_size_freq, BET_COL, null);
    D.bars = false;
  }
  // Gauges glide every frame (eased) while live, so the arcs sweep smoothly rather
  // than snapping when a resources reading lands.
  if (store.latest.resources && (D.gauges || store.status === "live")) {
    const r = store.latest.resources;
    drawGauge($("g-cpu"), anim.cpu / 100, COL["accent-2"]);
    drawGauge($("g-gpu"), anim.gpu / 100, COL["green"]);
    drawGauge($("g-ram"), anim.ram / 100, COL["amber"]);
    $("gsub-cpu").textContent = r.cpu_pct != null ? Math.round(anim.cpu) + "%" : "—";
    $("gsub-gpu").textContent = r.gpu_mem_used_gb != null ? `${r.gpu_mem_used_gb.toFixed(1)}/${(r.gpu_mem_total_gb || 0).toFixed(0)}GB` : "—";
    $("gsub-ram").textContent = r.ram_used_gb != null ? `${r.ram_used_gb.toFixed(0)}/${(r.ram_total_gb || 0).toFixed(0)}GB` : "—";
    $("kpi-threads").textContent = r.threads != null ? r.threads + " threads" : "—";
    D.gauges = false;
  }
  requestAnimationFrame(frame);
}

function buildGauges() {
  const el = $("gauges"); el.__built = true; el.innerHTML = "";
  for (const [id, label] of [["cpu", "CPU"], ["gpu", "GPU"], ["ram", "RAM"]]) {
    const g = document.createElement("div"); g.className = "gauge";
    g.innerHTML = `<canvas id="g-${id}"></canvas><div class="g-label">${label}</div><div class="g-sub" id="gsub-${id}">—</div>`;
    el.appendChild(g);
  }
}

// redraw all charts on resize
let resizeRAF = 0;
window.addEventListener("resize", () => {
  cancelAnimationFrame(resizeRAF);
  resizeRAF = requestAnimationFrame(() => { for (const k in store.dirty) store.dirty[k] = true; store.dirty.grid = true; });
});

// ── SSE connection ──────────────────────────────────────────────────────
function connect() {
  const es = new EventSource("/events");
  es.onmessage = (ev) => {
    let batch; try { batch = JSON.parse(ev.data); } catch (_) { return; }
    for (const e of batch) { try { store.apply(e); } catch (err) { console.error("apply", err, e); } }
  };
  es.onopen = () => { if (store.status === "connecting") { $("stat-status").innerHTML = '<span class="dot idle"></span>waiting for events'; } };
  es.onerror = () => { /* EventSource auto-reconnects; leave status as-is */ };
}

// Dismiss the completion banner: the button, a click on the backdrop, or Escape.
function dismissCompletion() { $("done-overlay").hidden = true; }
$("done-dismiss").addEventListener("click", dismissCompletion);
$("done-overlay").addEventListener("click", (e) => { if (e.target === $("done-overlay")) dismissCompletion(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") dismissCompletion(); });

// try to secure notification permission up front (best-effort; needs no gesture in most browsers)
try { if ("Notification" in window && Notification.permission === "default") Notification.requestPermission().catch(() => {}); } catch (_) {}

buildGauges();
connect();
requestAnimationFrame(frame);
