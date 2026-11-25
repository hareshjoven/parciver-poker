// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
/* FLOPSEARCH M2 viewer. Data-only: the watcher (scripts/watch_m2.py) reads m2.json after each saved chunk and
   sends the curves, the 50-block preview and the flop latencies; the page only extrapolates the pace and draws. */
"use strict";
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
const num = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
const sgn = (v, d = 1) => num(v) ? `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}` : "—";

function duration(s) {
  if (!num(s)) return "—";
  s = Math.max(0, Math.round(s));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60), q = s % 60;
  const p = (x) => String(x).padStart(2, "0");
  return d ? `${d}d ${h}h ${p(m)}m` : h ? `${h}h ${p(m)}m ${p(q)}s` : m ? `${m}m ${p(q)}s` : `${q}s`;
}
function log(text, ts, kind = "note") {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${new Date(ts || Date.now()).toLocaleTimeString()}</span><span class="kind ${esc(kind)}">${esc(kind)}</span><span class="msg">${esc(text)}</span>`;
  $("event-log").prepend(li);
}

const state = { config: null, m2: null, alive: null, res: null, ended: null, endReason: null, lastLine: null, cores: null, idle: null, hits: {} };
const done = () => !!state.m2?.summary;
const over = () => done() || state.alive === false;

// ---- header ---------------------------------------------------------------------------------------------
/** The run's start: the first chunk's end minus its time (the watcher may start later than the run). */
function started() {
  const m = state.m2;
  if (m && num(m.first_ended) && num(m.first_seconds)) return m.first_ended - m.first_seconds;
  return state.config?.started ?? null;
}
function lastSave() {
  const p = state.m2?.pace;
  return p?.length ? p[p.length - 1][5] : null;
}
function renderHeader() {
  const c = state.config, m = state.m2;
  if (!c) return;
  $("sub").textContent = `reads ${c.out} only${c.pid ? ` · run pid ${c.pid}` : ""} · ${c.rule}`;
  if (!m) {  // before the first save: the run's own CPU says it is working
    $("stat-pace").textContent = num(state.cores) ? `run using ${state.cores.toFixed(1)} cores` : "—";
    updateClocks();
    return;
  }
  const cur = m.curve[m.curve.length - 1];
  $("stat-mean").textContent = cur ? sgn(cur[1]) : "—";
  $("stat-mean").style.color = cur ? css(cur[1] >= 0 ? "--ok" : "--hot") : "";
  $("stat-mean-sub").textContent = cur && num(cur[2]) ? `bb/100 ± ${cur[2].toFixed(1)} se` : "bb/100 so far";
  $("stat-deals").textContent = m.deals_done.toLocaleString();
  $("stat-deals-sub").textContent = `of ${m.deals.toLocaleString()} · ${(100 * m.deals_done / m.deals).toFixed(1)}%`;
  $("stat-pace").textContent = m.deals_done ? `${(m.seconds / m.deals_done).toFixed(2)} s/deal` +
    (m.latency ? ` · flop ${m.latency.median.toFixed(2)} s` : "") : "—";
  const res = state.res;
  $("stat-res").textContent = [res && num(res.gpu_pct) ? `${res.gpu_pct.toFixed(0)}%` + (num(res.gpu_mem_used_gb) ? ` ${res.gpu_mem_used_gb.toFixed(1)} GB` : "") : "—",
    res && num(res.cpu_pct) ? `${res.cpu_pct.toFixed(0)}%` : "—",
    res && num(res.ram_used_gb) ? `${res.ram_used_gb.toFixed(1)}/${res.ram_total_gb.toFixed(0)} GB` : "—"].join(" · ");
  updateClocks();
}
function updateClocks() {
  const m = state.m2;
  if (!state.config) return;
  const now = Date.now() / 1000, t0 = started(), last = lastSave();
  let text = num(t0) ? duration((over() && num(last) ? last : now) - t0) : "—";
  if (done()) text += " · complete";
  else if (over()) text += " · stopped";
  else if (m?.deals_done) {
    const per = m.seconds / m.deals_done, left = (m.deals - m.deals_done) * per - (num(last) ? now - last : 0);
    text += ` · ${duration(Math.max(0, left))} left`;
  }
  $("stat-eta").textContent = text;
  let dot = "live", label = m?.deals_done ? "running" : "starting (first save after one chunk)";
  if (done()) { dot = "done"; label = "complete · reading written"; }
  else if (state.alive === false) { dot = "err"; label = "M2 exited before its last deal"; }
  else if (m?.deals_done && num(last)) {
    const n = m.bars.length ? m.bars[m.bars.length - 1][2] : 250, due = n * m.seconds / m.deals_done;
    if (now - last > Math.max(900, 3 * due)) { dot = "err"; label = `no save for ${duration(now - last)}: stalled?`; }
  }
  if (!over() && state.alive && num(state.idle) && state.idle > 300) {  // the watcher's IDLE_WARN
    dot = "err"; label = `no CPU used for ${duration(state.idle)}: hung?`;
  }
  if (state.ended && !over()) { dot = "idle"; label = "watcher stopped"; }
  $("stat-status").innerHTML = `<span class="dot ${dot}"></span>${esc(label)}`;
  const b = $("banner");
  if (state.alive === false && !done()) {
    b.textContent = `M2 stopped at ${m?.deals_done ?? 0} deals: ${state.lastLine || "no log line"}. Report before re-running anything.`;
    b.classList.add("on");
  } else b.classList.remove("on");
}

// ---- verdict ---------------------------------------------------------------------------------------------
function renderVerdict() {
  const m = state.m2;
  if (!m) return;
  const s = m.summary || m.preview, vd = $("verdict");
  if (!s) { vd.className = "verdict wait"; vd.textContent = "waiting for 100 deals"; $("ci").innerHTML = ""; }
  else {
    const down = s.hi < 0;
    if (m.summary) { vd.className = `verdict ${down ? "no" : "yes"}`; vd.textContent = down ? "DOWN · (b) fails" : "not DOWN · (b) holds → ADOPT"; }
    else { vd.className = "verdict wait"; vd.textContent = `preview at ${s.deals.toLocaleString()} deals: ${down ? "below 0" : "not below 0"} (decides nothing yet)`; }
    const span = Math.max(Math.abs(s.lo), Math.abs(s.hi), 1) * 1.15, X = (v) => `${50 + 50 * v / span}%`;
    $("ci").innerHTML = `<div class="zero" style="left:50%"></div><div class="band" style="left:${X(s.lo)};width:calc(${X(s.hi)} - ${X(s.lo)})"></div><div class="pt" style="left:${X(s.mean)}"></div>`;
  }
  const rows = [];
  if (s) rows.push(["mean", `${sgn(s.mean, 2)} bb/100`], ["95% t (50 blocks)", `[${sgn(s.lo, 2)}, ${sgn(s.hi, 2)}]`]);
  if (num(m.a_share)) rows.push(["flop searcher: search decides", `${(100 * m.a_share).toFixed(0)}%`]);
  if (num(m.b_share)) rows.push(["turn searcher: search decides", `${(100 * m.b_share).toFixed(0)}%`]);
  const c = m.calls || {}, n = m.deals_done || 1;
  if (m.deals_done) rows.push(["solves / deal (flop · turn · river)", `${((c["3"] || 0) / n).toFixed(2)} · ${((c["4"] || 0) / n).toFixed(2)} · ${((c["5"] || 0) / n).toFixed(2)}`]);
  $("kv").innerHTML = rows.map(([k, v]) => `<span class="k">${esc(k)}</span><span class="v">${esc(v)}</span>`).join("");
}

// ---- charts ----------------------------------------------------------------------------------------------
function setup(canvas) {
  const dpr = window.devicePixelRatio || 1, r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height };
}
function niceTicks(lo, hi, count = 5) {
  const span = hi - lo || 1, raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => span / s <= count) || 10 * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function empty(ctx, w, h, text) {
  ctx.fillStyle = css("--ink-3"); ctx.font = `12px ${css("--sans")}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, w / 2, h / 2);
}
/** Axes on [x0, x1] × [lo, hi]; returns the scales. */
function axes(ctx, w, h, x0, x1, lo, hi, fmtX = (t) => t.toLocaleString()) {
  const L = 52, R = 18, T = 10, B = 24;
  const X = (v) => L + (v - x0) / (x1 - x0 || 1) * (w - L - R), Y = (v) => T + (1 - (v - lo) / (hi - lo || 1)) * (h - T - B);
  ctx.font = `11px ${css("--mono")}`; ctx.lineWidth = 1;
  for (const t of niceTicks(lo, hi, 5)) {
    ctx.strokeStyle = css(t === 0 ? "--ink-3" : "--border"); ctx.beginPath(); ctx.moveTo(L, Y(t)); ctx.lineTo(w - R, Y(t)); ctx.stroke();
    ctx.fillStyle = css("--ink-3"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(String(t), L - 6, Y(t));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of niceTicks(x0, x1, 5)) ctx.fillText(fmtX(t), X(t), h - B + 7);
  return { X, Y, L, R, T, B };
}
function drawCum() {
  const id = "c-cum", { ctx, w, h } = setup($(id)), m = state.m2;
  state.hits[id] = [];
  if (!m?.curve.length) return empty(ctx, w, h, "first save at 250 deals");
  const pts = m.curve, band = pts.map(([d, y, se]) => [d, y, num(se) ? 1.96 * se : 0]);
  const ys = band.flatMap(([, y, e]) => [y - e, y + e]).concat([0]);
  let lo = Math.min(...ys), hi = Math.max(...ys); const pad = 0.06 * (hi - lo || 1); lo -= pad; hi += pad;
  const { X, Y } = axes(ctx, w, h, 0, m.deals, lo, hi);
  const a = css("--armA");
  ctx.fillStyle = a; ctx.globalAlpha = 0.18; ctx.beginPath();
  band.forEach(([d, y, e], i) => { if (i) ctx.lineTo(X(d), Y(y + e)); else ctx.moveTo(X(d), Y(y + e)); });
  [...band].reverse().forEach(([d, y, e]) => ctx.lineTo(X(d), Y(y - e)));
  ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
  ctx.strokeStyle = a; ctx.lineWidth = 2; ctx.beginPath();
  pts.forEach(([d, y], i) => { if (i) ctx.lineTo(X(d), Y(y)); else ctx.moveTo(X(d), Y(y)); });
  ctx.stroke(); ctx.fillStyle = a;
  for (const [d, y, se] of pts) {
    ctx.beginPath(); ctx.arc(X(d), Y(y), 2.5, 0, 2 * Math.PI); ctx.fill();
    state.hits[id].push({ x: X(d), y: Y(y), html: `<div class="th">${d.toLocaleString()} deals</div><div class="tr"><span>cumulative</span><span>${sgn(y)} bb/100</span></div>` +
      (num(se) ? `<div class="tr"><span>95%</span><span>[${sgn(y - 1.96 * se)}, ${sgn(y + 1.96 * se)}]</span></div>` : "") });
  }
}
function drawBars() {
  const id = "c-bars", { ctx, w, h } = setup($(id)), m = state.m2;
  state.hits[id] = [];
  if (!m?.bars.length) return empty(ctx, w, h, "first save at 250 deals");
  const ys = m.bars.map((b) => b[1]).concat([0]);
  const lim = Math.max(...ys.map(Math.abs)) * 1.1 || 1;
  const { X, Y } = axes(ctx, w, h, 0, m.deals, -lim, lim);
  for (const [d, y, n] of m.bars) {
    ctx.fillStyle = css(y >= 0 ? "--ok" : "--hot");
    const bw = Math.max(2, (X(n) - X(0)) * 0.8), mid = X(d - n / 2), y0 = Y(0), y1 = Y(y);
    ctx.fillRect(mid - bw / 2, Math.min(y0, y1), bw, Math.abs(y1 - y0));
    state.hits[id].push({ x: mid, y: y1, html: `<div class="th">deals ${(d - n).toLocaleString()}–${(d - 1).toLocaleString()}</div><div class="tr"><span>chunk</span><span>${sgn(y)} bb/100</span></div>` });
  }
}
function drawHist() {
  const id = "c-hist", { ctx, w, h } = setup($(id)), m = state.m2;
  state.hits[id] = [];
  const L = m?.latency;
  $("lat-kpi").textContent = L ? `median ${L.median.toFixed(2)} s · p95 ${L.p95.toFixed(2)} s · max ${L.max.toFixed(2)} s · ${L.n.toLocaleString()} solves` : "—";
  if (!m?.hist.length) return empty(ctx, w, h, "no flop solve saved yet");
  const bin = m.hist_bin, xmax = m.hist.length * bin;
  let first = m.hist.findIndex((c) => c > 0), last = m.hist.length - 1 - [...m.hist].reverse().findIndex((c) => c > 0);
  const x0 = Math.max(0, Math.floor((first - 2) * bin)), x1 = Math.min(xmax, Math.max(2.2, Math.ceil((last + 3) * bin)));
  const top = Math.max(...m.hist) * 1.1;
  const { X, Y } = axes(ctx, w, h, x0, x1, 0, top, (t) => `${t}s`);
  ctx.fillStyle = css("--armA");
  m.hist.forEach((c, i) => {
    if (!c) return;
    const a = X(i * bin), b = X((i + 1) * bin);
    ctx.fillRect(a + 0.5, Y(c), Math.max(1, b - a - 1), Y(0) - Y(c));
    state.hits[id].push({ x: (a + b) / 2, y: Y(c), html: `<div class="th">${(i * bin).toFixed(2)}–${((i + 1) * bin).toFixed(2)} s</div><div class="tr"><span>solves</span><span>${c}</span></div>` });
  });
  ctx.strokeStyle = css("--hot"); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(X(2), Y(top)); ctx.lineTo(X(2), Y(0)); ctx.stroke(); ctx.setLineDash([]);
  if (L) {
    ctx.strokeStyle = css("--amber"); ctx.beginPath(); ctx.moveTo(X(L.median), Y(top)); ctx.lineTo(X(L.median), Y(0)); ctx.stroke();
  }
}
function drawPace() {
  const id = "c-pace", { ctx, w, h } = setup($(id)), m = state.m2;
  state.hits[id] = [];
  if (!m?.pace.length) return empty(ctx, w, h, "first save at 250 deals");
  const top = Math.max(...m.pace.flatMap((p) => p.slice(1, 5))) * 1.15;
  const { X, Y } = axes(ctx, w, h, 0, m.deals, 0, top);
  const series = [[1, "--armA", "s/deal"], [2, "--amber", "flop solves/deal"], [3, "--ok", "turn solves/deal"], [4, "--hot", "river solves/deal"]];
  for (const [k, col, name] of series) {
    ctx.strokeStyle = css(col); ctx.lineWidth = k === 1 ? 2 : 1.5; ctx.beginPath();
    m.pace.forEach((p, i) => { if (i) ctx.lineTo(X(p[0]), Y(p[k])); else ctx.moveTo(X(p[0]), Y(p[k])); });
    ctx.stroke();
    ctx.fillStyle = css(col);
    for (const p of m.pace) {
      ctx.beginPath(); ctx.arc(X(p[0]), Y(p[k]), 2, 0, 2 * Math.PI); ctx.fill();
      state.hits[id].push({ x: X(p[0]), y: Y(p[k]), html: `<div class="th">chunk to ${p[0].toLocaleString()}</div><div class="tr"><span>${name}</span><span>${p[k].toFixed(2)}</span></div>` });
    }
  }
}

function render() { renderHeader(); renderVerdict(); drawCum(); drawBars(); drawHist(); drawPace(); }

// ---- events ----------------------------------------------------------------------------------------------
let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } }
function finish() {
  const m = state.m2, s = m?.summary;
  if (s) {
    $("done-title").textContent = s.down ? "M2: DOWN · (b) fails" : "M2: not DOWN · (b) holds";
    $("done-detail").textContent = `${sgn(s.mean, 2)} bb/100, 95% [${sgn(s.lo, 2)}, ${sgn(s.hi, 2)}] over ${m.deals_done.toLocaleString()} deals. ` +
      `Flop latency median ${s.flop_median.toFixed(2)} s, p95 ${s.flop_p95.toFixed(2)} s.`;
    if (s.down) { $("done-badge").textContent = "!"; $("done-badge").classList.add("err"); }
  } else {
    $("done-badge").textContent = "!"; $("done-badge").classList.add("err"); $("done-title").textContent = "M2 stopped";
    $("done-detail").textContent = `At ${m?.deals_done ?? 0} deals: ${state.lastLine || "no log line"}.`;
  }
  $("done-overlay").hidden = false;
}
function applyEvent(e) {
  const ts = Number(e.ts) * 1000 || Date.now();
  if (e.type === "run_start") {
    if (e.config?.study !== "m2") {
      $("sub").textContent = `this run is not an M2 watch (study: ${e.config?.study ?? e.variant ?? "unknown"})`;
      events.close();
      return;
    }
    Object.assign(state, { config: e.config, m2: null, alive: null, ended: null, endReason: null, cores: null, idle: null });
    $("event-log").innerHTML = "";
    log(`watching ${e.config.out}${e.config.pid ? ` (pid ${e.config.pid})` : ""}`, ts);
  } else if (e.type === "metrics") {
    if (e.resources) state.res = e.resources;
    if ("m2_alive" in e) state.alive = e.m2_alive;
    if (e.m2_last_line) state.lastLine = e.m2_last_line;
    if ("m2_cores" in e) state.cores = e.m2_cores;
    if ("m2_idle_s" in e) state.idle = e.m2_idle_s;
  } else if (e.type === "checkpoint") {
    if (e.m2) state.m2 = e.m2;
  } else if (e.type === "milestone") {
    log(e.message || e.kind, ts, e.kind === "warning" ? "warning" : "note");
  } else if (e.type === "run_end") {
    state.ended = ts; state.endReason = e.reason;
    log(e.reason === "completed" ? "M2 complete" : `watch ended (${e.summary?.note ?? e.reason})`, ts, e.reason === "completed" ? "note" : "warning");
    if (over()) finish();
  }
  schedule();
}

document.addEventListener("mousemove", (ev) => {
  const tip = $("tip"), el = ev.target;
  const hits = el && el.tagName === "CANVAS" ? state.hits[el.id] : null;
  if (!hits || !hits.length) { tip.hidden = true; return; }
  const r = el.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
  let best = null, bd = Infinity;
  for (const h of hits) { const d = Math.hypot(h.x - mx, h.y - my); if (d < bd) { bd = d; best = h; } }
  if (!best || bd > 14) { tip.hidden = true; return; }
  tip.innerHTML = best.html; tip.hidden = false;
  tip.style.left = `${r.left + best.x + 12}px`; tip.style.top = `${r.top + best.y - 8}px`;
});

const events = new EventSource("/events");
events.onmessage = (m) => JSON.parse(m.data).forEach(applyEvent);
let failures = 0;
events.onopen = () => { failures = 0; };
events.onerror = () => {
  failures += 1;
  if (failures >= 3) { events.close(); log("dashboard server stopped (the watcher exited); M2 is unaffected", Date.now(), "warning"); }
  else if (failures === 1) log("telemetry reconnecting…", Date.now(), "warning");
};
window.addEventListener("resize", schedule);
$("done-dismiss").onclick = () => { $("done-overlay").hidden = true; };
setInterval(updateClocks, 1000);
render();
