/* Response-convergence study viewer. Data-only: every number comes from the
   producer (scripts/response_convergence.py); nothing is recomputed here except
   replicate means for plotting. */
"use strict";
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const OPP_COLOR = { "anchor": "--s-anchor", "nit(0.35)": "--s-nit", "station(0.35)": "--s-station" };
// Sequential blue ramp for iteration budget N: larger N = more salient on the dark surface.
// Colors are interpolated evenly between the ramp ends for however many budgets the
// run has, so a short ladder never lands on two near-identical light stops.
const N_RAMP_ENDS = [[0x1c, 0x5c, 0xab], [0xcd, 0xe2, 0xfb]];
function nColor(i, count) {
  const f = count > 1 ? i / (count - 1) : 1, [a, b] = N_RAMP_ENDS;
  return "#" + a.map((v, k) => Math.round(v + (b[k] - v) * f).toString(16).padStart(2, "0")).join("");
}
const Z99 = 2.576;

const state = { config: null, rows: [], keys: new Set(), progress: null, rate: null, resources: null,
                solve: null, solveAt: 0, started: null, finished: null, updatesDone: 0, hits: {} };

const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "—";
const sgn = (v, d = 1) => Number.isFinite(Number(v)) ? (v >= 0 ? "+" : "") + Number(v).toFixed(d) : "—";
const iters = (n) => n >= 1e6 ? `${+(n / 1e6).toFixed(n % 1e6 ? 1 : 0)}M` : n >= 1e3 ? `${n / 1e3}k` : String(n);
const tauLabel = (t) => t === 0 ? "0" : t.toExponential(0).replace("e-", "e−");
function duration(s) {
  if (!Number.isFinite(s)) return "—";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), q = s % 60, p = (x) => String(x).padStart(2, "0");
  return h ? `${h}h ${p(m)}m ${p(q)}s` : m ? `${m}m ${p(q)}s` : `${q}s`;
}
function log(text, ts) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${new Date(ts || Date.now()).toLocaleTimeString()}</span><span class="kind note">rc</span><span class="msg">${esc(text)}</span>`;
  $("event-log").prepend(li);
}

// ---- data shaping ------------------------------------------------------
function candidate(row, ti) { return row.candidates[ti]; }
/** mean over replicates of candidate ti for (opponent, n); eval-noise-only 99% half-width. */
function agg(opp, n, ti) {
  const rs = state.rows.filter((r) => r.opponent === opp && r.n === n);
  if (!rs.length) return null;
  const cs = rs.map((r) => candidate(r, ti));
  const mean = cs.reduce((a, c) => a + c.gain, 0) / cs.length;
  const half = Z99 * Math.sqrt(cs.reduce((a, c) => a + c.se * c.se, 0)) / cs.length;
  const kept = cs.reduce((a, c) => a + c.kept_frac, 0) / cs.length;
  return { mean, half, kept, reps: cs, tau: cs[0].tau };
}

// ---- canvas plumbing ----------------------------------------------------
function setup(canvas) {
  const dpr = window.devicePixelRatio || 1, r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height };
}
function niceTicks(lo, hi, count = 5) {
  const span = hi - lo || 1, raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
/** Frame with a y axis that always includes 0; returns scale helpers. */
function frame(ctx, w, h, xs, ys, xTicks, pad = { l: 52, r: 96, t: 12, b: 28 }) {
  let lo = Math.min(0, ...ys), hi = Math.max(0, ...ys);
  const m = (hi - lo) * 0.08 || 1; lo -= m; hi += m;
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const X = (v) => pad.l + (x1 === x0 ? 0.5 : (v - x0) / (x1 - x0)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
  ctx.font = `11px ${css("--mono")}`; ctx.lineWidth = 1;
  for (const v of niceTicks(lo, hi)) {  // recessive grid
    ctx.strokeStyle = css("--border"); ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke();
    ctx.fillStyle = css("--ink-3"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(sgn(v, 0), pad.l - 8, Y(v));
  }
  ctx.strokeStyle = css("--ink-3"); ctx.setLineDash([4, 4]); ctx.beginPath();  // the 0 line matters here
  ctx.moveTo(pad.l, Y(0)); ctx.lineTo(w - pad.r, Y(0)); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = css("--ink-3"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const [v, label] of xTicks) ctx.fillText(label, X(v), h - pad.b + 8);
  return { X, Y };
}
function whisker(ctx, x, yLo, yHi, color) {
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
  ctx.moveTo(x, yLo); ctx.lineTo(x, yHi); ctx.moveTo(x - 4, yLo); ctx.lineTo(x + 4, yLo); ctx.moveTo(x - 4, yHi); ctx.lineTo(x + 4, yHi); ctx.stroke();
}
function dot(ctx, x, y, color, r = 4) {
  ctx.fillStyle = color; ctx.strokeStyle = css("--panel"); ctx.lineWidth = 2;  // 2px surface ring
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
}

// ---- main chart: gain vs N at tau=0 --------------------------------------
function drawMain() {
  const canvas = $("chart-main"), { ctx, w, h } = setup(canvas), hits = [];
  const ladder = state.config?.ladder || [], opps = state.config?.opponents || [];
  if (!ladder.length) return;
  const xs = ladder.map(Math.log10), ys = [];
  for (const r of state.rows) { const c = candidate(r, 0); ys.push(c.lo99, c.hi99); }
  const { X, Y } = frame(ctx, w, h, xs, ys.length ? ys : [0, 1], ladder.map((n) => [Math.log10(n), `${iters(n)} / seat`]));
  opps.forEach((opp) => {
    const color = css(OPP_COLOR[opp] || "--accent"), pts = [];
    for (const n of ladder) {
      const a = agg(opp, n, 0); if (!a) continue;
      pts.push([X(Math.log10(n)), Y(a.mean)]);
      a.reps.forEach((c, i) => {
        const x = X(Math.log10(n)) + (i - (a.reps.length - 1) / 2) * 10;  // dodge replicates
        whisker(ctx, x, Y(c.lo99), Y(c.hi99), color); dot(ctx, x, Y(c.gain), color);
        hits.push({ x, y: Y(c.gain), html: `<div class="th">${esc(opp)} · N=${iters(n)}/seat · rep ${i}</div>`
          + `<div class="tr"><span class="k">gain</span><span>${sgn(c.gain)} bb/100</span></div>`
          + `<div class="tr"><span class="k">99% interval</span><span>${sgn(c.lo99)} … ${sgn(c.hi99)}</span></div>` });
      });
    }
    if (pts.length > 1) { ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); }
    if (pts.length) {  // direct label at the line end (text in ink, dot carries identity)
      const [x, y] = pts[pts.length - 1];
      ctx.fillStyle = css("--ink-2"); ctx.font = `12px ${css("--sans")}`; ctx.textAlign = "left"; ctx.textBaseline = "middle";
      ctx.fillText(opp, x + 16, y);
    }
  });
  state.hits["chart-main"] = hits;
  $("legend-main").innerHTML = opps.map((o) => `<span class="li"><i style="background:${css(OPP_COLOR[o] || "--accent")}"></i>${esc(o)}</span>`).join("");
}

// ---- small multiples: gain vs tau, one line per N -----------------------
function drawMultiples() {
  const opps = state.config?.opponents || [], taus = state.config?.taus || [], ladder = state.config?.ladder || [];
  const host = $("multiples");
  if (host.children.length !== opps.length) {
    host.innerHTML = opps.map((o, i) => `<div><h3><i style="background:${css(OPP_COLOR[o] || "--accent")}"></i>${esc(o)}</h3><div class="canvas-wrap"><canvas id="mult-${i}"></canvas></div></div>`).join("");
  }
  $("legend-n").innerHTML = ladder.map((n, i) => `<span class="li"><i style="background:${nColor(i, ladder.length)}"></i>N=${iters(n)}</span>`).join("");
  opps.forEach((opp, oi) => {
    const canvas = $(`mult-${oi}`), { ctx, w, h } = setup(canvas), hits = [];
    const ys = [];
    for (const n of ladder) taus.forEach((_, ti) => { const a = agg(opp, n, ti); if (a) ys.push(a.mean - a.half, a.mean + a.half); });
    const { X, Y } = frame(ctx, w, h, taus.map((_, i) => i), ys.length ? ys : [0, 1], taus.map((t, i) => [i, tauLabel(t)]), { l: 46, r: 14, t: 10, b: 26 });
    ladder.forEach((n, ni) => {
      const color = nColor(ni, ladder.length), pts = [];
      taus.forEach((t, ti) => {
        const a = agg(opp, n, ti); if (!a) return;
        const x = X(ti), y = Y(a.mean); pts.push([x, y]);
        const base = agg(opp, n, 0);
        hits.push({ x, y, html: `<div class="th">${esc(opp)} · N=${iters(n)}/seat · τ=${tauLabel(t)}</div>`
          + `<div class="tr"><span class="k">mean gain</span><span>${sgn(a.mean)} ± ${fmt(a.half)}</span></div>`
          + `<div class="tr"><span class="k">vs τ=0</span><span>${sgn(a.mean - base.mean)}</span></div>`
          + `<div class="tr"><span class="k">response kept</span><span>${fmt(100 * a.kept, 1)}%</span></div>`
          + `<div class="tr"><span class="k">replicates</span><span>${a.reps.map((c) => sgn(c.gain)).join(" / ")}</span></div>` });
      });
      if (pts.length > 1) { ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke(); }
      pts.forEach(([x, y]) => dot(ctx, x, y, color, 3.5));
    });
    state.hits[`mult-${oi}`] = hits;
  });
}

// ---- table, stats, clocks -----------------------------------------------
function drawTable() {
  const taus = state.config?.taus || [];
  $("table").querySelector("thead").innerHTML = `<tr><th>N/seat</th><th>opponent</th><th>rep</th><th>solve</th><th>infosets</th>${taus.map((t) => `<th>τ=${tauLabel(t)}</th>`).join("")}</tr>`;
  const rows = [...state.rows].sort((a, b) => a.n - b.n || a.opponent.localeCompare(b.opponent) || a.rep - b.rep);
  $("table").querySelector("tbody").innerHTML = rows.map((r) => {
    const best = Math.max(...r.candidates.map((c) => c.gain));
    return `<tr><td>${iters(r.n)}</td><td>${esc(r.opponent)}</td><td>${r.rep}</td><td>${duration(r.solve_s)}</td><td>${(r.infosets.reduce((a, b) => a + b, 0) / 1e6).toFixed(1)}M</td>`
      + r.candidates.map((c) => `<td class="${c.gain === best ? "best" : c.gain < 0 ? "neg" : ""}">${sgn(c.gain)} <span class="k">±${fmt(Z99 * c.se)}</span></td>`).join("") + "</tr>";
  }).join("");
  $("table-kpi").textContent = `${state.rows.length} measurements × ${taus.length} τ`;
}
function updateClocks() {
  const now = state.finished || Date.now();
  $("stat-elapsed").textContent = duration(state.started ? (now - state.started) / 1000 : NaN);
  const total = state.config?.total_updates, done = state.updatesDone;
  let eta = "—";
  if (state.finished) eta = "done";
  else if (total && state.rate) {
    const inFlight = state.solve ? Math.min(state.solve.expected_s, state.solve.elapsed_s + (Date.now() - state.solveAt) / 1000) * state.rate : 0;
    eta = duration((total - done - inFlight) / state.rate * 1.04);  // +4%: evaluation + table conversion
  }
  $("stat-eta").textContent = eta;
  if (!state.solve) {
    $("solve-label").textContent = state.finished ? "done" : state.started ? "evaluating candidates…" : "—";
    $("solve-bar").style.width = state.finished ? "100%" : "0%";
  } else if (!state.finished) {
    const el = state.solve.elapsed_s + (Date.now() - state.solveAt) / 1000;
    $("solve-bar").style.width = `${Math.min(100, 100 * el / Math.max(1, state.solve.expected_s)).toFixed(1)}%`;
    $("solve-label").textContent = `${state.solve.label} · ${duration(el)} / ~${duration(state.solve.expected_s)}`;
  }
}
function render() {
  const c = state.config;
  if (c) {
    $("sub").textContent = `response convergence · ${c.opponents.length} opponents × ${c.ladder.length} budgets × ${c.reps} reps × ${c.taus.length} τ`;
    $("predictions").innerHTML = Object.entries(c.predictions || {}).map(([k, v]) => `<li><b>${esc(k)}</b> ${esc(v)}</li>`).join("");
  }
  const p = state.progress;
  if (p) {
    $("stat-progress").textContent = `${p.completed} / ${p.total}`;
    $("stat-current").textContent = p.completed < p.total ? `${p.opponent} · N=${iters(p.n)} · rep ${p.rep}` : "all measured";
  }
  if (c?.total_updates) {
    const f = Math.min(1, state.updatesDone / c.total_updates);
    $("run-label").textContent = `${(state.updatesDone / 1e6).toFixed(0)}M of ${(c.total_updates / 1e6).toFixed(0)}M response updates`;
    $("run-pct").textContent = `${(100 * f).toFixed(1)}%`; $("run-bar").style.width = `${(100 * f).toFixed(1)}%`;
  }
  $("stat-rate").textContent = state.rate ? `${(state.rate / 1e3).toFixed(0)}k/s` : "—";
  const res = state.resources;
  $("stat-cpu").textContent = res?.cpu_pct != null ? `CPU ${fmt(res.cpu_pct, 0)}% · ${res.threads ?? "—"} threads` : "—";
  $("stat-ram").textContent = res?.ram_used_gb != null ? `${fmt(res.ram_used_gb, 1)} / ${fmt(res.ram_total_gb, 0)} GB` : "—";
  const ladder = c?.ladder || [];
  const latest = [...ladder].reverse().find((n) => agg("anchor", n, 0));
  if (latest) {
    const a = agg("anchor", latest, 0);
    $("stat-null").textContent = `${sgn(a.mean)} bb/100`;
    $("stat-null-sub").textContent = `N=${iters(latest)}/seat · ±${fmt(a.half)} (99%) · converged BR ≥ 0`;
  }
  drawMain(); drawMultiples(); drawTable(); updateClocks();
}

// ---- events ---------------------------------------------------------------
let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } }
function applyEvent(e) {
  const ts = Number(e.ts) * 1000 || Date.now();
  if (e.type === "run_start") { state.config = e.config; state.started = ts; log("study started", ts); }
  else if (e.type === "metrics") {
    if (e.resources) state.resources = e.resources;
    if (Number.isFinite(e.it_per_s)) state.rate = e.it_per_s;
    if (Number.isInteger(e.iteration)) state.updatesDone = Math.max(state.updatesDone, e.iteration);
    state.solve = e.rc_solve || null; state.solveAt = Date.now();
  } else if (e.type === "checkpoint") {
    if (e.rc_progress) { state.progress = e.rc_progress; if (e.rc_progress.rate) state.rate = e.rc_progress.rate; }
    if (e.rc_row) { const r = e.rc_row, k = `${r.n}:${r.opponent}:${r.rep}`; if (!state.keys.has(k)) { state.keys.add(k); state.rows.push(r); } }
  } else if (e.type === "milestone") log(e.message || e.kind, ts);
  else if (e.type === "run_end") {
    state.finished = ts; state.solve = null;
    $("done-detail").textContent = e.reason === "completed" ? `${state.rows.length} measurements · ${duration((ts - state.started) / 1000)}` : `ended: ${e.reason}`;
    $("done-overlay").hidden = false; log(`run ended (${e.reason})`, ts);
  }
  schedule();
}

// ---- hover ------------------------------------------------------------------
document.addEventListener("mousemove", (ev) => {
  const tip = $("tip"), el = ev.target;
  const hits = el && el.tagName === "CANVAS" ? state.hits[el.id] : null;
  if (!hits) { tip.hidden = true; return; }
  const r = el.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
  let best = null, bd = 14 * 14;  // hit target larger than the mark
  for (const hIt of hits) { const d = (hIt.x - mx) ** 2 + (hIt.y - my) ** 2; if (d < bd) { bd = d; best = hIt; } }
  if (!best) { tip.hidden = true; return; }
  tip.innerHTML = best.html; tip.hidden = false;
  tip.style.left = `${r.left + best.x}px`; tip.style.top = `${r.top + best.y - 6}px`;
});

const events = new EventSource("/events");
events.onmessage = (m) => JSON.parse(m.data).forEach(applyEvent);
events.onerror = () => log("telemetry reconnecting…");
window.addEventListener("resize", schedule);
$("done-dismiss").onclick = () => { $("done-overlay").hidden = true; };
if (typeof setInterval !== "undefined") setInterval(updateClocks, 1000);
render();
