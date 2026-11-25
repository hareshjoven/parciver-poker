/* FLOPSEARCH M3 viewer. Data-only: the watcher (scripts/watch_m3.py) reads the three outputs after each saved chunk
   and sends each run's curves, previews and summaries; the page only extrapolates the pace and draws. */
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

const state = { config: null, runs: [], alive: null, res: null, ended: null, lastLine: null, cores: null, idle: null, pick: null, hits: {} };
const done = () => state.runs.length === 3 && state.runs.every((r) => r.summary);
const over = () => done() || state.alive === false;
/** The run being played: the first without a reading (the last one once all are read). */
const current = () => state.runs.find((r) => !r.summary) || state.runs[state.runs.length - 1] || null;
const shown = () => state.runs.find((r) => r.name === state.pick) || current();
const better = (r, v) => r.mode === "lbr" ? v < 0 : v > 0;  // good for A
const lastSave = (r) => r?.pace?.length ? r.pace[r.pace.length - 1][4] : null;
const started = (r) => r && num(r.first_ended) && num(r.first_seconds) ? r.first_ended - r.first_seconds : null;

// ---- header ---------------------------------------------------------------------------------------------
function renderHeader() {
  const c = state.config, r = current();
  if (!c) return;
  $("sub").textContent = `reads runs/flop-eval/m3-*.json only${c.pid ? ` · run pid ${c.pid}` : ""} · ${c.rule}`;
  const k = r ? state.runs.indexOf(r) + 1 : 0;
  $("stat-deals-label").textContent = r ? `run ${k} of 3 · ${r.name}` : "deals";
  if (!r?.started || !r.deals_done) {
    $("stat-mean-label").textContent = r ? `A − B · ${r.name}` : "A − B"; $("stat-mean-sub").textContent = "bb/100 so far";
    $("stat-mean").textContent = "—"; $("stat-mean").style.color = "";
    $("stat-deals").textContent = r?.deals_done ? r.deals_done.toLocaleString() : "0";
    $("stat-deals-sub").textContent = r?.deals ? `of ${r.deals.toLocaleString()}` : "starting";
    $("stat-pace").textContent = num(state.cores) ? `run using ${state.cores.toFixed(1)} cores` : "—";
  } else {
    const cur = r.curve[r.curve.length - 1];
    $("stat-mean-label").textContent = `A − B · ${r.name}`;
    $("stat-mean").textContent = sgn(cur[1]);
    $("stat-mean").style.color = css(better(r, cur[1]) ? "--ok" : "--hot");
    $("stat-mean-sub").textContent = `bb/100 ± ${num(cur[2]) ? cur[2].toFixed(1) : "—"} se · ${r.mode === "lbr" ? "lower" : "higher"} is better for A`;
    $("stat-deals").textContent = r.deals_done.toLocaleString();
    $("stat-deals-sub").textContent = `of ${r.deals.toLocaleString()} · ${(100 * r.deals_done / r.deals).toFixed(1)}%`;
    $("stat-pace").textContent = `${(r.seconds / r.deals_done).toFixed(2)} s/deal` + (num(state.cores) ? ` · ${state.cores.toFixed(1)} cores` : "");
  }
  const res = state.res;
  $("stat-res").textContent = [res && num(res.gpu_pct) ? `${res.gpu_pct.toFixed(0)}%` + (num(res.gpu_mem_used_gb) ? ` ${res.gpu_mem_used_gb.toFixed(1)} GB` : "") : "—",
    res && num(res.cpu_pct) ? `${res.cpu_pct.toFixed(0)}%` : "—",
    res && num(res.ram_used_gb) ? `${res.ram_used_gb.toFixed(1)}/${res.ram_total_gb.toFixed(0)} GB` : "—"].join(" · ");
  updateClocks();
}
function updateClocks() {
  if (!state.config) return;
  const r = current(), now = Date.now() / 1000;
  const t0 = started(state.runs[0]) ?? state.config.started, ends = state.runs.map(lastSave).filter(num);
  const last = ends.length ? Math.max(...ends) : null;
  let text = num(t0) ? duration((over() && num(last) ? last : now) - t0) : "—";
  if (done()) text += " · complete";
  else if (over()) text += " · stopped";
  else if (r?.deals_done) {
    const per = r.seconds / r.deals_done, l = lastSave(r);
    text += ` · this run ${duration(Math.max(0, (r.deals - r.deals_done) * per - (num(l) ? now - l : 0)))} left`;
  }
  $("stat-eta").textContent = text;
  let dot = "live", label = r?.deals_done ? `running ${r.name}` : "starting (first save after one chunk)";
  if (done()) { dot = "done"; label = "complete · all three readings written"; }
  else if (state.alive === false) { dot = "err"; label = "M3 exited before its last run's last deal"; }
  else if (r?.deals_done && num(lastSave(r))) {
    const n = r.bars.length ? r.bars[r.bars.length - 1][2] : 100, due = n * r.seconds / r.deals_done;
    if (now - lastSave(r) > Math.max(900, 3 * due)) { dot = "err"; label = `no save for ${duration(now - lastSave(r))}: stalled?`; }
  }
  if (!over() && state.alive && num(state.idle) && state.idle > 300) { dot = "err"; label = `no CPU used for ${duration(state.idle)}: hung?`; }
  if (state.ended && !over()) { dot = "idle"; label = "watcher stopped"; }
  $("stat-status").innerHTML = `<span class="dot ${dot}"></span>${esc(label)}`;
  const b = $("banner");
  if (state.alive === false && !done()) {
    b.textContent = `M3 stopped in ${r?.name ?? "?"} at ${r?.deals_done ?? 0} deals: ${state.lastLine || "no log line"}. Report before re-running anything.`;
    b.classList.add("on");
  } else b.classList.remove("on");
}

// ---- the runs table -------------------------------------------------------------------------------------------
function ci(r, s, d = 1) { return s ? `${sgn(s.mean, d)} [${sgn(s.lo, d)}, ${sgn(s.hi, d)}]` : "—"; }
function renderRuns() {
  const head = "<tr><th>run</th><th>deals</th><th>A − B, bb/100</th><th>A</th><th>B</th><th>search decides (A · B)</th><th>s/deal</th></tr>";
  const cur = current(), sel = shown();
  const rows = state.runs.map((r) => {
    if (!r.started) return `<tr data-run="${esc(r.name)}"${r === sel ? ' class="sel"' : ""}><td>${esc(r.name)}<div class="sub">${esc(r.title)}</div></td><td class="muted" colspan="6">not started</td></tr>`;
    const s = r.summary || r.preview, d = s?.a_minus_b;
    const word = !d ? (r.deals_done ? "preview after 100 deals" : "") :
      d.lo > 0 || d.hi < 0 ? (better(r, d.mean) ? "A better" : "A worse") : "no resolved difference";
    const cls = !d ? "muted" : d.lo > 0 || d.hi < 0 ? (better(r, d.mean) ? "good" : "bad") : "";
    const tag = r.summary ? "reading" : s ? `preview at ${s.deals.toLocaleString()}` : "";
    const post = r.mode === "lbr" && s?.postflop_a_minus_b ? `<div class="sub">postflop part ${ci(r, s.postflop_a_minus_b)}</div>` : "";
    const pct = r.deals ? 100 * r.deals_done / r.deals : 0;
    return `<tr data-run="${esc(r.name)}"${r === sel ? ' class="sel"' : ""}><td>${esc(r.name)}${r === cur && !over() ? " ▸" : ""}<div class="sub">${esc(r.title)}</div></td>` +
      `<td>${r.deals_done.toLocaleString()} / ${(r.deals ?? 0).toLocaleString()}<div class="bar"><i style="width:${pct.toFixed(1)}%"></i></div></td>` +
      `<td class="${cls}">${ci(r, d)}<div class="sub">${esc(tag)}${word ? ` · ${esc(word)}` : ""}</div>${post}</td>` +
      `<td>${ci(r, s?.a)}</td><td>${ci(r, s?.b)}</td>` +
      `<td>${r.mode === "lbr" ? "—" : `${num(r.a_share) ? (100 * r.a_share).toFixed(0) : "—"}% · ${num(r.b_share) ? (100 * r.b_share).toFixed(0) : "—"}%`}</td>` +
      `<td>${r.deals_done ? (r.seconds / r.deals_done).toFixed(2) : "—"}</td></tr>`;
  });
  $("runs").innerHTML = head + rows.join("");
}
$("runs").addEventListener("click", (ev) => {
  const tr = ev.target.closest("tr[data-run]");
  if (tr) { state.pick = tr.dataset.run === current()?.name ? null : tr.dataset.run; schedule(); }
});

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
function axes(ctx, w, h, x0, x1, lo, hi) {
  const L = 52, R = 18, T = 10, B = 24;
  const X = (v) => L + (v - x0) / (x1 - x0 || 1) * (w - L - R), Y = (v) => T + (1 - (v - lo) / (hi - lo || 1)) * (h - T - B);
  ctx.font = `11px ${css("--mono")}`; ctx.lineWidth = 1;
  for (const t of niceTicks(lo, hi, 5)) {
    ctx.strokeStyle = css(t === 0 ? "--ink-3" : "--border"); ctx.beginPath(); ctx.moveTo(L, Y(t)); ctx.lineTo(w - R, Y(t)); ctx.stroke();
    ctx.fillStyle = css("--ink-3"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(String(t), L - 6, Y(t));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of niceTicks(x0, x1, 5)) ctx.fillText(t.toLocaleString(), X(t), h - B + 7);
  return { X, Y };
}
function line(ctx, pts, X, Y, col, width, dash = []) {
  ctx.strokeStyle = col; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath();
  pts.forEach(([x, y], i) => { if (i) ctx.lineTo(X(x), Y(y)); else ctx.moveTo(X(x), Y(y)); });
  ctx.stroke(); ctx.setLineDash([]);
}
function drawCum() {
  const id = "c-cum", { ctx, w, h } = setup($(id)), r = shown();
  state.hits[id] = [];
  if (r) {
    $("cum-title").textContent = `Cumulative A − B · ${r.name} (${r.title})`;
    $("cum-note").textContent = `bb/100; ${r.mode === "lbr" ? "what LBR wins from each arm: below 0 = LBR wins less from A (good for A)" : "each arm against the bot: above 0 = A wins more (good for A)"}. The band is the per-deal 95% interval, a preview; the reading is 50 blocks at ${r.deals?.toLocaleString() ?? "—"} deals.`;
  }
  if (!r?.curve?.length) return empty(ctx, w, h, r?.started === false ? "this run has not started" : "first save after one chunk");
  const pts = r.curve, ys = pts.flatMap(([, y, se, a, b]) => [y - 1.96 * (se || 0), y + 1.96 * (se || 0), a, b]).concat([0]);
  let lo = Math.min(...ys), hi = Math.max(...ys); const pad = 0.06 * (hi - lo || 1); lo -= pad; hi += pad;
  const { X, Y } = axes(ctx, w, h, 0, r.deals, lo, hi);
  const a = css("--armA");
  ctx.fillStyle = a; ctx.globalAlpha = 0.18; ctx.beginPath();
  pts.forEach(([d, y, se], i) => { const e = 1.96 * (se || 0); if (i) ctx.lineTo(X(d), Y(y + e)); else ctx.moveTo(X(d), Y(y + e)); });
  [...pts].reverse().forEach(([d, y, se]) => ctx.lineTo(X(d), Y(y - 1.96 * (se || 0))));
  ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
  line(ctx, pts.map((p) => [p[0], p[3]]), X, Y, css("--ok"), 1.5, [4, 3]);
  line(ctx, pts.map((p) => [p[0], p[4]]), X, Y, css("--armB"), 1.5, [4, 3]);
  line(ctx, pts.map((p) => [p[0], p[1]]), X, Y, a, 2);
  ctx.fillStyle = a;
  for (const [d, y, se, ya, yb] of pts) {
    ctx.beginPath(); ctx.arc(X(d), Y(y), 2.5, 0, 2 * Math.PI); ctx.fill();
    state.hits[id].push({ x: X(d), y: Y(y), html: `<div class="th">${d.toLocaleString()} deals</div><div class="tr"><span>A − B</span><span>${sgn(y)} bb/100</span></div>` +
      (num(se) ? `<div class="tr"><span>95%</span><span>[${sgn(y - 1.96 * se)}, ${sgn(y + 1.96 * se)}]</span></div>` : "") +
      `<div class="tr"><span>A</span><span>${sgn(ya)}</span></div><div class="tr"><span>B</span><span>${sgn(yb)}</span></div>` });
  }
}
function drawBars() {
  const id = "c-bars", { ctx, w, h } = setup($(id)), r = shown();
  state.hits[id] = [];
  if (!r?.bars?.length) return empty(ctx, w, h, "first save after one chunk");
  const lim = Math.max(...r.bars.map((b) => Math.abs(b[1]))) * 1.1 || 1;
  const { X, Y } = axes(ctx, w, h, 0, r.deals, -lim, lim);
  for (const [d, y, n] of r.bars) {
    ctx.fillStyle = css(better(r, y) ? "--ok" : "--hot");
    const bw = Math.max(2, (X(n) - X(0)) * 0.8), mid = X(d - n / 2), y0 = Y(0), y1 = Y(y);
    ctx.fillRect(mid - bw / 2, Math.min(y0, y1), bw, Math.abs(y1 - y0));
    state.hits[id].push({ x: mid, y: y1, html: `<div class="th">deals ${(d - n).toLocaleString()}–${(d - 1).toLocaleString()}</div><div class="tr"><span>A − B</span><span>${sgn(y)} bb/100</span></div>` });
  }
}
function drawPace() {
  const id = "c-pace", { ctx, w, h } = setup($(id)), r = shown();
  state.hits[id] = [];
  if (!r?.pace?.length) return empty(ctx, w, h, "first save after one chunk");
  const top = Math.max(...r.pace.flatMap((p) => p.slice(1, 4))) * 1.15;
  const { X, Y } = axes(ctx, w, h, 0, r.deals, 0, top);
  for (const [k, col, name] of [[1, "--amber", "s/deal"], [2, "--armA", "A's solver calls/deal"], [3, "--armB", "B's solver calls/deal"]]) {
    line(ctx, r.pace.map((p) => [p[0], p[k]]), X, Y, css(col), k === 1 ? 2 : 1.5);
    ctx.fillStyle = css(col);
    for (const p of r.pace) {
      ctx.beginPath(); ctx.arc(X(p[0]), Y(p[k]), 2, 0, 2 * Math.PI); ctx.fill();
      state.hits[id].push({ x: X(p[0]), y: Y(p[k]), html: `<div class="th">chunk to ${p[0].toLocaleString()}</div><div class="tr"><span>${name}</span><span>${p[k].toFixed(2)}</span></div>` });
    }
  }
}

function render() { renderHeader(); renderRuns(); drawCum(); drawBars(); drawPace(); }

// ---- events ----------------------------------------------------------------------------------------------
let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } }
function finish() {
  if (done()) {
    $("done-title").textContent = "M3 complete (descriptive)";
    $("done-detail").textContent = state.runs.map((r) => `${r.name}: A − B ${ci(r, r.summary.a_minus_b, 2)} bb/100`).join(" · ");
  } else {
    const r = current();
    $("done-badge").textContent = "!"; $("done-badge").classList.add("err"); $("done-title").textContent = "M3 stopped";
    $("done-detail").textContent = `In ${r?.name ?? "?"} at ${r?.deals_done ?? 0} deals: ${state.lastLine || "no log line"}.`;
  }
  $("done-overlay").hidden = false;
}
function applyEvent(e) {
  const ts = Number(e.ts) * 1000 || Date.now();
  if (e.type === "run_start") {
    if (e.config?.study !== "m3") {
      $("sub").textContent = `this run is not an M3 watch (study: ${e.config?.study ?? e.variant ?? "unknown"})`;
      events.close();
      return;
    }
    Object.assign(state, { config: e.config, runs: [], alive: null, ended: null, cores: null, idle: null });
    $("event-log").innerHTML = "";
    log(`watching runs/flop-eval/m3-*.json${e.config.pid ? ` (pid ${e.config.pid})` : ""}`, ts);
  } else if (e.type === "metrics") {
    if (e.resources) state.res = e.resources;
    if ("m3_alive" in e) state.alive = e.m3_alive;
    if (e.m3_last_line) state.lastLine = e.m3_last_line;
    if ("m3_cores" in e) state.cores = e.m3_cores;
    if ("m3_idle_s" in e) state.idle = e.m3_idle_s;
  } else if (e.type === "checkpoint") {
    if (e.m3) state.runs = e.m3.runs;
  } else if (e.type === "milestone") {
    log(e.message || e.kind, ts, e.kind === "warning" ? "warning" : "note");
  } else if (e.type === "run_end") {
    state.ended = ts;
    log(e.reason === "completed" ? "M3 complete" : `watch ended (${e.summary?.note ?? e.reason})`, ts, e.reason === "completed" ? "note" : "warning");
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
  if (failures >= 3) { events.close(); log("dashboard server stopped (the watcher exited); M3 is unaffected", Date.now(), "warning"); }
  else if (failures === 1) log("telemetry reconnecting…", Date.now(), "warning");
};
window.addEventListener("resize", schedule);
$("done-dismiss").onclick = () => { $("done-overlay").hidden = true; };
setInterval(updateClocks, 1000);
render();
