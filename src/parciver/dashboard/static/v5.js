/* VERIFY V5 viewer. Data-only: the watcher (scripts/watch_v5.py) reads the queue's log, the steps' logs and
   their --out files, and sends each step's progress, the F − D preview and v5_paired.py's reading; the page
   only extrapolates the pace and draws. */
"use strict";
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
const num = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
const sgn = (v, d = 1) => num(v) ? `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}` : "—";
const ARM = { D: "--armD", F: "--armF" };
// the charts: (canvas, step kind, run key in the step's runs)
const CHARTS = [["c-lbr-v1", "lbr", "v1/turn"], ["c-lbr-v2", "lbr", "v2-2B/turn"], ["c-h2h", "h2h", "v2-2B vs v1"]];

function duration(s) {
  if (!num(s)) return "—";
  s = Math.max(0, Math.round(s));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60), q = s % 60;
  const p = (x) => String(x).padStart(2, "0");
  return d ? `${d}d ${h}h ${p(m)}m` : h ? `${h}h ${p(m)}m` : m ? `${m}m ${p(q)}s` : `${q}s`;
}
const clock = (t) => num(t) ? new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "—";
function log(text, ts, kind = "note") {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${new Date(ts || Date.now()).toLocaleTimeString()}</span><span class="kind ${esc(kind)}">${esc(kind)}</span><span class="msg">${esc(text)}</span>`;
  $("event-log").prepend(li);
}

const state = { config: null, v5: null, alive: null, res: null, ended: null, endReason: null, hits: {} };

// ---- progress and pace -----------------------------------------------------------------------------
const conf = (s) => state.config.steps.find((x) => x.step === s.step);
/** The deals a step plays in all (LBR: one run per candidate). */
const planned = (s) => s.kind === "lbr" ? conf(s).deals * state.config.lbr_runs : conf(s).deals;
const played = (s) => Object.values(s.runs).reduce((a, r) => a + r.deals_done, 0);
const secs = (s) => Object.values(s.runs).reduce((a, r) => a + (r.seconds || 0), 0);
const armD = (s) => state.v5.steps.find((x) => x.kind === s.kind && x.arm === "D");
/** Arm F's time over arm D's on the same runs and deal counts (LBR's candidates differ in pace), or null. */
function fRatio(s) {
  const d = armD(s);
  let f = 0, dd = 0;
  for (const [k, r] of Object.entries(s.runs)) {
    const q = d?.runs[k];
    if (r.deals_done && q?.deals_done) { f += r.seconds; dd += q.seconds / q.deals_done * r.deals_done; }
  }
  return dd > 0 ? f / dd : null;
}
/** Seconds per deal: the step's own once it has saved 50 deals (arm F: arm D's whole pace × F's ratio on the same
    runs, when D is complete); before that a guess (arm F: its arm D's × the expected factor; otherwise the 3 Oct
    runs' pace). */
function pace(s) {
  const k = played(s), c = state.config;
  if (k >= 50) {
    const d = s.arm === "F" && armD(s), ratio = d && d.status === "done" ? fRatio(s) : null;
    return { v: num(ratio) ? ratio * secs(d) / played(d) : secs(s) / k, guess: false };
  }
  if (s.arm === "F") {
    const d = armD(s);
    const p = d && played(d) >= 50 ? secs(d) / played(d) : c.prior_s_per_deal[d?.step];
    return num(p) ? { v: p * c.f_factor, guess: true } : null;
  }
  const p = c.prior_s_per_deal[s.step];
  return num(p) ? { v: p, guess: true } : null;
}
function over() {
  const v = state.v5;
  return !!(v?.queue_end || state.alive === false || v?.steps.some((s) => s.status === "failed"));
}
/** Seconds left: each unfinished step's deals left at its pace, the running one counted down from its last save. */
function eta() {
  const v = state.v5;
  if (!v || over()) return null;
  const now = Date.now() / 1000;
  let seconds = 0, guess = false, unknown = [];
  for (const s of v.steps) {
    if (s.status === "done" || s.kind === "read") continue;
    const p = pace(s);
    if (!p) { unknown.push(s.step); continue; }
    let left = (planned(s) - played(s)) * p.v;
    if (s.status === "running") left -= now - (num(s.saved) && s.saved > (s.started || 0) ? s.saved : s.started || now);
    seconds += Math.max(0, left);
    guess = guess || p.guess;
  }
  return { seconds, guess, unknown };
}
const running = () => state.v5?.steps.find((s) => s.status === "running");

// ---- header -----------------------------------------------------------------------------------------
function renderHeader() {
  const c = state.config, v = state.v5;
  if (!c) return;
  $("sub").textContent = `reads ${c.dir} only${c.pid ? ` · queue pid ${c.pid}` : ""} · ${c.hands_per_deal} hands per deal`;
  $("scope-note").textContent = `Same code for every step (the snapshot C:/tmp/verify/v5code); each script resumes its --out, so a restarted queue re-runs finished steps instantly. ` +
    `Progress moves in saves of 250 deals per run. Results are bb/100 ± one standard error per deal (LBR: LBR's winnings; h2h: v2-2B's). ` +
    `Steps not yet started are timed by a guess: arm F at ${c.f_factor}× its arm D; R8'-D and R7' at the 3 Oct runs' pace.`;
  if (!v) return;
  const deal = v.steps.filter((s) => s.kind !== "read");
  const n = deal.reduce((a, s) => a + played(s), 0), N = deal.reduce((a, s) => a + planned(s), 0);
  $("stat-deals").textContent = n.toLocaleString();
  $("stat-deals-sub").textContent = `of ${N.toLocaleString()} · ${(100 * n / N).toFixed(1)}% · ${v.steps.filter((s) => s.status === "done").length} / ${v.steps.length} steps done`;
  const r = running();
  if (r) {
    $("stat-step").textContent = r.step;
    $("stat-step-sub").textContent = `${r.label}${r.arm ? ` · arm ${r.arm}` : ""} · ${played(r).toLocaleString()} / ${planned(r).toLocaleString()} deals`;
  } else {
    $("stat-step").textContent = v.queue_end ? (v.queue_end === "QUEUE DONE" ? "done" : "stopped") : "—";
    $("stat-step-sub").textContent = v.queue_end ? v.queue_end.toLowerCase() : "between steps";
  }
  const p = r && pace(r);
  let pt = p ? `${p.v.toFixed(2)} s/deal${p.guess ? " (guess)" : ""}` : "—";
  const ratio = r?.arm === "F" ? fRatio(r) : null;
  if (num(ratio)) pt += ` · ${ratio.toFixed(1)}× arm D`;
  $("stat-pace").textContent = pt;
  const res = state.res;
  $("stat-res").textContent = [res && num(res.gpu_pct) ? `${res.gpu_pct.toFixed(0)}%` + (num(res.gpu_mem_used_gb) ? ` ${res.gpu_mem_used_gb.toFixed(1)} GB` : "") : "—",
    res && num(res.cpu_pct) ? `${res.cpu_pct.toFixed(0)}%` : "—",
    res && num(res.ram_used_gb) ? `${res.ram_used_gb.toFixed(1)}/${res.ram_total_gb.toFixed(0)} GB` : "—"].join(" · ");
  $("stat-res").title = "The whole machine (nvidia-smi and Windows counters), not the queue alone.";
  updateClocks();
}
function updateClocks() {
  const c = state.config, v = state.v5;
  if (!c || !v) return;
  const now = Date.now() / 1000, first = v.steps[0].started;
  const ends = v.steps.map((s) => s.ended).filter(num);
  const end = over() && ends.length ? Math.max(...ends) : now;
  let text = num(first) ? duration(end - first) : "—";
  const e = eta();
  if (v.queue_end === "QUEUE DONE") text += " · complete";
  else if (over()) text += " · stopped";
  else if (e) text += ` · ${e.guess ? "~" : ""}${duration(e.seconds)} left${e.unknown.length ? ` + ${e.unknown.join(", ")}` : ""}`;
  $("stat-eta").textContent = text;
  $("stat-eta").title = "Elapsed: since the queue's first step started. ETA: each unfinished step's deals at its pace (\"~\": some steps' pace is still a guess).";

  const r = running(), failed = v.steps.find((s) => s.status === "failed");
  let dot = "live", label = r ? `running ${r.step}` : "between steps";
  if (v.queue_end === "QUEUE DONE") { dot = "done"; label = "complete · reading written"; }
  else if (failed) { dot = "err"; label = `STOPPED: ${failed.step} failed (exit ${failed.exit})`; }
  else if (v.queue_end) { dot = "err"; label = "queue stopped"; }
  else if (state.alive === false) { dot = "err"; label = "the queue exited without its last line"; }
  else if (r) {
    const p = pace(r), last = num(r.saved) && r.saved > (r.started || 0) ? r.saved : r.started;
    const due = p ? 250 * p.v : 600;
    if (num(last) && now - last > Math.max(900, 3 * due)) { dot = "err"; label = `no save for ${duration(now - last)}: stalled?`; }
  }
  if (state.ended && !over()) { dot = "idle"; label = "watcher stopped"; }
  $("stat-status").innerHTML = `<span class="dot ${dot}"></span>${esc(label)}`;

  const b = $("banner"), bad = failed || (v.queue_end && v.queue_end !== "QUEUE DONE") || (state.alive === false && !v.queue_end);
  if (bad) {
    const s = failed || r || v.steps.filter((x) => x.started).pop();
    const said = s ? [...new Set([s.errors?.[s.errors.length - 1], s.last_line].filter(Boolean))] : [];
    b.textContent = `The queue stopped${s ? ` at ${s.step} (log C:/tmp/verify/v5-${s.step}.log)` : ""}${said.length ? `: ${said.join(" | ")}` : ""}. Report before re-running anything.`;
    b.classList.add("on");
  } else b.classList.remove("on");
}

// ---- steps table ------------------------------------------------------------------------------------
function result(s) {
  if (s.kind === "read") return s.reading ? `<b>${esc(s.reading.reading)}</b>` : "—";
  const runs = Object.entries(s.runs);
  if (!runs.length) return "—";
  return runs.map(([k, r]) => {
    const name = s.kind === "lbr" ? `${k.split("/")[0]} ` : "";
    const share = num(r.search_share) ? ` · search ${(100 * r.search_share).toFixed(0)}%` : "";
    return `${esc(name)}${num(r.mean) ? `${sgn(r.mean)} ± ${r.se.toFixed(1)}` : "—"}${share}`;
  }).join("<br>");
}
function renderSteps() {
  const v = state.v5;
  if (!v) return;
  const now = Date.now() / 1000;
  $("steps").querySelector("thead").innerHTML = "<tr><th>step</th><th>arm</th><th class=\"l\">what</th><th>status</th><th>deals</th><th>s/deal</th><th>bb/100 so far</th><th>started</th><th>took</th><th class=\"l\">latest</th></tr>";
  $("steps").querySelector("tbody").innerHTML = v.steps.map((s) => {
    const k = played(s), N = s.kind === "read" ? 0 : planned(s), p = s.kind === "read" ? null : pace(s);
    const per = Object.entries(s.runs).map(([key, r]) => `${key}: ${r.deals_done}/${r.deals}`).join(", ");
    const prog = s.kind === "read" ? "—" : `${k.toLocaleString()} / ${N.toLocaleString()}<div class="bar"><i style="width:${(100 * k / N).toFixed(1)}%;background:var(${ARM[s.arm] || "--ink-3"})"></i></div>`;
    const took = num(s.started) ? duration((num(s.ended) ? s.ended : now) - s.started) : "—";
    // the latest line; red when it, or an error since the last progress line, may explain a failure
    const late = s.status === "failed" || s.errors?.length
      ? `<td class="err">${esc(s.status === "failed" ? s.last_line : s.errors[s.errors.length - 1])}</td>` : `<td class="wrap">${esc(s.last_line || "")}</td>`;
    return `<tr><td>${esc(s.step)}</td><td>${s.arm ? `<span class="arm ${s.arm}">${s.arm}</span>` : ""}</td><td class="l">${esc(s.label)}</td>` +
      `<td><span class="st ${s.status}">${s.status}${s.status === "failed" ? ` (${s.exit})` : ""}</span></td>` +
      `<td title="${esc(per)}">${prog}</td><td class="${p?.guess ? "dim" : ""}">${p ? `${p.guess ? "~" : ""}${p.v.toFixed(2)}` : "—"}</td>` +
      `<td>${result(s)}</td><td class="dim">${clock(s.started)}</td><td class="dim">${s.status === "running" || num(s.ended) ? took : "—"}</td>${late}</tr>`;
  }).join("");
}

// ---- charts -----------------------------------------------------------------------------------------
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
function drawCurves([id, kind, key]) {
  const { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  const v = state.v5;
  if (!v) return;
  const arms = ["D", "F"].map((a) => [a, v.steps.find((s) => s.kind === kind && s.arm === a)?.runs[key]]).filter(([, r]) => r?.curve?.length);
  const N = conf(v.steps.find((s) => s.kind === kind && s.arm === "D")).deals;
  if (!arms.length) {
    ctx.fillStyle = css("--ink-3"); ctx.font = `12px ${css("--sans")}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("no saved deals yet", w / 2, h / 2);
    return;
  }
  const ys = arms.flatMap(([, r]) => r.curve.map((p) => p[1])).concat([0]);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  const pad = 0.08 * (hi - lo || 1); lo -= pad; hi += pad;
  const L = 50, R = 22, T = 10, B = 24;
  const X = (d) => L + d / N * (w - L - R), Y = (y) => T + (1 - (y - lo) / (hi - lo)) * (h - T - B);
  ctx.font = `11px ${css("--mono")}`; ctx.lineWidth = 1;
  for (const t of niceTicks(lo, hi, 5)) {
    ctx.strokeStyle = css(t === 0 ? "--ink-3" : "--border"); ctx.beginPath(); ctx.moveTo(L, Y(t)); ctx.lineTo(w - R, Y(t)); ctx.stroke();
    ctx.fillStyle = css("--ink-3"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(String(t), L - 6, Y(t));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of niceTicks(0, N, 5)) ctx.fillText(t.toLocaleString(), X(t), h - B + 7);
  for (const [a, r] of arms) {
    ctx.strokeStyle = css(ARM[a]); ctx.lineWidth = 2; ctx.setLineDash(a === "F" ? [6, 3] : []);
    ctx.beginPath();
    r.curve.forEach(([d, y], i) => { if (i) ctx.lineTo(X(d), Y(y)); else ctx.moveTo(X(d), Y(y)); });
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css(ARM[a]);
    for (const [d, y] of r.curve) {
      ctx.beginPath(); ctx.arc(X(d), Y(y), 2.5, 0, 2 * Math.PI); ctx.fill();
      state.hits[id].push({ x: X(d), y: Y(y), html: `<div class="th">arm ${a} · ${d.toLocaleString()} deals</div><div class="tr"><span>cumulative</span><span>${sgn(y)} bb/100</span></div>` +
        (d === r.deals_done && num(r.se) ? `<div class="tr"><span>± se</span><span>${r.se.toFixed(1)}</span></div>` : "") });
    }
  }
}

// ---- F − D -------------------------------------------------------------------------------------------
function renderDeltas() {
  const v = state.v5, c = state.config;
  if (!v || !c) return;
  const read = v.steps.find((s) => s.kind === "read")?.reading;
  const rows = read ? read.deltas : v.deltas;
  $("deltas").querySelector("thead").innerHTML = `<tr><th>run${read ? " · final" : " · preview"}</th><th>deals</th><th>F − D</th><th>se</th><th>Bonferroni (t ${c.t_bonferroni})</th>` +
    `${read ? `<th>95% (t ${c.t95})</th><th>s/deal D → F</th>` : ""}<th>excludes 0</th></tr>`;
  const keys = Object.keys(rows || {});
  $("deltas").querySelector("tbody").innerHTML = keys.length ? keys.map((k) => {
    const x = rows[k];
    if (!x) return `<tr><td>${esc(k)}</td><td class="dim" colspan="${read ? 7 : 5}">fewer than 100 common deals</td></tr>`;
    const ex = x.lo > 0 || x.hi < 0;
    const n = read ? (k.startsWith("lbr") ? 2500 : 10000) : x.deals;
    return `<tr><td>${esc(k)}</td><td>${n.toLocaleString()}</td><td>${sgn(x.mean, 2)}</td><td class="dim">${x.se.toFixed(2)}</td>` +
      `<td>[${sgn(x.lo, 2)}, ${sgn(x.hi, 2)}]</td>` +
      (read ? `<td class="dim">[${sgn(x.lo95, 2)}, ${sgn(x.hi95, 2)}]</td><td class="dim">${x.seconds_per_deal.D.toFixed(2)} → ${x.seconds_per_deal.F.toFixed(2)}</td>` : "") +
      `<td class="${ex ? "hot" : "ok"}">${ex ? "yes" : "no"}</td></tr>`;
  }).join("") : `<tr><td colspan="6" class="dim">arm F has not played yet: F − D appears as it plays</td></tr>`;
  const vd = $("verdict");
  if (read) {
    const yes = read.reading === "K2 MATTERS";
    vd.className = `verdict ${yes ? "yes" : "no"}`;
    vd.textContent = `${read.reading}${read.excludes_0?.length ? ` (${read.excludes_0.join(", ")})` : ""}`;
  } else {
    vd.className = "verdict wait";
    vd.textContent = "The reading comes from records/verify/v5_paired.py (step \"read\") once all four GPU runs are complete.";
  }
  $("delta-note").textContent = read
    ? `v5_paired.py's reading (${c.blocks} blocks per run, bb/100). K2 MATTERS if any Bonferroni interval excludes 0; the 95% intervals are descriptive.`
    : `Preview, decides nothing: the same ${c.blocks}-block reading on the deals both arms have saved so far (bb/100; a positive F − D = more LBR winnings, or more v2-2B winnings, with float64 terminal sums).`;
}

function render() {
  renderHeader(); renderSteps(); CHARTS.forEach(drawCurves); renderDeltas();
}

// ---- events -------------------------------------------------------------------------------------------
let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } }
function finish(ts) {
  const v = state.v5, read = v?.steps.find((s) => s.kind === "read")?.reading;
  const r7 = v?.steps.find((s) => s.step === "r7")?.runs["v2-2B vs v1"];
  const first = v?.steps[0].started, ends = (v?.steps || []).map((s) => s.ended).filter(num);
  if (v?.queue_end === "QUEUE DONE") {
    $("done-detail").textContent = `${read ? `${read.reading}${read.excludes_0?.length ? ` (${read.excludes_0.join(", ")})` : ""}` : "no reading"}. ` +
      `R7' river h2h: ${r7 && num(r7.mean) ? `${sgn(r7.mean)} ± ${r7.se.toFixed(1)} bb/100 over ${r7.deals_done.toLocaleString()} deals` : "—"}. ` +
      `${num(first) && ends.length ? `Took ${duration(Math.max(...ends) - first)}.` : ""}`;
  } else {
    $("done-badge").textContent = "!"; $("done-badge").classList.add("err"); $("done-title").textContent = "V5 queue stopped";
    const f = v?.steps.find((s) => s.status === "failed");
    $("done-detail").textContent = f ? `${f.step} exited with ${f.exit}: see C:/tmp/verify/v5-${f.step}.log.` : "The queue's process exited without its last line.";
  }
  $("done-overlay").hidden = false;
}
function applyEvent(e) {
  const ts = Number(e.ts) * 1000 || Date.now();
  if (e.type === "run_start") {
    if (e.config?.study !== "v5") {
      $("sub").textContent = `this run is not a V5 watch (study: ${e.config?.study ?? e.variant ?? "unknown"})`;
      events.close();
      return;
    }
    // a reconnect replays the whole history: start again from it
    Object.assign(state, { config: e.config, v5: null, alive: null, ended: null, endReason: null });
    $("event-log").innerHTML = "";
    log(`watching ${e.config.dir}${e.config.pid ? ` (queue pid ${e.config.pid})` : ""}`, ts);
  } else if (e.type === "metrics") {
    if (e.resources) state.res = e.resources;
    if ("v5_alive" in e) state.alive = e.v5_alive;
  } else if (e.type === "checkpoint") {
    if (e.v5) state.v5 = e.v5;
  } else if (e.type === "milestone") {
    log(e.message || e.kind, ts, e.kind === "warning" ? "warning" : "note");
  } else if (e.type === "run_end") {
    state.ended = ts; state.endReason = e.reason;
    log(e.reason === "completed" ? "V5 complete" : `watch ended (${e.summary?.note ?? e.reason})`, ts, e.reason === "completed" ? "note" : "warning");
    if (over()) finish(ts);
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
  if (failures >= 3) {
    events.close();
    log("dashboard server stopped (the watcher exited); the queue is unaffected", Date.now(), "warning");
  } else if (failures === 1) {
    log("telemetry reconnecting…", Date.now(), "warning");
  }
};
window.addEventListener("resize", schedule);
$("done-dismiss").onclick = () => { $("done-overlay").hidden = true; };
setInterval(updateClocks, 1000);
render();
