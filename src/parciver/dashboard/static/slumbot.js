// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
/* Slumbot live viewer. Data-only: the watcher (scripts/watch_slumbot.py) reads the run's hands, status and events
   and sends one summary at a time; the page draws it and counts the clocks down. */
"use strict";
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
const num = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
const sgn = (v, d = 1) => num(v) ? `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}` : "—";
const cls = (v) => num(v) ? (v > 0 ? "pos" : v < 0 ? "neg" : "") : "";
const pct = (v, d = 0) => num(v) ? `${(100 * v).toFixed(d)}%` : "—";
const int = (v) => num(v) ? Number(v).toLocaleString() : "—";
const SUIT = { c: "♣", d: "♦", h: "♥", s: "♠" };
const SERIES = [["total", "--total", "all"], ["adj", "--adj", "all-in adjusted"], ["sd", "--sd", "showdown"], ["nsd", "--nsd", "without showdown"]];

function duration(s) {
  if (!num(s)) return "—";
  s = Math.max(0, Math.round(s));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60), q = s % 60;
  const p = (x) => String(x).padStart(2, "0");
  return d ? `${d}d ${h}h ${p(m)}m` : h ? `${h}h ${p(m)}m` : m ? `${m}m ${p(q)}s` : `${q}s`;
}
const utcSec = (u) => u ? Date.parse(u) / 1000 : null;
function log(text, ts, kind = "note") {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${new Date(ts || Date.now()).toLocaleTimeString()}</span><span class="kind ${esc(kind)}">${esc(kind)}</span><span class="msg">${esc(text)}</span>`;
  const ul = $("event-log");
  ul.prepend(li);
  while (ul.children.length > 300) ul.lastChild.remove();
}
function card(c, big = false) {
  if (!c) return `<span class="card back${big ? " big" : ""}">··</span>`;
  return `<span class="card ${c[1]}${big ? " big" : ""}">${esc(c[0] === "T" ? "10" : c[0])}${SUIT[c[1]] || "?"}</span>`;
}
const cards = (xs, big = false, n = 0) => `<span class="cards">${(xs || []).map((c) => card(c, big)).join("")}${"".padEnd(Math.max(0, n - (xs || []).length)).split("").map(() => card(null, big)).join("")}</span>`;
const streetsHtml = (ss) => `<div class="streets">${(ss || []).map(([st, acts]) => `<div><span class="st">${esc(st)}</span>${acts.map(([w, t]) => `<span class="a ${w}">${w === "us" ? "P" : "S"}: ${esc(t)}</span>`).join("")}</div>`).join("")}</div>`;
const streetsShort = (ss) => (ss || []).map(([, acts]) => acts.map(([w, t]) => `<span class="${w === "us" ? "pos" : ""}" style="color:var(--${w})">${esc(t.replace(/^(bet|raise) /, (m) => m[0].toUpperCase()).replace(" all-in", "!"))}</span>`).join(" ")).join(" <span class=\"dim\">/</span> ");

const state = { config: null, s: null, res: null, alive: null, ended: null, hits: {}, started: null };

// ---- header ----------------------------------------------------------------------------------------
function renderHeader() {
  const s = state.s, c = state.config;
  if (!s) return;
  const ph = s.phases[s.focus] || {}, st = s.status || {};
  $("sub").innerHTML = `${esc(s.run.mode)} · <span class="phase-tag ${esc(s.focus)}">${esc(s.focus)}</span> · ${esc(s.run.dir)} · seed ${esc(s.run.seed)} · ${esc(s.run.commit)}`;
  $("stat-hands").textContent = int(ph.n || 0);
  $("stat-hands-sub").textContent = `of ${int(s.target)} ${s.focus}${s.focus === "pilot" ? " (discarded)" : ""}${s.lost.all ? ` · ${s.lost.all} lost` : ""}`;
  $("stat-won").innerHTML = `<span class="${cls(ph.total_bb)}">${sgn(ph.total_bb, 1)}</span><span class="unit"> bb</span>`;
  $("stat-won-sub").textContent = num(ph.total_bb) ? `${sgn(ph.total_bb * 100, 0)} chips` : "—";
  $("stat-rate").innerHTML = num(ph.mean) ? `<span class="${cls(ph.mean)}">${sgn(ph.mean, 1)}</span><span class="unit"> bb/100</span>` : "—";
  $("stat-rate-sub").textContent = num(ph.lo) ? `95% [${sgn(ph.lo, 1)}, ${sgn(ph.hi, 1)}]` : "needs 2 hands";
  const t = s.timing;
  $("stat-pace").textContent = num(t.recent) ? `${t.recent.toFixed(2)} s/hand · ${(3600 / t.recent).toFixed(0)} hands/h` : "—";
  const r = state.res;
  $("stat-res").textContent = r ? `${num(r.gpu_pct) ? Math.round(r.gpu_pct) + "%" : "—"} · ${num(r.cpu_pct) ? Math.round(r.cpu_pct) + "%" : "—"} · ${num(r.ram_used_gb) ? r.ram_used_gb.toFixed(1) + " GB" : "—"}` : "—";
  let dot = "live", label = `playing · session ${st.session ?? "—"} · ${int(st.session_hands)} hands`;
  const age = Date.now() / 1000 - (utcSec(st.utc) || 0);
  if (st.state === "done") { dot = "done"; label = `done: ${st.message || ""}`; }
  else if (st.state && st.state !== "playing") { dot = "err"; label = `${st.state.toUpperCase()}: ${st.message}`; }
  else if (state.alive === false) { dot = "err"; label = "the runner exited without its last line"; }
  else if (num(age) && age > 120) { dot = "err"; label = `no answer for ${duration(age)}: waiting on Slumbot or the GPU?`; }
  if (state.ended) { dot = "idle"; label = "watcher stopped"; }
  $("stat-status").innerHTML = `<span class="dot ${dot}"></span>${esc(label)}`;
  const b = $("banner"), msg = st.state && st.state !== "playing" && st.state !== "done" ? `${st.state.toUpperCase()}: ${st.message}` : "";
  b.className = `banner${msg ? " on" : ""}${st.state === "paused" ? " info" : ""}`;
  b.textContent = msg;
  updateClocks();
}
function updateClocks() {
  const s = state.s;
  if (!s) return;
  const start = utcSec(s.run.started), now = Date.now() / 1000;
  const st = s.status || {};
  const eta = num(s.timing.eta) && st.state === "playing" ? Math.max(0, s.timing.eta - (now - (utcSec(st.utc) || now))) : null;
  $("stat-eta").textContent = `${num(start) ? duration(now - start) : "—"} · ${num(eta) ? `${duration(eta)} (${int(s.timing.left)} hands)` : "—"}`;
}

// ---- hand in play ----------------------------------------------------------------------------------
function renderCurrent() {
  const c = state.s?.current;
  if (!c) { $("cur-kpi").textContent = "—"; return; }
  const boardN = 5;
  const done = num(c.win);
  $("cur-title").textContent = c.live ? "Hand in play" : "Last hand";
  $("cur-kpi").innerHTML = `hand ${int(c.hand)} · ${esc(c.phase)} · we are ${esc(c.seat)}${done ? ` · <span class="${cls(c.win)}">${sgn(c.win, 1)} bb</span>` : ""}`;
  const o = c.options;
  const opts = o ? `<div style="margin-top:12px"><div class="sub" style="margin-bottom:4px">our last decision · ${esc(o.street)}${num(o.seconds) ? ` · ${o.seconds.toFixed(2)} s` : ""}${o.solves ? ` · ${o.solves} GPU solve${o.solves > 1 ? "s" : ""}` : o.street === "turn" || o.street === "river" ? " · search (no new solve)" : " · blueprint"}</div>` +
    o.rows.map(([l, p], i) => `<div class="opt-row${i === o.pick ? " pick" : ""}"><span>${esc(l)}${i === o.pick ? " ◀" : ""}</span><span class="tr"><i style="width:${(100 * p).toFixed(1)}%"></i></span><span style="text-align:right">${(100 * p).toFixed(1)}%</span></div>`).join("") + "</div>" : "";
  $("current").innerHTML = `<div class="table-felt">
      <span class="k">parciver</span><span>${cards(c.hole, true, 2)}</span>
      <span class="k">board</span><span>${cards(c.board, true, boardN)}</span>
      <span class="k">Slumbot</span><span>${cards(c.bot, true, 2)}</span>
    </div><div style="margin-top:12px">${streetsHtml(c.streets)}</div>${opts}`;
}

// ---- curve -----------------------------------------------------------------------------------------
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
function empty(ctx, w, h, text) {
  ctx.fillStyle = css("--ink-3"); ctx.font = `12px ${css("--sans")}`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, w / 2, h / 2);
}
function drawCurve() {
  const { ctx, w, h } = setup($("c-curve"));
  state.hits["c-curve"] = [];
  const cv = state.s?.curve;
  if (!cv || !cv.x.length) return empty(ctx, w, h, "no finished hand yet");
  const ys = SERIES.flatMap(([k]) => cv[k]).concat([0]);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  const pad = 0.08 * (hi - lo || 1); lo -= pad; hi += pad;
  const N = Math.max(cv.x[cv.x.length - 1], 10);
  const L = 54, R = 16, T = 10, B = 24;
  const X = (d) => L + d / N * (w - L - R), Y = (y) => T + (1 - (y - lo) / (hi - lo)) * (h - T - B);
  ctx.font = `11px ${css("--mono")}`; ctx.lineWidth = 1;
  for (const t of niceTicks(lo, hi, 5)) {
    ctx.strokeStyle = css(t === 0 ? "--ink-3" : "--border"); ctx.beginPath(); ctx.moveTo(L, Y(t)); ctx.lineTo(w - R, Y(t)); ctx.stroke();
    ctx.fillStyle = css("--ink-3"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(String(t), L - 6, Y(t));
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of niceTicks(0, N, 6)) ctx.fillText(t.toLocaleString(), X(t), h - B + 7);
  for (const [k, color] of SERIES.slice().reverse()) {
    ctx.strokeStyle = css(color); ctx.lineWidth = k === "total" ? 2.2 : 1.4; ctx.setLineDash(k === "adj" ? [5, 3] : []);
    ctx.beginPath();
    cv.x.forEach((d, i) => { const y = Y(cv[k][i]); if (i) ctx.lineTo(X(d), y); else ctx.moveTo(X(d), y); });
    ctx.stroke(); ctx.setLineDash([]);
  }
  cv.x.forEach((d, i) => state.hits["c-curve"].push({ x: X(d), y: Y(cv.total[i]), html: `<div class="th">${d.toLocaleString()} hands</div>` +
    SERIES.map(([k, , name]) => `<div class="tr"><span>${name}</span><span>${sgn(cv[k][i], 1)} bb</span></div>`).join("") +
    `<div class="tr"><span>rate</span><span>${sgn(cv.total[i] / d * 100, 1)} bb/100</span></div>` }));
  const s = state.s;
  $("curve-note").textContent = `${s.focus} hands. Red: won or lost at showdown; green: without one. Gold: all-in hands scored at their equity when both hands were shown (descriptive).`;
}

// ---- win rate --------------------------------------------------------------------------------------
function rateRow(name, r, extra = "") {
  if (!r || !r.n) return `<tr><td>${esc(name)}</td><td colspan="4" class="dim">no hands</td></tr>`;
  return `<tr><td>${esc(name)}</td><td>${int(r.n)}</td><td class="${cls(r.mean)}">${sgn(r.mean, 1)}</td>` +
    `<td class="dim">${num(r.lo) ? `± ${((r.hi - r.lo) / 2).toFixed(1)}` : "—"}</td><td class="dim">${num(r.sd) ? (r.sd / 100).toFixed(1) : "—"}${extra}</td></tr>`;
}
function renderRates() {
  const s = state.s;
  if (!s) return;
  const ph = s.phases[s.focus];
  $("rates").querySelector("thead").innerHTML = `<tr><th>${esc(s.focus)}</th><th>hands</th><th>bb/100</th><th>95%</th><th>sd bb</th></tr>`;
  const rows = [rateRow("all", ph), rateRow("as SB (button)", s.seats.SB), rateRow("as BB", s.seats.BB), rateRow("all-in adjusted", s.adjusted)];
  if (s.focus === "counted" && s.phases.pilot) rows.push(rateRow("pilot (discarded)", s.phases.pilot));
  $("rates").querySelector("tbody").innerHTML = rows.join("");
  const v = $("verdict");
  const final = s.focus !== "pilot" && ph && ph.n >= s.target;
  v.className = `verdict ${ph?.verdict || ""}`;
  v.textContent = ph?.verdict ? `${ph.verdict}${final ? "" : " (so far; decides nothing until N)"}` : "—";
  $("rate-note").textContent = `bb/100 = mean chips per hand (bb 100 chips). WIN if the 95% interval is above 0 at N = ${int(s.run.hands)}, LOSE if below, else UNRESOLVED (PREREG). N is fixed: the running reading stops nothing.`;
}

// ---- distribution ----------------------------------------------------------------------------------
function binLabel(i, mag) {
  const n = mag.length, z = n + 1;
  if (i === z) return "0";
  const j = i < z ? n - i : i - (n + 2), lo = j ? mag[j - 1] : 0, hi = mag[j];
  const side = i < z ? "−" : "+";
  return hi === undefined ? `${side}${lo}+ bb` : `${side}${lo}…${hi} bb`;
}
function drawHist() {
  const { ctx, w, h } = setup($("c-hist"));
  state.hits["c-hist"] = [];
  const s = state.s, hs = s?.hist;
  if (!hs || !hs.counts.some((x) => x)) return empty(ctx, w, h, "no finished hand yet");
  const n = hs.counts.length, max = Math.max(...hs.counts), total = hs.counts.reduce((a, b) => a + b, 0);
  const L = 8, R = 8, T = 8, B = 22, bw = (w - L - R) / n, z = hs.mag.length + 1;
  ctx.font = `10px ${css("--mono")}`; ctx.textAlign = "center"; ctx.textBaseline = "top";
  hs.counts.forEach((c, i) => {
    const bh = max ? c / max * (h - T - B) : 0, x = L + i * bw, y = h - B - bh;
    ctx.fillStyle = css(i < z ? "--red" : i > z ? "--green" : "--ink-3");
    ctx.globalAlpha = 0.85; ctx.fillRect(x + 1, y, bw - 2, bh); ctx.globalAlpha = 1;
    state.hits["c-hist"].push({ x: x + bw / 2, y: Math.min(y, h - B - 6), html: `<div class="th">${binLabel(i, hs.mag)}</div><div class="tr"><span>hands</span><span>${c.toLocaleString()} (${(100 * c / total).toFixed(1)}%)</span></div>` });
  });
  ctx.fillStyle = css("--ink-3");
  [[0, `−${hs.mag[hs.mag.length - 1]}`], [z, "0"], [n - 1, `+${hs.mag[hs.mag.length - 1]}`]].forEach(([i, t]) => ctx.fillText(t, L + (i + 0.5) * bw, h - B + 6));
  const [wn, ls, ti] = s.wlt;
  $("wlt").textContent = `won ${int(wn)} · lost ${int(ls)} · tied ${int(ti)}`;
}

// ---- style -----------------------------------------------------------------------------------------
function renderHud() {
  const s = state.s;
  if (!s) return;
  const u = s.hud.us, b = s.hud.bot;
  const row = (name, k, f, tip) => `<tr title="${esc(tip)}"><td>${name}</td><td>${f(u[k])}</td><td>${f(b[k])}</td></tr>`;
  $("hud").querySelector("thead").innerHTML = `<tr><th>over ${int(u.hands)} hands</th><th style="color:var(--us)">parciver</th><th style="color:var(--bot)">Slumbot</th></tr>`;
  $("hud").querySelector("tbody").innerHTML = [
    row("VPIP", "vpip", pct, "put chips in preflop by choice (a call or a raise)"),
    row("PFR", "pfr", pct, "raised preflop"),
    row("WTSD", "wtsd", pct, "went to showdown, of hands that reached the flop"),
    row("W$SD", "wsd", pct, "won at showdown, of showdowns"),
    row("AF", "af", (v) => num(v) ? v.toFixed(2) : "—", "postflop bets and raises per call"),
  ].join("");
  const names = ["preflop", "flop", "turn", "river"];
  $("mix").querySelector("thead").innerHTML = `<tr><th>actions</th><th>fold</th><th>check</th><th>call</th><th>bet/raise</th><th>n</th></tr>`;
  $("mix").querySelector("tbody").innerHTML = names.flatMap((n, i) => [["us", "P"], ["bot", "S"]].map(([w, t]) => {
    const r = s.mix[w][i];
    return `<tr><td><span style="color:var(--${w})">${t}</span> ${n}</td>${r.slice(0, 4).map((x) => `<td>${r[4] ? pct(x) : "—"}</td>`).join("")}<td class="dim">${int(r[4])}</td></tr>`;
  })).join("");
}

// ---- hands -----------------------------------------------------------------------------------------
function handRow(r, long = true) {
  return `<tr><td>${int(r.hand)}</td><td class="l">${esc(r.seat)}</td><td class="l">${cards(r.hole)}</td><td class="l">${cards(r.board)}</td>` +
    `<td class="l">${r.bot ? cards(r.bot) : "<span class=\"dim\">—</span>"}</td>` +
    (long ? `<td class="l" style="white-space:normal">${streetsShort(r.streets)}</td>` : "") +
    `<td class="${cls(r.win)}">${sgn(r.win, 1)}</td>${long ? `<td class="dim">${num(r.s) ? r.s.toFixed(1) : "—"}</td>` : ""}</tr>`;
}
function renderHands() {
  const s = state.s;
  if (!s) return;
  $("recent").querySelector("thead").innerHTML = `<tr><th>#</th><th class="l">seat</th><th class="l">ours</th><th class="l">board</th><th class="l">Slumbot</th><th class="l">action (P parciver, S Slumbot)</th><th>bb</th><th>s</th></tr>`;
  $("recent").querySelector("tbody").innerHTML = s.recent.length ? s.recent.map((r) => handRow(r)).join("") : `<tr><td colspan="8" class="dim">no finished hand yet</td></tr>`;
  $("top").querySelector("thead").innerHTML = `<tr><th>#</th><th class="l">seat</th><th class="l">ours</th><th class="l">board</th><th class="l">Slumbot</th><th>bb</th></tr>`;
  $("top").querySelector("tbody").innerHTML = s.top.length ? s.top.map((r) => handRow(r, false)).join("") : `<tr><td colspan="6" class="dim">—</td></tr>`;
  const c = s.checks;
  $("checks").textContent = `payoff checks: ${int(c.ok)} ok · ${int(c.unchecked)} unchecked · ${int(c.mismatch)} mismatch`;
}

// ---- pace and health -------------------------------------------------------------------------------
function renderTiming() {
  const s = state.s;
  if (!s) return;
  const t = s.timing, st = s.status || {};
  const tb = t.think_by_street || [];
  const kv = [
    ["seconds per hand (all · last 200)", `${num(t.s_per_hand) ? t.s_per_hand.toFixed(2) : "—"} · ${num(t.recent) ? t.recent.toFixed(2) : "—"}`],
    ["thinking · waiting on Slumbot", num(t.s_per_hand) ? `${t.think.toFixed(2)} · ${(t.s_per_hand - t.think).toFixed(2)} s` : "—"],
    ["think per decision (pre · flop · turn · river)", tb.map((x) => num(x) ? x.toFixed(2) : "—").join(" · ")],
    ["GPU solves per hand", num(t.solves_per_hand) ? t.solves_per_hand.toFixed(2) : "—"],
    ["search", `turn ${s.run.search.turn_iters} it · river ${s.run.search.river_iters} it${s.run.search.terminals64 ? " · float64 terminals" : ""}`],
    ["network retries", int(st.retried || 0)],
    ["session · its hands", `${st.session ?? "—"} · ${int(st.session_hands)}`],
    ["runner pid", `${st.pid ?? "—"}${state.alive === false ? " (exited)" : ""}`],
  ];
  if (s.pilot_event) kv.push(["pilot projection", num(s.pilot_event.projected_hours) ? `${s.pilot_event.projected_hours.toFixed(1)} h for N (limit ${s.pilot_event.limit_hours} h)` : "none: no pilot hand finished"]);
  if (s.extra) kv.push(["Slumbot also sent", Object.entries(s.extra).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ")]);
  $("timing").innerHTML = kv.map(([k, v]) => `<span class="k">${esc(k)}</span><span>${esc(v)}</span>`).join("");
  const L = s.lost, frac = L.limit ? Math.min(1, L.n / L.limit) : 0;
  $("lost-label").textContent = `lost hands (${s.focus}): ${L.n} of ${L.limit} allowed (0.5% of N)${L.all !== L.n ? ` · ${L.all} in all` : ""}`;
  $("lost-meter").className = `meter${frac > 0.5 ? " warn" : ""}`;
  $("lost-meter").querySelector("i").style.width = `${(100 * frac).toFixed(1)}%`;
  $("lost-note").textContent = L.recent.length ? `last: hand ${L.recent[L.recent.length - 1].hand}: ${L.recent[L.recent.length - 1].reason}` : "none lost";
}

function render() {
  renderHeader(); renderCurrent(); drawCurve(); renderRates(); drawHist(); renderHud(); renderHands(); renderTiming();
}

// ---- events ----------------------------------------------------------------------------------------
let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } }
function finish() {
  const s = state.s, st = s?.status || {};
  if (st.state === "done") {
    const ph = s.phases[s.focus] || {};
    $("done-detail").textContent = `${int(ph.n)} ${s.focus} hands: ${sgn(ph.mean, 1)} bb/100, 95% [${sgn(ph.lo, 1)}, ${sgn(ph.hi, 1)}] → ${ph.verdict}. ${st.message || ""}`;
  } else {
    $("done-badge").textContent = "!"; $("done-badge").classList.add("err");
    $("done-title").textContent = st.state === "paused" ? "Run paused: your decision" : st.state === "failed" ? "Mock check failed" : "Run stopped";
    $("done-detail").textContent = st.message || "";
  }
  $("done-overlay").hidden = false;
}
let shownEnd = null;
function applyEvent(e) {
  const ts = Number(e.ts) * 1000 || Date.now();
  if (e.type === "run_start") {
    if (e.config?.study !== "slumbot") {
      $("sub").textContent = `this run is not a Slumbot watch (study: ${e.config?.study ?? e.variant ?? "unknown"})`;
      events.close();
      return;
    }
    Object.assign(state, { config: e.config, ended: null });
    if (!state.started) log(`watching ${e.config.dir}`, ts);
  } else if (e.type === "metrics") {
    if (e.resources) state.res = e.resources;
    if ("runner_alive" in e) state.alive = e.runner_alive;
  } else if (e.type === "checkpoint") {
    if (e.slumbot) state.s = e.slumbot;
    state.started = true;
    const st = state.s?.status;
    if (st && st.state && st.state !== "playing" && shownEnd !== `${st.state}:${st.utc}`) { shownEnd = `${st.state}:${st.utc}`; finish(); }
  } else if (e.type === "milestone") {
    if (e.kind === "rotate") { reconnect(); return; }
    log(e.message || e.kind, ts, e.kind === "warning" ? "warning" : "note");
  } else if (e.type === "run_end") {
    state.ended = ts;
    log(`watcher ended (${e.summary?.note ?? e.reason})`, ts, "warning");
  }
  schedule();
}

document.addEventListener("mousemove", (ev) => {
  const tip = $("tip"), el = ev.target;
  const hits = el && el.tagName === "CANVAS" ? state.hits[el.id] : null;
  if (!hits || !hits.length) { tip.hidden = true; return; }
  const r = el.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
  let best = null, bd = Infinity;
  for (const h of hits) { const d = el.id === "c-curve" ? Math.abs(h.x - mx) : Math.hypot(h.x - mx, h.y - my); if (d < bd) { bd = d; best = h; } }
  if (!best || bd > (el.id === "c-curve" ? 30 : 40)) { tip.hidden = true; return; }
  tip.innerHTML = best.html; tip.hidden = false;
  tip.style.left = `${Math.min(r.left + best.x + 12, window.innerWidth - 260)}px`; tip.style.top = `${r.top + best.y - 8}px`;
});

let events = null, failures = 0;
function connect() {
  events = new EventSource("/events");
  events.onmessage = (m) => JSON.parse(m.data).forEach(applyEvent);
  events.onopen = () => { failures = 0; };
  events.onerror = () => {
    failures += 1;
    if (failures >= 3) {
      events.close();
      log("dashboard server stopped (the watcher exited); the run is unaffected", Date.now(), "warning");
    } else if (failures === 1) {
      log("telemetry reconnecting…", Date.now(), "warning");
    }
  };
}
function reconnect() {  // the watcher started a new log: read it from its start
  events.close();
  setTimeout(connect, 300);
}
connect();
window.addEventListener("resize", schedule);
$("done-dismiss").onclick = () => { $("done-overlay").hidden = true; };
setInterval(updateClocks, 1000);
render();
