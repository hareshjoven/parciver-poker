// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
/* PREREG15 D1 re-solve viewer. Data-only: the watcher (scripts/watch_resolve15.py)
   tails the job's OUT file and sends its rows and resolve15.py's summary statistics;
   the page only extrapolates the pace and draws. */
"use strict";
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
const num = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
const L10 = Math.log10;
const GROUPS = ["TAIL", "A", "B"];
const GCOL = { TAIL: "--gT", A: "--gA", B: "--gB" };
const gcol = (g) => css(GCOL[g.split("+")[0]] || "--ink-3");
const SRC_NAME = ["recorded", "random"];
const atName = (at) => `${at[0]}/${SRC_NAME[at[1]] ?? at[1]}`;

/** Pots as a percentage with sensible precision: 0.0123 -> "1.23%". */
function pct(v, digits) {
  if (!num(v)) return "—";
  const p = 100 * v, a = Math.abs(p);
  const d = digits ?? (a >= 10 ? 1 : a >= 1 ? 2 : a >= 0.1 ? 3 : 4);
  return `${p.toFixed(d)}%`;
}
function tickPct(v) {  // axis labels: 1e-4 -> "0.01%"
  const p = 100 * v;
  return `${p >= 1 ? +p.toFixed(0) : +p.toPrecision(1)}%`;
}
function duration(s) {
  if (!num(s)) return "—";
  s = Math.max(0, Math.round(s));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60), q = s % 60;
  const p = (x) => String(x).padStart(2, "0");
  return d ? `${d}d ${h}h ${p(m)}m` : h ? `${h}h ${p(m)}m ${p(q)}s` : m ? `${m}m ${p(q)}s` : `${q}s`;
}
const mean = (xs) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
function median(xs) {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
function log(text, ts, kind = "note") {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${new Date(ts || Date.now()).toLocaleTimeString()}</span><span class="kind ${esc(kind)}">${esc(kind)}</span><span class="msg">${esc(text)}</span>`;
  $("event-log").prepend(li);
}

const state = { config: null, head: null, rows: [], stats: null, mtime: null, summary: null, stopped: null,
                alive: null, res: null, ended: null, endReason: null, hits: {}, table: "kept" };

// ---- pace -----------------------------------------------------------------------------------------
function total() {
  const c = state.config;
  return state.head?.rows ?? (c ? Object.values(c.groups).reduce((a, b) => a + b, 0) : 0);
}
/** Seconds left: the rows left at the mean pace of the last 100 rows without a walk, plus the C2 walks still due
    (the tail's rows over 5% and A's first 40) at their extra cost so far. */
function eta() {
  const c = state.config, rows = state.rows, s = state.stats;
  if (!c || !rows.length || state.summary || state.stopped || state.alive === false || state.ended) return null;
  const recent = rows.slice(-100);
  let plain = recent.filter((r) => !("e_walk" in r)).map((r) => r.seconds);
  if (!plain.length) plain = rows.filter((r) => !("e_walk" in r)).map((r) => r.seconds);
  if (!plain.length) return null;  // only C2 rows so far (the tail's first 81): no plain pace yet
  const walked = rows.filter((r) => "e_walk" in r);
  const perRow = mean(plain);
  const walkA = walked.filter((r) => r.group.startsWith("A")), walkT = walked.filter((r) => r.group === "TAIL");
  const extra = Math.max(0, mean((walkA.length ? walkA : walked).map((r) => r.seconds)) - perRow) || 0;
  const tailOver = rows.filter((r) => r.group === "TAIL" && r.e_stored > c.c2_hi).length;
  const walksLeft = Math.max(0, c.c2_tail - tailOver) + Math.max(0, c.c2_a - walkA.length);
  const left = total() - rows.length;
  return { perRow, left, seconds: left * perRow + walksLeft * extra, walksLeft, extra, tailWalk: mean(walkT.map((r) => r.seconds)) };
}

// ---- header -----------------------------------------------------------------------------------------
function renderHeader() {
  const c = state.config, s = state.stats, h = state.head, n = state.rows.length, N = total();
  if (c) {
    $("sub").textContent = `${h ? `code ${h.head.slice(0, 7)} · ${h.gpu} · torch ${h.torch}` : "waiting for the job's first line"} · ` +
      `FusedSolver(terminals64=True) · 500 iterations · γ 5 · float32`;
    $("scope-note").textContent = `Reads ${c.out} (and its stderr) only; it cannot slow the job. Statistics mirror resolve15.py's summary ` +
      `(keep-better: a row keeps the lower of the stored and the re-solved e). Checks stop the job on failure: C1 |game_new − game_stored| ≤ e_new + e_stored + 1e-5 ` +
      `(every row); C2 |e_readout − e_walk| ≤ ${c.c2_tol} pot (the tail's ${c.c2_tail} rows over 5% and A's first ${c.c2_a}). Pre-registration: ${c.prereg}.`;
  }
  $("stat-rows").textContent = c ? n.toLocaleString() : "—";
  $("stat-rows-sub").textContent = N ? `of ${N.toLocaleString()} · ${(100 * n / N).toFixed(1)}%` : "—";
  const t = s?.tail;
  if (t) {
    $("stat-tail").textContent = `${pct(t.median_before, 2)} → ${pct(t.median_after, 2)}`;
    $("stat-tail-sub").textContent = `${t.rows.toLocaleString()} tail rows so far · ${t.improved} better, ${t.worsened} worse`;
  }
  const r = state.res;
  const parts = [r && num(r.gpu_pct) ? `${r.gpu_pct.toFixed(0)}%` + (num(r.gpu_mem_used_gb) ? ` ${r.gpu_mem_used_gb.toFixed(1)} GB` : "") : "—",
                 r && num(r.cpu_pct) ? `${r.cpu_pct.toFixed(0)}%` : "—",
                 r && num(r.ram_used_gb) ? `${r.ram_used_gb.toFixed(1)}/${r.ram_total_gb.toFixed(0)} GB` : "—"];
  $("stat-res").textContent = parts.join(" · ");
  $("stat-res").title = "The whole machine (nvidia-smi and Windows counters), not the job alone.";
  const e = eta();
  $("stat-pace").textContent = e ? `${e.perRow.toFixed(1)} s/row` + (e.walksLeft ? ` · ${e.walksLeft} walks left` : "") : "—";
  updateClocks();
}
function updateClocks() {
  const c = state.config, rows = state.rows;
  if (!c) return;
  const now = Date.now() / 1000;
  const solved = rows.reduce((a, r) => a + r.seconds, 0);
  const start = num(c.started) ? c.started : num(state.mtime) ? state.mtime - solved : null;
  const over = state.summary || state.stopped || state.alive === false;
  const end = over && num(state.mtime) ? state.mtime : now;
  let text = start ? duration(end - start) : "—";
  const e = eta();
  if (state.summary) text += " · complete";
  else if (state.stopped) text += " · stopped";
  else if (e && num(state.mtime)) {
    const left = state.mtime + e.seconds - now;
    text += left > 0 ? ` · ${duration(left)} left` : " · due";
  } else if (!over && rows.length) text += " · ETA after the first row without a walk";
  $("stat-eta").textContent = text;
  $("stat-eta").title = `Elapsed: since the job's process started${num(c.started) ? "" : " (estimated: the rows' own seconds)"}. ` +
    "ETA: the rows left at the mean pace of the last 100 rows without a C2 walk, plus the walks still due, counting down from the last finished row.";
  const since = num(state.mtime) ? now - state.mtime : null;
  const last = rows[rows.length - 1];
  $("last-row").textContent = last ? `last row ${last.j}/${SRC_NAME[last.source]} · ${last.group} · ${duration(since)} ago` : "—";
  let dot = "live", label = "re-solving";
  const typical = rows.length ? median(rows.slice(-50).map((r) => r.seconds)) : NaN;
  if (state.summary) { dot = "done"; label = "complete · summary written"; }
  else if (state.stopped) { dot = "err"; label = `STOPPED: ${state.stopped.reason} failed`; }
  else if (state.alive === false) { dot = "err"; label = "job exited without its summary"; }
  else if (!rows.length) { dot = "idle"; label = state.head ? "first row solving" : "waiting for the job"; }
  else if (num(since) && num(typical) && since > Math.max(120, 10 * typical)) { dot = "err"; label = `no row for ${duration(since)}: stalled?`; }
  if (state.ended && !state.summary && !state.stopped && state.alive !== false) { dot = "idle"; label = "watcher stopped"; }
  $("stat-status").innerHTML = `<span class="dot ${dot}"></span>${esc(label)}`;
  const b = $("banner");
  if (state.stopped) { b.textContent = `The job stopped: check ${state.stopped.reason} failed on row ${atName(state.stopped.at)}. No patch is written; report before anything else.`; b.classList.add("on"); }
  else if (state.alive === false && !state.summary) { b.textContent = `The job's process exited without its summary line: see ${c.err}.`; b.classList.add("on"); }
  else b.classList.remove("on");
}

// ---- progress ---------------------------------------------------------------------------------------
function renderProgress() {
  const c = state.config, s = state.stats;
  if (!c) return;
  const N = total() || 1, done = s?.done || {};
  // B's own segment is the B rows not already solved as A (the overlap is solved once, in A's place)
  const overlap = Object.values(c.groups).reduce((a, b) => a + b, 0) - N;
  const bOnly = state.rows.filter((r) => r.group === "B").length;
  const segs = [["TAIL", c.groups.TAIL, done.TAIL || 0], ["A", c.groups.A, done.A || 0], ["B", c.groups.B - overlap, bOnly]];
  $("segbar").innerHTML = segs.map(([g, n, k]) =>
    `<div class="seg" style="flex:${n} 1 0"><div class="track"><div class="fill" style="width:${(100 * k / n).toFixed(2)}%;background:var(${GCOL[g]})"></div></div>` +
    `<div class="sl"><b style="color:var(${GCOL[g]})">${g === "TAIL" ? "tail" : g === "B" && overlap ? "B only" : g}</b><span>${k.toLocaleString()} / ${n.toLocaleString()}</span></div></div>`).join("");
  if (!s) return;
  const a = s.all, c1 = s.c1, c2 = s.c2;
  const tiles = [
    ["C1 · consistency", `${c1.rows - c1.fails} / ${c1.rows}`, num(c1.worst) ? `worst margin ${c1.worst.toExponential(1)} pot (≤ 1e-5)` : "", c1.fails ? "bad" : "good"],
    ["C2 · float64 walk", `${c2.rows - c2.fails} / ${c2.rows}`, num(c2.max_gap) ? `max gap ${c2.max_gap.toExponential(1)} pot (≤ ${c.c2_tol})` : "no walks yet", c2.fails ? "bad" : c2.rows ? "good" : ""],
    ["re-solve better", a.improved.toLocaleString(), a.rows ? `${(100 * a.improved / a.rows).toFixed(1)}% of rows · kept` : "", ""],
    ["re-solve worse", a.worsened.toLocaleString(), "the stored solve stays", ""],
    ["tail still > 5%", s.tail ? String(s.tail.over_5_after) : "—", s.tail ? `after keep-better · > 10%: ${s.tail.over_10_after}` : "", s.tail?.over_10_after ? "warn" : ""],
    ["largest kept e", pct(a.kept_max, 2), "over every row so far", ""],
    ["mean value move", num(a.mean_value_move) ? `${(100 * a.mean_value_move).toFixed(3)}%` : "—", "range-weighted |Δ values|, pot", ""],
  ];
  $("tiles").innerHTML = tiles.map(([k, v, sub, cls]) =>
    `<div class="tile ${cls}"><span class="tl">${esc(k)}</span><strong>${esc(v)}</strong><small>${esc(sub)}</small></div>`).join("");
}

// ---- drawing helpers ----------------------------------------------------------------------------------
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
/** Log-axis ticks (values, not logs) at 1, 2, 5 × 10^k inside [lo, hi]. */
function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(L10(lo)); e <= Math.ceil(L10(hi)); e++) {
    for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
  }
  return out.length > 7 ? out.filter((v) => Math.abs(L10(v) - Math.round(L10(v))) < 1e-9) : out;
}
function logDomain(vals, padLo = 1.6, padHi = 1.6) {
  const v = vals.filter((x) => num(x) && x > 0);
  if (!v.length) return [1e-4, 1e-1];
  return [Math.min(...v) / padLo, Math.max(...v) * padHi];
}
function axisY(ctx, ticks, Y, L, R, w, fmt) {
  ctx.font = `11px ${css("--mono")}`; ctx.lineWidth = 1;
  for (const v of ticks) {
    const y = Y(v);
    ctx.strokeStyle = css("--border"); ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(w - R, y); ctx.stroke();
    ctx.fillStyle = css("--ink-3"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(fmt(v), L - 6, y);
  }
}
function hit(id, x, y, html, mode = "xy") { state.hits[id].push({ x, y, html, mode }); }
function rowTip(r, extra = "") {
  const kept = Math.min(r.e_stored, r.e_new);
  return `<div class="th">${esc(r.j)}/${esc(SRC_NAME[r.source])} · ${esc(r.group)}</div>` +
    `<div class="tr"><span>stored e</span><span>${pct(r.e_stored)}</span></div><div class="tr"><span>re-solved e</span><span>${pct(r.e_new)}</span></div>` +
    `<div class="tr"><span>kept</span><span>${pct(kept)}</span></div><div class="tr"><span>value move</span><span>${pct(r.value_move)}</span></div>` +
    ("e_walk" in r ? `<div class="tr"><span>C2 gap</span><span>${Math.abs(r.e_new - r.e_walk).toExponential(1)}</span></div>` : "") +
    `<div class="tr"><span>seconds</span><span>${r.seconds}</span></div>${extra}`;
}

// ---- scatter: stored vs re-solved ---------------------------------------------------------------------
function drawScatter() {
  const id = "chart-scatter", { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  const rows = state.rows;
  if (!rows.length) return;
  const [lo, hi] = logDomain(rows.flatMap((r) => [r.e_stored, r.e_new]), 1.4, 1.4);
  const L = 52, R = 14, T = 10, B = 30;
  const X = (v) => L + (L10(Math.max(v, lo)) - L10(lo)) / (L10(hi) - L10(lo)) * (w - L - R);
  const Y = (v) => T + (1 - (L10(Math.max(v, lo)) - L10(lo)) / (L10(hi) - L10(lo))) * (h - T - B);
  const ticks = logTicks(lo, hi);
  axisY(ctx, ticks, Y, L, R, w, tickPct);
  ctx.fillStyle = css("--ink-3"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const v of ticks) {
    ctx.strokeStyle = css("--border"); ctx.beginPath(); ctx.moveTo(X(v), T); ctx.lineTo(X(v), h - B); ctx.stroke();
    ctx.fillText(tickPct(v), X(v), h - B + 7);
  }
  ctx.textAlign = "right"; ctx.fillText("stored →", w - R, h - B + 18);
  ctx.save(); ctx.translate(12, T); ctx.rotate(-Math.PI / 2); ctx.textAlign = "right"; ctx.fillText("re-solved →", 0, 0); ctx.restore();
  ctx.setLineDash([4, 4]); ctx.strokeStyle = css("--ink-3");
  for (const v of [0.03, 0.05, 0.1]) if (v > lo && v < hi) {
    ctx.beginPath(); ctx.moveTo(X(v), T); ctx.lineTo(X(v), h - B); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(w - R, Y(v)); ctx.stroke();
  }
  ctx.setLineDash([]); ctx.strokeStyle = css("--ink-2"); ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(X(lo), Y(lo)); ctx.lineTo(X(hi), Y(hi)); ctx.stroke();
  for (const r of rows) {
    const x = X(r.e_stored), y = Y(r.e_new);
    ctx.fillStyle = gcol(r.group); ctx.globalAlpha = 0.75;
    ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 2 * Math.PI); ctx.fill(); ctx.globalAlpha = 1;
    hit(id, x, y, rowTip(r));
  }
}

// ---- readings ------------------------------------------------------------------------------------------
function renderReadings() {
  const s = state.stats, sum = state.summary, c = state.config;
  if (!c) return;
  const T = sum?.tail || s?.tail, T5 = sum?.tail_over_5 || s?.tail_over_5;
  const cols = [["all tail rows", T], ["stored e > 5%", T5]];
  $("tail-tbl").querySelector("thead").innerHTML = `<tr><th>tail${sum ? " · final" : " · so far"}</th>${cols.map(([k]) => `<th>${k}</th>`).join("")}</tr>`;
  const line = (k, f, cls) => `<tr><td>${k}</td>${cols.map(([, t]) => `<td class="${cls ? cls(t) : ""}">${t ? f(t) : "—"}</td>`).join("")}</tr>`;
  $("tail-tbl").querySelector("tbody").innerHTML = [
    line("rows", (t) => t.rows.toLocaleString()),
    line("re-solve better / worse", (t) => `${t.improved} / ${t.worsened}`),
    line("median e stored", (t) => pct(t.median_before, 2)),
    line("median e kept", (t) => pct(t.median_after, 2), () => "ok"),
    line("still > 5% / > 10%", (t) => `${t.over_5_after} / ${t.over_10_after}`, (t) => t?.over_10_after ? "warm" : ""),
  ].join("");
  for (const g of ["A", "B"]) drawReading(g, sum?.[g] ? { ...sum[g], final: true, limit: c.limits[g] } : s?.[g]);
}
function drawReading(g, r) {
  const id = `chart-${g}`, { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  const c = state.config, lim = c.limits[g], n = c.groups[g];
  const done = state.stats?.done?.[g] || 0;
  if (!r) {
    $(`rv-${g}`).innerHTML = `<span class="wait">${done ? `${done} row so far: an interval needs 2` : `no ${g} rows yet: the samples follow the tail`}</span>`;
    return;
  }
  const [lo, hi] = r.ci95;
  const xs = [0, lim, lo, hi, r.mean_d];
  const span = Math.max(...xs) - Math.min(...xs) || lim;
  const a = Math.min(...xs) - 0.12 * span, b = Math.max(...xs) + 0.12 * span;
  const L = 8, R = 8, X = (v) => L + (v - a) / (b - a) * (w - L - R), y = h / 2 - 6;
  ctx.font = `10.5px ${css("--mono")}`; ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const [v, label, col, dash] of [[0, "0", "--ink-3", []], [lim, `${pct(lim, 2)} limit`, "--warm", [4, 3]]]) {
    ctx.strokeStyle = css(col); ctx.setLineDash(dash); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(v), 4); ctx.lineTo(X(v), h - 18); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css(col); ctx.fillText(label, X(v), h - 15);
  }
  const col = css(GCOL[g]);
  ctx.strokeStyle = col; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(X(lo), y); ctx.lineTo(X(hi), y); ctx.stroke();
  for (const v of [lo, hi]) { ctx.beginPath(); ctx.moveTo(X(v), y - 6); ctx.lineTo(X(v), y + 6); ctx.stroke(); }
  ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(r.mean_d), y, 4.5, 0, 2 * Math.PI); ctx.fill();
  hit(id, X(r.mean_d), y, `<div class="th">${g} · ${r.rows} rows</div><div class="tr"><span>mean d</span><span>${pct(r.mean_d, 4)}</span></div>` +
    `<div class="tr"><span>95% interval</span><span>${pct(lo, 4)} … ${pct(hi, 4)}</span></div>`, "x");
  const yes = r.reading === "RE-SOLVE-ALL" || r.reading === "EXTEND";
  const status = r.final ? (state.summary ? "the reading" : "all rows in; the job's summary decides") : `preview at ${r.rows} / ${n} rows: would read`;
  $(`rv-${g}`).innerHTML = `${status} <b class="${yes ? "yes" : "no"}">${esc(r.reading)}</b> · mean d ${pct(r.mean_d, 4)} · interval ${pct(lo, 4)} … ${pct(hi, 4)}` +
    (num(r.median_value_move) ? ` · median value move ${pct(r.median_value_move, 3)}` : "");
}

// ---- solving order ---------------------------------------------------------------------------------------
function drawOrder() {
  const id = "chart-order", { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  const rows = state.rows, N = total();
  if (!rows.length || !N) return;
  const [lo, hi] = logDomain(rows.flatMap((r) => [r.e_stored, r.e_new]), 1.4, 1.4);
  const L = 52, R = 10, T = 10, B = 24;
  const X = (i) => L + i / Math.max(1, N - 1) * (w - L - R);
  const Y = (v) => T + (1 - (L10(Math.max(v, lo)) - L10(lo)) / (L10(hi) - L10(lo))) * (h - T - B);
  axisY(ctx, logTicks(lo, hi), Y, L, R, w, tickPct);
  ctx.fillStyle = css("--ink-3"); ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.font = `11px ${css("--mono")}`;
  for (const k of niceTicks(0, N, 6)) ctx.fillText(String(k), X(k), h - B + 7);
  const c = state.config, tailEnd = c.groups.TAIL, aEnd = tailEnd + c.groups.A;
  ctx.strokeStyle = css("--border-2"); ctx.setLineDash([3, 3]);
  for (const k of [tailEnd, aEnd]) { ctx.beginPath(); ctx.moveTo(X(k), T); ctx.lineTo(X(k), h - B); ctx.stroke(); }
  ctx.setLineDash([]);
  rows.forEach((r, i) => {
    const x = X(i);
    ctx.fillStyle = css("--ink-3"); ctx.globalAlpha = 0.5; ctx.fillRect(x - 1, Y(r.e_stored) - 1, 2, 2);
    ctx.globalAlpha = 0.85; ctx.fillStyle = gcol(r.group); ctx.fillRect(x - 1.2, Y(r.e_new) - 1.2, 2.4, 2.4); ctx.globalAlpha = 1;
    hit(id, x, Y(r.e_new), rowTip(r, `<div class="tr"><span>order</span><span>${i + 1}</span></div>`), "x");
  });
}

// ---- seconds per row -------------------------------------------------------------------------------------
function drawPace() {
  const id = "chart-pace", { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  const rows = state.rows, N = total();
  if (!rows.length || !N) return;
  const secs = rows.map((r) => r.seconds), med = median(secs), cap = Math.max(4 * med, 1);
  const ymax = Math.min(Math.max(...secs), cap) * 1.12;
  const L = 44, R = 10, T = 10, B = 24;
  const bw = (w - L - R) / N;
  const X = (i) => L + i * bw;
  const Y = (s) => T + (1 - s / ymax) * (h - T - B);
  axisY(ctx, niceTicks(0, ymax, 4), Y, L, R, w, (v) => `${v}s`);
  ctx.fillStyle = css("--ink-3"); ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.font = `11px ${css("--mono")}`;
  for (const k of niceTicks(0, N, 6)) ctx.fillText(String(k), X(k), h - B + 7);
  rows.forEach((r, i) => {
    const s = Math.min(r.seconds, cap), walk = "e_walk" in r;
    ctx.fillStyle = walk ? css("--warm") : gcol(r.group); ctx.globalAlpha = r.seconds > cap ? 1 : 0.7;
    ctx.fillRect(X(i), Y(s), Math.max(1, bw), Y(0) - Y(s)); ctx.globalAlpha = 1;
    hit(id, X(i) + bw / 2, Y(s), `<div class="th">${esc(r.j)}/${esc(SRC_NAME[r.source])} · ${esc(r.group)}${walk ? " · C2 walk" : ""}</div>` +
      `<div class="tr"><span>seconds</span><span>${r.seconds}${r.seconds > cap ? " (clipped)" : ""}</span></div><div class="tr"><span>order</span><span>${i + 1}</span></div>`, "x");
  });
  ctx.strokeStyle = css("--ink-2"); ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(L, Y(med)); ctx.lineTo(w - R, Y(med)); ctx.stroke(); ctx.setLineDash([]);
  const e = eta();
  $("pace-kpi").textContent = `median ${med.toFixed(1)} s` + (e ? ` · ETA pace ${e.perRow.toFixed(1)} s/row` : "") +
    ` · solving ${duration(secs.reduce((a, b) => a + b, 0))}`;
}

// ---- rows table -------------------------------------------------------------------------------------------
function renderRows() {
  const rows = state.rows.map((r, i) => ({ ...r, i, kept: Math.min(r.e_stored, r.e_new) }));
  let sel;
  if (state.table === "kept") sel = rows.sort((a, b) => b.kept - a.kept).slice(0, 14);
  else if (state.table === "worse") sel = rows.filter((r) => r.d < 0).sort((a, b) => a.d - b.d).slice(0, 14);
  else sel = rows.slice(-14).reverse();
  $("rows").querySelector("thead").innerHTML = "<tr><th>row</th><th>group</th><th>order</th><th>stored e</th><th>re-solved e</th><th>kept</th><th>d</th><th>value move</th><th>C1</th><th>C2 gap</th><th>s</th></tr>";
  $("rows").querySelector("tbody").innerHTML = sel.length ? sel.map((r) => {
    const cls = r.kept >= 0.1 ? "hot" : r.kept >= 0.05 ? "warm" : "";
    const gap = "e_walk" in r ? Math.abs(r.e_new - r.e_walk) : null;
    return `<tr><td>${r.j}/${SRC_NAME[r.source]}</td><td style="text-align:left"><span class="gtag" style="background:${gcol(r.group)}"></span>${esc(r.group)}</td><td class="dim">${r.i + 1}</td>` +
      `<td>${pct(r.e_stored)}</td><td class="${r.d > 0 ? "ok" : r.d < 0 ? "warm" : ""}">${pct(r.e_new)}</td><td class="${cls}">${pct(r.kept)}</td>` +
      `<td>${pct(r.d)}</td><td class="dim">${pct(r.value_move)}</td><td class="${r.c1 ? "ok" : "hot"}">${r.c1 ? "pass" : "FAIL"}</td>` +
      `<td class="${gap === null ? "dim" : r.c2 ? "ok" : "hot"}">${gap === null ? "—" : gap.toExponential(1)}</td><td class="dim">${r.seconds}</td></tr>`;
  }).join("") : `<tr><td colspan="11" class="dim">none so far</td></tr>`;
}

function render() {
  renderHeader(); renderProgress(); drawScatter(); renderReadings(); drawOrder(); drawPace(); renderRows();
}

// ---- events -------------------------------------------------------------------------------------------------
let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } }
let announced = false;
function announce() {
  if (announced) return;
  announced = true;
  const s = state.summary, t = s.tail || {};
  const rd = (g) => s[g] ? `${g}: ${s[g].reading} (mean d ${pct(s[g].mean_d, 4)}, interval ${pct(s[g].ci95[0], 4)} … ${pct(s[g].ci95[1], 4)})` : `${g}: —`;
  $("done-detail").textContent = `${state.rows.length.toLocaleString()} rows; solving time ${duration(s.seconds)}${state.head?.resumed ? " (the last session)" : ""}. Tail median ${pct(t.median_before, 2)} → ${pct(t.median_after, 2)}; ` +
    `still over 5%: ${t.over_5_after}, over 10%: ${t.over_10_after}. ${rd("A")}. ${rd("B")}. C1 ${s.c1_all ? "pass" : "FAIL"}, C2 ${s.c2_all ? "pass" : "FAIL"} (${s.c2_rows} rows)` +
    `${s.patch ? `; patch: ${s.rows} kept re-solves → ${s.patch}` : "; no patch written"}.`;
  $("done-overlay").hidden = false;
}
function applyEvent(e) {
  const ts = Number(e.ts) * 1000 || Date.now();
  if (e.type === "run_start") {
    if (e.config?.study !== "resolve15") {
      $("sub").textContent = `this run is not a re-solve watch (study: ${e.config?.study ?? e.variant ?? "unknown"})`;
      events.close();
      return;
    }
    // a reconnect replays the whole history: start again from it
    Object.assign(state, { config: e.config, head: null, rows: [], stats: null, mtime: null, summary: null, stopped: null,
                           alive: null, ended: null, endReason: null });
    $("event-log").innerHTML = "";
    log(`watching ${e.config.out}${e.config.pid ? ` (pid ${e.config.pid})` : ""}`, ts);
  } else if (e.type === "metrics") {
    if (e.resources) state.res = e.resources;
    if ("r15_alive" in e) state.alive = e.r15_alive;
  } else if (e.type === "checkpoint") {
    if (e.r15_head) state.head = e.r15_head;
    if (e.r15_rows) state.rows.push(...e.r15_rows);
    if (e.r15_stats) state.stats = e.r15_stats;
    if (num(e.r15_mtime)) state.mtime = e.r15_mtime;
    if (e.r15_summary) state.summary = e.r15_summary;
    if (e.r15_stopped) state.stopped = e.r15_stopped;
  } else if (e.type === "milestone") {
    log(e.message || e.kind, ts, e.kind === "warning" ? "warning" : "note");
  } else if (e.type === "run_end") {
    state.ended = ts; state.endReason = e.reason;
    if (e.reason === "completed" && state.summary) announce();
    else if (state.stopped) {
      $("done-badge").textContent = "!"; $("done-badge").classList.add("err"); $("done-title").textContent = "Re-solve stopped";
      $("done-detail").textContent = `Check ${state.stopped.reason} failed on row ${atName(state.stopped.at)}. No patch was written.`;
      $("done-overlay").hidden = false;
    }
    log(e.reason === "completed" ? "re-solve complete" : `watch ended (${e.summary?.note ?? e.reason})`, ts, e.reason === "completed" ? "note" : "warning");
  }
  schedule();
}

document.addEventListener("mousemove", (ev) => {
  const tip = $("tip"), el = ev.target;
  const hits = el && el.tagName === "CANVAS" ? state.hits[el.id] : null;
  if (!hits || !hits.length) { tip.hidden = true; return; }
  const r = el.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
  let best = null, bd = Infinity;
  for (const h of hits) {
    const d = h.mode === "x" ? Math.abs(h.x - mx) : Math.hypot(h.x - mx, h.y - my);
    if (d < bd) { bd = d; best = h; }
  }
  if (!best || bd > (best.mode === "xy" ? 10 : 24)) { tip.hidden = true; return; }
  tip.innerHTML = best.html; tip.hidden = false;
  tip.style.left = `${r.left + best.x + 12}px`; tip.style.top = `${r.top + (best.mode === "x" ? my : best.y) - 8}px`;
});
$("switch").addEventListener("click", (ev) => {
  const k = ev.target?.dataset?.k;
  if (!k) return;
  state.table = k;
  for (const b of $("switch").querySelectorAll("button")) b.classList.toggle("on", b.dataset.k === k);
  renderRows();
});

const events = new EventSource("/events");
events.onmessage = (m) => JSON.parse(m.data).forEach(applyEvent);
let failures = 0;
events.onopen = () => { failures = 0; };
events.onerror = () => {
  failures += 1;
  if (failures >= 3) {
    events.close();
    log("dashboard server stopped (the watcher exited); the re-solve is unaffected", Date.now(), "warning");
  } else if (failures === 1) {
    log("telemetry reconnecting…", Date.now(), "warning");
  }
};
window.addEventListener("resize", schedule);
$("done-dismiss").onclick = () => { $("done-overlay").hidden = true; };
setInterval(updateClocks, 1000);
render();
