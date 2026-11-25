// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
/* Turn-value data (D1) viewer. Data-only: every statistic comes from the
   watcher (scripts/watch_turn_data.py), which reads the generator's finished
   shards; the page only sums per-shard counts, extrapolates the pace, and draws. */
"use strict";
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
const num = (v) => v !== null && v !== undefined && Number.isFinite(Number(v));
const L10 = Math.log10;
const SRC = ["--s0", "--s1"];
const SRC_NAME = ["recorded", "random"];
const PALETTE = ["--hot", "--warm", "--s1", "--s0", "--ok"];

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

const state = { config: null, shards: new Map(), global: null, checks: {}, tdlog: null, res: null,
                watchStarted: null, watchEnded: null, endReason: null, hits: {}, agg: null };

// ---- aggregation ---------------------------------------------------------------
function aggregate() {
  const c = state.config;
  const ks = [...state.shards.keys()].sort((a, b) => a - b);
  const nb = c ? c.hist_edges.length - 1 : 0, ns = c ? c.spr_edges.length - 1 : 0;
  const a = { ks, rows: 0, bad: 0, nonpos: 0, hist: [new Array(nb).fill(0), new Array(nb).fill(0)],
              spr: Array.from({ length: ns }, () => ({ n: 0, sum: 0, max: 0, over: [0, 0, 0] })),
              worst: [], intervals: [], tFirst: Infinity, tLast: -Infinity };
  let prev = null;
  for (const k of ks) {
    const st = state.shards.get(k);
    a.rows += st.rows;
    a.tFirst = Math.min(a.tFirst, st.t); a.tLast = Math.max(a.tLast, st.t);
    st.src.forEach((d, s) => {
      a.bad += d.bad || 0; a.nonpos += d.nonpos || 0;
      (d.hist || []).forEach((x, i) => { a.hist[s][i] += x; });
      (d.spr || []).forEach((b, i) => {
        const t = a.spr[i]; t.n += b.n; t.sum += b.sum; t.max = Math.max(t.max, b.max || 0);
        b.over.forEach((x, o) => { t.over[o] += x; });
      });
    });
    for (const w of st.worst || []) if (num(w.e)) a.worst.push(w);
    if (prev && k === prev.k + 1) a.intervals.push({ k, dt: st.t - prev.t, full: st.rows === c.rows_per_shard });
    prev = st;
  }
  a.worst.sort((x, y) => y.e - x.e);
  a.worst = a.worst.slice(0, 15);
  const full = a.intervals.filter((x) => x.full).map((x) => x.dt);
  a.medAll = median(full);
  a.med = median(full.slice(-20));
  a.lastDt = full.length ? full[full.length - 1] : NaN;
  if (c && ks.length) {
    a.start = a.tFirst - (num(a.medAll) ? a.medAll : 0);
    a.left = c.rows_total - a.rows;
    a.finishAt = a.tLast + (num(a.med) ? a.left / c.rows_per_shard * a.med : NaN);
  }
  a.done = c && ks.length >= c.shards_total;
  return a;
}

// ---- header ---------------------------------------------------------------------
function renderHeader() {
  const c = state.config, a = state.agg, g = state.global;
  if (c) {
    const m = c.meta || {};
    $("sub").textContent = `${String(c.data).split(/[\\/]/).pop()} · ${m.iters} DCFR iterations · γ ${m.gamma} · ${m.dtype} · ${c.rows_total.toLocaleString()} solves`;
    $("scope-note").textContent = `Generator: ${m.iters} iterations, strategy weights γ ${m.gamma}, ${m.dtype} on the fused GPU solver, ` +
      `shards of ${m.shard} solves; code ${m.code}${m.resumed?.length ? ` (resumed ${m.resumed.length}×)` : ""}. ` +
      `This page is a read-only watcher: it reads finished shards and generate.log, never the shard being written, and cannot slow the generator.`;
  }
  if (c && a) {
    $("stat-rows").innerHTML = `${a.rows.toLocaleString()}`;
    $("stat-rows-sub").textContent = `of ${c.rows_total.toLocaleString()} · ${a.ks.length} / ${c.shards_total} shards`;
    const frac = c.rows_total ? a.rows / c.rows_total : 0;
    $("run-label").textContent = `${a.ks.length} of ${c.shards_total} shards · ${a.rows.toLocaleString()} of ${c.rows_total.toLocaleString()} solves`;
    $("run-pct").textContent = `${(100 * frac).toFixed(1)}%`;
    $("run-bar").style.width = `${(100 * frac).toFixed(2)}%`;
  }
  if (g?.all?.n) {
    $("stat-expl").innerHTML = `${pct(g.all.p50, 2)} / ${pct(g.all.p99, 2)}`;
    $("stat-expl-sub").textContent = `mean ${pct(g.all.mean, 2)} · max ${pct(g.all.max, 1)} of pot`;
  }
  const r = state.res, l = state.tdlog;
  const parts = [];
  parts.push(l ? `${l.gpu_mem_gib.toFixed(2)} GiB` : "—");
  if (r && num(r.gpu_pct)) parts[0] += ` ${r.gpu_pct.toFixed(0)}%`;
  parts.push(r && num(r.cpu_pct) ? `${r.cpu_pct.toFixed(0)}%` : "—");
  parts.push(r && num(r.ram_used_gb) ? `${r.ram_used_gb.toFixed(1)}/${r.ram_total_gb.toFixed(0)} GB` : "—");
  $("stat-res").textContent = parts.join(" · ");
  $("stat-res").title = "GPU: memory the generator's torch reserves (from generate.log) and utilization (nvidia-smi); CPU and RAM: the whole machine.";
  $("stat-pace").textContent = `${l ? l.solves_per_s.toFixed(2) + " solves/s" : "—"} · ${a && num(a.med) ? (a.med / 60).toFixed(1) + " min/shard" : "—"}`;
  updateClocks();
}
function updateClocks() {
  const a = state.agg, c = state.config;
  if (!a || !c || !a.ks.length) return;
  const now = Date.now() / 1000;
  const end = a.done ? a.tLast : now;
  let text = duration(end - a.start);
  if (a.done) text += " · complete";
  else if (num(a.finishAt)) {
    const left = a.finishAt - now;
    text += left > 0 ? ` · ${duration(left)} left` : " · due";
  }
  $("stat-eta").textContent = text;
  $("stat-eta").title = "Elapsed since the first shard began (wall clock, restarts included). ETA: the solves left at the median pace of the last 20 shards, counting down from the last finished shard.";
  const since = now - a.tLast;
  $("last-shard").textContent = a.done ? "all shards written" : `last shard ${a.ks[a.ks.length - 1]} · ${duration(since)} ago`;
  let dot = "live", label = "generating";
  if (a.done) { dot = "done"; label = "complete"; }
  else if (num(a.med) && since > 3 * a.med) { dot = "err"; label = `no shard for ${duration(since)}: stalled?`; }
  if (state.watchEnded && !a.done) { dot = "idle"; label = "watcher stopped"; }
  $("stat-status").innerHTML = `<span class="dot ${dot}"></span>${esc(label)}`;
}

// ---- tiles ----------------------------------------------------------------------
function renderTiles() {
  const c = state.config, a = state.agg, g = state.global;
  if (!c || !a) return;
  const all = g?.all || {}, s0 = g?.src?.[0] || {}, s1 = g?.src?.[1] || {};
  const share = (x) => all.n ? `${(100 * x / all.n).toFixed(2)}% of rows` : "";
  const w = a.worst[0];
  const tiles = [
    ["rows solved", a.rows.toLocaleString(), `${c.pairs.toLocaleString()} situations × 2 range sources`, ""],
    ["non-finite values", String(a.bad), "NaN / inf exploitability or game value", a.bad ? "bad" : "good"],
    ["exploitability ≤ 0", String(a.nonpos), "float32 rounding at or below zero", a.nonpos ? "warn" : "good"],
    ["mean · recorded / random", `${pct(s0.mean, 2)} / ${pct(s1.mean, 2)}`, "of pot", ""],
    [`over ${tickPct(c.over[0])} of pot`, num(all.over?.[0]) ? all.over[0].toLocaleString() : "—", share(all.over?.[0] || 0), ""],
    [`over ${tickPct(c.over[2])} of pot`, num(all.over?.[2]) ? all.over[2].toLocaleString() : "—", share(all.over?.[2] || 0), all.over?.[2] ? "warn" : ""],
    ["largest", w ? pct(w.e, 2) : "—", w ? `situation ${w.j} · ${SRC_NAME[w.s]} ranges` : "", ""],
  ];
  $("tiles").innerHTML = tiles.map(([k, v, s, cls]) =>
    `<div class="tile ${cls}"><span class="tl">${esc(k)}</span><strong>${esc(v)}</strong><small>${esc(s)}</small></div>`).join("");
}

// ---- drawing helpers ----------------------------------------------------------------
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
function diamond(ctx, x, y, r, fill, stroke) {
  ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke(); }
}
function hit(id, x, y, html, mode = "xy") { state.hits[id].push({ x, y, html, mode }); }

/** Color for a shard's largest exploitability: dim blue (low) to amber to red (≥ 10% of pot). */
function heat(v) {
  if (!num(v)) return css("--border");
  const t = Math.max(0, Math.min(1, (L10(v) - L10(0.003)) / (L10(0.1) - L10(0.003))));
  const stops = [[36, 50, 74], [138, 180, 248], [249, 171, 0], [242, 139, 130]];
  const f = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(f)), u = f - i;
  const c = stops[i].map((x, k) => Math.round(x + (stops[i + 1][k] - x) * u));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
const shardMax = (st) => Math.max(...st.src.map((d) => (num(d.max) ? d.max : 0)));
function shardTip(st) {
  const rows = st.src.map((d, s) => `<div class="tr"><span>${SRC_NAME[s]}</span><span>mean ${pct(d.mean)} · p99 ${pct(d.p99)} · max ${pct(d.max)}</span></div>`).join("");
  return `<div class="th">shard ${st.k} · situations ${st.first}–${st.last}</div>${rows}<div class="tr"><span>finished</span><span>${new Date(st.t * 1000).toLocaleString()}</span></div>`;
}

// ---- shard strip --------------------------------------------------------------------
function drawStrip() {
  const c = state.config, a = state.agg, id = "strip";
  const { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  if (!c || !a) return;
  const n = c.shards_total, cw = w / n;
  for (let k = 0; k < n; k++) {
    const st = state.shards.get(k), x = k * cw;
    if (st) {
      ctx.fillStyle = heat(shardMax(st)); ctx.fillRect(x, 4, Math.max(1, cw - (cw > 3 ? 0.6 : 0)), h - 8);
      hit(id, x + cw / 2, h / 2, shardTip(st), "x");
    } else {
      ctx.fillStyle = css("--bg-2"); ctx.fillRect(x, 4, Math.max(1, cw - 0.6), h - 8);
    }
  }
  if (!a.done && a.ks.length < n) {  // the shard in flight
    const k = a.ks.length ? a.ks[a.ks.length - 1] + 1 : 0;
    ctx.strokeStyle = css("--accent"); ctx.globalAlpha = Math.floor(Date.now() / 1000) % 2 ? 1 : 0.45; ctx.lineWidth = 1.5;
    ctx.strokeRect(k * cw + 0.5, 2.5, Math.max(2, cw - 1), h - 5); ctx.globalAlpha = 1;
  }
}

// ---- exploitability per shard ------------------------------------------------------------
function drawShards() {
  const c = state.config, a = state.agg, id = "chart-shards";
  const { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  if (!c || !a || !a.ks.length) return;
  const sts = a.ks.map((k) => state.shards.get(k));
  const vals = sts.flatMap((st) => st.src.flatMap((d) => [d.mean, d.p99, d.max]));
  const [lo, hi] = logDomain(vals, 1.3, 1.3);
  const L = 56, R = 12, T = 10, B = 26;
  const X = (k) => L + (k + 0.5) / c.shards_total * (w - L - R);
  const Y = (v) => T + (1 - (L10(Math.max(v, lo)) - L10(lo)) / (L10(hi) - L10(lo))) * (h - T - B);
  axisY(ctx, logTicks(lo, hi), Y, L, R, w, tickPct);
  ctx.fillStyle = css("--ink-3"); ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const k of niceTicks(0, c.shards_total, 8)) ctx.fillText(String(k), X(k), h - B + 7);
  const line = (key, s, width, alpha) => {
    ctx.strokeStyle = css(SRC[s]); ctx.lineWidth = width; ctx.globalAlpha = alpha; ctx.beginPath();
    let started = false, prevK = null;
    for (const st of sts) {
      const v = st.src[s][key];
      if (!num(v) || v <= 0) { started = false; continue; }
      if (!started || st.k !== prevK + 1) ctx.moveTo(X(st.k), Y(v)); else ctx.lineTo(X(st.k), Y(v));
      started = true; prevK = st.k;
    }
    ctx.stroke(); ctx.globalAlpha = 1;
  };
  line("p99", 0, 1, 0.5); line("p99", 1, 1, 0.5); line("mean", 0, 2, 1); line("mean", 1, 2, 1);
  ctx.fillStyle = css("--warm");
  for (const st of sts) {
    const m = shardMax(st);
    if (m > 0) { ctx.beginPath(); ctx.arc(X(st.k), Y(m), 1.9, 0, 2 * Math.PI); ctx.fill(); }
    hit(id, X(st.k), Y(st.src[0].mean || lo), shardTip(st), "x");
  }
}

// ---- histogram ----------------------------------------------------------------------------
function drawHist() {
  const c = state.config, a = state.agg, id = "chart-hist";
  const { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  if (!c || !a || !a.rows) return;
  const E = c.hist_edges, tot = a.hist[0].map((x, i) => x + a.hist[1][i]);
  let i0 = tot.findIndex((x) => x > 0), i1 = tot.length - 1 - [...tot].reverse().findIndex((x) => x > 0);
  if (i0 < 0) return;
  i0 = Math.max(0, i0 - 1); i1 = Math.min(tot.length - 1, i1 + 1);
  const lo = E[i0], hi = E[i1 + 1], ymax = Math.max(...tot) * 1.08;
  const L = 48, R = 10, T = 18, B = 26;
  const X = (lg) => L + (lg - lo) / (hi - lo) * (w - L - R);
  const Y = (v) => T + (1 - v / ymax) * (h - T - B);
  axisY(ctx, niceTicks(0, ymax, 4), Y, L, R, w, (v) => v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(v));
  ctx.fillStyle = css("--ink-3"); ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.font = `11px ${css("--mono")}`;
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) ctx.fillText(tickPct(Math.pow(10, e)), X(e), h - B + 7);
  for (let i = i0; i <= i1; i++) {
    const x0 = X(E[i]) + 0.5, x1 = X(E[i + 1]) - 0.5;
    let base = 0;
    a.hist.forEach((hs, s) => {
      if (!hs[i]) return;
      ctx.fillStyle = css(SRC[s]); ctx.globalAlpha = 0.85;
      ctx.fillRect(x0, Y(base + hs[i]), Math.max(1, x1 - x0), Y(base) - Y(base + hs[i]));
      base += hs[i];
    });
    ctx.globalAlpha = 1;
    // The watcher clips into the end bins: the first holds everything below its upper edge (zero and
    // negative included), the last everything at or above its lower edge.
    const span = i === 0 ? `&lt; ${tickPct(Math.pow(10, E[1]))}` : i === E.length - 2 ? `≥ ${tickPct(Math.pow(10, E[i]))}`
      : `${tickPct(Math.pow(10, E[i]))} – ${tickPct(Math.pow(10, E[i + 1]))}`;
    if (tot[i]) hit(id, (x0 + x1) / 2, Y(tot[i]), `<div class="th">${span} of pot</div>` +
      `<div class="tr"><span>recorded</span><span>${a.hist[0][i].toLocaleString()}</span></div><div class="tr"><span>random</span><span>${a.hist[1][i].toLocaleString()}</span></div>`, "x");
  }
  const g = state.global?.all;
  c.over.forEach((o) => {
    const x = X(L10(o));
    if (x < L || x > w - R) return;
    ctx.strokeStyle = css("--hot"); ctx.globalAlpha = 0.75; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, h - B); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    ctx.fillStyle = css("--hot"); ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.font = `10px ${css("--mono")}`;
    ctx.fillText(tickPct(o), x + 3, T - 14);
  });
  $("hist-kpi").textContent = `${a.rows.toLocaleString()} rows`;
  if (g?.n) {
    $("hist-note").textContent = c.over.map((o, k) => `> ${tickPct(o)}: ${g.over[k].toLocaleString()} (${(100 * g.over[k] / g.n).toFixed(2)}%)`).join(" · ") +
      `. Exact quantiles: p90 ${pct(g.p90)}, p99 ${pct(g.p99)}, p99.9 ${pct(g.p999)}.`;
  }
}

// ---- stack-to-pot bins --------------------------------------------------------------------
function drawSpr() {
  const c = state.config, a = state.agg, id = "chart-spr";
  const { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  if (!c || !a || !a.rows) return;
  const edges = c.spr_edges, bins = a.spr.map((b, i) => ({ ...b, i })).filter((b) => b.n > 0);
  if (!bins.length) return;
  const [lo, hi] = logDomain(bins.flatMap((b) => [b.sum / b.n, b.max]), 1.5, 1.3);
  const narrow = w < 520, L = 58, R = narrow ? 8 : 190, T = 6, B = 24;
  const X = (v) => L + (L10(Math.max(v, lo)) - L10(lo)) / (L10(hi) - L10(lo)) * (w - L - R);
  const rowH = (h - T - B) / bins.length;
  ctx.font = `11px ${css("--mono")}`;
  for (const v of logTicks(lo, hi)) {
    ctx.strokeStyle = css("--border"); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(v), T); ctx.lineTo(X(v), h - B); ctx.stroke();
    ctx.fillStyle = css("--ink-3"); ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(tickPct(v), X(v), h - B + 6);
  }
  bins.forEach((b, r) => {
    const y = T + rowH * r + rowH * 0.18, bh = rowH * 0.64, mean = b.sum / b.n;
    const lab = edges[b.i + 1] === null ? `≥${edges[b.i]}` : `${edges[b.i]}–${edges[b.i + 1]}`;
    ctx.fillStyle = css("--ink-2"); ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(lab, L - 8, y + bh / 2);
    ctx.fillStyle = heat(b.max); ctx.globalAlpha = 0.35; ctx.fillRect(L, y, X(mean) - L, bh); ctx.globalAlpha = 1;
    ctx.fillStyle = css("--accent"); ctx.fillRect(X(mean) - 1, y, 2, bh);
    ctx.strokeStyle = heat(b.max); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(b.max), y - 1); ctx.lineTo(X(b.max), y + bh + 1); ctx.stroke();
    const s1 = 100 * b.over[0] / b.n, s2 = 100 * b.over[1] / b.n;
    if (!narrow) {
      ctx.fillStyle = s2 >= 1 ? css("--hot") : s1 >= 1 ? css("--warm") : css("--ink-3"); ctx.textAlign = "left";
      ctx.fillText(`${s1.toFixed(1)}% · ${s2.toFixed(1)}%  n ${b.n.toLocaleString()}`, w - R + 10, y + bh / 2);
    }
    hit(id, X(mean), y + bh / 2, `<div class="th">stack-to-pot ${lab}</div><div class="tr"><span>rows</span><span>${b.n.toLocaleString()}</span></div>` +
      `<div class="tr"><span>mean</span><span>${pct(mean)}</span></div><div class="tr"><span>largest</span><span>${pct(b.max)}</span></div>` +
      c.over.map((o, k) => `<div class="tr"><span>over ${tickPct(o)}</span><span>${b.over[k].toLocaleString()} (${(100 * b.over[k] / b.n).toFixed(2)}%)</span></div>`).join(""), "y");
  });
}

// ---- reference checks -------------------------------------------------------------------
function checkRows() {
  const refs = [], probes = [], harness = [];
  const labels = Object.keys(state.checks).sort();
  for (const label of labels) {
    for (const o of state.checks[label] || []) {
      if (o.check === "harness") harness.push({ label, ...o });
      else if (o.dtype && num(o.e_walk64) && num(o.e_stored)) probes.push({ label, ...o });
      else if (num(o.e_stored) && num(o.e_ref) && num(o.iters)) refs.push({ label, ...o });
    }
  }
  return { refs, probes, harness, labels };
}
const labelColor = (label, labels) => label === "random" ? css("--ok") : label === "tail" ? css("--hot") : css(PALETTE[(labels.indexOf(label) + 2) % PALETTE.length]);
function drawPrecision() {
  const id = "chart-prec", { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  const { refs, probes, harness, labels } = checkRows();
  const at500 = refs.filter((r) => r.iters === 500);
  const pts = [...at500.map((r) => ({ x: r.e_stored, y: r.e_ref, r, kind: "ref" })),
               ...probes.map((p) => ({ x: p.e_stored, y: p.e_walk64, r: p, kind: p.dtype }))];
  const verdicts = [];
  if (!pts.length) {
    ctx.fillStyle = css("--ink-3"); ctx.font = `12px ${css("--sans")}`; ctx.textAlign = "center";
    ctx.fillText(Object.keys(state.checks).length ? "no reference results yet" : "no reference checks attached (watcher --check)", w / 2, h / 2);
  } else {
    const [lo, hi] = logDomain(pts.flatMap((p) => [p.x, p.y]), 2, 2);
    const L = 56, R = 12, T = 10, B = 30;
    const side = Math.min(w - L - R, (h - T - B) * 1.9);
    const X = (v) => L + (L10(v) - L10(lo)) / (L10(hi) - L10(lo)) * side;
    const Y = (v) => T + (1 - (L10(v) - L10(lo)) / (L10(hi) - L10(lo))) * (h - T - B);
    const ticks = logTicks(lo, hi);
    axisY(ctx, ticks, Y, L, w - L - side, w, tickPct);
    ctx.fillStyle = css("--ink-3"); ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const v of ticks) ctx.fillText(tickPct(v), X(v), h - B + 6);
    ctx.fillText("stored (float32 GPU) →", L + side / 2, h - 13);
    ctx.save(); ctx.translate(12, T + (h - T - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("float64 reference →", 0, -6); ctx.restore();
    // agreement band (reference within 2x of stored) and the diagonal
    ctx.fillStyle = css("--accent"); ctx.globalAlpha = 0.07; ctx.beginPath();
    ctx.moveTo(X(lo), Y(lo * 2)); ctx.lineTo(X(hi / 2), Y(hi)); ctx.lineTo(X(hi), Y(hi)); ctx.lineTo(X(hi), Y(hi / 2)); ctx.lineTo(X(lo * 2), Y(lo)); ctx.lineTo(X(lo), Y(lo));
    ctx.closePath(); ctx.fill(); ctx.globalAlpha = 1;
    ctx.strokeStyle = css("--ink-3"); ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(lo), Y(lo)); ctx.lineTo(X(hi), Y(hi)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = css("--ink-3"); ctx.font = `10px ${css("--sans")}`; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
    ctx.fillText("below: stored worse than the reference", X(hi) - 4, h - B - 4);
    for (const p of pts) {
      const x = X(p.x), y = Y(p.y), r = p.r;
      const tag = `${r.label} · situation ${r.j} · ${SRC_NAME[r.source] ?? r.source}`;
      if (p.kind === "ref") {
        ctx.fillStyle = labelColor(r.label, labels); ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 2 * Math.PI); ctx.fill();
        hit(id, x, y, `<div class="th">${esc(tag)}</div><div class="tr"><span>stored (float32 GPU, 500 it)</span><span>${pct(r.e_stored)}</span></div>` +
          `<div class="tr"><span>float64 reference, 500 it</span><span>${pct(r.e_ref)}</span></div><div class="tr"><span>reference / stored</span><span>${(r.e_ref / r.e_stored).toFixed(3)}</span></div>`);
      } else {
        diamond(ctx, x, y, 5.5, null, p.kind === "float32" ? css("--warm") : css("--s0"));
        hit(id, x, y, `<div class="th">${esc(tag)} · probe</div><div class="tr"><span>generic ${esc(p.kind)} DCFR (CPU), 500 it</span><span>${pct(p.y)}</span></div>` +
          `<div class="tr"><span>its own ${esc(p.kind)} measure</span><span>${pct(r.e_own)}</span></div><div class="tr"><span>stored</span><span>${pct(r.e_stored)}</span></div>`);
      }
    }
    for (const r of at500) {
      const ratio = r.e_ref / r.e_stored;
      const gameOk = num(r.g_stored) && num(r.g_ref) ? Math.abs(r.g_stored - r.g_ref) <= r.e_stored + r.e_ref : null;
      const cls = ratio < 0.5 ? "fail" : ratio > 2 ? "info" : "pass";
      const word = ratio < 0.5 ? `stored ${(1 / ratio).toFixed(1)}× worse` : ratio > 2 ? `stored ${ratio.toFixed(1)}× better` : "agrees";
      verdicts.push(`<div class="verdict"><span class="vk ${cls}">${esc(word)}</span><span class="vm">${esc(r.label)} · ${esc(r.j)}/${esc(SRC_NAME[r.source] ?? r.source)}: stored ${pct(r.e_stored)} vs float64 ${pct(r.e_ref)} (×${ratio.toFixed(2)})${gameOk === null ? "" : gameOk ? " · game value consistent" : " · GAME VALUE INCONSISTENT"}</span></div>`);
    }
    // PREREG3's reading, where a row has both probes and its float64 reference at 500
    for (const p32 of probes.filter((p) => p.dtype === "float32")) {
      const ref = at500.find((r) => r.j === p32.j && r.source === p32.source);
      if (!ref) continue;
      const e32 = p32.e_walk64, st = p32.e_stored, er = ref.e_ref;
      const reading = e32 >= st / 2 ? ["fail", "generic float32"] : e32 <= 2 * er ? ["pass", "kernel-specific"] : ["info", "partly generic"];
      verdicts.push(`<div class="verdict"><span class="vk ${reading[0]}">${reading[1]}</span><span class="vm">pre-registered reading · ${esc(p32.j)}/${esc(SRC_NAME[p32.source] ?? p32.source)}: generic float32 DCFR ${pct(e32)} vs stored ${pct(st)} and float64 ${pct(er)}</span></div>`);
    }
  }
  for (const hc of harness) {
    verdicts.push(`<div class="verdict"><span class="vk ${hc.ok ? "pass" : "fail"}">harness ${hc.ok ? "ok" : "FAILED"}</span><span class="vm">${esc(hc.j)}/${esc(SRC_NAME[hc.source] ?? hc.source)}: probe solver = reference to ${num(hc.max_abs_dev) ? Number(hc.max_abs_dev).toExponential(1) : "—"} after ${esc(hc.iters)} iterations</span></div>`);
  }
  $("verdicts").innerHTML = verdicts.join("") || "";
}
function drawConvergence() {
  const id = "chart-conv", { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  const { refs, probes } = checkRows();
  const groups = new Map();
  for (const r of refs) {
    const key = `${r.label}|${r.j}|${r.source}`;
    if (!groups.has(key)) groups.set(key, { label: r.label, j: r.j, source: r.source, stored: r.e_stored, pts: [] });
    groups.get(key).pts.push([r.iters, r.e_ref]);
  }
  const series = [...groups.values()].filter((g) => g.pts.length >= 2 || probes.some((p) => p.j === g.j && p.source === g.source));
  if (!series.length) {
    ctx.fillStyle = css("--ink-3"); ctx.font = `12px ${css("--sans")}`; ctx.textAlign = "center";
    ctx.fillText("no multi-iteration reference runs yet", w / 2, h / 2);
    return;
  }
  const its = series.flatMap((g) => g.pts.map((p) => p[0]));
  const xlo = Math.min(...its) / 1.4, xhi = Math.max(...its) * 1.4;
  const ys = series.flatMap((g) => [g.stored, ...g.pts.map((p) => p[1])]).concat(probes.map((p) => p.e_walk64));
  const [lo, hi] = logDomain(ys, 1.6, 1.4);
  const L = 56, R = 96, T = 10, B = 26;
  const X = (v) => L + (L10(v) - L10(xlo)) / (L10(xhi) - L10(xlo)) * (w - L - R);
  const Y = (v) => T + (1 - (L10(v) - L10(lo)) / (L10(hi) - L10(lo))) * (h - T - B);
  axisY(ctx, logTicks(lo, hi), Y, L, R, w, tickPct);
  ctx.fillStyle = css("--ink-3"); ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.font = `11px ${css("--mono")}`;
  for (const v of [...new Set(its)].sort((a, b) => a - b)) ctx.fillText(String(v), X(v), h - B + 7);
  series.forEach((g, i) => {
    const col = css(PALETTE[i % PALETTE.length]);
    g.pts.sort((a, b) => a[0] - b[0]);
    ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]); ctx.globalAlpha = 0.85;
    ctx.beginPath(); ctx.moveTo(X(xlo * 1.15), Y(g.stored)); ctx.lineTo(X(xhi / 1.15), Y(g.stored)); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    ctx.lineWidth = 2; ctx.beginPath();
    g.pts.forEach(([it, e], k) => { if (k) ctx.lineTo(X(it), Y(e)); else ctx.moveTo(X(it), Y(e)); });
    ctx.stroke();
    ctx.fillStyle = col;
    for (const [it, e] of g.pts) {
      ctx.beginPath(); ctx.arc(X(it), Y(e), 3.5, 0, 2 * Math.PI); ctx.fill();
      hit(id, X(it), Y(e), `<div class="th">${esc(g.j)}/${esc(SRC_NAME[g.source] ?? g.source)} · float64, ${esc(it)} iterations</div><div class="tr"><span>exploitability</span><span>${pct(e)}</span></div><div class="tr"><span>stored at 500</span><span>${pct(g.stored)}</span></div><div class="tr"><span>stored / reference</span><span>×${(g.stored / e).toFixed(1)}</span></div>`);
    }
    for (const p of probes.filter((q) => q.j === g.j && q.source === g.source)) {
      const x = X(500 * (p.dtype === "float32" ? 0.93 : 1.07)), y = Y(p.e_walk64);
      diamond(ctx, x, y, 5, null, p.dtype === "float32" ? css("--warm") : css("--s0"));
      hit(id, x, y, `<div class="th">${esc(g.j)}/${esc(SRC_NAME[g.source] ?? g.source)} · generic ${esc(p.dtype)} probe</div><div class="tr"><span>500 iterations</span><span>${pct(p.e_walk64)}</span></div>`);
    }
    ctx.fillStyle = col; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.font = `11px ${css("--mono")}`;
    ctx.fillText(`${g.j}/${String(SRC_NAME[g.source] ?? g.source).slice(0, 4)}`, w - R + 8, Y(g.stored));
  });
}

// ---- pace -----------------------------------------------------------------------------------
function drawPace() {
  const c = state.config, a = state.agg, id = "chart-pace";
  const { ctx, w, h } = setup($(id));
  state.hits[id] = [];
  if (!c || !a || !a.intervals.length) return;
  const iv = a.intervals.filter((x) => x.full).slice(-160);
  if (!iv.length) return;
  const med = a.medAll, cap = 3 * med;
  const ymax = Math.max(...iv.map((x) => Math.min(x.dt, cap))) / 60 * 1.12;
  const L = 44, R = 10, T = 10, B = 26;
  const k0 = iv[0].k, k1 = iv[iv.length - 1].k;
  const bw = (w - L - R) / Math.max(1, k1 - k0 + 1);
  const X = (k) => L + (k - k0) * bw;
  const Y = (m) => T + (1 - m / ymax) * (h - T - B);
  axisY(ctx, niceTicks(0, ymax, 4), Y, L, R, w, (v) => `${v}m`);
  ctx.fillStyle = css("--ink-3"); ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.font = `11px ${css("--mono")}`;
  for (const k of niceTicks(k0, k1, 6)) ctx.fillText(String(k), X(k) + bw / 2, h - B + 7);
  for (const x of iv) {
    const m = Math.min(x.dt, cap) / 60, clipped = x.dt > cap;
    ctx.fillStyle = clipped ? css("--warm") : css("--accent"); ctx.globalAlpha = clipped ? 0.9 : 0.6;
    ctx.fillRect(X(x.k) + 0.5, Y(m), Math.max(1, bw - 1), Y(0) - Y(m)); ctx.globalAlpha = 1;
    hit(id, X(x.k) + bw / 2, Y(m), `<div class="th">shard ${x.k}</div><div class="tr"><span>since shard ${x.k - 1}</span><span>${duration(x.dt)}</span></div>`, "x");
  }
  ctx.strokeStyle = css("--ink-2"); ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(L, Y(med / 60)); ctx.lineTo(w - R, Y(med / 60)); ctx.stroke(); ctx.setLineDash([]);
  $("pace-kpi").textContent = `median ${(a.medAll / 60).toFixed(1)} · last 20 ${(a.med / 60).toFixed(1)} · last ${(a.lastDt / 60).toFixed(1)} min`;
}

// ---- worst rows -------------------------------------------------------------------------------
function renderWorst() {
  const c = state.config, a = state.agg;
  if (!c || !a) return;
  const bb = c.bb || 1, unit = c.bb ? "bb" : "chips";
  const { refs } = checkRows();
  $("worst").querySelector("thead").innerHTML = `<tr><th>#</th><th>situation</th><th>ranges</th><th>pot (${unit})</th><th>stack (${unit})</th><th>stack/pot</th><th>exploitability</th><th>game value</th><th>float64 check, 500 it</th></tr>`;
  $("worst").querySelector("tbody").innerHTML = a.worst.map((r, i) => {
    const ref = refs.find((x) => x.j === r.j && x.source === r.s && x.iters === 500);
    const cls = r.e >= 0.05 ? "hot" : r.e >= 0.02 ? "warm" : "";
    const chk = ref ? `<td class="${ref.e_ref / ref.e_stored < 0.5 ? "hot" : "ok"}">${pct(ref.e_ref)} (×${(ref.e_ref / ref.e_stored).toFixed(2)})</td>` : `<td class="dim">—</td>`;
    return `<tr><td>${i + 1}</td><td>${r.j}</td><td style="text-align:left">${SRC_NAME[r.s]}</td><td>${(2 * r.half / bb).toLocaleString()}</td><td>${(r.stack / bb).toLocaleString()}</td>` +
      `<td>${num(r.spr) ? r.spr.toFixed(1) : "—"}</td><td class="${cls}">${pct(r.e, 2)}</td><td class="dim">${num(r.g) ? r.g.toFixed(4) : "—"}</td>${chk}</tr>`;
  }).join("");
  $("worst-kpi").textContent = a.worst.length ? `of ${a.rows.toLocaleString()} rows` : "—";
}

function render() {
  state.agg = aggregate();
  renderHeader(); renderTiles(); drawStrip(); drawShards(); drawHist(); drawSpr(); drawPrecision(); drawConvergence(); drawPace(); renderWorst();
}

// ---- events ---------------------------------------------------------------------------------------
let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } }
let announced = false;
function applyEvent(e) {
  const ts = Number(e.ts) * 1000 || Date.now();
  if (e.type === "run_start") {
    if (e.config?.study !== "turn-data") {
      $("sub").textContent = `this run is not a turn-data watch (study: ${e.config?.study ?? e.variant ?? "unknown"})`;
      events.close();
      return;
    }
    state.config = e.config; state.watchStarted = ts;
    log(`watching ${e.config.data}`, ts);
  } else if (e.type === "metrics") {
    if (e.resources) state.res = e.resources;
    if (e.td_log) state.tdlog = e.td_log;
  } else if (e.type === "checkpoint") {
    if (e.td_config && state.config) Object.assign(state.config, e.td_config);  // generate.json rewritten (a resume)
    if (e.td_shard) state.shards.set(e.td_shard.k, e.td_shard);  // a shard read again replaces its entry
    if (e.td_global) state.global = e.td_global;
    if (e.td_checks) state.checks = e.td_checks;
  } else if (e.type === "milestone") {
    log(e.message || e.kind, ts, e.kind === "warning" ? "warning" : "note");
  } else if (e.type === "run_end") {
    state.watchEnded = ts; state.endReason = e.reason;
    if (e.reason === "completed" && !announced) {
      announced = true;
      const g = state.global?.all;
      $("done-detail").textContent = `${(e.summary?.rows ?? 0).toLocaleString()} solves in ${e.summary?.shards} shards.` +
        (g?.n ? ` Exploitability: median ${pct(g.p50, 2)}, p99 ${pct(g.p99, 2)}, max ${pct(g.max, 1)} of pot; non-finite values: ${state.agg ? state.agg.bad : "?"}.` : "");
      $("done-overlay").hidden = false;
    }
    log(e.reason === "completed" ? "generation complete" : `watcher stopped (${e.reason})`, ts, e.reason === "completed" ? "note" : "warning");
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
    const d = h.mode === "x" ? Math.abs(h.x - mx) : h.mode === "y" ? Math.abs(h.y - my) : Math.hypot(h.x - mx, h.y - my);
    if (d < bd) { bd = d; best = h; }
  }
  if (!best || bd > (best.mode === "xy" ? 14 : 24)) { tip.hidden = true; return; }
  tip.innerHTML = best.html; tip.hidden = false;
  tip.style.left = `${r.left + best.x}px`; tip.style.top = `${r.top + (best.mode === "x" ? my : best.y) - 8}px`;
});

const events = new EventSource("/events");
events.onmessage = (m) => JSON.parse(m.data).forEach(applyEvent);
let failures = 0;
events.onopen = () => { failures = 0; };
events.onerror = () => {
  failures += 1;
  if (failures >= 3) {
    events.close();
    log("dashboard server stopped (the watcher exited); the generator is unaffected", Date.now(), "warning");
  } else if (failures === 1) {
    log("telemetry reconnecting…", Date.now(), "warning");
  }
};
window.addEventListener("resize", schedule);
$("done-dismiss").onclick = () => { $("done-overlay").hidden = true; };
setInterval(() => { updateClocks(); if (state.agg && !state.agg.done) drawStrip(); }, 1000);
render();
