// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
/* Bot scorecard viewer. Data-only: every number comes from the producer
   (scripts/scorecard.py evaluate); nothing is recomputed here except the
   progress fraction and a rough ETA. Final row numbers are the receipt's. */
"use strict";
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const Z95 = 1.96;
const SRC = [["blueprint", "--s-blueprint"], ["missing key", "--s-missing"], ["off-tree", "--s-off"], ["search", "--s-search"]];

const state = { config: null, prog: {}, finals: {}, current: null, currentAt: 0, resources: null,
                unitsDone: 0, unitsTotal: 0, started: null, finished: null, reason: null, summary: null, hits: {} };

const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
const num = (v) => Number.isFinite(Number(v));
const fmt = (v, d = 1) => num(v) ? Number(v).toFixed(d) : "—";
const sgn = (v, d = 1) => num(v) ? (v >= 0 ? "+" : "−") + Math.abs(Number(v)).toFixed(d) : "—";
const ci = (o, d = 1) => o && num(o.lo) ? `[${sgn(o.lo, d)}, ${sgn(o.hi, d)}]` : "";
function duration(s) {
  if (!num(s)) return "—";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), q = s % 60, p = (x) => String(x).padStart(2, "0");
  return h ? `${h}h ${p(m)}m` : m ? `${m}m ${p(q)}s` : `${q}s`;
}
function log(text, ts, kind = "note") {
  const li = document.createElement("li");
  li.innerHTML = `<span class="t">${new Date(ts || Date.now()).toLocaleTimeString()}</span><span class="kind ${esc(kind)}">${esc(kind)}</span><span class="msg">${esc(text)}</span>`;
  $("event-log").prepend(li);
}

// ---- row model ------------------------------------------------------------
function rowNames() { return state.config ? [...state.config.rows.map((r) => r.name), "LBR"] : []; }
function rowDef(name) { return state.config?.rows.find((r) => r.name === name) || null; }
function rowDeals(name) { return name === "LBR" ? state.config?.lbr?.deals : rowDef(name)?.deals; }
function opponentText(name) {
  if (name === "LBR") return "local best response (exploitability ≥)";
  const r = rowDef(name);
  if (!r) return "";
  if (r.kind === "agent") return "frozen baseline bot · headline";
  if (r.kind === "leak") return `baseline with a ${r.family} leak (${r.strength})`;
  if (r.kind === "perturbed") return `baseline with bet sizes ×${r.perturb}`;
  return `built-in ${r.builtin}`;
}
/** Lower is better only for LBR. */
function statusClass(name, status) {
  if (!status || status.startsWith("UNCLEAR")) return "flat";
  const up = status.startsWith("UP") || status.startsWith("MORE");
  const good = name === "LBR" ? !up : up;
  return good ? "up" : "down";
}
function statusText(name, status) {
  if (!status) return "";
  if (status.startsWith("UNCLEAR")) return "unclear";
  if (name === "LBR") return status.startsWith("MORE") ? "more exploitable" : "less exploitable";
  return status.startsWith("UP") ? "better" : "worse";
}
function valueStatus(o) { return !o ? null : o.lo > 0 ? "UP" : o.hi < 0 ? "DOWN" : "UNCLEAR"; }
function prevOf(name) {
  const p = state.config?.previous;
  if (!p) return null;
  return name === "LBR" ? p.lbr : p.rows?.[name] || null;
}
function rowStatus(name) {
  if (state.finals[name]) return "done";
  const pr = state.prog[name];
  if (pr && pr.done < pr.total) return "running";
  if (state.current?.row === name && !state.finished) return "running";
  return "queued";
}

// ---- header -----------------------------------------------------------------
function renderHeader() {
  const c = state.config;
  if (c) {
    const cand = c.candidate || {};
    const search = cand.search ? `search ${cand.search.streets} ×${cand.search.river_iters}` : "blueprint only";
    $("sub").textContent = `${c.benchmark_id} · ${cand.label ?? "candidate"} (${search}) vs frozen ${c.baseline?.file ?? "baseline"}`;
    const hands = (c.rows.reduce((a, r) => a + r.deals, 0) + c.lbr.deals) * 2;
    $("scope-note").textContent = `Scope: the play agent at ${c.stakes.sb}/${c.stakes.bb}, ${c.stakes.stack / c.stakes.bb}bb heads-up; ` +
      `${hands.toLocaleString()} duplicate hands on fixed seeds, each row in ${c.chunks} chunks. A development diagnostic on this benchmark, not a promotion decision.`;
  }
  const head = c?.headline;
  const f = head && state.finals[head];
  if (f) {
    $("stat-strength").innerHTML = `${sgn(f.mean)} <span class="unit">bb/100</span>`;
    const st = valueStatus(f), cls = st === "UP" ? "st-up" : st === "DOWN" ? "st-down" : "st-flat";
    $("stat-strength-sub").innerHTML = `${esc(ci(f))} · <span class="${cls}">${st === "UP" ? "stronger than baseline" : st === "DOWN" ? "weaker than baseline" : "no resolved difference"}</span>`;
  } else if (head && state.prog[head]?.partial) {
    const p = state.prog[head];
    $("stat-strength").innerHTML = `${sgn(p.partial.mean)} <span class="unit">bb/100</span>`;
    $("stat-strength-sub").textContent = `live estimate ± ${fmt(Z95 * p.partial.se)} · ${p.done.toLocaleString()}/${p.total.toLocaleString()} deals`;
  }
  const l = state.finals.LBR, pl = prevOf("LBR");
  if (l) {
    $("stat-lbr").innerHTML = `${sgn(l.mean)} <span class="unit">bb/100</span>`;
    let sub = esc(ci(l));
    if (l.change) sub += ` · <span class="st-${statusClass("LBR", l.change.status) === "up" ? "up" : statusClass("LBR", l.change.status) === "down" ? "down" : "flat"}">${sgn(l.change.mean)} vs ${esc(state.config.previous.label)}</span>`;
    // Postflop only: LBR just calls preflop, so preflop-ended hands are not exploitation.
    if (l.postflop) {
      sub += `<br>postflop ${sgn(l.postflop.mean)} ${esc(ci(l.postflop))}`;
      const pc = l.postflop.change;
      if (pc) sub += ` · <span class="st-${statusClass("LBR", pc.status) === "up" ? "up" : statusClass("LBR", pc.status) === "down" ? "down" : "flat"}">${sgn(pc.mean)} vs ${esc(state.config.previous.label)}</span>`;
    }
    $("stat-lbr-sub").innerHTML = sub;
  } else if (state.prog.LBR?.partial) {
    const p = state.prog.LBR;
    $("stat-lbr").innerHTML = `${sgn(p.partial.mean)} <span class="unit">bb/100</span>`;
    $("stat-lbr-sub").textContent = `live estimate ± ${fmt(Z95 * p.partial.se)} · ${p.done}/${p.total} deals`;
  } else {
    $("stat-lbr").textContent = "—";
    $("stat-lbr-sub").textContent = pl ? `${state.config.previous.label}: ${sgn(pl.mean)} ${ci(pl)} · measured last` : "measured last";
  }
  const total = state.unitsTotal || (c ? c.rows.reduce((a, r) => a + r.deals, 0) + c.lbr.deals : 0);
  const done = state.finished && state.reason === "completed" ? total : state.unitsDone;
  const frac = total ? done / total : 0;
  $("stat-progress").textContent = total ? `${(100 * frac).toFixed(1)}% · ${done.toLocaleString()} / ${total.toLocaleString()} deals` : "—";
  $("run-label").textContent = total ? `${done.toLocaleString()} of ${total.toLocaleString()} deals` : "—";
  $("run-pct").textContent = `${(100 * frac).toFixed(1)}%`;
  $("run-bar").style.width = `${(100 * frac).toFixed(2)}%`;
  const res = state.resources;
  $("stat-res").textContent = res ? `${fmt(res.cpu_pct, 0)}% · ${fmt(res.ram_used_gb, 1)}/${fmt(res.ram_total_gb, 0)} GB` : "—";
  updateClocks();
  const dot = state.finished ? (state.reason === "completed" ? "done" : "err") : state.started ? "live" : "idle";
  const label = state.finished ? (state.reason === "completed" ? "complete" : state.reason) : state.started ? "running" : "connecting";
  $("stat-status").innerHTML = `<span class="dot ${dot}"></span>${esc(label)}`;
}
function updateClocks() {
  if (!state.started) return;
  const end = state.finished || Date.now(), el = (end - state.started) / 1000;
  let eta = "";
  if (!state.finished && state.unitsDone > 0 && state.unitsTotal) {
    eta = ` · ~${duration(el / state.unitsDone * (state.unitsTotal - state.unitsDone))} left`;
  }
  $("stat-eta").textContent = `${duration(el)}${eta}`;
  $("stat-eta").title = "ETA is rough: it assumes every remaining deal costs the average so far, but rows differ in cost (search rows and LBR are slower).";
  const cur = state.current;
  if (cur && !state.finished) {
    const e = cur.elapsed_s + (Date.now() - state.currentAt) / 1000;
    $("current-label").textContent = `${cur.row} · chunk ${duration(e)}${num(cur.expected_s) ? ` / ~${duration(cur.expected_s)}` : ""}`;
    renderRows();
  } else {
    $("current-label").textContent = state.finished ? (state.reason === "completed" ? "all rows measured" : state.reason) : "—";
  }
}

// ---- row cards -----------------------------------------------------------------
function renderRows() {
  const out = rowNames().map((name) => {
    const st = rowStatus(name), f = state.finals[name], p = state.prog[name], total = rowDeals(name) || 1;
    let value = `<span class="rv"><small>queued</small></span>`, frac = 0, kpi = opponentText(name);
    if (f) {
      frac = 1;
      if (f.change) {
        const cls = statusClass(name, f.change.status);
        kpi = `vs ${state.config.previous.label}: ${sgn(f.change.mean)} ${ci(f.change)}`;
        value = `<div class="rv">${sgn(f.mean)} <small>${esc(ci(f))}</small></div>` +
                `<div class="rk"><span class="st ${cls}">${esc(statusText(name, f.change.status))}</span> ${esc(kpi)}</div>`;
        kpi = opponentText(name);
      } else {
        value = `<div class="rv">${sgn(f.mean)} <small>${esc(ci(f))}</small></div>`;
      }
    } else if (p || st === "running") {
      const done = p?.done || 0;
      let within = 0;
      const cur = state.current;
      if (cur && cur.row === name && num(cur.expected_s) && cur.expected_s > 0) {
        const e = cur.elapsed_s + (Date.now() - state.currentAt) / 1000;
        within = Math.min(0.95, e / cur.expected_s) * (total / (state.config?.chunks || 10));
      }
      frac = Math.min(1, (done + within) / total);
      value = p?.partial ? `<div class="rv">${sgn(p.partial.mean)} <small>± ${fmt(Z95 * p.partial.se)} live</small></div>`
                         : `<div class="rv"><small>first chunk running…</small></div>`;
    }
    const secs = p?.seconds ? ` · ${duration(p.seconds)}` : "";
    const statusLabel = st === "running" ? `${(p?.done || 0).toLocaleString()}/${total.toLocaleString()}` : st === "done" ? `done${secs}` : "queued";
    return `<div class="row-card ${st}"><div class="rt"><span>${esc(name)}</span><span class="rs">${esc(statusLabel)}</span></div>` +
           `${value}<div class="rb"><div style="width:${(100 * frac).toFixed(1)}%"></div></div><div class="rk">${esc(kpi)}</div></div>`;
  });
  $("rows").innerHTML = out.join("");
}

// ---- change chart (forest plot) ---------------------------------------------------
function setup(canvas) {
  const dpr = window.devicePixelRatio || 1, r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr));
  const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height };
}
function niceTicks(lo, hi, count = 6) {
  const span = hi - lo || 1, raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || 10 * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function drawChange() {
  const c = state.config, canvas = $("chart-change");
  const { ctx, w, h } = setup(canvas);
  state.hits["chart-change"] = [];
  if (!c) return;
  const hasPrev = !!c.previous;
  $("change-title").textContent = hasPrev ? `Change vs ${c.previous.label} · paired on identical cards` : "Every row · this evaluation (no previous evaluation to compare)";
  $("change-note").textContent = hasPrev
    ? "Candidate minus previous, per row, on the same deals (paired per-block difference, ~95% interval). Green = resolved improvement (for LBR: less exploitable), red = resolved regression, grey = unresolved. Rows fill in as they finish."
    : "Candidate's bb/100 per row with ~95% intervals. Pass --previous <receipt> to see paired changes instead.";
  $("change-legend").innerHTML = hasPrev
    ? `<span class="lg"><i class="sw" style="background:var(--up)"></i>better</span><span class="lg"><i class="sw" style="background:var(--down)"></i>worse</span><span class="lg"><i class="sw" style="background:var(--flat)"></i>unclear</span>`
    : `<span class="lg"><i class="sw" style="background:var(--cur)"></i>candidate</span>`;
  const names = rowNames();
  const items = names.map((name) => {
    const f = state.finals[name];
    if (!f) return { name, pending: true };
    const o = hasPrev ? f.change : f;
    return o ? { name, mean: o.mean, lo: o.lo, hi: o.hi, status: hasPrev ? f.change.status : null, f } : { name, pending: true };
  });
  const vals = items.filter((i) => !i.pending).flatMap((i) => [i.lo, i.hi]);
  let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  if (!vals.length) { lo = -10; hi = 10; }
  if (hasPrev) { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m; }
  const pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad;
  const narrow = w < 520;
  const L = narrow ? 12 : Math.min(190, w * 0.3), R = narrow ? 12 : 18, T = narrow ? 14 : 8, B = 26;
  const X = (v) => L + (v - lo) / (hi - lo) * (w - L - R);
  const rowH = (h - T - B) / names.length, Y = (i) => T + rowH * (i + 0.5);
  ctx.font = `11px ${css("--mono")}`; ctx.lineWidth = 1;
  for (const v of niceTicks(lo, hi)) {
    ctx.strokeStyle = css("--border"); ctx.beginPath(); ctx.moveTo(X(v), T); ctx.lineTo(X(v), h - B); ctx.stroke();
    ctx.fillStyle = css("--ink-3"); ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.fillText(sgn(v, 0), X(v), h - B + 7);
  }
  ctx.strokeStyle = css("--ink-2"); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(0), T); ctx.lineTo(X(0), h - B); ctx.stroke(); ctx.setLineDash([]);
  items.forEach((it, i) => {
    const y = Y(i);
    ctx.fillStyle = it.name === c.headline ? css("--ink") : css("--ink-2");
    ctx.font = `${it.name === c.headline ? "600 " : ""}12px ${css("--mono")}`;
    const label = it.name === "LBR" ? "LBR (lower better)" : it.name;
    if (narrow) { ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText(label, L, y - 7); }
    else { ctx.textAlign = "right"; ctx.textBaseline = "middle"; ctx.fillText(label, L - 12, y); }
    if (it.pending) {
      ctx.fillStyle = css("--ink-3"); ctx.textAlign = "left"; ctx.font = `11px ${css("--sans")}`;
      ctx.fillText(rowStatus(it.name) === "running" ? "measuring…" : "pending", X(0) + 8, y);
      return;
    }
    const col = hasPrev ? css(`--${statusClass(it.name, it.status)}`) : css("--cur");
    ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(X(it.lo), y); ctx.lineTo(X(it.hi), y); ctx.stroke();
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(it.mean), y, 4.5, 0, 2 * Math.PI); ctx.fill();
    ctx.lineWidth = 1;
    const f = it.f, prev = prevOf(it.name);
    let html = `<div class="th">${esc(it.name)}</div><div class="tr"><span>candidate</span><span>${sgn(f.mean)} ${esc(ci(f))}</span></div>`;
    if (prev) html += `<div class="tr"><span>${esc(c.previous.label)}</span><span>${sgn(prev.mean)} ${esc(ci(prev))}</span></div>`;
    if (f.change) html += `<div class="tr"><span>paired change</span><span>${sgn(f.change.mean)} ${esc(ci(f.change))}</span></div>`;
    state.hits["chart-change"].push({ x: X(it.mean), y, html });
  });
}

// ---- tables & sources ----------------------------------------------------------------
function renderTable() {
  const c = state.config;
  if (!c) return;
  const hasPrev = !!c.previous;
  $("rows-table").querySelector("thead").innerHTML = `<tr><th>row</th><th>opponent</th><th>candidate</th><th></th>` +
    (hasPrev ? `<th>${esc(c.previous.label)}</th><th></th><th>change</th><th></th><th>status</th>` : "") + `<th>deals</th><th>time</th></tr>`;
  $("rows-table").querySelector("tbody").innerHTML = rowNames().map((name) => {
    const f = state.finals[name], p = state.prog[name], prev = prevOf(name);
    const cand = f ? `<td>${sgn(f.mean)}</td><td class="ci">${esc(ci(f))}</td>`
      : p?.partial ? `<td>${sgn(p.partial.mean)}</td><td class="ci">live ±${fmt(Z95 * p.partial.se)}</td>` : `<td>—</td><td></td>`;
    let prevCells = "";
    if (hasPrev) {
      const ch = f?.change;
      prevCells = `<td>${prev ? sgn(prev.mean) : "—"}</td><td class="ci">${esc(ci(prev))}</td>` +
        `<td>${ch ? sgn(ch.mean) : "—"}</td><td class="ci">${esc(ci(ch))}</td>` +
        `<td>${ch ? `<span class="st ${statusClass(name, ch.status)}">${esc(statusText(name, ch.status))}</span>` : ""}</td>`;
    }
    const deals = rowDeals(name) || 0;
    return `<tr class="${name === c.headline ? "headline" : ""}"><td>${esc(name)}</td><td style="text-align:left;color:var(--ink-3)">${esc(opponentText(name))}</td>${cand}${prevCells}` +
      `<td>${(p?.done ?? (f ? deals : 0)).toLocaleString()}/${deals.toLocaleString()}</td><td>${p?.seconds ? duration(p.seconds) : "—"}</td></tr>`;
  }).join("");
  const nDone = Object.keys(state.finals).length;
  $("table-kpi").textContent = `${nDone} / ${rowNames().length} rows final`;
}
function renderSources() {
  const rows = rowNames().filter((n) => n !== "LBR").map((name) => {
    const s = state.finals[name]?.sources || state.prog[name]?.partial?.sources;
    if (!s) return null;
    const total = s.reduce((a, b) => a + b, 0) || 1;
    const segs = SRC.map(([label, v], k) => {
      const pct = 100 * (s[k] || 0) / total;
      return pct > 0 ? `<div class="bar-seg" style="width:${pct}%;background:var(${v})" title="${esc(label)} ${pct.toFixed(1)}%"><span>${pct >= 9 ? pct.toFixed(0) + "%" : ""}</span></div>` : "";
    }).join("");
    return `<div class="src-row"><span>${esc(name)}</span><div class="bar-track">${segs}</div></div>`;
  }).filter(Boolean);
  $("sources").innerHTML = rows.length ? rows.join("") : `<div class="empty">no decisions yet</div>`;
}
function renderHistory() {
  const c = state.config;
  if (!c) return;
  const rows = [...(c.history || [])];
  const head = state.finals[c.headline], l = state.finals.LBR;
  $("history").querySelector("thead").innerHTML = `<tr><th>date (UTC)</th><th>label</th><th>agent</th><th>strength</th><th></th><th>LBR</th><th></th><th>postflop LBR</th><th>off-tree</th></tr>`;
  const line = (r, cur) => {
    const agent = r.search ? `search ${r.search.streets} ×${r.search.river_iters}` : "blueprint";
    return `<tr class="${cur ? "current" : ""}"><td>${esc(r.started_utc ? r.started_utc.slice(0, 16).replace("T", " ") : "now")}</td><td style="text-align:left">${esc(r.label)}${cur ? " (this run)" : ""}</td>` +
      `<td style="text-align:left;color:var(--ink-3)">${esc(agent)}</td><td>${r.strength ? sgn(r.strength.mean) : "…"}</td><td class="ci">${esc(ci(r.strength))}</td>` +
      `<td>${r.lbr ? sgn(r.lbr.mean) : "…"}</td><td class="ci">${esc(ci(r.lbr))}</td><td>${r.lbr_postflop ? sgn(r.lbr_postflop.mean) : "—"}</td><td>${num(r.off_tree_share) ? (100 * r.off_tree_share).toFixed(1) + "%" : "—"}</td></tr>`;
  };
  const cand = c.candidate || {};
  const current = { label: cand.label, search: cand.search, strength: head || null, lbr: l || null,
                    lbr_postflop: l?.postflop || null, off_tree_share: state.summary?.scores?.off_tree_share };
  $("history").querySelector("tbody").innerHTML = rows.map((r) => line(r, false)).join("") + line(current, true);
  $("history-kpi").textContent = `${rows.length} earlier`;
}

function render() {
  renderHeader(); renderRows(); drawChange(); renderTable(); renderSources(); renderHistory();
}

// ---- events -------------------------------------------------------------------------
let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; render(); }); } }
function applyEvent(e) {
  const ts = Number(e.ts) * 1000 || Date.now();
  if (e.type === "run_start") {
    if (e.config?.study !== "scorecard") {
      $("sub").textContent = `this run is not a scorecard evaluation (study: ${e.config?.study ?? e.variant ?? "unknown"})`;
      events.close();
      return;
    }
    state.config = e.config; state.started = ts; state.unitsTotal = e.target_iters || 0;
    log(`evaluating ${e.config?.candidate?.label ?? ""} on ${e.config?.benchmark_id ?? ""}`, ts);
  } else if (e.type === "metrics") {
    if (e.resources) state.resources = e.resources;
    if (Number.isInteger(e.iteration)) state.unitsDone = Math.max(state.unitsDone, e.iteration);
    if (e.sc_current) { state.current = e.sc_current; state.currentAt = Date.now(); }
  } else if (e.type === "checkpoint") {
    if (e.sc_progress) {
      const p = e.sc_progress, prev = state.prog[p.row];
      state.prog[p.row] = { done: p.done, total: p.total, partial: p.partial, seconds: (prev?.seconds || 0) + (p.chunk_s || 0) };
      state.unitsDone = Math.max(state.unitsDone, p.units_done); state.unitsTotal = p.units_total || state.unitsTotal;
    }
    if (e.sc_row) state.finals[e.sc_row.name] = e.sc_row;
  } else if (e.type === "milestone") log(e.message || e.kind, ts);
  else if (e.type === "run_end") {
    state.finished = ts; state.reason = e.reason; state.summary = e.summary; state.current = null;
    const ok = e.reason === "completed";
    $("done-badge").textContent = ok ? "✓" : "!"; $("done-badge").className = ok ? "done-badge" : "done-badge err";
    $("done-title").textContent = ok ? "Evaluation complete" : `Evaluation ${e.reason}`;
    if (ok) {
      const ch = e.summary?.change;
      const parts = [];
      if (ch) {
        parts.push(`Strength vs ${ch.previous_label}: ${sgn(ch.strength.mean)} ${ci(ch.strength)} → ${statusText(state.config.headline, ch.strength.status)}.`);
        parts.push(`LBR: ${sgn(ch.exploitability_lbr.mean)} ${ci(ch.exploitability_lbr)} → ${statusText("LBR", ch.exploitability_lbr.status)}.`);
        const pc = ch.exploitability_lbr_postflop;
        if (pc) parts.push(`Postflop LBR: ${sgn(pc.mean)} ${ci(pc)} → ${statusText("LBR", pc.status)}.`);
      } else {
        const s = e.summary?.scores;
        if (s) parts.push(`Strength ${sgn(s.strength.mean)} ${ci(s.strength)}; LBR ${sgn(s.exploitability_lbr.mean)} ${ci(s.exploitability_lbr)}.`);
      }
      parts.push(`${duration((ts - state.started) / 1000)} · receipt ${String(e.summary?.receipt ?? "").split(/[\\/]/).pop()}`);
      $("done-detail").textContent = parts.join(" ");
    } else {
      $("done-detail").textContent = e.summary?.error || "stopped before completion; no receipt was written";
    }
    $("done-overlay").hidden = false;
    log(`run ended (${e.reason})`, ts, ok ? "note" : "warning");
  }
  schedule();
}

document.addEventListener("mousemove", (ev) => {
  const tip = $("tip"), el = ev.target;
  const hits = el && el.tagName === "CANVAS" ? state.hits[el.id] : null;
  if (!hits) { tip.hidden = true; return; }
  const r = el.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
  let best = null, bd = Infinity;
  for (const hit of hits) { const d = Math.abs(hit.y - my) * 4 + Math.abs(hit.x - mx) * 0.05; if (d < bd && Math.abs(hit.y - my) < 16) { bd = d; best = hit; } }
  if (!best) { tip.hidden = true; return; }
  tip.innerHTML = best.html; tip.hidden = false;
  tip.style.left = `${r.left + best.x}px`; tip.style.top = `${r.top + best.y - 8}px`;
});

const events = new EventSource("/events");
events.onmessage = (m) => JSON.parse(m.data).forEach(applyEvent);
// The server lives in the evaluation script: when the script exits, stop
// retrying after a few attempts instead of warning forever.
let failures = 0;
events.onopen = () => { failures = 0; };
events.onerror = () => {
  failures += 1;
  if (failures >= 3) {
    events.close();
    log(state.finished ? "dashboard server stopped (evaluation finished)"
                       : "dashboard server stopped: the evaluation script exited; re-open with scorecard.py view",
        Date.now(), state.finished ? "note" : "warning");
  } else if (!state.finished && failures === 1) {
    log("telemetry reconnecting…", Date.now(), "warning");
  }
};
window.addEventListener("resize", schedule);
$("done-dismiss").onclick = () => { $("done-overlay").hidden = true; };
setInterval(updateClocks, 1000);
render();
