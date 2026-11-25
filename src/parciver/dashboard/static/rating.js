// parciver rating page: renders the `rating_view` event that scripts/rating.py builds from receipts.
// Nothing is computed here except layout: every number comes from the view.
"use strict";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt = (x, d = 1) => (x === undefined || x === null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const sgn = (x, d = 1) => (x >= 0 ? "+" : "") + fmt(x, d);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const Z = 1.959963984540054;
let view = null, selected = null, points = [];

function status(text, cls) { $("m-status").innerHTML = `<span class="dot ${cls}"></span>${esc(text)}`; }

function applyEvent(e) {
  if (e.type === "rating_progress") {
    const pct = e.total ? (100 * e.done) / e.total : 0;
    $("flight-what").textContent = e.what;
    $("flight-kpi").textContent = `${e.done.toLocaleString()} / ${e.total.toLocaleString()} deals`;
    $("flight-bar").style.width = `${pct}%`;
    status("running", "live");
  } else if (e.type === "rating_view") {
    view = e;
    if (selected === null || !view.rows.some((r) => r.receipt === selected)) {
      selected = view.headline ? view.headline.receipt : (view.rows.length ? view.rows[view.rows.length - 1].receipt : null);
    }
    render();
  } else if (e.type === "rating_error") {
    status(`view error: ${e.error}`, "idle");
  } else if (e.type === "run_end") {
    $("flight-what").textContent = "idle (run finished)";
    status("finished", "done");
  }
}

function render() {
  const v = view;
  $("sub").textContent = `pool ${v.pool} · ${v.anchors.length} anchors · ${(v.rating_deals_weighted ?? v.rating_deals).toLocaleString()} deals vs weighted anchors, ${v.rating_deals.toLocaleString()} otherwise`;
  $("m-pool").textContent = `${v.pool} ${v.pool_hash}`;
  $("m-id").textContent = v.identity;
  $("m-built").textContent = v.built_utc.replace("T", " ").replace("+00:00", "Z");
  const h = v.headline;
  if (h) {
    const r = h.rating;
    $("pr").className = "pr-big";
    $("pr").textContent = fmt(r.pr, 0);
    const c = v.change;
    const ch = !c ? "" : c.error ? ` · vs ${esc(c.from)}: ${esc(c.error)}` : ` · vs ${esc(c.from)} ${sgn(c.mean, 1)} (95% ${sgn(c.lo, 1)} to ${sgn(c.hi, 1)})`;
    const raw = h.rating_raw ? ` · AIVAT (raw ${fmt(h.rating_raw.pr, 0)}, 95% ${fmt(h.rating_raw.lo, 0)}–${fmt(h.rating_raw.hi, 0)})` : "";
    $("pr-sub").innerHTML = `${esc(h.label)} <span class="mono">[${esc(h.fingerprint)}]</span> · 95% ${fmt(r.lo, 0)}–${fmt(r.hi, 0)}${raw} · <span class="chip ${h.state}">${h.state}</span>${ch}`;
    if (h.lbr) {
      $("lbr").innerHTML = `${fmt(h.lbr.mean, 0)} <span class="unit">mbb/hand</span>`;
      const post = h.lbr_postflop ? ` · postflop ${fmt(h.lbr_postflop.mean, 0)}` : "";
      $("lbr-sub").textContent = `±${fmt(Z * h.lbr.se, 0)} (95%) · ${h.lbr.n.toLocaleString()} deals${post} · ${h.label}`;
    } else {
      $("lbr").textContent = "—";
      $("lbr-sub").textContent = "the headline receipt has no LBR";
    }
  } else {
    $("pr").className = "pr-big none";
    $("pr").textContent = "no verified rating";
    $("pr-sub").textContent = v.headline_why || "no current bot set (rating.py current --label …)";
    $("lbr").textContent = "—";
    $("lbr-sub").textContent = "shown beside the rating, never folded into it";
  }
  renderWeights(v);
  renderTable(v);
  renderMatrix(v);
  renderSelected(v);
  drawChart();
}

function renderWeights(v) {
  const max = Math.max(...Object.values(v.weights), 1e-9);
  $("weights").innerHTML = v.anchors.map((j) => {
    const w = v.weights[j];
    const def = v.anchor_defs.find((a) => a.id === j);
    const what = def.kind === "leak" ? `${def.of} + ${def.family} leak ${def.strength}` : def.path.split(/[\\/]/).slice(-2, -1)[0];
    return `<div class="wrow" title="${esc(what)}"><span>${esc(j)}</span><div class="wtrack"><div style="width:${(100 * w) / max}%"></div></div><span>${fmt(w, 3)}</span></div>`;
  }).join("");
  $("cal-state").textContent = v.calibration_error ? "DOES NOT CHECK" : v.calibration_state;
  const a = v.acceptance || {}, li = [];
  // ok: true PASS, false FAIL, null (not decided yet: e.g. the reference is STALE) PENDING.
  const line = (ok, name, text) => li.push(`<li><span class="${ok === true ? "ok" : ok === false ? "fail" : "rep"}">${ok === true ? "PASS" : ok === false ? "FAIL" : "PENDING"}</span><span>${esc(name)}: ${esc(text)}</span></li>`);
  if (v.calibration_error) line(false, "calibration recheck", v.calibration_error);
  if (v.c6) line({ VERIFIED: true, INVALID: false }[v.c6.state] ?? null, `C6 vs ${v.c6.against} (gating)`, v.c6.why);
  if (a.C1_replay) line(a.C1_replay.ok, "C1 replay", a.C1_replay.detail);
  if (a.C2_seed_consistency) line(a.C2_seed_consistency.ok, "C2 seed consistency", Object.entries(a.C2_seed_consistency.anchors).map(([j, x]) => `${j} ${sgn(x.rating_seeds)} vs ${x.calibration_seeds === undefined ? "0" : sgn(x.calibration_seeds)}`).join(", "));
  if (a.C3_nash) line(a.C3_nash.ok, "C3 Nash check", `max(Ap)/max|A| = ${a.C3_nash.max_Ap_over_maxA.toExponential(2)}`);
  if (a.C4_role_symmetry) line(a.C4_role_symmetry.ok, "C4 role symmetry", `${a.C4_role_symmetry.exact_pairs}/${a.C4_role_symmetry.pairs} pairs negate exactly, deal by deal, when seats are swapped`);
  $("checks").innerHTML = li.join("");
}

function renderTable(v) {
  $("tbl").querySelector("thead").innerHTML = "<tr><th>configuration</th><th>state</th><th>PR</th><th>95% interval</th><th>deals</th><th>LBR mbb/h</th><th>search failed</th><th>engine+code</th><th>receipt</th><th>note</th></tr>";
  const rows = [...v.rows].reverse();
  $("tbl").querySelector("tbody").innerHTML = rows.map((r) => {
    const x = r.rating;
    const cls = [r.anchor ? "anchor" : "", r.headline ? "head" : "", r.state === "STALE" ? "stale" : ""].join(" ");
    const dl = r.deals ? Object.values(r.deals) : [];
    const deals = !dl.length ? "—" : Math.min(...dl) === Math.max(...dl) ? dl[0].toLocaleString() : `${Math.min(...dl).toLocaleString()}–${Math.max(...dl).toLocaleString()}`;
    return `<tr class="${cls}" data-r="${esc(r.receipt)}" style="cursor:pointer">
      <td>${esc(r.label)}${r.headline ? " ★" : ""}</td>
      <td><span class="chip ${r.state}">${r.state}</span></td>
      <td>${x ? fmt(x.pr, 1) : "—"}${r.rating_raw ? `<div class="dim">raw ${fmt(r.rating_raw.pr, 1)}</div>` : ""}</td><td class="dim">${x ? `${fmt(x.lo, 1)} – ${fmt(x.hi, 1)}` : "—"}</td>
      <td class="dim">${deals}</td>
      <td>${r.lbr ? `${fmt(r.lbr.mean, 0)} ±${fmt(Z * r.lbr.se, 0)}` : "—"}</td>
      <td class="dim">${r.search_failed ?? "—"}</td><td class="dim">${esc(r.identity || "—")}</td>
      <td class="dim">${esc(r.receipt)}</td><td class="why">${r.state === "VERIFIED" ? "" : esc(r.why)}</td></tr>`;
  }).join("");
  $("tbl").querySelectorAll("tbody tr").forEach((tr) => tr.addEventListener("click", () => { selected = tr.dataset.r; renderSelected(view); drawChart(); }));
  const nv = v.rows.filter((r) => r.state === "VERIFIED").length;
  $("tbl-kpi").textContent = `${nv} verified of ${v.rows.length}`;
}

function renderMatrix(v) {
  const k = v.anchors.length;
  if (!v.A) { $("mx").querySelector("thead").innerHTML = ""; $("mx").querySelector("tbody").innerHTML = `<tr><td class="dim">no calibration</td></tr>`; return; }
  $("mx").querySelector("thead").innerHTML = `<tr><th>row \\ col</th>${v.anchors.map((j) => `<th>${esc(j)}</th>`).join("")}</tr>`;
  const max = Math.max(...v.A.flat().map(Math.abs), 1e-9);
  const g = css("--green"), rd = css("--red");
  $("mx").querySelector("tbody").innerHTML = v.anchors.map((ri, i) => `<tr><td>${esc(ri)}</td>${Array.from({ length: k }, (_, j) => {
    if (i === j) return `<td class="dim">·</td>`;
    const a = v.A[i][j], al = Math.min(0.55, Math.abs(a) / max * 0.55);
    const bg = `${a >= 0 ? g : rd}${Math.round(al * 255).toString(16).padStart(2, "0")}`;
    return `<td class="c" style="background:${bg}" title="± ${fmt(v.A_se[i][j], 2)}">${sgn(a, 1)}</td>`;
  }).join("")}</tr>`).join("");
}

function renderSelected(v) {
  const r = v.rows.find((x) => x.receipt === selected);
  const head = $("sel").querySelector("thead"), body = $("sel").querySelector("tbody");
  if (!r || !r.anchors) { head.innerHTML = ""; body.innerHTML = `<tr><td class="dim">nothing selected</td></tr>`; $("sel-kpi").textContent = "—"; return; }
  $("sel-kpi").textContent = r.label;
  head.innerHTML = "<tr><th>anchor</th><th>weight</th><th>win mbb/h</th><th>± 95%</th><th>vs A0 (paired)</th><th>± 95%</th></tr>";
  body.innerHTML = v.anchors.filter((j) => r.anchors[j]).map((j) => {
    const a = r.anchors[j], p = r.paired && r.paired[j];
    return `<tr><td>${esc(j)}</td><td class="dim">${fmt(v.weights[j], 3)}</td><td>${sgn(a.mean)}</td><td class="dim">${fmt(Z * a.se)}</td>
      <td>${p ? sgn(p.mean) : "—"}</td><td class="dim">${p ? fmt(Z * p.se) : "—"}</td></tr>`;
  }).join("");
}

function drawChart() {
  const cv = $("chart"), ctx = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  points = [];
  if (!view) return;
  const rows = view.rows.filter((r) => r.rating);
  if (!rows.length) { ctx.fillStyle = css("--ink-3"); ctx.font = `13px ${css("--sans")}`; ctx.fillText("no ratings yet", 12, 24); return; }
  const lo = Math.min(1500, ...rows.map((r) => r.rating.lo)), hi = Math.max(1500, ...rows.map((r) => r.rating.hi));
  const pad = Math.max(2, (hi - lo) * 0.08), y0 = lo - pad, y1 = hi + pad;
  const L = 56, R = 12, T = 12, B = 44;
  const X = (i) => L + (rows.length === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (rows.length - 1));
  const Y = (v) => T + ((y1 - v) * (H - T - B)) / (y1 - y0);
  ctx.font = `11px ${css("--mono")}`; ctx.fillStyle = css("--ink-3"); ctx.strokeStyle = css("--border"); ctx.lineWidth = 1;
  const step = niceStep((y1 - y0) / 5);
  for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
    ctx.beginPath(); ctx.moveTo(L, Y(v)); ctx.lineTo(W - R, Y(v)); ctx.stroke();
    ctx.fillText(fmt(v, step < 1 ? 1 : 0), 6, Y(v) + 4);
  }
  ctx.strokeStyle = css("--ink-3"); ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(L, Y(1500)); ctx.lineTo(W - R, Y(1500)); ctx.stroke(); ctx.setLineDash([]);
  const col = { VERIFIED: css("--green"), STALE: css("--ink-3"), INVALID: css("--red") };
  rows.forEach((r, i) => {
    const x = X(i), c = col[r.state];
    ctx.strokeStyle = c; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, Y(r.rating.lo)); ctx.lineTo(x, Y(r.rating.hi)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - 5, Y(r.rating.lo)); ctx.lineTo(x + 5, Y(r.rating.lo));
    ctx.moveTo(x - 5, Y(r.rating.hi)); ctx.lineTo(x + 5, Y(r.rating.hi)); ctx.stroke();
    ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, Y(r.rating.pr), r.receipt === selected ? 6 : 4, 0, 2 * Math.PI); ctx.fill();
    ctx.save(); ctx.translate(x, H - B + 8); ctx.rotate(-Math.PI / 9); ctx.fillStyle = css("--ink-2");
    ctx.fillText(r.label.length > 18 ? r.label.slice(0, 17) + "…" : r.label, -4, 10); ctx.restore();
    points.push({ x, y: Y(r.rating.pr), r });
  });
}

function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

$("chart").addEventListener("mousemove", (ev) => {
  const b = $("chart").getBoundingClientRect(), mx = ev.clientX - b.left, my = ev.clientY - b.top;
  const hit = points.find((p) => Math.hypot(p.x - mx, p.y - my) < 9);
  const tip = $("tip");
  if (!hit) { tip.hidden = true; return; }
  const r = hit.r, x = r.rating;
  tip.innerHTML = `${esc(r.label)} <span class="chip ${r.state}">${r.state}</span><br>PR ${fmt(x.pr)} (95% ${fmt(x.lo)}–${fmt(x.hi)})<br>${x.n.toLocaleString()} paired deals<br>${esc(r.agent)}`;
  tip.style.left = `${ev.clientX + 14}px`; tip.style.top = `${ev.clientY + 10}px`; tip.hidden = false;
});
$("chart").addEventListener("mouseleave", () => { $("tip").hidden = true; });
window.addEventListener("resize", drawChart);

const events = new EventSource("/events");
events.onopen = () => status("connected", "live");
events.onerror = () => status("disconnected", "idle");
events.onmessage = (m) => JSON.parse(m.data).forEach(applyEvent);
