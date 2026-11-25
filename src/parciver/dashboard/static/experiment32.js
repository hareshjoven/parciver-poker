/* Locked S3.2 full-100bb telemetry view. The client renders records only; every
 * verdict is re-derived in Python and arrives in `step32` / run_end.summary.
 * The live telemetry cards are presentation-only: they never steer the run. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  config: null, rows: [], rowKeys: new Set(), thresholds: {}, report: null,
  progress: { completed: 0, total: 45 }, calibrations: {},
  proxyEvents: [], proxyKeys: new Set(),
  startedAt: null, finishedAt: null, lastEventAt: null,
  lastEvent: "", stage: "waiting for locked run…",
};
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
const COL = { border:css("border"), muted:css("ink-3"), accent:css("accent"),
  violet:css("accent-2"), green:css("green"), amber:css("amber"), red:css("red") };

function n(value, digits = 3) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}
function eventMs(event) {
  const ts = Number(event?.ts);
  return Number.isFinite(ts) ? ts * 1000 : Date.now();
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#39;"}[ch]));
}
function addLog(text, timestamp = null) {
  const row = document.createElement("li");
  const when = timestamp == null ? new Date() : new Date(timestamp);
  row.innerHTML = `<span class="t">${when.toLocaleTimeString([], { hour:"2-digit", minute:"2-digit" })}</span><span class="kind note">S3.2</span><span class="msg">${escapeHtml(text)}</span>`;
  $("event-log").prepend(row);
  while ($("event-log").children.length > 80) $("event-log").removeChild($("event-log").lastChild);
}

function applyEvent(event) {
  const timestamp = eventMs(event);
  state.lastEventAt = timestamp;
  state.lastEvent = event.message || event.kind || event.type || "telemetry";
  if (event.type === "run_start") {
    state.config = event.config || {};
    state.startedAt = timestamp;
    const total = Number(state.config.seeds?.length) * Number(state.config.population?.length);
    if (Number.isFinite(total) && total > 0) state.progress.total = total;
    state.stage = "initializing locked workflow";
    $("game-name").textContent = "S3.2 · full 100bb · " + (state.config.protocol || "unknown protocol");
    addLog("run started · frozen hashes verified", timestamp);
  } else if (event.type === "checkpoint") {
    if (event.step32_thresholds) state.thresholds[event.step32_thresholds.seed] = event.step32_thresholds;
    if (event.step32_row) {
      const rowKey = `${event.step32_row.seed}:${event.step32_row.label}`;
      if (!state.rowKeys.has(rowKey)) {
        state.rowKeys.add(rowKey); state.rows.push(event.step32_row);
        addLog(`${event.step32_row.seed} · ${event.step32_row.label} complete`, timestamp);
      }
      state.stage = `case complete · ${event.step32_row.seed} ${event.step32_row.label}`;
    } else if (event.step32_thresholds) {
      state.stage = `thresholds calibrated · seed ${event.step32_thresholds.seed}`;
    }
    if (event.step32_progress) state.progress = event.step32_progress;
    if (event.step32) { state.report = event.step32; state.stage = "final report ready"; }
  } else if (event.type === "milestone") {
    if (event.s3_2_calibration) {
      const calibration = event.s3_2_calibration;
      state.calibrations[`${calibration.seed}:${calibration.label}`] = { ...calibration, timestamp };
      state.stage = `calibration · seed ${calibration.seed} ${calibration.label}`;
    } else if (event.s3_2_proxy) {
      const proxy = event.s3_2_proxy;
      const proxyKey = `${proxy.seed}:${proxy.label}:${proxy.alpha}`;
      if (!state.proxyKeys.has(proxyKey)) { state.proxyKeys.add(proxyKey); state.proxyEvents.push({ ...proxy, timestamp }); }
      state.stage = `proxy search · seed ${proxy.seed} ${proxy.label} · α=${n(proxy.alpha, 2)} ${proxy.passed ? "pass" : "reject"}`;
    } else {
      state.stage = event.kind || "milestone";
    }
    addLog(event.message || event.kind, timestamp);
  } else if (event.type === "run_end") {
    state.report = event.summary || state.report;
    state.finishedAt = timestamp;
    state.stage = state.report?.verdict || event.reason || "run ended";
    const overlay = $("done-overlay"); overlay.hidden = false;
    $("done-title").textContent = state.report?.verdict || event.reason;
    $("done-detail").textContent = state.report?.failure || state.report?.note || state.report?.safety_language || "Run ended.";
    if (state.report && !state.report.all_pass) { $("done-badge").textContent = "!"; $("done-badge").className = "done-badge err"; }
  }
  render();
}

function formatDuration(seconds) {
  if (!Number.isFinite(Number(seconds)) || seconds < 0) return "—";
  let remaining = Math.round(Number(seconds));
  const days = Math.floor(remaining / 86400); remaining %= 86400;
  const hours = Math.floor(remaining / 3600); remaining %= 3600;
  const minutes = Math.floor(remaining / 60); const secs = remaining % 60;
  if (days) return `${days}d ${hours}h`;
  if (hours) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  return `${minutes}m ${String(secs).padStart(2, "0")}s`;
}
function median(values) {
  const xs = values.filter((value) => Number.isFinite(Number(value))).map(Number).sort((a, b) => a - b);
  if (!xs.length) return null;
  const mid = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
}
function proxyCadenceSeconds() {
  const times = state.proxyEvents.map((event) => Number(event.timestamp)).filter(Number.isFinite).sort((a, b) => a - b);
  return median(times.slice(1).map((time, i) => (time - times[i]) / 1000));
}
function expectedCalibrations() {
  const seeds = Array.isArray(state.config?.seeds) ? state.config.seeds : [];
  const labels = Array.isArray(state.config?.population)
    ? state.config.population.filter((spec) => spec.split === "fit").map((spec) => spec.label)
    : [];
  const keys = [];
  seeds.forEach((seed) => labels.forEach((label) => keys.push(`${seed}:${label}`)));
  return keys.length ? keys : Object.keys(state.calibrations);
}
function renderCalibrations() {
  const host = $("calibration-cards");
  const keys = expectedCalibrations();
  host.innerHTML = keys.map((key) => {
    const calibration = state.calibrations[key];
    const [seed, ...labelParts] = key.split(":");
    const label = labelParts.join(":");
    const active = state.stage.includes(`${seed} ${label}`);
    if (!calibration) return `<article class="cal-card pending${active ? " active" : ""}"><div class="cal-title"><span>seed ${escapeHtml(seed)} · ${escapeHtml(label)}</span><span class="cal-status">${active ? "running" : "queued"}</span></div><div class="cal-values"><span>mean <b>—</b> bb/100</span><span>99% CI <b>—</b></span></div></article>`;
    return `<article class="cal-card done"><div class="cal-title"><span>seed ${escapeHtml(seed)} · ${escapeHtml(label)}</span><span class="cal-status">complete</span></div><div class="cal-values"><span>mean <b>${n(calibration.mean, 1)}</b> bb/100</span><span>99% CI <b>[${n(calibration.lower, 1)}, ${n(calibration.upper, 1)}]</b></span><span>SE <b>${n(calibration.standard_error, 2)}</b></span></div></article>`;
  }).join("");
}
function renderPhaseStrip() {
  const host = $("phase-strip");
  const calibrationTotal = expectedCalibrations().length || 9;
  const calibrationDone = Object.keys(state.calibrations).length;
  const completed = Number.isFinite(Number(state.progress.completed)) ? Number(state.progress.completed) : state.rows.length;
  const total = Number.isFinite(Number(state.progress.total)) ? Number(state.progress.total) : 45;
  const reportDone = !!state.report;
  const calibrationStatus = calibrationDone >= calibrationTotal ? "done" : state.stage.includes("calibration") ? "active" : calibrationDone ? "done" : "active";
  const caseStatus = reportDone ? "done" : completed ? "active" : state.stage.includes("proxy") || state.stage.includes("case") ? "active" : "queued";
  const reportStatus = reportDone ? "done" : "queued";
  const phase = (label, detail, status) => `<div class="phase-step ${status}">${label}<span>${detail}</span></div>`;
  host.innerHTML = phase("reference", "frozen anchor", state.startedAt == null ? "active" : "done")
    + phase("calibration", `${calibrationDone}/${calibrationTotal} responses`, calibrationStatus)
    + phase("case evaluation", `${completed}/${total} cases`, caseStatus)
    + phase("report", reportDone ? "verdict ready" : "awaiting cases", reportStatus);
}
function renderTelemetry() {
  const now = state.finishedAt || Date.now();
  const elapsed = state.startedAt == null ? null : (now - state.startedAt) / 1000;
  $("stat-elapsed").textContent = formatDuration(elapsed);
  const completed = Number.isFinite(Number(state.progress.completed)) ? Number(state.progress.completed) : state.rows.length;
  const total = Number.isFinite(Number(state.progress.total)) ? Number(state.progress.total) : 45;
  $("stat-workload").textContent = `${completed} / ${total}`;
  const fraction = total > 0 ? Math.max(0, Math.min(1, completed / total)) : 0;
  $("run-progress-label").textContent = `${completed} of ${total} cases complete`;
  $("run-progress-percent").textContent = `${(100 * fraction).toFixed(1)}%`;
  $("run-progress-bar").style.width = `${(100 * fraction).toFixed(1)}%`;
  const typicalCase = median(state.rows.map((row) => row.compute_s));
  const cadence = proxyCadenceSeconds();
  const remaining = Math.max(0, total - completed);
  let eta = null;
  let etaNote = "needs a measured case";
  if (typicalCase != null) { eta = typicalCase * remaining; etaNote = "based on median case time"; }
  else if (cadence != null && remaining > 0) {
    const stepsPerCase = Math.max(3, Math.round(state.proxyEvents.length / Math.max(1, completed + 1)));
    eta = Math.max(0, cadence * (remaining * stepsPerCase - state.proxyEvents.length));
    etaNote = "rough · proxy-search cadence";
  }
  $("stat-eta").textContent = state.report ? "complete" : eta == null ? "building estimate…" : `~${formatDuration(eta)}`;
  $("stat-eta-note").textContent = state.report ? "final report received" : etaNote;
  $("stat-throughput").textContent = typicalCase != null ? `~${n(3600 / typicalCase, 2)} cases/hour` : cadence != null ? `~${n(3600 / cadence, 2)} alpha checks/hour` : "waiting for work rate";
  $("stat-last-event").textContent = state.lastEvent || "—";
  $("run-stage").textContent = state.stage;
}

function canvas(id) {
  const el = $(id), rect = el.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  el.width = Math.max(1, Math.round(rect.width * dpr)); el.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = el.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: rect.width, h: rect.height };
}
function frame(ctx, w, h) {
  ctx.clearRect(0, 0, w, h); ctx.strokeStyle = COL.border; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(36, 8); ctx.lineTo(36, h - 24); ctx.lineTo(w - 8, h - 24); ctx.stroke();
}
function noData(ctx, w, h) { ctx.fillStyle = COL.muted; ctx.textAlign = "center"; ctx.fillText("waiting for completed cases…", w / 2, h / 2); }

function drawSeries(id, rows, value, color, baseline = null, intervals = false) {
  const { ctx, w, h } = canvas(id); frame(ctx, w, h);
  if (!rows.length) return noData(ctx, w, h);
  let values = rows.flatMap((r) => intervals ? [r.lower, r.upper, value(r)] : [value(r)]).filter(Number.isFinite);
  if (baseline) values = values.concat(rows.map(baseline).filter(Number.isFinite));
  let lo = Math.min(0, ...values), hi = Math.max(0.001, ...values); if (hi === lo) hi = lo + 1;
  const x = (i) => 42 + i * Math.max(1, (w - 56) / Math.max(1, rows.length - 1));
  const y = (v) => 8 + (hi - v) * (h - 36) / (hi - lo);
  if (baseline) { ctx.strokeStyle = COL.amber; ctx.setLineDash([4, 4]); ctx.beginPath();
    rows.forEach((r, i) => i ? ctx.lineTo(x(i), y(baseline(r))) : ctx.moveTo(x(i), y(baseline(r)))); ctx.stroke(); ctx.setLineDash([]); }
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.beginPath();
  rows.forEach((r, i) => { const v = value(r); i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)); }); ctx.stroke();
  rows.forEach((r, i) => { if (intervals) { ctx.globalAlpha = .55; ctx.fillRect(x(i) - 1, y(r.upper), 2, Math.max(1, y(r.lower) - y(r.upper))); ctx.globalAlpha = 1; }
    ctx.beginPath(); ctx.arc(x(i), y(value(r)), 3, 0, Math.PI * 2); ctx.fill(); });
  ctx.fillStyle = COL.muted; ctx.textAlign = "right"; ctx.fillText(n(hi, 2), 32, 13); ctx.fillText(n(lo, 2), 32, h - 24);
}

function drawAlpha() {
  const { ctx, w, h } = canvas("c-alpha"); frame(ctx, w, h); if (!state.rows.length) return noData(ctx, w, h);
  const bw = Math.max(2, (w - 50) / state.rows.length), y = (v) => 8 + (1 - v) * (h - 36);
  state.rows.forEach((r, i) => { const x = 40 + i * bw; ctx.globalAlpha = .46; ctx.fillStyle = COL.violet; ctx.fillRect(x, y(r.alpha_conf), bw * .42, h - 24 - y(r.alpha_conf));
    ctx.globalAlpha = 1; ctx.fillStyle = COL.accent; ctx.fillRect(x + bw * .45, y(r.alpha), bw * .42, h - 24 - y(r.alpha)); });
  ctx.fillStyle = COL.muted; ctx.textAlign = "right"; ctx.fillText("1.0", 32, 13); ctx.fillText("0", 32, h - 24);
}

function drawModel() {
  const held = state.rows.filter((r) => r.split === "held" && r.klass === "suboptimal");
  const { ctx, w, h } = canvas("c-model"); frame(ctx, w, h); if (!held.length) return noData(ctx, w, h);
  const bw = Math.max(3, (w - 50) / held.length), y = (v) => 8 + (1 - Math.max(0, Math.min(1, v))) * (h - 36);
  held.forEach((r, i) => { const x = 40 + i * bw; ctx.fillStyle = r.coverage >= .95 ? COL.green : COL.red;
    ctx.fillRect(x, y(r.coverage), bw * .55, h - 24 - y(r.coverage));
    const improve = Math.max(0, Math.min(1, (r.reference_log_loss - r.model_log_loss) / Math.max(.001, r.reference_log_loss)));
    ctx.fillStyle = COL.accent; ctx.fillRect(x + bw * .58, y(improve), bw * .28, h - 24 - y(improve)); });
  ctx.strokeStyle = COL.amber; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(36, y(.95)); ctx.lineTo(w - 8, y(.95)); ctx.stroke(); ctx.setLineDash([]);
}

function renderTable() {
  const body = $("case-table").querySelector("tbody");
  body.innerHTML = state.rows.map((r) => `<tr><td>${r.seed}</td><td>${r.label}</td><td>${r.split}</td>` +
    `<td>${n(r.gain)}</td><td>[${n(r.lower)}, ${n(r.upper)}]</td><td>${n(r.alpha, 2)}</td>` +
    `<td class="${r.coverage >= .95 ? "pass" : "fail"}">${n(100 * r.coverage, 1)}%</td>` +
    `<td>${n(r.proxy_loss_upper)}</td><td>${n(r.proxy_budget)}</td><td>${n(r.compute_s, 0)}s</td><td>${n(r.peak_memory_gb, 2)}</td></tr>`).join("");
}

function renderGates() {
  const gates = state.report?.gates;
  if (gates) {
    $("gates-list").innerHTML = Object.entries(gates).map(([name, ok]) => `<li class="${ok ? "pass" : "fail"}">${ok ? "PASS" : "FAIL"} · ${name}</li>`).join("");
    return;
  }
  // Older stopped runs may contain only a schema-valid terminal note instead
  // of a full report. Keep the failure visible rather than leaving a blank
  // gate panel; no verdict is inferred from the incomplete record.
  if (state.report) {
    const detail = state.report.failure || state.report.note || "terminal report unavailable";
    $("gates-list").innerHTML = `<li class="fail">FAIL · artifact/completeness · ${escapeHtml(detail)}</li>`;
  }
}

function render() {
  const held = state.rows.filter((r) => r.split === "held" && r.klass === "suboptimal");
  $("stat-verdict").textContent = state.report?.verdict || "running";
  $("stat-verdict").className = state.report ? (state.report.all_pass ? "pass" : "fail") : "";
  $("stat-progress").textContent = `${state.progress.completed || state.rows.length} / ${state.progress.total || 45}`;
  $("stat-gain").textContent = held.length ? n(held.reduce((a, r) => a + r.gain, 0) / held.length) + " bb/100" : "—";
  $("stat-coverage").textContent = held.length ? n(100 * Math.min(...held.map((r) => r.coverage)), 1) + "%" : "—";
  drawSeries("c-product", held, (r) => r.gain, COL.green, (r) => r.delta_sub, true);
  drawAlpha();
  drawSeries("c-proxy", state.rows, (r) => r.proxy_loss_upper, COL.red, (r) => r.proxy_budget, false);
  drawModel(); renderTable(); renderGates(); renderCalibrations(); renderPhaseStrip();
  renderTelemetry();
}

const events = new EventSource("/events");
events.onmessage = (message) => { JSON.parse(message.data).forEach(applyEvent); };
events.onerror = () => addLog("telemetry reconnecting…");
window.addEventListener("resize", render);
$("done-dismiss").onclick = () => { $("done-overlay").hidden = true; };
if (typeof requestAnimationFrame === "function") {
  const liveFrame = () => { renderTelemetry(); requestAnimationFrame(liveFrame); };
  requestAnimationFrame(liveFrame);
} else if (typeof setInterval === "function") setInterval(renderTelemetry, 1000);
render();
