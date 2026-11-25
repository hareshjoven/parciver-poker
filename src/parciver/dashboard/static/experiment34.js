// Copyright (c) 2026 Haresh Joven. MIT License (see LICENSE).
/* S3.4 data-only telemetry. Verdicts are computed by Python, never here. */
"use strict";
const $=(id)=>document.getElementById(id);
const state={config:null,rows:[],keys:new Set(),progress:{stage:"observation",completed:0,total:27},report:null,started:null,finished:null,last:"",stageStart:null,etaFinishMs:null};
const esc=(v)=>String(v??"").replace(/[&<>"']/g,(c)=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
const n=(v,d=3)=>Number.isFinite(Number(v))?Number(v).toFixed(d):"—";
const eventMs=(e)=>Number.isFinite(Number(e?.ts))?Number(e.ts)*1000:Date.now();
function duration(s){if(!Number.isFinite(s))return"—";s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),q=s%60,pad=(x)=>String(x).padStart(2,"0");return h?`${h}h ${pad(m)}m ${pad(q)}s`:m?`${m}m ${pad(q)}s`:`${q}s`;}
function updateProgress(p,ts){const prev=state.progress||{};const done=Number(p.completed)||0;if(!state.stageStart||prev.stage!==p.stage)state.stageStart={ms:ts,done:done};state.progress=p;const total=Number(p.total)||0,remaining=Math.max(0,total-done),doneInStage=done-(state.stageStart?state.stageStart.done:0);if(remaining<=0)state.etaFinishMs=ts;else if(doneInStage>=1)state.etaFinishMs=ts+((ts-state.stageStart.ms)/doneInStage)*remaining;else state.etaFinishMs=null;}
function updateClocks(){const now=state.finished||Date.now();$("stat-elapsed").textContent=duration(state.started?(now-state.started)/1000:NaN);let eta="—";if(state.finished)eta="done";else if(!state.started)eta="—";else if(state.progress&&Number(state.progress.total)&&(Number(state.progress.completed)||0)>=Number(state.progress.total))eta="finishing…";else if(state.etaFinishMs){const rem=(state.etaFinishMs-Date.now())/1000;eta=rem>0?duration(rem):"finishing…";}else eta="estimating…";$("stat-eta").textContent=eta;}
function log(text,ts){const li=document.createElement("li");li.innerHTML=`<span class="t">${new Date(ts||Date.now()).toLocaleTimeString()}</span><span class="kind note">S3.4</span><span class="msg">${esc(text)}</span>`;$("event-log").prepend(li);}
function applyEvent(e){const ts=eventMs(e);state.last=e.message||e.kind||e.type||"telemetry";
  if(e.type==="run_start"){state.config=e.config||{};state.started=ts;$("game-name").textContent=`S3.4 · ${state.config.protocol||"unknown"}`;log("locked run started",ts);}
  else if(e.type==="checkpoint"){
    if(e.step34_observation_row){const r=e.step34_observation_row,k=`${r.seed}:${r.label}`;if(!state.keys.has(k)){state.keys.add(k);state.rows.push(r);log(`${r.seed} · ${r.label} observation complete`,ts);}}
    if(e.step34_progress)updateProgress(e.step34_progress,ts);
    if(e.step34_observation)state.report=e.step34_observation;
    if(e.step34)state.report=e.step34;
  }else if(e.type==="milestone")log(e.message||e.kind,ts);
  else if(e.type==="run_end"){state.report=e.summary||state.report;state.finished=ts;$("done-overlay").hidden=false;$("done-title").textContent=state.report?.verdict||e.reason;$("done-detail").textContent=state.report?.failure||state.report?.safety_language||"Run ended.";if(!state.report?.all_pass){$("done-badge").textContent="!";$("done-badge").className="done-badge err";}}
  render();}
function render(){const p=state.progress,total=Number(p.total)||27,done=Number(p.completed)||state.rows.length,frac=Math.max(0,Math.min(1,done/total));
  $("stat-verdict").textContent=state.report?.verdict||"running";$("stat-verdict").className=state.report?(state.report.all_pass?"pass":"fail"):"";
  $("run-stage").textContent=p.stage||"observation";$("stat-stage").textContent=p.stage||"observation";$("stat-progress").textContent=`${done} / ${total}`;$("stat-workload").textContent=p.stage==="downstream"?"observation passed · downstream":"complete family before gate";
  $("stat-workers").textContent=state.config?.threads??"—";$("stat-protocol").textContent=state.config?.protocol??"—";$("stat-last-event").textContent=state.last||"—";
  $("run-progress-label").textContent=`${done} of ${total} ${p.stage||"observation"} cases`;$("run-progress-percent").textContent=`${(frac*100).toFixed(1)}%`;$("run-progress-bar").style.width=`${(frac*100).toFixed(1)}%`;
  updateClocks();
  $("stat-coverage").textContent=state.rows.length?`${n(100*Math.min(...state.rows.map(r=>r.effective_coverage)),2)}%`:"—";$("stat-ece").textContent=state.rows.length?n(Math.max(...state.rows.map(r=>r.ece)),4):"—";
  $("observation-table").querySelector("tbody").innerHTML=state.rows.map(r=>`<tr><td>${r.seed}</td><td>${esc(r.label)}</td><td class="${r.effective_coverage>=.99&&r.zero_weight_decisions===0?"pass":"fail"}">${n(100*r.effective_coverage,2)}%</td><td>${r.three_level_zero_decisions??"—"}</td><td>${r.recovered_three_level_zero_decisions??"—"}</td><td class="${r.zero_weight_decisions===0?"pass":"fail"}">${r.zero_weight_decisions??"—"}</td><td>${n(r.model_nll,4)}</td><td>${n(r.reference_nll,4)}</td><td>${n(r.shadow_nll,4)}</td><td class="${r.ece<=.03?"pass":"fail"}">${n(r.ece,4)}</td><td>${n(r.compute_s,0)}s</td><td>${n(r.peak_memory_gb,2)}</td></tr>`).join("");
  const gates=state.report?.gates;if(gates)$("gates-list").innerHTML=Object.entries(gates).map(([name,ok])=>`<li class="${ok?"pass":"fail"}">${ok?"PASS":"FAIL"} · ${esc(name)}</li>`).join("");}
const events=new EventSource("/events");events.onmessage=(m)=>JSON.parse(m.data).forEach(applyEvent);events.onerror=()=>log("telemetry reconnecting…");window.addEventListener("resize",render);$("done-dismiss").onclick=()=>{$("done-overlay").hidden=true;};if(typeof setInterval!=="undefined")setInterval(updateClocks,1000);render();

