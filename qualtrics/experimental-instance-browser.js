"use strict";

const ui = Object.fromEntries(["phase","explanation","previous","next","open-case","summary","cases","preview"].map((id) => [id.replace("-", ""), document.getElementById(id)]));
const query = new URLSearchParams(location.search);
ui.phase.value = query.get("phase") === "testing" ? "testing" : "training";
ui.explanation.value = ["counterfactual","attribution","none"].includes(query.get("explanation")) ? query.get("explanation") : "counterfactual";
let selected = Math.max(0, Number(query.get("case")) || 0);

function dataset(){ return window.EXPERIMENTAL_ACTIONABLE_PREVIEW_DATA?.datasets?.diabetes; }
function cases(){ const data=dataset(); return ui.phase.value === "training" ? data.training_pool : data.test_pool; }
function number(value, digits=3){ return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—"; }
function target(payload){ return dataset().labels[1-Number(payload.prediction.value)]; }
function changeText(payload){
  const original=payload.raw_feature_values, edited=payload.counterfactual.raw_feature_values;
  return payload.counterfactual.raw_selected_feature_names.map((name) => {
    const index=payload.raw_feature_names.indexOf(name), delta=Number(edited[index])-Number(original[index]);
    const range=payload.raw_feature_ranges[index], normalized=Math.abs(delta)/(Number(range[1])-Number(range[0]));
    return `${payload.feature_names[index]} (${delta>=0?"+":""}${Number(delta.toFixed(2))}; ${number(normalized,2)} range)`;
  }).join(" + ");
}
function caseUrl(payload){
  const testing=ui.phase.value === "testing";
  const params=new URLSearchParams({appId:"diabetes",xaiType:testing?"none":ui.explanation.value,split:testing?"test":"train",instanceId:String(payload.instance_id),showPrediction:"1",counterfactualSimulation:testing?"1":"0"});
  if(testing){ params.set("immutableFeatures","glucose"); params.set("maxChangedFeatures","1"); }
  return `../experimental-iframe.html?${params}`;
}
function render(){
  const rows=cases(); selected=Math.max(0,Math.min(rows.length-1,selected)); const payload=rows[selected]; const opt=payload.counterfactual.optimization; const testing=ui.phase.value === "testing";
  ui.summary.innerHTML=[
    `<div><strong>Case</strong>${selected+1} of ${rows.length} <span class="badge">cluster ${payload.selection_cluster}</span></div>`,
    `<div><strong>Direction</strong>${payload.prediction.label} → ${target(payload)}</div>`,
    `<div><strong>${testing?"Best feasible edit":"Counterfactual"}</strong>${changeText(payload)}</div>`,
    `<div><strong>Normalized L1</strong>${number(opt.normalized_l1)}</div>`,
    `<div><strong>Target-class 3NN distance</strong>${number(opt.target_class_3nn_gower)}</div>`,
    `<div><strong>Combined objective</strong>${number(opt.objective_value)}</div>`,
    testing ? `<div class="constraint"><strong>Test constraint</strong>Glucose is fixed. The edited attribute must move by at least 0.10 of its range. Changing one attribute locks all remaining attributes; returning it to the original value unlocks them.</div>` : `<div class="constraint"><strong>Training constraint</strong>Exactly two actionable attributes change, each by 0.15–0.25 of its range; Age is excluded. Two balanced, disjoint pair families provide diversity.</div>`,
  ].join("");
  ui.cases.innerHTML="";
  rows.forEach((row,index)=>{ const tr=document.createElement("tr"); if(index===selected)tr.className="selected"; const values=[index+1,row.instance_id,row.selection_cluster,row.prediction.label,changeText(row),number(row.counterfactual.optimization.objective_value)]; values.forEach((value)=>{const td=document.createElement("td");td.textContent=value;tr.appendChild(td)}); tr.onclick=()=>{selected=index;render()};ui.cases.appendChild(tr)});
  const url=caseUrl(payload); ui.preview.src=url; ui.opencase.href=url; ui.previous.disabled=selected===0;ui.next.disabled=selected===rows.length-1;
  history.replaceState(null,"",`?phase=${ui.phase.value}&explanation=${ui.explanation.value}&case=${selected}`);
}
ui.phase.onchange=()=>{selected=0;render()};ui.explanation.onchange=render;ui.previous.onclick=()=>{selected--;render()};ui.next.onclick=()=>{selected++;render()};
window.addEventListener("message",(event)=>{if(event.source===ui.preview.contentWindow&&event.data?.type==="counterfactual-ui:iframe-height")ui.preview.style.height=`${Math.max(500,Math.min(900,Number(event.data.height)||0))}px`});
render();
