"use strict";

const DATASET = "diabetes";
const PAGE_PARAMS = new URLSearchParams(window.location.search);
const DATA = window.EXPERIMENT_DATA;
const BUNDLE = DATA?.datasets?.[DATASET];
const STORAGE_KEY = "diabetes-experiment-self-test-runs-v1";
const CONDITIONS = {
    none: "No explanation",
    attribution: "Attribution",
    counterfactual: "Counterfactual",
};
const ANALYSIS_ID_BY_UI_ID = {
    130100: 160100,
    130101: 160101,
    130102: 160102,
    130103: 160103,
    130104: 160104,
    130105: 160105,
    130106: 160106,
    130107: 160107,
    130108: 160108,
    130109: 160111,
    130110: 160110,
    130111: 160109,
    130200: 160200,
    130201: 160201,
    130202: 160202,
    130203: 160203,
    130204: 160204,
    130205: 160210,
    130206: 160211,
    130207: 160212,
    130208: 160213,
    130209: 160214,
    130300: 160205,
    130301: 160206,
    130302: 160209,
    130303: 160207,
    130304: 160208,
    130305: 160215,
    130306: 160219,
    130307: 160216,
    130308: 160217,
    130309: 160218,
};

const elements = {
    content: document.querySelector("#content"),
    contentHead: document.querySelector("#content-head"),
    title: document.querySelector("#case-title"),
    subtitle: document.querySelector("#case-subtitle"),
    pill: document.querySelector("#condition-pill"),
    caseList: document.querySelector("#case-list"),
    actions: document.querySelector("#case-actions"),
    submitTest: document.querySelector("#submit-test"),
    next: document.querySelector("#next-case"),
    newRun: document.querySelector("#new-run"),
    downloadJson: document.querySelector("#download-json"),
    downloadCsv: document.querySelector("#download-csv"),
};

const state = {
    screen: "setup",
    condition: "attribution",
    cases: [],
    currentIndex: 0,
    caseStartedAt: null,
    trainingRecords: [],
    testingRecords: [],
    latestChanges: [],
    latestDisplayedValues: null,
    latestRawValues: null,
    editEvents: [],
    completedRun: null,
};

function displayedFeatureName(name) {
    return String(name) === "Blood Glucose" ? "Glucose" : String(name);
}

function latestIds(poolName) {
    const metadata = BUNDLE?.metadata ?? {};
    if (poolName === "training_pool") {
        return metadata.qualtrics_v1_6_training_ids || metadata.qualtrics_v1_4_training_ids || [];
    }
    const byPrediction = metadata.qualtrics_v1_6_testing_ids_by_prediction ||
        metadata.qualtrics_v1_4_testing_ids_by_prediction || {};
    return [
        ...(byPrediction["0"] || byPrediction[0] || []),
        ...(byPrediction["1"] || byPrediction[1] || []),
    ];
}

function latestPool(poolName, phase) {
    const wantedIds = new Set(latestIds(poolName).map(Number));
    const pool = BUNDLE?.[poolName] ?? [];
    if (!wantedIds.size) {
        return pool.filter((payload) => payload.experimental_phase === phase);
    }
    const selected = pool.filter((payload) => wantedIds.has(Number(payload.instance_id)));
    return selected.length ? selected : pool.filter((payload) => payload.experimental_phase === phase);
}

function latestTrainingBlocks() {
    const metadata = BUNDLE?.metadata ?? {};
    const blocks = metadata.qualtrics_v1_6_training_blocks || metadata.qualtrics_v1_4_training_blocks;
    if (!blocks) return [latestIds("training_pool")];
    return Object.values(blocks);
}

function latestTestingBlocks() {
    const metadata = BUNDLE?.metadata ?? {};
    const byPrediction = metadata.qualtrics_v1_6_testing_ids_by_prediction ||
        metadata.qualtrics_v1_4_testing_ids_by_prediction || {};
    return [byPrediction["0"] || byPrediction[0] || [], byPrediction["1"] || byPrediction[1] || []];
}

function shuffled(values) {
    const copy = [...values];
    for (let index = copy.length - 1; index > 0; index -= 1) {
        const swapIndex = Math.floor(Math.random() * (index + 1));
        [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
    }
    return copy;
}

function buildCases() {
    const trainingById = new Map(latestPool("training_pool", "training").map((payload) => [Number(payload.instance_id), payload]));
    let blockedTraining = shuffled(latestTrainingBlocks())
        .flatMap((block) => shuffled(block.map(Number).map((id) => trainingById.get(id)).filter(Boolean)));
    if (!blockedTraining.length) {
        blockedTraining = shuffled(latestPool("training_pool", "training"));
    }
    const training = blockedTraining.map((payload, index) => ({
        phase: "training",
        phaseIndex: index,
        payload,
    }));
    const testingById = new Map(latestPool("test_pool", "testing").map((payload) => [Number(payload.instance_id), payload]));
    let directionBlocks = shuffled(latestTestingBlocks().map((block) =>
        shuffled(block.map(Number).map((id) => testingById.get(id)).filter(Boolean))
    ));
    if (!directionBlocks.flat().length) {
        const testingPool = latestPool("test_pool", "testing");
        directionBlocks = shuffled([0, 1].map((prediction) =>
            shuffled(testingPool.filter((payload) => Number(payload.prediction.value) === prediction))
        ));
    }
    const testing = directionBlocks.flat().map((payload, index) => ({
        phase: "testing",
        phaseIndex: index,
        payload,
    }));
    return [...training, ...testing];
}

function caseKey(item) {
    return `${item.phase}:${item.payload.instance_id}`;
}

function analysisId(payload) {
    return ANALYSIS_ID_BY_UI_ID[Number(payload.instance_id)] || Number(payload.instance_id);
}

function featurePair(payload) {
    return payload.feature_pair_names?.join(" + ") || payload.feature_pair_key || "not listed";
}

function completedKeys() {
    return new Set([
        ...state.trainingRecords.map((record) => record.caseKey),
        ...state.testingRecords.map((record) => record.caseKey),
    ]);
}

function renderCaseList() {
    elements.caseList.innerHTML = "";
    if (!state.cases.length) state.cases = buildCases();
    const trainingCount = state.cases.filter((item) => item.phase === "training").length;
    const testingCount = state.cases.filter((item) => item.phase === "testing").length;
    document.querySelector("#case-summary").textContent = `${trainingCount} training · ${testingCount} testing`;
    const complete = completedKeys();
    ["training", "testing"].forEach((phase) => {
        const title = document.createElement("div");
        title.className = "case-group-title";
        const count = state.cases.filter((item) => item.phase === phase).length;
        title.textContent = `${phase} · ${count} cases`;
        elements.caseList.appendChild(title);
        state.cases.forEach((item, absoluteIndex) => {
            if (item.phase !== phase) return;
            const button = document.createElement("button");
            button.type = "button";
            button.className = "case-item";
            if (state.screen === "run" && absoluteIndex === state.currentIndex) button.classList.add("current");
            if (complete.has(caseKey(item))) button.classList.add("complete");
            button.disabled = state.screen === "run" && absoluteIndex !== state.currentIndex;
            button.innerHTML = [
                `<span class="case-number">${item.phase === "training" ? "T" : "S"}${item.phaseIndex + 1}</span>`,
                `<span class="case-id">UI ${item.payload.instance_id} · v2.0 ${analysisId(item.payload)}</span>`,
                `<span class="case-status">${complete.has(caseKey(item)) ? "✓" : "○"}</span>`,
            ].join("");
            elements.caseList.appendChild(button);
        });
    });
}

function loadHistory() {
    try {
        const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
}

function saveRun(run) {
    const history = loadHistory();
    history.push(run);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(-20)));
}

function historyMarkup() {
    const history = loadHistory();
    if (!history.length) return "";
    return `
      <section class="history">
        <h3>Previous completed runs</h3>
        <div class="table-wrap"><table>
          <thead><tr><th>Date</th><th>Condition</th><th>Training</th><th>Valid flips</th><th>Successful proximity</th></tr></thead>
          <tbody>${history.slice().reverse().map((run) => `
            <tr>
              <td>${new Date(run.completedAt).toLocaleString()}</td>
              <td>${CONDITIONS[run.condition] ?? run.condition}</td>
              <td>${formatPercent(run.summary.trainingAccuracy)}</td>
              <td>${formatPercent(run.summary.validity)}</td>
              <td>${formatNumber(run.summary.successfulProximity)}</td>
            </tr>`).join("")}</tbody>
        </table></div>
      </section>`;
}

function renderSetup() {
    state.screen = "setup";
    elements.contentHead.hidden = true;
    elements.actions.hidden = true;
    elements.newRun.hidden = true;
    elements.downloadJson.hidden = true;
    elements.downloadCsv.hidden = true;
    const trainingCount = latestPool("training_pool", "training").length;
    const testingCount = latestPool("test_pool", "testing").length;
    elements.content.innerHTML = `
      <section class="setup">
        <h2>Try the experiment yourself</h2>
        <p>You will first predict the AI output for the current ${trainingCount} training profiles and review the feedback available in your selected condition. You will then edit the current ${testingCount} testing profiles. Testing feedback is withheld until the final results, as in the participant experiment.</p>
        <div class="setup-grid">
          <div class="setup-box"><strong>Training · ${trainingCount} cases</strong><p>Guess Diabetes or No Diabetes, then review the correct prediction and condition-specific explanation.</p></div>
          <div class="setup-box"><strong>Testing · ${testingCount} cases</strong><p>Make the smallest changes you think will reverse each AI prediction. Every submitted edit is recorded locally.</p></div>
        </div>
        <label class="control">Explanation condition
          <select id="condition-select">
            <option value="none">No explanation</option>
            <option value="attribution" selected>Attribution</option>
            <option value="counterfactual">Counterfactual</option>
          </select>
        </label>
        <button id="start-run" class="primary" type="button">Start self-test</button>
        ${historyMarkup()}
      </section>`;
    const condition = document.querySelector("#condition-select");
    condition.value = state.condition;
    condition.addEventListener("change", () => { state.condition = condition.value; });
    document.querySelector("#start-run").addEventListener("click", startRun);
    renderCaseList();
}

function startRun() {
    state.screen = "run";
    state.cases = buildCases();
    state.currentIndex = 0;
    state.trainingRecords = [];
    state.testingRecords = [];
    state.latestChanges = [];
    state.latestDisplayedValues = null;
    state.latestRawValues = null;
    state.editEvents = [];
    state.completedRun = null;
    renderCurrentCase();
}

function iframeUrl(item, options) {
    const query = new URLSearchParams({
        appId: DATASET,
        xaiType: options.xaiType,
        split: item.phase === "training" ? "train" : "test",
        instanceId: String(item.payload.instance_id),
        showPrediction: options.showPrediction ? "1" : "0",
        counterfactualSimulation: options.simulation ? "1" : "0",
        simulationMode: "any",
    });
    if (PAGE_PARAMS.get("dataPreview")) {
        query.set("dataPreview", PAGE_PARAMS.get("dataPreview"));
    }
    if (options.simulationValues) query.set("simulationValues", JSON.stringify(options.simulationValues));
    return `../iframe.html?${query}`;
}

function makeIframe(item, options) {
    const iframe = document.createElement("iframe");
    iframe.id = "experiment-frame";
    iframe.className = "case-frame";
    iframe.title = `${item.phase} case ${item.phaseIndex + 1}`;
    iframe.src = iframeUrl(item, options);
    return iframe;
}

function renderCurrentCase() {
    const item = state.cases[state.currentIndex];
    if (!item) return finishRun();
    state.caseStartedAt = performance.now();
    state.latestChanges = [];
    state.latestDisplayedValues = null;
    state.latestRawValues = null;
    elements.contentHead.hidden = false;
    elements.actions.hidden = false;
    elements.pill.textContent = CONDITIONS[state.condition];
    elements.title.textContent = `${item.phase === "training" ? "Training" : "Testing"} case ${item.phaseIndex + 1}`;
    elements.subtitle.textContent = `UI ${item.payload.instance_id} · v2.0 ${analysisId(item.payload)} · ${featurePair(item.payload)} · ${state.currentIndex + 1} of ${state.cases.length}`;
    elements.content.innerHTML = "";
    elements.submitTest.hidden = true;
    elements.next.hidden = true;

    if (item.phase === "training") renderTrainingCase(item);
    else renderTestingCase(item);
    renderCaseList();
}

function renderTrainingCase(item) {
    elements.content.appendChild(makeIframe(item, {
        xaiType: "none",
        showPrediction: false,
        simulation: false,
    }));
    const panel = document.createElement("div");
    panel.className = "question-panel";
    panel.innerHTML = "<p>What warning do you think the AI issues for this profile?</p>";
    const answers = document.createElement("div");
    answers.className = "answers";
    (item.payload.prediction_labels ?? ["Diabetes", "No Diabetes"]).forEach((label, value) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "answer-button";
        button.textContent = label;
        button.addEventListener("click", () => submitTrainingAnswer(item, value, panel));
        answers.appendChild(button);
    });
    panel.appendChild(answers);
    elements.content.appendChild(panel);
}

function submitTrainingAnswer(item, selectedValue, panel) {
    if (state.trainingRecords.some((record) => record.caseKey === caseKey(item))) return;
    const correctValue = Number(item.payload.prediction.value);
    const correct = selectedValue === correctValue;
    state.trainingRecords.push({
        caseKey: caseKey(item),
        phase: "training",
        case: item.phaseIndex + 1,
        instanceId: item.payload.instance_id,
        analysisInstanceId: analysisId(item.payload),
        condition: state.condition,
        selectedValue,
        selectedLabel: item.payload.prediction_labels[selectedValue],
        correctValue,
        correctLabel: item.payload.prediction.label,
        correct: Number(correct),
        responseMs: Math.round(performance.now() - state.caseStartedAt),
    });
    panel.querySelectorAll("button").forEach((button, index) => {
        button.disabled = true;
        button.classList.toggle("selected", index === selectedValue);
    });
    const feedback = document.createElement("div");
    feedback.className = `feedback ${correct ? "correct" : "incorrect"}`;
    feedback.textContent = correct
        ? `Correct. The AI predicts ${item.payload.prediction.label}.`
        : `Incorrect. The AI predicts ${item.payload.prediction.label}.`;
    panel.appendChild(feedback);
    const oldFrame = document.querySelector("#experiment-frame");
    oldFrame.replaceWith(makeIframe(item, {
        xaiType: state.condition,
        showPrediction: true,
        simulation: false,
    }));
    elements.next.hidden = false;
    elements.next.textContent = state.currentIndex === state.cases.length - 1 ? "View results" : "Next case";
    renderCaseList();
}

function renderTestingCase(item) {
    elements.content.appendChild(makeIframe(item, {
        xaiType: "none",
        showPrediction: true,
        simulation: true,
    }));
    const note = document.createElement("div");
    note.id = "testing-note";
    note.className = "feedback neutral";
    note.textContent = "Make at least one change. Your result will be shown after all testing cases.";
    elements.content.appendChild(note);
    elements.submitTest.hidden = false;
    elements.submitTest.disabled = true;
}

function rawObject(payload, values = payload.raw_feature_values) {
    return Object.fromEntries(payload.raw_feature_names.map((name, index) => [name, values[index]]));
}

function numericDifferent(first, second) {
    const a = Number(first);
    const b = Number(second);
    return Number.isFinite(a) && Number.isFinite(b) ? Math.abs(a - b) > 1e-9 : String(first) !== String(second);
}

function normalizedDifference(payload, first, second, index) {
    const range = payload.raw_feature_ranges[index];
    if (payload.feature_types[index] === "categorical") {
        return String(first) === String(second) ? 0 : 1;
    }
    const span = Number(range[1]) - Number(range[0]);
    return span > 0 ? Math.abs(Number(first) - Number(second)) / span : 0;
}

function subsetPlausibility(payload, changedValues) {
    const distances = latestPool("training_pool", "training").map((training) => {
        const distance = payload.raw_feature_names.reduce((sum, name, index) => {
            return sum + normalizedDifference(payload, changedValues[name], training.raw_feature_values[index], index);
        }, 0) / payload.raw_feature_names.length;
        return distance;
    });
    return 1 - Math.min(...distances);
}

function scoreTestingCase(item) {
    const payload = item.payload;
    const original = rawObject(payload);
    const changed = state.latestRawValues;
    const changedIndices = payload.raw_feature_names
        .map((name, index) => numericDifferent(original[name], changed[name]) ? index : -1)
        .filter((index) => index >= 0);
    const proximity = changedIndices.reduce((sum, index) => {
        const name = payload.raw_feature_names[index];
        return sum + normalizedDifference(payload, original[name], changed[name], index);
    }, 0);
    const prediction = window.StaticModel.predictDataset(DATA, DATASET, changed);
    const originalPrediction = Number(payload.prediction.value);
    const targetPrediction = 1 - originalPrediction;
    const valid = Number(prediction.value) === targetPrediction;
    const targetProbability = Number(prediction.probabilities[targetPrediction]);
    const minimal = rawObject(payload, payload.counterfactual.raw_feature_values ?? payload.counterfactual.feature_values);
    const distanceToBundledMinimal = payload.raw_feature_names.reduce((sum, name, index) => {
        return sum + normalizedDifference(payload, changed[name], minimal[name], index);
    }, 0);
    const ageIndex = payload.raw_feature_names.indexOf("age");
    const actionability = ageIndex < 0 || !changedIndices.includes(ageIndex) ? 1 : 0;
    const editEventCount = state.editEvents.filter((event) => event.caseKey === caseKey(item)).length;

    return {
        caseKey: caseKey(item),
        phase: "testing",
        case: item.phaseIndex + 1,
        instanceId: payload.instance_id,
        analysisInstanceId: analysisId(payload),
        condition: state.condition,
        originalPrediction,
        originalLabel: payload.prediction.label,
        targetPrediction,
        targetLabel: payload.prediction_labels[targetPrediction],
        changedPrediction: Number(prediction.value),
        changedLabel: prediction.label,
        valid: Number(valid),
        changedAttributeCount: changedIndices.length,
        changedAttributes: changedIndices.map((index) => displayedFeatureName(payload.feature_names[index])).join(" | "),
        proximity,
        sparsity: 1 - changedIndices.length / payload.raw_feature_names.length,
        actionability,
        subsetPlausibility: subsetPlausibility(payload, changed),
        targetProbability,
        probabilityBoundaryGap: Math.abs(targetProbability - 0.5),
        distanceToBundledMinimal,
        editEventCount,
        originalValues: original,
        changedValues: { ...changed },
        changes: state.latestChanges,
        responseMs: Math.round(performance.now() - state.caseStartedAt),
    };
}

function submitTestingCase() {
    const item = state.cases[state.currentIndex];
    if (!item || item.phase !== "testing" || !state.latestRawValues || !state.latestChanges.length) return;
    state.testingRecords.push(scoreTestingCase(item));
    elements.submitTest.hidden = true;
    elements.next.hidden = false;
    elements.next.textContent = state.currentIndex === state.cases.length - 1 ? "View results" : "Next case";
    const note = document.querySelector("#testing-note");
    note.textContent = "Changes recorded. The prediction result remains hidden until the end.";
    document.querySelector("#experiment-frame").style.pointerEvents = "none";
    renderCaseList();
}

function average(records, property) {
    const values = records.map((record) => Number(record[property])).filter(Number.isFinite);
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function summarizeRun() {
    const successful = state.testingRecords.filter((record) => record.valid === 1);
    return {
        trainingAccuracy: average(state.trainingRecords, "correct"),
        validity: average(state.testingRecords, "valid"),
        successfulProximity: average(successful, "proximity"),
        successfulChangedAttributes: average(successful, "changedAttributeCount"),
        successfulSparsity: average(successful, "sparsity"),
        actionability: average(state.testingRecords, "actionability"),
        successfulSubsetPlausibility: average(successful, "subsetPlausibility"),
        successfulBoundaryGap: average(successful, "probabilityBoundaryGap"),
        successfulDistanceToBundledMinimal: average(successful, "distanceToBundledMinimal"),
    };
}

function finishRun() {
    state.screen = "results";
    const run = {
        version: DATA?.version,
        dataset: DATASET,
        condition: state.condition,
        completedAt: new Date().toISOString(),
        summary: summarizeRun(),
        trainingRecords: state.trainingRecords,
        testingRecords: state.testingRecords,
        editEvents: state.editEvents,
    };
    state.completedRun = run;
    saveRun(run);
    renderResults();
}

function formatPercent(value) {
    return value === null || value === undefined ? "—" : `${Math.round(value * 100)}%`;
}

function formatNumber(value, digits = 3) {
    return value === null || value === undefined || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(digits);
}

function renderResults() {
    const run = state.completedRun;
    const summary = run.summary;
    elements.contentHead.hidden = false;
    elements.title.textContent = "Your results";
    const trainingCount = state.cases.filter((item) => item.phase === "training").length;
    const testingCount = state.cases.filter((item) => item.phase === "testing").length;
    elements.subtitle.textContent = `${CONDITIONS[state.condition]} · ${trainingCount} training and ${testingCount} testing cases`;
    elements.pill.textContent = CONDITIONS[state.condition];
    elements.actions.hidden = true;
    elements.newRun.hidden = false;
    elements.downloadJson.hidden = false;
    elements.downloadCsv.hidden = false;
    elements.content.innerHTML = `
      <section class="results">
        <h2>Performance summary</h2>
        <div class="metrics">
          ${metricMarkup(formatPercent(summary.trainingAccuracy), "Training accuracy")}
          ${metricMarkup(formatPercent(summary.validity), "Valid counterfactuals")}
          ${metricMarkup(formatNumber(summary.successfulProximity), "Proximity among valid edits · lower is better")}
          ${metricMarkup(formatNumber(summary.successfulChangedAttributes, 2), "Attributes changed among valid edits")}
          ${metricMarkup(formatPercent(summary.actionability), "Actionable edits · Age unchanged")}
          ${metricMarkup(formatNumber(summary.successfulSubsetPlausibility), "Training-subset plausibility · higher is better")}
        </div>
        <p class="result-note">Proximity is the sum of range-normalized changes. Boundary gap is the successful target probability’s distance from 0.50; lower values mean less overshooting. Plausibility is one minus the nearest average Gower distance to the 12 displayed training profiles.</p>
        <h3>Testing cases</h3>
        <div class="table-wrap"><table>
          <thead><tr><th>Case</th><th>Instance</th><th>Direction</th><th>Predicted after edit</th><th>Target probability</th><th>Result</th><th>Changed</th><th>Proximity</th><th>Sparsity</th><th>Actionable</th><th>Plausibility</th><th>Boundary gap</th></tr></thead>
          <tbody>${run.testingRecords.map((record) => `
            <tr>
              <td>${record.case}</td><td>${record.instanceId}</td><td>${record.originalLabel} → ${record.targetLabel}</td>
              <td>${record.changedLabel}</td><td>${formatPercent(record.targetProbability)}</td>
              <td class="${record.valid ? "good" : "bad"}">${record.valid ? "Valid" : "Not flipped"}</td>
              <td title="${record.changedAttributes}">${record.changedAttributeCount}</td>
              <td>${formatNumber(record.proximity)}</td><td>${formatNumber(record.sparsity)}</td>
              <td class="${record.actionability ? "good" : "bad"}">${record.actionability ? "Yes" : "No"}</td>
              <td>${formatNumber(record.subsetPlausibility)}</td><td>${formatNumber(record.probabilityBoundaryGap)}</td>
            </tr>`).join("")}</tbody>
        </table></div>
      </section>`;
    renderCaseList();
}

function metricMarkup(value, label) {
    return `<div class="metric"><div class="metric-value">${value}</div><div class="metric-label">${label}</div></div>`;
}

function download(name, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

function csvCell(value) {
    const text = typeof value === "object" ? JSON.stringify(value) : String(value ?? "");
    return `"${text.replaceAll('"', '""')}"`;
}

function recordEditEvent(eventType, item, data) {
    state.editEvents.push({
        eventNumber: state.editEvents.length + 1,
        eventType,
        recordedAt: new Date().toISOString(),
        elapsedMs: Math.round(performance.now() - state.caseStartedAt),
        caseKey: caseKey(item),
        phase: item.phase,
        case: item.phaseIndex + 1,
        instanceId: item.payload.instance_id,
        analysisInstanceId: analysisId(item.payload),
        condition: state.condition,
        changes: Array.isArray(data.changes) ? data.changes : [],
        changedDisplayedValues: Array.isArray(data.changedDisplayedValues) ? data.changedDisplayedValues : null,
        changedRawFeatureValues: data.changedRawFeatureValues ? { ...data.changedRawFeatureValues } : null,
    });
}

function downloadCsv() {
    const rows = state.completedRun.testingRecords;
    const columns = [
        "condition", "case", "instanceId", "originalLabel", "targetLabel", "changedLabel", "valid",
        "changedAttributeCount", "changedAttributes", "proximity", "sparsity", "actionability",
        "subsetPlausibility", "targetProbability", "probabilityBoundaryGap", "distanceToBundledMinimal",
        "editEventCount",
        "originalValues", "changedValues", "responseMs",
    ];
    const csv = [columns.map(csvCell).join(","), ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(","))].join("\r\n");
    download(`diabetes-self-test-${state.condition}.csv`, csv, "text/csv;charset=utf-8");
}

window.addEventListener("message", (event) => {
    const iframe = document.querySelector("#experiment-frame");
    if (!iframe || event.source !== iframe.contentWindow || !event.data) return;
    if (event.data.type === "counterfactual-ui:iframe-height") {
        iframe.style.height = `${Math.max(440, Math.min(900, Number(event.data.height) || 0))}px`;
        return;
    }
    if (event.data.type === "counterfactual-ui:simulation-change") {
        const item = state.cases[state.currentIndex];
        if (item?.phase === "testing") recordEditEvent("simulation-change", item, event.data);
        state.latestChanges = Array.isArray(event.data.changes) ? event.data.changes : [];
        state.latestDisplayedValues = Array.isArray(event.data.changedDisplayedValues)
            ? [...event.data.changedDisplayedValues]
            : null;
        state.latestRawValues = event.data.changedRawFeatureValues
            ? { ...event.data.changedRawFeatureValues }
            : null;
        elements.submitTest.disabled = state.latestChanges.length === 0;
        const note = document.querySelector("#testing-note");
        if (note) note.textContent = state.latestChanges.length
            ? `${state.latestChanges.length} attribute${state.latestChanges.length === 1 ? "" : "s"} changed. Submit when ready.`
            : "Make at least one change. Your result will be shown after all testing cases.";
    }
});

elements.submitTest.addEventListener("click", submitTestingCase);
elements.next.addEventListener("click", () => {
    state.currentIndex += 1;
    renderCurrentCase();
});
elements.newRun.addEventListener("click", renderSetup);
elements.downloadJson.addEventListener("click", () => {
    download(`diabetes-self-test-${state.condition}.json`, JSON.stringify(state.completedRun, null, 2), "application/json");
});
elements.downloadCsv.addEventListener("click", downloadCsv);

if (!BUNDLE || !window.StaticModel) {
    elements.content.innerHTML = "<p class='bad'>The static diabetes experiment bundle or browser model could not be loaded.</p>";
} else {
    state.cases = buildCases();
    renderSetup();
}
