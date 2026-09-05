"use strict";

const controls = {
    dataset: document.querySelector("#dataset"),
    split: document.querySelector("#split"),
    explanation: document.querySelector("#explanation"),
    previous: document.querySelector("#previous"),
    next: document.querySelector("#next"),
    openCase: document.querySelector("#open-case"),
    summary: document.querySelector("#summary"),
    attempts: document.querySelector("#attempts"),
    cases: document.querySelector("#cases"),
    preview: document.querySelector("#preview"),
};

const params = new URLSearchParams(window.location.search);
const CURRENT_DATASET = "diabetes";
const STORAGE_KEY = "diabetes-current-interface-self-test-attempts-v1";
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

controls.dataset.value = CURRENT_DATASET;
controls.split.value = params.get("split") === "test" ? "test" : "train";
controls.explanation.value = ["attribution", "counterfactual", "none"].includes(params.get("explanation"))
    ? params.get("explanation")
    : "counterfactual";
let selectedIndex = Math.max(0, Number(params.get("case")) || 0);
let messageSequence = 0;

function bundle() {
    return window.EXPERIMENT_DATA?.datasets?.[CURRENT_DATASET];
}

function currentIds() {
    const metadata = bundle()?.metadata || {};
    if (controls.split.value === "train") {
        return metadata.qualtrics_v1_6_training_ids || metadata.qualtrics_v1_4_training_ids || [];
    }
    const byPrediction = metadata.qualtrics_v1_6_testing_ids_by_prediction ||
        metadata.qualtrics_v1_4_testing_ids_by_prediction || {};
    return [
        ...(byPrediction["0"] || byPrediction[0] || []),
        ...(byPrediction["1"] || byPrediction[1] || []),
    ];
}

function pool() {
    const data = bundle();
    if (!data) return [];
    const source = controls.split.value === "train" ? data.training_pool : data.test_pool;
    const byId = new Map((source || []).map((payload) => [Number(payload.instance_id), payload]));
    const ids = currentIds().map(Number);
    const selected = ids.map((id) => byId.get(id)).filter(Boolean);
    return selected.length ? selected : source || [];
}

function analysisId(payload) {
    return ANALYSIS_ID_BY_UI_ID[Number(payload.instance_id)] || "";
}

function targetLabel(payload) {
    const labels = bundle()?.labels || ["Diabetes", "No Diabetes"];
    const original = Number(payload.prediction?.value);
    return labels[1 - original] || String(1 - original);
}

function directionLabel(payload) {
    return `${payload.prediction?.label || "original"} to ${targetLabel(payload)}`;
}

function caseUrl(payload) {
    const query = new URLSearchParams({
        appId: CURRENT_DATASET,
        xaiType: controls.split.value === "test" ? "none" : controls.explanation.value,
        split: controls.split.value,
        instanceId: String(payload.instance_id),
        showPrediction: "1",
        counterfactualSimulation: controls.split.value === "test" ? "1" : "0",
    });
    if (params.get("dataPreview")) query.set("dataPreview", params.get("dataPreview"));
    return `../iframe.html?${query}`;
}

function updateUrl() {
    const query = new URLSearchParams({
        dataset: CURRENT_DATASET,
        split: controls.split.value,
        explanation: controls.explanation.value,
        case: String(selectedIndex),
    });
    if (params.get("dataPreview")) query.set("dataPreview", params.get("dataPreview"));
    history.replaceState(null, "", `?${query}`);
}

function readLogs() {
    try {
        const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
        return Array.isArray(parsed) ? parsed : [];
    } catch (_error) {
        return [];
    }
}

function writeLogs(logs) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(logs));
}

function logsForCurrentCondition(logs = readLogs()) {
    return logs.filter((row) => row.condition === controls.explanation.value);
}

function logsForSelectedCase(payload, logs = readLogs()) {
    return logs.filter((row) =>
        row.condition === controls.explanation.value &&
        Number(row.uiInstanceId) === Number(payload.instance_id)
    );
}

function describeLastAttempt(logs) {
    const latest = [...logs].reverse().find((row) => row.eventType === "simulation-change");
    if (!latest) return "No saved edits for this selected case yet.";
    const changes = latest.payload?.changes || [];
    if (!changes.length) return "Last saved edit had no changed attributes.";
    return changes.map((change) => {
        const before = Number(change.originalValue);
        const after = Number(change.newValue);
        const direction = Number.isFinite(before) && Number.isFinite(after)
            ? after > before ? "increased" : after < before ? "decreased" : "unchanged"
            : "changed";
        return `${change.attributeName} ${direction}`;
    }).join("; ");
}

function downloadLogs() {
    const logs = logsForCurrentCondition();
    const blob = new Blob([JSON.stringify({
        exportedAt: new Date().toISOString(),
        dataset: CURRENT_DATASET,
        condition: controls.explanation.value,
        note: "Testing iframes hide explanations; condition records the training condition selected in this browser.",
        attempts: logs,
    }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `diabetes-self-test-${controls.explanation.value}-attempts.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
}

function clearConditionLogs() {
    const logs = readLogs();
    writeLogs(logs.filter((row) => row.condition !== controls.explanation.value));
    render();
}

function renderAttemptPanel(payload) {
    const allLogs = readLogs();
    const conditionLogs = logsForCurrentCondition(allLogs);
    const selectedLogs = logsForSelectedCase(payload, allLogs);
    controls.attempts.innerHTML = [
        `<div><strong>Recorded condition</strong>${controls.explanation.value}</div>`,
        `<div><strong>Condition log</strong>${conditionLogs.length} event(s)</div>`,
        `<div><strong>This case</strong>${selectedLogs.length} event(s)</div>`,
        `<div><strong>Last edit</strong>${describeLastAttempt(selectedLogs)}</div>`,
        `<div class="attempt-actions">` +
            `<button id="download-attempts" type="button">Download condition log</button>` +
            `<button id="clear-attempts" type="button">Clear condition log</button>` +
        `</div>`,
    ].join("");
    document.querySelector("#download-attempts").addEventListener("click", downloadLogs);
    document.querySelector("#clear-attempts").addEventListener("click", clearConditionLogs);
}

function render() {
    const cases = pool();
    if (!Array.isArray(cases) || cases.length === 0) {
        controls.summary.innerHTML = '<div class="empty">No current diabetes cases found. Regenerate static experiment data.</div>';
        return;
    }
    selectedIndex = Math.max(0, Math.min(cases.length - 1, selectedIndex));
    const payload = cases[selectedIndex];
    const phase = controls.split.value === "train" ? "Training" : "Testing";
    const pair = payload.feature_pair_names?.join(" + ") || payload.feature_pair_key || "—";
    const testingNote = controls.split.value === "test"
        ? `<div><strong>Target</strong>${directionLabel(payload)}</div>`
        : "";
    controls.summary.innerHTML = [
        `<div><strong>Case</strong>${selectedIndex + 1} of ${cases.length}</div>`,
        `<div><strong>UI ID</strong>${payload.instance_id}</div>`,
        `<div><strong>v2.0 ID</strong>${analysisId(payload) || "not mapped"}</div>`,
        `<div><strong>AI label</strong>${payload.prediction.label}</div>`,
        testingNote,
        `<div><strong>Phase</strong>${phase}</div>`,
        `<div><strong>Top pair</strong>${pair}</div>`,
    ].join("");
    renderAttemptPanel(payload);

    controls.cases.innerHTML = "";
    cases.forEach((candidate, index) => {
        const row = document.createElement("tr");
        if (index === selectedIndex) row.className = "selected";
        const candidatePair = candidate.feature_pair_names?.join(" + ") || candidate.feature_pair_key || "—";
        [
            index + 1,
            candidate.instance_id,
            analysisId(candidate) || "",
            candidate.prediction.label,
            candidatePair,
        ].forEach((value) => {
            const cell = document.createElement("td");
            cell.textContent = String(value);
            row.appendChild(cell);
        });
        row.addEventListener("click", () => {
            selectedIndex = index;
            render();
        });
        controls.cases.appendChild(row);
    });
    const url = caseUrl(payload);
    controls.preview.src = url;
    controls.openCase.href = url;
    controls.previous.disabled = selectedIndex === 0;
    controls.next.disabled = selectedIndex === cases.length - 1;
    updateUrl();
}

function selectedPayload() {
    return pool()[selectedIndex] || null;
}

function recordIframeEvent(eventType, payload) {
    if (controls.split.value !== "test") return;
    const selected = selectedPayload();
    if (!selected) return;
    const logs = readLogs();
    logs.push({
        sequence: ++messageSequence,
        recordedAt: new Date().toISOString(),
        phase: controls.split.value,
        condition: controls.explanation.value,
        uiInstanceId: Number(selected.instance_id),
        analysisInstanceId: analysisId(selected) || null,
        casePosition: selectedIndex + 1,
        direction: directionLabel(selected),
        eventType,
        payload,
    });
    writeLogs(logs);
    renderAttemptPanel(selected);
}

controls.split.addEventListener("change", () => {
    selectedIndex = 0;
    render();
});
controls.explanation.addEventListener("change", render);
controls.previous.addEventListener("click", () => {
    selectedIndex -= 1;
    render();
});
controls.next.addEventListener("click", () => {
    selectedIndex += 1;
    render();
});
window.addEventListener("message", (event) => {
    if (event.source !== controls.preview.contentWindow || !event.data) return;
    if (event.data.type === "counterfactual-ui:iframe-height") {
        controls.preview.style.height = `${Math.max(420, Math.min(900, Number(event.data.height) || 0))}px`;
        return;
    }
    if (event.data.type === "counterfactual-ui:simulation-change") {
        recordIframeEvent("simulation-change", {
            changes: event.data.changes || [],
            changedRawFeatureValues: event.data.changedRawFeatureValues || null,
        });
        return;
    }
    if (event.data.type === "counterfactual-ui:simulation-feedback") {
        recordIframeEvent("simulation-feedback", {
            feedback: event.data.feedback || null,
            prediction: event.data.prediction || null,
        });
    }
});

render();
