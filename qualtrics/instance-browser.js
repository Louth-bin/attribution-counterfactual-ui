"use strict";

const controls = {
    dataset: document.querySelector("#dataset"),
    split: document.querySelector("#split"),
    explanation: document.querySelector("#explanation"),
    previous: document.querySelector("#previous"),
    next: document.querySelector("#next"),
    openCase: document.querySelector("#open-case"),
    summary: document.querySelector("#summary"),
    cases: document.querySelector("#cases"),
    preview: document.querySelector("#preview"),
};

const params = new URLSearchParams(window.location.search);
controls.dataset.value = params.get("dataset") === "safelimit" ? "safelimit" : "housing";
controls.split.value = params.get("split") === "test" ? "test" : "train";
controls.explanation.value = ["attribution", "counterfactual", "none"].includes(params.get("explanation"))
    ? params.get("explanation")
    : "attribution";
let selectedIndex = Math.max(0, Math.min(9, Number(params.get("case")) || 0));

function pool() {
    const bundle = window.EXPERIMENT_DATA?.datasets?.[controls.dataset.value];
    return controls.split.value === "train" ? bundle?.training_pool : bundle?.test_pool;
}

function caseUrl(payload) {
    const query = new URLSearchParams({
        appId: controls.dataset.value,
        xaiType: controls.split.value === "test" ? "none" : controls.explanation.value,
        split: controls.split.value,
        instanceId: String(payload.instance_id),
        showPrediction: "1",
        counterfactualSimulation: controls.split.value === "test" ? "1" : "0",
    });
    return `../iframe.html?${query}`;
}

function updateUrl() {
    const query = new URLSearchParams({
        dataset: controls.dataset.value,
        split: controls.split.value,
        explanation: controls.explanation.value,
        case: String(selectedIndex),
    });
    history.replaceState(null, "", `?${query}`);
}

function render() {
    const cases = pool();
    if (!Array.isArray(cases) || cases.length === 0) {
        controls.summary.innerHTML = '<div class="empty">No cases found. Regenerate static experiment data.</div>';
        return;
    }
    selectedIndex = Math.max(0, Math.min(cases.length - 1, selectedIndex));
    const payload = cases[selectedIndex];
    const phase = controls.split.value === "train" ? "Training" : "Testing";
    controls.explanation.disabled = controls.split.value === "test";
    controls.explanation.title = controls.split.value === "test"
        ? "Testing intentionally hides the explanation."
        : "Choose the explanation shown for this training profile.";
    const pair = payload.feature_pair_names?.join(" + ") || payload.feature_pair_key || "—";
    controls.summary.innerHTML = [
        `<div><strong>Case</strong>${selectedIndex + 1} of ${cases.length}</div>`,
        `<div><strong>Instance ID</strong>${payload.instance_id}</div>`,
        `<div><strong>AI label</strong>${payload.prediction.label}</div>`,
        `<div><strong>Phase</strong>${phase}</div>`,
        `<div><strong>Top pair</strong>${pair}</div>`,
    ].join("");
    controls.cases.innerHTML = "";
    cases.forEach((candidate, index) => {
        const row = document.createElement("tr");
        if (index === selectedIndex) row.className = "selected";
        const candidatePair = candidate.feature_pair_names?.join(" + ") || candidate.feature_pair_key || "—";
        [index + 1, candidate.instance_id, candidate.prediction.label, candidatePair].forEach((value) => {
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

[controls.dataset, controls.split].forEach((control) => {
    control.addEventListener("change", () => {
        selectedIndex = 0;
        render();
    });
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
    if (event.source !== controls.preview.contentWindow || event.data?.type !== "counterfactual-ui:iframe-height") return;
    controls.preview.style.height = `${Math.max(420, Math.min(900, Number(event.data.height) || 0))}px`;
});

render();
