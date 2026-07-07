const urlParams = new URLSearchParams(window.location.search);
const apiBaseUrl = resolveApiBaseUrl(urlParams.get("apiBaseUrl"));

const DATASET_COPY = {
    diabetes: {
        trainingQuestion: "Is a patient with this profile diabetic or non-diabetic?",
        originalTitle: "Original patient case",
        explanationTitle: "Explanation and feedback",
        testOriginalTitle: "Original patient case",
        simulationTitle: "Counterfactual simulation",
    },
    ceramic: {
        trainingQuestion: "Which firing outcome do you think the AI will predict?",
        originalTitle: "Original kiln batch",
        explanationTitle: "Explanation and feedback",
        testOriginalTitle: "Original kiln batch",
        simulationTitle: "Counterfactual simulation",
    },
    safelimit: {
        trainingQuestion: "Do you think this person is above or below the safe limit?",
        originalTitle: "Original person case",
        explanationTitle: "Explanation and feedback",
        testOriginalTitle: "Original person case",
        simulationTitle: "Counterfactual simulation",
    },
};

const state = {
    metadata: null,
    cases: [],
    currentIndex: 0,
    answers: new Map(),
    payloadCache: new Map(),
    renderToken: 0,
    attributeOrderSeed: null,
    randomizeAttributes: true,
};

function isLocalHost(hostname) {
    return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

function appendApiPath(baseUrl) {
    const url = new URL(baseUrl, window.location.href);
    const pathParts = url.pathname.split("/").filter(Boolean);

    if (isLocalHost(url.hostname)) {
        url.port = "5000";
        if (pathParts[pathParts.length - 1] === "api") {
            pathParts.pop();
        }
    } else if (pathParts[pathParts.length - 1] !== "api") {
        pathParts.push("api");
    }

    url.pathname = `/${pathParts.join("/")}`;
    return url.toString();
}

function resolveApiBaseUrl(configuredApiBaseUrl) {
    if (configuredApiBaseUrl) {
        return appendApiPath(configuredApiBaseUrl);
    }

    if (window.location.origin && window.location.origin !== "null") {
        return appendApiPath(window.location.origin);
    }

    return appendApiPath("http://127.0.0.1:5000");
}

function buildApiUrl(path) {
    const baseUrl = apiBaseUrl.endsWith("/") ? apiBaseUrl : `${apiBaseUrl}/`;
    return new URL(path, baseUrl);
}

function getDataset() {
    return document.querySelector("#experiment_dataset").value;
}

function getModel() {
    return document.querySelector("#experiment_model").value;
}

function getExplanationType() {
    return document.querySelector("#experiment_explanation").value;
}

function getFaceFiguresEnabled() {
    return document.querySelector("#experiment_face_figures").checked;
}

function getRandomizeAttributesEnabled() {
    return document.querySelector("#experiment_randomize_attributes").checked;
}

function getCopy() {
    return DATASET_COPY[getDataset()] ?? {
        trainingQuestion: "What do you think the AI will predict?",
        originalTitle: "Original case",
        explanationTitle: "Explanation and feedback",
        testOriginalTitle: "Original case",
        simulationTitle: "Counterfactual simulation",
    };
}

function clampCount(input, fallback) {
    const min = Number(input.min || 0);
    const max = Number(input.max || 100);
    const parsed = Number(input.value);
    const value = Number.isFinite(parsed) ? Math.round(parsed) : fallback;
    input.value = String(Math.min(Math.max(value, min), max));
    return Number(input.value);
}

function sampleInstanceIds(totalCount, requestedCount) {
    const ids = Array.from({ length: totalCount }, (_, index) => index);
    shuffleArray(ids);
    return ids.slice(0, Math.min(requestedCount, totalCount));
}

function shuffleArray(values) {
    for (let i = values.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
}

function sampleBalancedPredictionInstanceIds(metadata, split, requestedCount, fallbackTotal) {
    const groups = metadata.prediction_instance_ids_by_split?.[split];
    const buckets = shuffleArray(Object.values(groups ?? {})
        .filter((instanceIds) => Array.isArray(instanceIds) && instanceIds.length > 0)
        .map((instanceIds) => shuffleArray([...instanceIds])));
    const availableCount = buckets.reduce((sum, bucket) => sum + bucket.length, 0);
    const targetCount = Math.min(requestedCount, availableCount || fallbackTotal);

    if (buckets.length <= 1 || targetCount <= 0) {
        return sampleInstanceIds(fallbackTotal, requestedCount);
    }

    const selectedIds = [];
    let bucketIndex = 0;
    while (selectedIds.length < targetCount && buckets.some((bucket) => bucket.length > 0)) {
        const availableBuckets = buckets.filter((bucket) => bucket.length > 0);
        const bucket = availableBuckets[bucketIndex % availableBuckets.length];
        selectedIds.push(bucket.pop());
        bucketIndex += 1;
    }
    return shuffleArray(selectedIds);
}

function createSessionSeed() {
    if (window.crypto?.getRandomValues) {
        const values = new Uint32Array(2);
        window.crypto.getRandomValues(values);
        return Array.from(values, (value) => value.toString(36)).join("-");
    }
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

async function fetchMetadata() {
    const endpoint = buildApiUrl("metadata");
    endpoint.searchParams.set("dataset", getDataset());
    endpoint.searchParams.set("model", getModel());
    const response = await fetch(endpoint);
    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload.error ?? `Metadata request failed with ${response.status}`);
    }
    return payload;
}

function buildCases(metadata, trainingCount, testCount) {
    const trainTotal = metadata.train_instance_count ?? metadata.available_instance_count ?? 0;
    const testTotal = metadata.test_instance_count ?? metadata.available_instance_count ?? 0;
    const sampledTrainingIds = sampleBalancedPredictionInstanceIds(
        metadata,
        "train",
        trainingCount,
        trainTotal
    );
    const trainingCases = sampledTrainingIds.map((instanceId) => ({
        phase: "training",
        split: "train",
        instanceId,
    }));
    const sampledTestIds = sampleBalancedPredictionInstanceIds(
        metadata,
        "test",
        testCount,
        testTotal
    );
    const simulationModes = buildOrderedSimulationModes(sampledTestIds.length);
    const testCases = sampledTestIds.map((instanceId, index) => ({
        phase: "test",
        split: "test",
        instanceId,
        simulationMode: simulationModes[index],
    }));
    return [...trainingCases, ...testCases];
}

function buildOrderedSimulationModes(count) {
    return Array.from({ length: count }, () => "any");
}

function caseKey(caseItem, xaiType = getExplanationType()) {
    return [
        getDataset(),
        getModel(),
        xaiType,
        caseItem.split,
        caseItem.instanceId,
    ].join(":");
}

async function getCasePayload(caseItem, xaiType = getExplanationType()) {
    const key = caseKey(caseItem, xaiType);
    if (state.payloadCache.has(key)) {
        return state.payloadCache.get(key);
    }

    const endpoint = buildApiUrl("explanations");
    endpoint.searchParams.set("dataset", getDataset());
    endpoint.searchParams.set("model", getModel());
    endpoint.searchParams.set("xaiMethod", "shap");
    endpoint.searchParams.set("xaiType", xaiType);
    endpoint.searchParams.set("explanationView", "persona");
    endpoint.searchParams.set("split", caseItem.split);
    endpoint.searchParams.set("instanceId", String(caseItem.instanceId));
    endpoint.searchParams.set("k", "2");

    const response = await fetch(endpoint);
    const payload = await response.json();
    if (!response.ok) {
        throw new Error(payload.error ?? `Explanation request failed with ${response.status}`);
    }

    state.payloadCache.set(key, payload);
    return payload;
}

function buildIframeSrc(caseItem, options = {}) {
    const query = new URLSearchParams({
        appId: getDataset(),
        AIModel: getModel(),
        expAlgorithm: "shap",
        xaiType: options.xaiType ?? "none",
        explanationView: "persona",
        split: caseItem.split,
        instanceId: String(caseItem.instanceId),
        k: "2",
        showPrediction: String(options.showPrediction ?? 0),
        counterfactualSimulation: String(options.counterfactualSimulation ?? 0),
        simulationMode: options.simulationMode ?? "any",
        faceFigures: String(getFaceFiguresEnabled() ? 1 : 0),
        apiBaseUrl,
    });
    if (state.randomizeAttributes && state.attributeOrderSeed) {
        query.set("attributeOrderSeed", state.attributeOrderSeed);
    }
    return `iframe.html?${query.toString()}`;
}

function createIframe(caseItem, options = {}) {
    const iframe = document.createElement("iframe");
    iframe.className = options.short ? "case-iframe case-iframe-short" : "case-iframe";
    iframe.classList.toggle("case-iframe-face", getFaceFiguresEnabled());
    iframe.src = buildIframeSrc(caseItem, options);
    iframe.title = options.title ?? "Case";
    return iframe;
}

window.addEventListener("message", (event) => {
    if (event.data?.type !== "counterfactual-ui:iframe-height") {
        return;
    }
    const iframe = [...document.querySelectorAll("iframe")]
        .find((candidate) => candidate.contentWindow === event.source);
    const height = Number(event.data.height);
    if (!iframe || !Number.isFinite(height)) {
        return;
    }
    iframe.style.height = `${Math.max(260, Math.ceil(height))}px`;
});

function updateStatus() {
    const phase = document.querySelector("#experiment_phase");
    const progress = document.querySelector("#experiment_progress");
    const prevButton = document.querySelector("#experiment_prev");
    const nextButton = document.querySelector("#experiment_next");

    if (state.cases.length === 0) {
        phase.textContent = "Ready";
        progress.textContent = "Choose a setup and start.";
        prevButton.disabled = true;
        nextButton.disabled = true;
        return;
    }

    const caseItem = state.cases[state.currentIndex];
    const phaseLabel = caseItem.phase === "training"
        ? "Training case"
        : `Test case (${getSimulationModeLabel(caseItem.simulationMode)})`;
    phase.textContent = phaseLabel;
    progress.textContent = `${state.currentIndex + 1} of ${state.cases.length} · ${getDataset().replaceAll("_", " ")} · instance ${caseItem.instanceId}`;
    prevButton.disabled = state.currentIndex === 0;
    nextButton.disabled = state.currentIndex >= state.cases.length - 1;
}

function getSimulationModeLabel(mode) {
    if (mode === "specific") {
        return "two specified attributes";
    }
    if (mode === "budget") {
        return "10-point budget";
    }
    return "open ended";
}

function showStageMessage(message, isError = false) {
    const stage = document.querySelector("#experiment_stage");
    stage.innerHTML = "";
    const panel = document.createElement("section");
    panel.className = "case-panel case-panel-wide empty-state";
    panel.textContent = message;
    if (isError) {
        panel.style.color = "rgba(174, 31, 32, 1)";
    }
    stage.appendChild(panel);
}

async function renderCurrentCase() {
    const token = ++state.renderToken;
    updateStatus();

    if (state.cases.length === 0) {
        showStageMessage("No cases have been started yet.");
        return;
    }

    const caseItem = state.cases[state.currentIndex];
    if (caseItem.phase === "training") {
        await renderTrainingCase(caseItem, token);
    } else {
        renderTestCase(caseItem);
    }
}

async function renderTrainingCase(caseItem, token) {
    const stage = document.querySelector("#experiment_stage");
    const copy = getCopy();
    stage.innerHTML = "";

    const layout = document.createElement("div");
    layout.className = "case-layout";

    const originalPanel = document.createElement("section");
    originalPanel.className = "case-panel";
    const originalTitle = document.createElement("h2");
    originalTitle.textContent = copy.originalTitle;
    originalPanel.appendChild(originalTitle);
    originalPanel.appendChild(createIframe(caseItem, {
        xaiType: "none",
        showPrediction: 0,
        short: true,
        title: copy.originalTitle,
    }));
    layout.appendChild(originalPanel);

    const answerPanel = document.createElement("section");
    answerPanel.className = "case-panel";
    const answerTitle = document.createElement("h2");
    answerTitle.textContent = copy.trainingQuestion;
    answerPanel.appendChild(answerTitle);
    const answerArea = document.createElement("div");
    answerArea.textContent = "Loading choices...";
    answerPanel.appendChild(answerArea);
    layout.appendChild(answerPanel);

    const explanationPanel = document.createElement("section");
    explanationPanel.className = "case-panel case-panel-wide";
    explanationPanel.hidden = true;
    const explanationTitle = document.createElement("h2");
    explanationTitle.textContent = copy.explanationTitle;
    explanationPanel.appendChild(explanationTitle);
    layout.appendChild(explanationPanel);

    stage.appendChild(layout);

    try {
        const payload = await getCasePayload(caseItem);
        if (token !== state.renderToken) {
            return;
        }
        renderAnswerChoices(caseItem, payload, answerArea, explanationPanel);
    } catch (error) {
        if (token === state.renderToken) {
            answerArea.textContent = String(error.message ?? error);
            answerArea.style.color = "rgba(174, 31, 32, 1)";
        }
    }
}

function renderAnswerChoices(caseItem, payload, answerArea, explanationPanel) {
    answerArea.innerHTML = "";
    const choices = document.createElement("div");
    choices.className = "answer-grid";
    const labels = payload.prediction_labels ?? ["Class 0", "Class 1"];
    const selectedAnswer = state.answers.get(caseKey(caseItem));

    labels.forEach((label, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "answer-choice";
        button.classList.toggle("answer-choice-selected", selectedAnswer === index);
        button.textContent = formatPredictionLabel(label);
        button.addEventListener("click", () => {
            state.answers.set(caseKey(caseItem), index);
            renderAnswerChoices(caseItem, payload, answerArea, explanationPanel);
        });
        choices.appendChild(button);
    });
    answerArea.appendChild(choices);

    if (selectedAnswer === undefined) {
        explanationPanel.hidden = true;
        return;
    }

    const correctAnswer = Number(payload.prediction?.value);
    const feedback = document.createElement("div");
    const isCorrect = selectedAnswer === correctAnswer;
    feedback.className = isCorrect
        ? "answer-feedback answer-feedback-correct"
        : "answer-feedback answer-feedback-incorrect";
    feedback.textContent = isCorrect
        ? `Correct. The AI predicts ${formatPredictionLabel(payload.prediction.label)}.`
        : `Incorrect. The AI predicts ${formatPredictionLabel(payload.prediction.label)}.`;
    answerArea.appendChild(feedback);

    explanationPanel.hidden = false;
    const existingIframe = explanationPanel.querySelector("iframe");
    if (existingIframe) {
        existingIframe.remove();
    }
    explanationPanel.appendChild(createIframe(caseItem, {
        xaiType: getExplanationType(),
        showPrediction: 1,
        title: "Persona explanation",
    }));
}

function formatPredictionLabel(label) {
    const labelText = String(label ?? "");
    if (getDataset() === "diabetes") {
        return labelText.toLowerCase().includes("no")
            ? "Non-Diabetic"
            : "Diabetic";
    }
    if (getDataset() === "safelimit") {
        return labelText;
    }
    return labelText;
}

function renderTestCase(caseItem) {
    const stage = document.querySelector("#experiment_stage");
    const copy = getCopy();
    stage.innerHTML = "";

    const stack = document.createElement("div");
    stack.className = "test-stack";

    const simulationPanel = document.createElement("section");
    simulationPanel.className = "case-panel";
    const simulationTitle = document.createElement("h2");
    simulationTitle.textContent = copy.simulationTitle;
    simulationPanel.appendChild(simulationTitle);
    simulationPanel.appendChild(createIframe(caseItem, {
        xaiType: "none",
        showPrediction: 1,
        counterfactualSimulation: 1,
        simulationMode: caseItem.simulationMode ?? "any",
        title: copy.simulationTitle,
    }));
    stack.appendChild(simulationPanel);

    stage.appendChild(stack);
}

async function startRunthrough() {
    const startButton = document.querySelector("#experiment_start");
    const trainingInput = document.querySelector("#experiment_training_count");
    const testInput = document.querySelector("#experiment_test_count");
    const trainingCount = clampCount(trainingInput, 10);
    const testCount = clampCount(testInput, 10);

    startButton.disabled = true;
    showStageMessage("Preparing random cases...");
    try {
        state.metadata = await fetchMetadata();
        state.cases = buildCases(state.metadata, trainingCount, testCount);
        state.currentIndex = 0;
        state.randomizeAttributes = getRandomizeAttributesEnabled();
        state.attributeOrderSeed = state.randomizeAttributes ? createSessionSeed() : null;
        state.answers.clear();
        state.payloadCache.clear();
        if (state.cases.length === 0) {
            showStageMessage("This setup has no cases to show.");
        } else {
            await renderCurrentCase();
        }
    } catch (error) {
        state.cases = [];
        updateStatus();
        showStageMessage(String(error.message ?? error), true);
    } finally {
        startButton.disabled = false;
    }
}

function goToCase(delta) {
    if (state.cases.length === 0) {
        return;
    }
    const nextIndex = Math.min(
        Math.max(state.currentIndex + delta, 0),
        state.cases.length - 1
    );
    if (nextIndex !== state.currentIndex) {
        state.currentIndex = nextIndex;
        renderCurrentCase();
    }
}

document.querySelector("#experiment_start").addEventListener("click", startRunthrough);
document.querySelector("#experiment_prev").addEventListener("click", () => goToCase(-1));
document.querySelector("#experiment_next").addEventListener("click", () => goToCase(1));

document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
        return;
    }

    if (event.key === "ArrowLeft") {
        goToCase(-1);
    } else if (event.key === "ArrowRight") {
        goToCase(1);
    }
});

updateStatus();
