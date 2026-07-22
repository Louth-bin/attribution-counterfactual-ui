const DATA = window.EXPERIMENT_DATA;
const DEFAULT_MODEL = DATA?.default_model ?? "mlp";

const DATASET_COPY = {
    diabetes: {
        label: "Diabetes",
        trainingQuestion: "Is a patient with this profile diabetic or non-diabetic?",
        originalTitle: "Original patient case",
        explanationTitle: "Explanation and feedback",
        simulationTitle: "Counterfactual simulation",
    },
    safelimit: {
        label: "SafeLimit",
        trainingQuestion: "Do you think this person is above or below the safe limit?",
        originalTitle: "Original person case",
        explanationTitle: "Explanation and feedback",
        simulationTitle: "Counterfactual simulation",
    },
};

const state = {
    cases: [],
    currentIndex: 0,
    answers: new Map(),
    screeningAnswers: new Map(),
    screeningQuestions: new Map(),
    counterfactualChanges: new Map(),
    attributeOrderSeed: null,
    randomizeAttributes: true,
    experimentStarted: false,
    lastShownStepKey: null,
};

function logStudyEvent(eventType, details = {}) {
    return window.ExperimentLogger?.log(eventType, details) ?? Promise.resolve(false);
}

function isRecordedPhase(step) {
    return step?.phase === "training" || step?.phase === "test";
}

function caseSnapshot(step) {
    if (!step?.payload) return null;
    return {
        phase: step.phase,
        caseId: step.id,
        instanceId: step.payload.instance_id,
        split: step.split,
        dataset: getDataset(),
        explanationType: getExplanationType(),
        attributeOrderSeed: state.attributeOrderSeed,
        payload: step.payload,
        currentSimulationValues: state.counterfactualChanges.get(caseKey(step)) ?? null,
    };
}

const DATASET_SCENARIOS = {
    diabetes: {
        title: "Patient profiles",
        intro: [
            ["In the following pages, you will see patient profiles described by ", { strong: "five attributes" }, ". ",
            "The patients are either ", { strong: "Diabetic" }, " or ", { strong: "Non-diabetic" }, "."],
        ],
        aiLabel: "diagnosis",
        attributes: {
            glucose: "Concentration of glucose in blood",
            blood_pressure: "Diastolic blood pressure of patient",
            insulin: "Insulin level 2 hours after glucose intake",
            bmi: "Body Mass Index",
            age: "Age in years",
        },
    },
    safelimit: {
        title: "Driver profiles",
        intro: [
            ["In the following pages, you will see driver profiles described by ", { strong: "five attributes" }, ".",
            "They are either ", { strong: "above "}, "the alcohol limit for driving", " or ", { strong: "below "}, "that limit."],
        ],
        aiLabel: "prediction",
        attributes: {
            units: "Amount of alcohol consumed",
            weight: "Weight of the driver in kilograms",
            duration: "Length of time spent drinking in minutes",
            gender: "Gender of the driver",
            stomach_fullness: "Whether the driver ate before or while drinking",
        },
    },
};

function getDataset() {
    return document.querySelector("#experiment_dataset").value;
}

function getExplanationType() {
    return document.querySelector("#experiment_explanation").value;
}

function getRandomizeAttributesEnabled() {
    return document.querySelector("#experiment_randomize_attributes").checked;
}

function applyUrlConfiguration() {
    const params = new URLSearchParams(window.location.search);
    const dataset = String(params.get("dataset") ?? "diabetes").toLowerCase();
    const explanation = String(params.get("explanation") ?? "attribution").toLowerCase();
    const validDatasets = new Set(["diabetes", "safelimit"]);
    const validExplanations = new Set(["attribution", "counterfactual", "none"]);

    if (!validDatasets.has(dataset)) {
        throw new Error(`Unknown dataset '${dataset}'. Use diabetes or safelimit.`);
    }
    if (!validExplanations.has(explanation)) {
        throw new Error(`Unknown explanation '${explanation}'. Use attribution, counterfactual, or none.`);
    }

    document.querySelector("#experiment_dataset").value = dataset;
    document.querySelector("#experiment_explanation").value = explanation;
}

function getDatasetBundle(dataset = getDataset()) {
    const bundle = DATA?.datasets?.[dataset];
    if (!bundle) {
        throw new Error(`Static data for dataset '${dataset}' is unavailable.`);
    }
    return bundle;
}

function validateDiabetesLabelMapping() {
    const bundle = getDatasetBundle("diabetes");
    const expectedLabels = ["Diabetes", "No Diabetes"];
    const metadataLabels = bundle.metadata?.prediction_labels ?? [];

    if (metadataLabels.length !== 2 || metadataLabels.some((label, index) => label !== expectedLabels[index])) {
        throw new Error("Diabetes label mapping is invalid: class 0 must be Diabetes and class 1 must be No Diabetes.");
    }

    const cases = [...(bundle.training_pool ?? []), ...(bundle.test_pool ?? [])];
    cases.forEach((payload) => {
        const labels = payload.prediction_labels ?? metadataLabels;
        const predictions = [payload.prediction, payload.counterfactual?.prediction].filter(Boolean);
        predictions.forEach((prediction) => {
            const classIndex = Number(prediction.value);
            if (labels[classIndex] !== prediction.label) {
                throw new Error(
                    `Diabetes label mapping is inconsistent for case ${payload.instance_id}: ` +
                    `class ${classIndex} is '${prediction.label}', expected '${labels[classIndex]}'.`
                );
            }
        });
    });
}

function getCopy() {
    return DATASET_COPY[getDataset()] ?? DATASET_COPY.diabetes;
}

function clampCount(input, fallback) {
    const min = Number(input.min || 0);
    const max = Number(input.max || 100);
    const parsed = Number(input.value);
    const value = Number.isFinite(parsed) ? Math.round(parsed) : fallback;
    input.value = String(Math.min(Math.max(value, min), max));
    return Number(input.value);
}

function shuffleArray(values) {
    for (let i = values.length - 1; i > 0; i -= 1) {
        const j = Math.floor(Math.random() * (i + 1));
        [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
}

function createSessionSeed() {
    if (window.crypto?.getRandomValues) {
        const values = new Uint32Array(2);
        window.crypto.getRandomValues(values);
        return Array.from(values, (value) => value.toString(36)).join("-");
    }
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function getPredictionKey(payload) {
    return String(payload.prediction?.value ?? payload.prediction?.label ?? "unknown");
}

function buildBalancedPredictionTargets(pool, requestedCount) {
    const availableCounts = new Map();
    pool.forEach((payload) => {
        const predictionKey = getPredictionKey(payload);
        availableCounts.set(predictionKey, (availableCounts.get(predictionKey) ?? 0) + 1);
    });

    const labels = shuffleArray([...availableCounts.keys()]);
    const targets = new Map(labels.map((label) => [label, 0]));
    let remaining = requestedCount;

    while (remaining > 0) {
        let assignedInRound = false;
        for (const label of labels) {
            if (remaining <= 0) {
                break;
            }
            const currentTarget = targets.get(label) ?? 0;
            if (currentTarget >= (availableCounts.get(label) ?? 0)) {
                continue;
            }
            targets.set(label, currentTarget + 1);
            remaining -= 1;
            assignedInRound = true;
        }
        if (!assignedInRound) {
            break;
        }
    }

    return targets;
}

function sampleBalancedTrainingPool(pool, requestedCount, pairKeys) {
    const pairSet = new Set(pairKeys);
    pool.forEach((payload) => pairSet.add(payload.feature_pair_key));
    const orderedKeys = shuffleArray([...pairSet]);
    const labelTargets = buildBalancedPredictionTargets(pool, requestedCount);
    const labelOrder = shuffleArray([...labelTargets.keys()]);
    const selectedCounts = new Map(labelOrder.map((label) => [label, 0]));
    const buckets = new Map(orderedKeys.map((key) => [key, new Map()]));

    pool.forEach((payload) => {
        const pairKey = payload.feature_pair_key;
        const predictionKey = getPredictionKey(payload);
        if (!buckets.has(pairKey)) {
            buckets.set(pairKey, new Map());
        }
        const pairBucket = buckets.get(pairKey);
        if (!pairBucket.has(predictionKey)) {
            pairBucket.set(predictionKey, []);
        }
        pairBucket.get(predictionKey).push(payload);
    });
    [...buckets.values()].forEach((pairBucket) => {
        [...pairBucket.values()].forEach(shuffleArray);
    });

    const selected = [];
    while (selected.length < requestedCount) {
        let selectedInRound = false;
        for (const key of orderedKeys) {
            const pairBucket = buckets.get(key);
            if (!pairBucket) {
                continue;
            }
            const eligibleLabels = labelOrder
                .filter((label) =>
                    (selectedCounts.get(label) ?? 0) < (labelTargets.get(label) ?? 0) &&
                    (pairBucket.get(label)?.length ?? 0) > 0
                )
                .sort((a, b) =>
                    ((labelTargets.get(b) ?? 0) - (selectedCounts.get(b) ?? 0)) -
                    ((labelTargets.get(a) ?? 0) - (selectedCounts.get(a) ?? 0))
                );
            const label = eligibleLabels[0];
            if (label) {
                selected.push(pairBucket.get(label).pop());
                selectedCounts.set(label, (selectedCounts.get(label) ?? 0) + 1);
                selectedInRound = true;
            }
            if (selected.length >= requestedCount) {
                break;
            }
        }
        if (!selectedInRound) {
            break;
        }
    }
    return selected;
}

function buildCases(trainingCount, testCount) {
    const bundle = getDatasetBundle();
    const pairKeys = bundle.metadata.all_feature_pair_keys ?? [];
    const trainingPayloads = sampleBalancedTrainingPool(
        bundle.training_pool,
        Math.min(trainingCount, bundle.training_pool.length),
        pairKeys
    );
    const testPayloads = shuffleArray([...bundle.test_pool])
        .slice(0, Math.min(testCount, bundle.test_pool.length));

    const trainingCases = trainingPayloads.map((payload) => ({
        phase: "training",
        split: "train",
        payload,
    }));
    const testCases = testPayloads.map((payload) => ({
        phase: "test",
        split: "test",
        payload,
    }));
    return [...trainingCases, ...testCases];
}

function caseKey(caseItem) {
    return [
        getDataset(),
        DEFAULT_MODEL,
        caseItem.split,
        caseItem.payload.instance_id,
    ].join(":");
}

function buildIframeSrc(caseItem, options = {}) {
    const query = new URLSearchParams({
        appId: getDataset(),
        AIModel: DEFAULT_MODEL,
        expAlgorithm: "shap",
        xaiType: options.xaiType ?? "none",
        explanationView: "persona",
        split: caseItem.split,
        instanceId: String(caseItem.payload.instance_id),
        k: "2",
        showPrediction: String(options.showPrediction ?? 0),
        counterfactualSimulation: String(options.counterfactualSimulation ?? 0),
        simulationMode: "any",
        faceFigures: "0",
    });
    if (state.randomizeAttributes && state.attributeOrderSeed) {
        query.set("attributeOrderSeed", state.attributeOrderSeed);
    }
    if (options.tutorialCallouts) {
        query.set("tutorialCallouts", options.tutorialCallouts);
    }
    if (options.counterfactualSimulation) {
        const savedChanges = state.counterfactualChanges.get(caseKey(caseItem));
        if (savedChanges) {
            query.set("simulationValues", JSON.stringify(savedChanges));
        }
    }
    return `iframe.html?${query.toString()}`;
}

function createIframe(caseItem, options = {}) {
    const iframe = document.createElement("iframe");
    iframe.className = options.short ? "case-iframe case-iframe-short" : "case-iframe";
    iframe.dataset.minHeight = options.short ? "210" : "260";
    iframe.dataset.caseKey = caseKey(caseItem);
    iframe.src = buildIframeSrc(caseItem, options);
    iframe.title = options.title ?? "Case";
    return iframe;
}

window.addEventListener("message", (event) => {
    if (event.data?.type === "counterfactual-ui:simulation-change") {
        const iframe = [...document.querySelectorAll("iframe")]
            .find((candidate) => candidate.contentWindow === event.source);
        const values = event.data.values;
        if (iframe?.dataset.caseKey && Array.isArray(values)) {
            const previousValues = state.counterfactualChanges.get(iframe.dataset.caseKey) ?? null;
            state.counterfactualChanges.set(iframe.dataset.caseKey, [...values]);
            const step = state.cases[state.currentIndex];
            if (isRecordedPhase(step)) {
                logStudyEvent("simulation_changed", {
                    phase: step.phase, caseId: step.id, instanceId: step.payload.instance_id,
                    previousValues, values, normalizedValues: event.data.normalizedValues ?? null,
                });
            }
        }
        return;
    }
    if (event.data?.type === "counterfactual-ui:screen-state") {
        const step = state.cases[state.currentIndex];
        if (isRecordedPhase(step)) {
            logStudyEvent("iframe_screen_state", {
                phase: step.phase, caseId: step.id, instanceId: step.payload.instance_id,
                screenState: event.data.screenState,
            });
        }
        return;
    }
    if (event.data?.type !== "counterfactual-ui:iframe-height") {
        return;
    }
    const iframe = [...document.querySelectorAll("iframe")]
        .find((candidate) => candidate.contentWindow === event.source);
    const height = Number(event.data.height);
    if (!iframe || !Number.isFinite(height)) {
        return;
    }
    const minHeight = Number(iframe.dataset.minHeight ?? 260);
    iframe.style.height = `${Math.max(minHeight, Math.ceil(height))}px`;
});

function createElement(tagName, className, textContent) {
    const element = document.createElement(tagName);
    if (className) {
        element.className = className;
    }
    if (textContent !== undefined) {
        element.textContent = textContent;
    }
    return element;
}

function appendFormattedText(container, parts) {
    (Array.isArray(parts) ? parts : [parts]).forEach((part) => {
        if (typeof part === "string" || typeof part === "number") {
            container.appendChild(document.createTextNode(String(part)));
            return;
        }
        if (part?.strong !== undefined) {
            const strong = document.createElement("strong");
            strong.textContent = String(part.strong);
            container.appendChild(strong);
        }
    });
}

function getScenario(dataset = getDataset()) {
    return DATASET_SCENARIOS[dataset] ?? DATASET_SCENARIOS.diabetes;
}

function getSampleCase(cases) {
    return cases.find((caseItem) => caseItem.phase === "training")
        ?? cases.find((caseItem) => caseItem.phase === "test")
        ?? null;
}

function getPoolCase(payload, split = "train") {
    return payload
        ? { phase: "training", split, payload }
        : null;
}

function getBasicTutorialSampleCase(cases) {
    const bundle = getDatasetBundle();
    return getPoolCase(bundle.training_pool?.[0])
        ?? getSampleCase(cases);
}

function getExplanationTutorialSampleCase(cases) {
    const bundle = getDatasetBundle();
    const pool = bundle.training_pool ?? [];
    const explanationType = getExplanationType();
    let payload = null;

    if (explanationType === "attribution") {
        payload = pool.find((candidate) => {
            const values = (candidate.attribution?.shown_feature_indices ?? [])
                .map((index) => Number(candidate.attribution?.values?.[index] ?? 0));
            return values.some((value) => value > 0) && values.some((value) => value < 0);
        });
    } else if (explanationType === "counterfactual") {
        const hasNumericIncreaseAndDecrease = (candidate) => {
            const deltas = (candidate.counterfactual?.selected_feature_names ?? [])
                .map((name) => {
                    const index = getFeatureIndex(candidate, name);
                    if (index < 0 || candidate.feature_types?.[index] === "categorical") {
                        return null;
                    }
                    return Number(candidate.counterfactual?.feature_values?.[index]) -
                        Number(candidate.feature_values?.[index]);
                })
                .filter((value) => value !== null && value !== 0 && Number.isFinite(value));
            return deltas.some((value) => value > 0) && deltas.some((value) => value < 0);
        };

        payload = pool.find((candidate) => getDataset() === "diabetes" && hasNumericIncreaseAndDecrease(candidate));
        payload ??= pool.find((candidate) => {
            const names = candidate.counterfactual?.selected_feature_names ?? [];
            if (getDataset() !== "safelimit" || !names.includes("Alcohol Units") || !names.includes("Gender")) {
                return false;
            }
            const unitsIndex = getFeatureIndex(candidate, "Alcohol Units");
            return Number(candidate.counterfactual?.feature_values?.[unitsIndex]) <
                Number(candidate.feature_values?.[unitsIndex]);
        });
    }

    return getPoolCase(payload ?? pool[0])
        ?? getSampleCase(cases);
}

function getFeatureDescription(dataset, rawFeatureName) {
    return getScenario(dataset).attributes[rawFeatureName]
        ?? rawFeatureName.replaceAll("_", " ");
}

function formatRangeValue(value) {
    if (Array.isArray(value)) {
        const separator = value.every((item) => typeof item === "number") ? " - " : ", ";
        return value.map(formatRangeValue).join(separator);
    }
    if (typeof value === "number") {
        return Number.isInteger(value) ? String(value) : String(Math.round(value * 10) / 10);
    }
    return String(value);
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function strongText(value) {
    return `<strong>${escapeHtml(value)}</strong>`;
}

function colorText(value, className) {
    return `<span class="${className}">${escapeHtml(value)}</span>`;
}

function formatTutorialValue(value) {
    if (typeof value === "number") {
        const rounded = Math.round(value * 10) / 10;
        return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
    }
    return String(value);
}

function getFeatureIndex(payload, featureName) {
    const normalized = String(featureName).toLowerCase();
    return payload.feature_names.findIndex((name) => String(name).toLowerCase() === normalized);
}

function getFeatureRangeExample(payload, featureName) {
    const index = getFeatureIndex(payload, featureName);
    const range = payload.feature_ranges?.[index];
    if (index < 0 || !Array.isArray(range) || range.length < 2) {
        return "";
    }
    const [min, max] = range;
    const value = payload.feature_values[index];
    const ratio = (Number(value) - Number(min)) / (Number(max) - Number(min));
    const position = ratio >= 0.7
        ? "well over half"
        : ratio >= 0.52
            ? "just over half"
            : ratio >= 0.45
                ? "about half"
                : "under half";
    return `e.g., for ${escapeHtml(featureName)}, the lowest value is ${escapeHtml(formatTutorialValue(min))} and the highest is ${escapeHtml(formatTutorialValue(max))}, so ${escapeHtml(formatTutorialValue(value))} is ${position} of the bar`;
}

function getCategoricalExample(payload, featureName) {
    const index = getFeatureIndex(payload, featureName);
    if (index < 0) {
        return "";
    }
    return `e.g., the value ${escapeHtml(formatTutorialValue(payload.feature_values[index]))} for ${escapeHtml(featureName)} is checked with a filled-in circle`;
}

function getShownAttributionEntries(payload) {
    const total = (payload.attribution?.values ?? [])
        .reduce((sum, value) => sum + Math.abs(Number(value) || 0), 0) || 1;
    return (payload.attribution?.shown_feature_indices ?? [])
        .map((index) => {
            const value = Number(payload.attribution?.values?.[index] ?? 0);
            return {
                index,
                name: payload.feature_names[index],
                value,
                percent: Math.round((Math.abs(value) / total) * 100),
                label: value < 0
                    ? payload.attribution?.direction_labels?.left
                    : payload.attribution?.direction_labels?.right,
                colorClass: value < 0 ? "tutorial-color-red" : "tutorial-color-blue",
            };
        })
        .filter((entry) => entry.name);
}

function getCounterfactualChangeEntries(payload) {
    const selectedNames = payload.counterfactual?.selected_feature_names ?? [];
    return selectedNames
        .map((name) => {
            const index = getFeatureIndex(payload, name);
            if (index < 0) {
                return null;
            }
            const originalValue = payload.feature_values[index];
            const updatedValue = payload.counterfactual?.feature_values?.[index];
            const type = payload.feature_types?.[index];
            const numericDelta = Number(updatedValue) - Number(originalValue);
            const isNumeric = type !== "categorical" && Number.isFinite(numericDelta);
            return {
                name,
                originalValue,
                updatedValue,
                isNumeric,
                delta: isNumeric ? numericDelta : null,
                direction: isNumeric
                    ? (numericDelta < 0 ? "decreases" : "increases")
                    : "changes",
                colorClass: isNumeric && numericDelta < 0 ? "tutorial-color-red" : "tutorial-color-blue",
            };
        })
        .filter(Boolean);
}

function getScenarioRows(dataset) {
    const bundle = getDatasetBundle(dataset);
    const displayNames = bundle.metadata.feature_names ?? [];
    const rawNames = bundle.metadata.raw_feature_names ?? [];
    const ranges = bundle.training_pool?.[0]?.feature_ranges ?? [];
    return rawNames.map((rawName, index) => ({
        attribute: displayNames[index] ?? rawName,
        description: getFeatureDescription(dataset, rawName),
        value: formatRangeValue(ranges[index] ?? ""),
    }));
}

function buildTutorialSteps(cases) {
    const basicSampleCase = getBasicTutorialSampleCase(cases);
    const explanationSampleCase = getExplanationTutorialSampleCase(cases);
    if (!basicSampleCase) {
        return [];
    }

    const explanationType = getExplanationType();
    const steps = [
        {
            phase: "tutorial-scenario",
            id: "scenario",
            title: getScenario().title,
            sampleCase: basicSampleCase,
        },
        {
            phase: "tutorial-basic",
            id: "basic-ui",
            title: "Basic interface",
            sampleCase: basicSampleCase,
        },
        {
            phase: "screening-basic",
            id: "basic-screening",
            title: "Check your understanding",
            sampleCase: basicSampleCase,
        },
    ];

    if (explanationType !== "none" && explanationSampleCase) {
        steps.push(
            {
                phase: "tutorial-explanation",
                id: "explanation-ui",
                title: "AI explanation",
                sampleCase: explanationSampleCase,
            },
            {
                phase: "screening-explanation",
                id: "explanation-screening",
                title: "Check your understanding",
                sampleCase: explanationSampleCase,
            }
        );
    }

    return steps;
}

function buildPhaseSteps(caseSteps) {
    const steps = [];
    let previousPhase = null;
    caseSteps.forEach((caseItem) => {
        if (caseItem.phase !== previousPhase) {
            steps.push({
                phase: `${caseItem.phase}-instructions`,
                id: `${caseItem.phase}-instructions`,
                title: caseItem.phase === "training" ? "Training Session Instructions" : "Testing Session Instructions",
            });
            previousPhase = caseItem.phase;
        }
        steps.push(caseItem);
    });
    return steps;
}

function isTutorialStep(step) {
    return String(step?.phase ?? "").startsWith("tutorial-") ||
        String(step?.phase ?? "").startsWith("screening-");
}

function getStepProgressLabel(step) {
    if (!step) {
        return "Choose a setup and start.";
    }
    if (isTutorialStep(step) || String(step.phase).endsWith("-instructions")) {
        return `${state.currentIndex + 1} of ${state.cases.length} - ${step.title}`;
    }
    return `${state.currentIndex + 1} of ${state.cases.length} - ${getCopy().label} - instance ${step.payload.instance_id}`;
}

function getPhaseLabel(step) {
    if (!step) {
        return "Ready";
    }
    if (step.phase === "tutorial-scenario") {
        return "Overview";
    }
    if (step.phase === "tutorial-basic" || step.phase === "tutorial-explanation") {
        return "Tutorial";
    }
    if (step.phase === "screening-basic" || step.phase === "screening-explanation") {
        return "Screening questions";
    }
    if (step.phase === "training-instructions" || step.phase === "test-instructions") {
        return "Instructions";
    }
    return step.phase === "training" ? "Training case" : "Test case";
}

function getScreeningKey(step, questionId) {
    return `${getDataset()}:${getExplanationType()}:${step.id}:${questionId}`;
}

function isScreeningStepComplete(step) {
    if (!step || !String(step.phase).startsWith("screening-")) {
        return true;
    }
    return getScreeningQuestions(step).every((question) => {
        const answer = state.screeningAnswers.get(getScreeningKey(step, question.id));
        if (question.type === "multi") {
            return Array.isArray(answer) && answer.length === question.correct.length;
        }
        return answer !== undefined;
    });
}

function updateStatus() {
    const phase = document.querySelector("#experiment_phase");
    const progress = document.querySelector("#experiment_progress");
    const prevButton = document.querySelector("#experiment_prev");
    const nextButton = document.querySelector("#experiment_next");

    if (!state.experimentStarted) {
        phase.textContent = "Overview";
        progress.textContent = "Domain introduction";
        prevButton.disabled = true;
        nextButton.disabled = true;
        updateBackdoorMenu();
        return;
    }

    const caseItem = state.cases[state.currentIndex];
    phase.textContent = getPhaseLabel(caseItem);
    progress.textContent = getStepProgressLabel(caseItem);
    prevButton.disabled = state.currentIndex === 0;
    nextButton.disabled = state.currentIndex >= state.cases.length - 1 ||
        !isScreeningStepComplete(caseItem);
    updateBackdoorMenu();
}

const BACKDOOR_LABELS = {
    "tutorial-scenario": "Domain introduction",
    "tutorial-basic": "Basic interface tutorial",
    "screening-basic": "Basic screening questions",
    "tutorial-explanation": "Explanation tutorial",
    "screening-explanation": "Explanation screening questions",
    "training-instructions": "Training session instructions",
    training: "First training case",
    "test-instructions": "Testing session instructions",
    test: "First testing case",
};

function updateBackdoorMenu() {
    const select = document.querySelector("#experiment_jump");
    if (!select) {
        return;
    }
    const firstIndexByPhase = new Map();
    state.cases.forEach((step, index) => {
        if (!firstIndexByPhase.has(step.phase)) {
            firstIndexByPhase.set(step.phase, index);
        }
    });
    select.innerHTML = "";
    if (firstIndexByPhase.size === 0) {
        select.appendChild(new Option("Start a runthrough first", ""));
        select.disabled = true;
        return;
    }
    Object.entries(BACKDOOR_LABELS).forEach(([phase, label]) => {
        if (firstIndexByPhase.has(phase)) {
            select.appendChild(new Option(label, String(firstIndexByPhase.get(phase))));
        }
    });
    const currentPhaseIndex = firstIndexByPhase.get(state.cases[state.currentIndex]?.phase);
    select.value = String(currentPhaseIndex ?? state.currentIndex);
    select.disabled = false;
}

function setBackdoorVisible(visible) {
    const backdoor = document.querySelector("#experiment_backdoor");
    const configuration = document.querySelector(".experiment-configuration");
    if (backdoor) {
        backdoor.hidden = !visible;
    }
    if (configuration) {
        configuration.hidden = !visible;
    }
}

function jumpToCase(index) {
    if (!Number.isInteger(index) || index < 0 || index >= state.cases.length) {
        return;
    }
    state.currentIndex = index;
    renderCurrentCase();
}

function showStageMessage(message, isError = false) {
    const stage = document.querySelector("#experiment_stage");
    stage.innerHTML = "";
    const panel = document.createElement("section");
    panel.className = "case-panel case-panel-wide empty-state";
    panel.textContent = message;
    if (isError) {
        panel.classList.add("empty-state-error");
    }
    stage.appendChild(panel);
}

function renderTutorialPage(title, body) {
    const stage = document.querySelector("#experiment_stage");
    stage.innerHTML = "";
    const panel = createElement("section", "tutorial-panel");
    const heading = createElement("h1", "tutorial-title", title);
    panel.appendChild(heading);
    panel.appendChild(body);
    stage.appendChild(panel);
}

function renderScenarioPage(step) {
    const dataset = getDataset();
    const scenario = getScenario(dataset);
    const body = createElement("div", "tutorial-scenario-layout");

    const intro = createElement("div", "tutorial-copy");
    scenario.intro.forEach((paragraph) => {
        const text = createElement("p");
        appendFormattedText(text, paragraph);
        intro.appendChild(text);
    });
    const aiParagraph = createElement("p");
    aiParagraph.textContent = `An AI can provide the correct ${scenario.aiLabel} for each profile based on these attributes.`;
    intro.appendChild(aiParagraph);
    body.appendChild(intro);

    const table = createElement("table", "scenario-table");
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    ["Attribute", "Description", dataset === "safelimit" ? "Range / Options" : "Value range"].forEach((header) => {
        headerRow.appendChild(createElement("th", "", header));
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    getScenarioRows(dataset).forEach((row) => {
        const tr = document.createElement("tr");
        tr.appendChild(createElement("td", "", row.attribute));
        tr.appendChild(createElement("td", "", row.description));
        tr.appendChild(createElement("td", "", row.value));
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    body.appendChild(table);

    renderTutorialPage(step.title, body);
}

function appendTutorialBullets(container, bullets) {
    const list = createElement("ol", "tutorial-bullets");
    bullets.forEach((bullet) => {
        const item = document.createElement("li");
        item.innerHTML = bullet;
        list.appendChild(item);
    });
    container.appendChild(list);
}

function appendTutorialList(container, items, options = {}) {
    const list = createElement(options.ordered ? "ol" : "ul", options.className ?? "tutorial-bullets");
    items.forEach((item) => {
        const element = document.createElement("li");
        element.innerHTML = item;
        list.appendChild(element);
    });
    container.appendChild(list);
}

function renderBasicTutorialPage(step) {
    const dataset = getDataset();
    const scenario = getScenario(dataset);
    const body = createElement("div", "tutorial-two-column");
    const copyPanel = createElement("div", "tutorial-copy");
    copyPanel.appendChild(createElement(
        "p",
        "",
        dataset === "diabetes"
            ? "Each patient profile is shown using the same basic interface."
            : "Each driver profile is shown using the same basic interface."
    ));
    appendTutorialBullets(copyPanel, [
        `The five <strong>Attributes</strong> describing the ${dataset === "diabetes" ? "patient" : "driver"}.`,
        "The <strong>Values</strong> of each attribute.",
        dataset === "safelimit"
            ? `Bars indicating how ${strongText("low/high")} a given value is for an attribute (${getFeatureRangeExample(step.sampleCase.payload, "Alcohol Units")}). Other attribute values are just ${strongText("checked")} (${getCategoricalExample(step.sampleCase.payload, "Gender")}).`
            : `Bars indicating how ${strongText("low/high")} a given value is for that attribute (${getFeatureRangeExample(step.sampleCase.payload, "Blood Pressure")}).`,
        `The selected box shows the <strong>AI ${scenario.aiLabel}</strong>.`,
    ]);
    body.appendChild(copyPanel);

    const preview = createElement("div", "tutorial-preview");
    preview.appendChild(createIframe(step.sampleCase, {
        xaiType: "none",
        showPrediction: 1,
        tutorialCallouts: "basic",
        title: "Basic interface example",
    }));
    body.appendChild(preview);

    renderTutorialPage(step.title, body);
}

function getValueQuestion(step) {
    const payload = step.sampleCase.payload;
    const valueIndex = Math.min(2, payload.feature_names.length - 1);
    const correctValue = String(payload.feature_values[valueIndex]);
    const choices = [
        correctValue,
        String(payload.feature_values[(valueIndex + 1) % payload.feature_values.length]),
        String(payload.feature_values[(valueIndex + 2) % payload.feature_values.length]),
        "Cannot be determined",
    ];
    return {
        id: "value",
        type: "single",
        prompt: `What is the value of ${payload.feature_names[valueIndex]} in the profile shown?`,
        choices: shuffleArray([...new Set(choices)]),
        correct: correctValue,
    };
}

function getBasicScreeningQuestions(step) {
    return [
        getValueQuestion(step),
        {
            id: "basic-ui",
            type: "single",
            prompt: "Which part of the basic interface helps you judge whether a numeric value is relatively low or high?",
            choices: getDataset() === "safelimit"
                ? ["Range / Options", "Attribute", "AI prediction", "Value"]
                : ["Low / High", "Attribute", "AI prediction", "Value"],
            correct: getDataset() === "safelimit" ? "Range / Options" : "Low / High",
        },
    ];
}

function getChangedAttributeNames(payload) {
    const selected = payload.counterfactual?.selected_feature_names ?? [];
    if (selected.length > 0) {
        return selected.map(String);
    }
    return payload.feature_names.filter((_, index) =>
        String(payload.feature_values[index]) !== String(payload.counterfactual?.feature_values?.[index])
    );
}

function getExplanationScreeningQuestions(step) {
    const payload = step.sampleCase.payload;
    if (getExplanationType() === "attribution") {
        const shownIndices = payload.attribution?.shown_feature_indices ?? [];
        const correctAttributes = shownIndices.map((index) => payload.feature_names[index]);
        const attributeChoices = shuffleArray([...payload.feature_names]);
        return [
            {
                id: "attribution-purpose",
                type: "single",
                prompt: "What does the attribution explanation show?",
                choices: [
                    "Which attributes most influenced the AI prediction",
                    "The original source of the dataset",
                    "A random list of unused attributes",
                    "The participant's final answer",
                ],
                correct: "Which attributes most influenced the AI prediction",
            },
            {
                id: "attribution-attributes",
                type: "multi",
                prompt: "Which attributes are highlighted as influential in the explanation shown?",
                choices: shuffleArray(attributeChoices),
                correct: correctAttributes,
            },
        ];
    }

    const changedAttributes = getChangedAttributeNames(payload);
    const attributeChoices = shuffleArray([...payload.feature_names]);
    return [
        {
            id: "counterfactual-purpose",
            type: "single",
            prompt: "What does the counterfactual explanation show?",
            choices: [
                "How some attribute values could change to get an alternative AI prediction",
                "How each attribute contributes to the current prediction",
                "How accurate the participant's answer was",
                "The order in which profiles are sampled",
            ],
            correct: "How some attribute values could change to get an alternative AI prediction",
        },
        {
            id: "counterfactual-attributes",
            type: "multi",
            prompt: "Which attributes changed in the counter-example shown?",
            choices: shuffleArray(attributeChoices),
            correct: changedAttributes,
        },
    ];
}

function getScreeningQuestions(step) {
    const key = getScreeningKey(step, "questions");
    if (state.screeningQuestions.has(key)) {
        return state.screeningQuestions.get(key);
    }
    const questions = step.phase === "screening-basic"
        ? getBasicScreeningQuestions(step)
        : getExplanationScreeningQuestions(step);
    state.screeningQuestions.set(key, questions);
    return questions;
}

function valuesMatchAsSets(first, second) {
    const firstValues = [...first].map(String).sort();
    const secondValues = [...second].map(String).sort();
    return firstValues.length === secondValues.length &&
        firstValues.every((value, index) => value === secondValues[index]);
}

function renderScreeningQuestion(step, question, container) {
    const questionPanel = createElement("div", "screening-question");
    questionPanel.appendChild(createElement("h3", "", question.prompt));
    const answers = createElement("div", question.type === "multi" ? "screening-choices screening-choices-multi" : "screening-choices");
    const key = getScreeningKey(step, question.id);
    const storedAnswer = state.screeningAnswers.get(key);

    question.choices.forEach((choice) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "screening-choice";
        const isSelected = question.type === "multi"
            ? Array.isArray(storedAnswer) && storedAnswer.includes(choice)
            : storedAnswer === choice;
        button.classList.toggle("screening-choice-selected", isSelected);
        button.textContent = choice;
        button.addEventListener("click", () => {
            if (question.type === "multi") {
                const currentAnswer = state.screeningAnswers.get(key);
                const nextAnswer = Array.isArray(currentAnswer) ? [...currentAnswer] : [];
                const existingIndex = nextAnswer.indexOf(choice);
                if (existingIndex >= 0) {
                    nextAnswer.splice(existingIndex, 1);
                } else {
                    nextAnswer.push(choice);
                }
                state.screeningAnswers.set(key, nextAnswer);
            } else {
                state.screeningAnswers.set(key, choice);
            }
            renderCurrentCase();
        });
        answers.appendChild(button);
    });
    questionPanel.appendChild(answers);

    const answer = state.screeningAnswers.get(key);
    const hasCompleteAnswer = question.type === "multi"
        ? Array.isArray(answer) && answer.length === question.correct.length
        : answer !== undefined;
    if (hasCompleteAnswer) {
        const isCorrect = question.type === "multi"
            ? valuesMatchAsSets(answer, question.correct)
            : String(answer) === String(question.correct);
        const feedback = createElement(
            "div",
            isCorrect ? "screening-feedback screening-feedback-correct" : "screening-feedback screening-feedback-incorrect",
            isCorrect ? "Correct" : "Try again"
        );
        questionPanel.appendChild(feedback);
    }

    container.appendChild(questionPanel);
}

function renderScreeningPage(step) {
    const body = createElement("div", "tutorial-two-column");
    const questionsPanel = createElement("div", "screening-panel");
    getScreeningQuestions(step).forEach((question) => {
        renderScreeningQuestion(step, question, questionsPanel);
    });
    body.appendChild(questionsPanel);

    const preview = createElement("div", "tutorial-preview");
    preview.appendChild(createIframe(step.sampleCase, {
        xaiType: step.phase === "screening-basic" ? "none" : getExplanationType(),
        showPrediction: 1,
        title: "Screening example",
    }));
    body.appendChild(preview);

    renderTutorialPage(step.title, body);
    updateStatus();
}

function getAiOutcomeNoun() {
    return getDataset() === "diabetes" ? "diagnosis" : "prediction";
}

function getSubjectNoun() {
    return getDataset() === "diabetes" ? "patient" : "driver";
}

function getOutcomePhrase(label) {
    if (getDataset() === "diabetes") {
        return `diagnosed as ${escapeHtml(String(label ?? "").replace("No Diabetes", "Non-Diabetic").replace("Diabetes", "Diabetic"))}`;
    }
    return `predicted as ${escapeHtml(label)}`;
}

function joinHtmlClauses(clauses) {
    if (clauses.length <= 1) {
        return clauses[0] ?? "";
    }
    return `${clauses.slice(0, -1).join(", ")} and ${clauses[clauses.length - 1]}`;
}

function buildAttributionTutorialCopy(payload) {
    const outcomeNoun = getAiOutcomeNoun();
    const subjectNoun = getSubjectNoun();
    const entries = getShownAttributionEntries(payload);
    const featureNames = joinHtmlClauses(entries.map((entry) => strongText(entry.name)));
    const redLabel = payload.attribution?.direction_labels?.left;
    const blueLabel = payload.attribution?.direction_labels?.right;
    const examples = entries.map((entry) => {
        const strength = entry.percent >= 50 ? "strong" : "low";
        const sign = entry.value < 0 ? "-" : "+";
        return `${strongText(entry.name)} shows a ${strength} influence for ${getOutcomePhrase(entry.label)} (${colorText(`${sign}${entry.percent}%`, entry.colorClass)})`;
    });

    const intro = createElement("div");
    intro.appendChild(createElement(
        "p",
        "",
        `You will be shown an explanation for the AI's ${outcomeNoun}. The explanation will show the two attributes that had the strongest influence on the AI's ${outcomeNoun}.`
    ));
    intro.appendChild(createElement("p", "", "Here the explanation shows:"));
    appendTutorialList(intro, [
        `The influence of the two most important attributes (${featureNames}).<ul class="tutorial-subpoints"><li>${colorText("Red bars", "tutorial-color-red")} show the attribute(s) that contribute to the ${subjectNoun} being ${getOutcomePhrase(redLabel)}.</li><li>${colorText("Blue bars", "tutorial-color-blue")} show the attribute(s) that contribute to the ${subjectNoun} being ${getOutcomePhrase(blueLabel)}.</li></ul>`,
        `A short sentence describing the ${outcomeNoun} and influences.`,
    ], { className: "tutorial-bullets tutorial-bullets-compact", ordered: true });
    intro.appendChild(createElement("p"));
    intro.lastChild.innerHTML = `Here, ${joinHtmlClauses(examples)}. The higher the number, the stronger the influence.`;
    return intro;
}

function buildCounterfactualTutorialCopy(payload) {
    const outcomeNoun = getAiOutcomeNoun();
    const entries = getCounterfactualChangeEntries(payload);
    const targetLabel = payload.counterfactual?.prediction?.label;
    const featureNames = joinHtmlClauses(entries.map((entry) => strongText(entry.name)));
    const numericDecrease = entries.find((entry) => entry.isNumeric && entry.delta < 0);
    const numericIncrease = entries.find((entry) => entry.isNumeric && entry.delta > 0);
    const categoricalChange = entries.find((entry) => !entry.isNumeric);
    const changeExamples = entries.map((entry) => {
        if (entry.isNumeric) {
            const amount = formatTutorialValue(Math.abs(entry.delta));
            return `${strongText(entry.name)} ${entry.direction} by ${colorText(amount, entry.colorClass)}`;
        }
        return `${strongText(entry.name)} changes to ${colorText(formatTutorialValue(entry.updatedValue), entry.colorClass)}`;
    });

    const intro = createElement("div");
    intro.appendChild(createElement(
        "p",
        "",
        `You will be shown an explanation for the AI's ${outcomeNoun}. The explanation shows a counter-example in which changes to two attributes alter the ${outcomeNoun}.`
    ));
    intro.appendChild(createElement("p", "", "Here the explanation shows:"));
    const changeDetails = [];
    if (numericDecrease) {
        changeDetails.push(`${colorText("Red bars", "tutorial-color-red")} show the decrease in attribute value(s) that changes the ${outcomeNoun} to be ${escapeHtml(targetLabel)} (${colorText(`-${formatTutorialValue(Math.abs(numericDecrease.delta))}`, "tutorial-color-red")}).`);
    }
    if (numericIncrease) {
        changeDetails.push(`${colorText("Blue bars", "tutorial-color-blue")} show the increase in attribute value(s) that changes the ${outcomeNoun} to be ${escapeHtml(targetLabel)} (${colorText(`+${formatTutorialValue(Math.abs(numericIncrease.delta))}`, "tutorial-color-blue")}).`);
    }
    if (categoricalChange) {
        changeDetails.push(`${colorText("Blue markers", "tutorial-color-blue")} show changed categorical value(s) that alter the ${outcomeNoun} to be ${escapeHtml(targetLabel)} (${colorText(formatTutorialValue(categoricalChange.updatedValue), "tutorial-color-blue")}).`);
    }
    const subpoints = (items) => `<ul class="tutorial-subpoints">${items.map((item) => `<li>${item}</li>`).join("")}</ul>`;
    appendTutorialList(intro, [
        `The two attributes (${featureNames}) in the counter-example that alter the ${outcomeNoun} when they change.${subpoints(changeDetails)}`,
        `A sentence describing the ${outcomeNoun} and counter-example.${subpoints([`Here, when ${joinHtmlClauses(changeExamples)}, the ${outcomeNoun} for the counter-example changes to be ${strongText(targetLabel)}.`])}`,
    ], { className: "tutorial-bullets tutorial-bullets-compact", ordered: true });
    return intro;
}

function renderExplanationTutorialPage(step) {
    const explanationType = getExplanationType();
    const body = createElement("div", "tutorial-two-column");
    const copyPanel = createElement("div", "tutorial-copy");
    if (explanationType === "attribution") {
        copyPanel.appendChild(buildAttributionTutorialCopy(step.sampleCase.payload));
    } else {
        copyPanel.appendChild(buildCounterfactualTutorialCopy(step.sampleCase.payload));
    }
    body.appendChild(copyPanel);

    const preview = createElement("div", "tutorial-preview");
    preview.appendChild(createIframe(step.sampleCase, {
        xaiType: explanationType,
        showPrediction: 1,
        tutorialCallouts: "explanation",
        title: "Explanation interface example",
    }));
    body.appendChild(preview);

    renderTutorialPage(step.title, body);
}

function renderPhaseInstructions(step) {
    const dataset = getDataset();
    const isTraining = step.phase === "training-instructions";
    const subject = dataset === "diabetes" ? "patient" : "driver";
    const outcome = dataset === "diabetes" ? "diagnosis" : "prediction";
    const body = createElement("div", "tutorial-copy phase-instructions");

    if (isTraining) {
        const opening = createElement("p");
        opening.innerHTML = `Congratulations on passing the screening questions. Your task now is to learn how to make the correct ${outcome} for each ${subject}.`;
        body.appendChild(opening);
        body.appendChild(createElement("p", "", `You will see 10 ${subject} profiles, and for each you will:`));
        appendTutorialList(body, [
            `Provide your ${outcome} of the ${subject}.`,
            `Review the correct ${outcome} made by the AI${getExplanationType() === "none" ? "." : " and its explanation."}`,
        ], { ordered: true, className: "tutorial-bullets" });
    } else {
        body.appendChild(createElement("p", "", "Congratulations on passing the training phase!"));
        body.appendChild(createElement(
            "p",
            "",
            `Now, you will see 10 ${subject} profiles and their ${outcome}, and for each you will suggest the smallest possible (minimal) change to their profile to change their ${outcome}.`
        ));
    }
    renderTutorialPage(step.title, body);
}

function renderCurrentCase() {
    updateStatus();
    if (state.cases.length === 0) {
        showStageMessage("No cases have been started yet.");
        return;
    }

    const caseItem = state.cases[state.currentIndex];
    if (isRecordedPhase(caseItem)) {
        const shownKey = `${caseKey(caseItem)}:${state.currentIndex}`;
        if (state.lastShownStepKey !== shownKey) {
            state.lastShownStepKey = shownKey;
            logStudyEvent("case_shown", caseSnapshot(caseItem));
        }
    }
    if (caseItem.phase === "tutorial-scenario") {
        renderScenarioPage(caseItem);
        return;
    }
    if (caseItem.phase === "tutorial-basic") {
        renderBasicTutorialPage(caseItem);
        return;
    }
    if (caseItem.phase === "screening-basic" || caseItem.phase === "screening-explanation") {
        renderScreeningPage(caseItem);
        return;
    }
    if (caseItem.phase === "tutorial-explanation") {
        renderExplanationTutorialPage(caseItem);
        return;
    }
    if (caseItem.phase === "training-instructions" || caseItem.phase === "test-instructions") {
        renderPhaseInstructions(caseItem);
        return;
    }
    if (caseItem.phase === "training") {
        renderTrainingCase(caseItem);
        return;
    }
    renderTestCase(caseItem);
}

function renderTrainingCase(caseItem) {
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
    renderAnswerChoices(caseItem, answerArea, explanationPanel);
}

function renderAnswerChoices(caseItem, answerArea, explanationPanel) {
    const payload = caseItem.payload;
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
            const previousAnswer = state.answers.get(caseKey(caseItem));
            state.answers.set(caseKey(caseItem), index);
            logStudyEvent("answer_selected", {
                phase: caseItem.phase,
                caseId: caseItem.id,
                instanceId: payload.instance_id,
                previousAnswer,
                selectedAnswer: index,
                selectedLabel: label,
                correctAnswer: Number(payload.prediction?.value),
                isCorrect: index === Number(payload.prediction?.value),
            });
            renderAnswerChoices(caseItem, answerArea, explanationPanel);
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
    explanationPanel.querySelector("iframe")?.remove();
    explanationPanel.appendChild(createIframe(caseItem, {
        xaiType: getExplanationType(),
        showPrediction: 1,
        title: "Explanation",
    }));
}

function formatPredictionLabel(label) {
    const labelText = String(label ?? "");
    if (getDataset() === "diabetes") {
        return labelText.toLowerCase().includes("no") ? "Non-Diabetic" : "Diabetic";
    }
    return labelText;
}

function formatPredictionLabelForDataset(dataset, label) {
    const labelText = String(label ?? "");
    if (dataset === "diabetes") {
        return labelText.toLowerCase().includes("no") ? "Non-Diabetic" : "Diabetic";
    }
    return labelText;
}

function renderTestCase(caseItem) {
    const stage = document.querySelector("#experiment_stage");
    const copy = getCopy();
    stage.innerHTML = "";

    const simulationPanel = document.createElement("section");
    simulationPanel.className = "case-panel";
    const simulationTitle = document.createElement("h2");
    simulationTitle.textContent = copy.simulationTitle;
    simulationPanel.appendChild(simulationTitle);
    simulationPanel.appendChild(createIframe(caseItem, {
        xaiType: "none",
        showPrediction: 1,
        counterfactualSimulation: 1,
        title: copy.simulationTitle,
    }));
    stage.appendChild(simulationPanel);
}

function startRunthrough() {
    const trainingInput = document.querySelector("#experiment_training_count");
    const testInput = document.querySelector("#experiment_test_count");
    const trainingCount = clampCount(trainingInput, 10);
    const testCount = clampCount(testInput, 10);

    try {
        if (getDataset() === "diabetes") {
            validateDiabetesLabelMapping();
        }
        const caseSteps = buildCases(trainingCount, testCount);
        state.cases = [
            ...buildTutorialSteps(caseSteps),
            ...buildPhaseSteps(caseSteps),
        ];
        state.currentIndex = 0;
        state.randomizeAttributes = getRandomizeAttributesEnabled();
        state.attributeOrderSeed = state.randomizeAttributes ? createSessionSeed() : null;
        state.answers.clear();
        state.screeningAnswers.clear();
        state.screeningQuestions.clear();
        state.counterfactualChanges.clear();
        if (state.cases.length === 0) {
            showStageMessage("This setup has no cases to show.");
        } else {
            renderCurrentCase();
        }
    } catch (error) {
        state.cases = [];
        updateStatus();
        showStageMessage(String(error.message ?? error), true);
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
        const previousStep = state.cases[state.currentIndex];
        if (isRecordedPhase(previousStep)) {
            logStudyEvent(delta > 0 ? "next_clicked" : "previous_clicked", caseSnapshot(previousStep));
        }
        state.currentIndex = nextIndex;
        state.lastShownStepKey = null;
        renderCurrentCase();
    }
}

document.querySelector("#experiment_start").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "Starting…";
    try {
        await window.ExperimentLogger.startSession({
            dataset: getDataset(),
            explanationType: getExplanationType(),
            trainingCount: Number(document.querySelector("#experiment_training_count").value),
            testCount: Number(document.querySelector("#experiment_test_count").value),
        });
        state.experimentStarted = true;
        button.hidden = true;
        document.querySelector("#experiment_prev").hidden = false;
        document.querySelector("#experiment_next").hidden = false;
        state.currentIndex = Math.min(1, state.cases.length - 1);
        state.lastShownStepKey = null;
        renderCurrentCase();
    } catch (error) {
        button.disabled = false;
        button.textContent = "Start";
        showStageMessage(`The study logger could not start: ${error.message ?? error}`, true);
    }
});
document.querySelector("#experiment_prev").addEventListener("click", () => goToCase(-1));
document.querySelector("#experiment_next").addEventListener("click", () => goToCase(1));
document.querySelector("#experiment_jump").addEventListener("change", (event) => {
    jumpToCase(Number(event.target.value));
});

try {
    applyUrlConfiguration();
    startRunthrough();
} catch (error) {
    showStageMessage(String(error.message ?? error), true);
}

document.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.shiftKey && event.key.toLowerCase() === "j") {
        event.preventDefault();
        const backdoor = document.querySelector("#experiment_backdoor");
        setBackdoorVisible(backdoor?.hidden ?? true);
        return;
    }
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

setBackdoorVisible(false);
