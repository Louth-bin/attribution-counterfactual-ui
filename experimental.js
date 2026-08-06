const DATA = window.EXPERIMENT_DATA;
const DEFAULT_MODEL = DATA?.default_model ?? "mlp";

const DATASET_COPY = {
    housing: {
        label: "Housing",
        subject: "house",
        trainingQuestion: "Is this profile Cheap or Expensive?",
        explanationTitle: "Explanation and feedback",
        simulationTitle: "Make changes",
    },
    loan: {
        label: "Loan Application",
        subject: "loan application",
        trainingQuestion: "Is this profile Approved or Rejected?",
        explanationTitle: "Explanation and feedback",
        simulationTitle: "Make changes",
    },
};

const TEST_SESSIONS = {
    housing: [
        {
            id: "homeowner",
            title: "Buyer Testing Session",
            intro: [
                ["You will now see 6 ", { strong: "expensive" }, " house profiles."],
                ["For each, a person wants to buy the house, but they want it to be ", { strong: "cheap" }, "."],
                ["As a compromise, they will look for a similar cheap house. For each ", { strong: "buyer" }, ", what house profile should they look for?"],
            ],
            blocks: [
                { id: "expensive-to-cheap", sourcePrediction: 1, targetPrediction: 0 },
            ],
        },
        {
            id: "developer",
            title: "Housing Developer Testing Session",
            intro: [
                ["You will now see 6 ", { strong: "cheap" }, " house profiles."],
                ["As a ", { strong: "housing developer" }, ", you want to make small changes so each house can be listed as expensive. What changes would you make?"],
            ],
            blocks: [
                { id: "cheap-to-expensive", sourcePrediction: 0, targetPrediction: 1 },
            ],
        },
    ],
    loan: [
        {
            id: "applicant",
            title: "Loan Applicant Testing Session",
            intro: [
                ["You will now see 6 ", { strong: "rejected" }, " loan applications."],
                ["Each ", { strong: "applicant" }, " wants their loan to be approved. What should they change?"],
            ],
            blocks: [
                { id: "rejected-to-approved", sourcePrediction: 0, targetPrediction: 1 },
            ],
        },
        {
            id: "financial-advisor",
            title: "Financial Advisor Testing Session",
            intro: [
                ["You will now see 6 loan applications: 3 ", { strong: "rejected" }, " and 3 ", { strong: "approved" }, "."],
                ["As a ", { strong: "financial advisor" }, ", you will suggest small changes that could reverse each decision."],
            ],
            blocks: [
                { id: "advice", sourcePrediction: 0, targetPrediction: 1, fraction: 0.5 },
                { id: "warning", sourcePrediction: 1, targetPrediction: 0, fraction: 0.5 },
            ],
        },
    ],
};

const state = {
    cases: [],
    currentIndex: 0,
    answers: new Map(),
    screeningAnswers: new Map(),
    screeningQuestions: new Map(),
    counterfactualChanges: new Map(),
    counterfactualRawValues: new Map(),
    attributeOrderSeed: null,
    randomizeAttributes: true,
    experimentStarted: false,
    lastShownStepKey: null,
};

function logStudyEvent(eventType, details = {}) {
    return window.ExperimentLogger?.log(eventType, details) ?? Promise.resolve(false);
}

let pendingSimulationEvent = null;
let pendingSimulationTimer = null;

function flushPendingSimulationEvent() {
    if (pendingSimulationTimer !== null) {
        clearTimeout(pendingSimulationTimer);
        pendingSimulationTimer = null;
    }
    if (!pendingSimulationEvent) return Promise.resolve(false);
    const details = pendingSimulationEvent;
    pendingSimulationEvent = null;
    return logStudyEvent("simulation_changed", details);
}

function queueSimulationEvent(details) {
    pendingSimulationEvent = details;
    if (pendingSimulationTimer !== null) clearTimeout(pendingSimulationTimer);
    pendingSimulationTimer = setTimeout(flushPendingSimulationEvent, 400);
}

function isRecordedPhase(step) {
    return step?.phase === "training" || step?.phase === "test";
}

function normalizePayloadValues(payload) {
    return (payload.feature_values ?? []).map((value, index) => {
        const range = payload.feature_ranges?.[index];
        if (payload.feature_types?.[index] === "categorical") {
            const optionIndex = Array.isArray(range) ? range.indexOf(value) : -1;
            return range?.length > 1 && optionIndex >= 0 ? optionIndex / (range.length - 1) : 0;
        }
        const min = Number(range?.[0]);
        const max = Number(range?.[1]);
        const numericValue = Number(value);
        return Number.isFinite(min) && Number.isFinite(max) && max !== min
            ? (numericValue - min) / (max - min)
            : null;
    });
}

function compactPrediction(prediction) {
    return prediction ? { value: prediction.value, label: prediction.label } : null;
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
        attributeNames: step.payload.feature_names,
        instanceValues: step.payload.feature_values,
        instanceNormalizedValues: normalizePayloadValues(step.payload),
        prediction: compactPrediction(step.payload.prediction),
        stakeholderSession: step.sessionId ?? null,
        direction: step.direction ?? null,
        profileName: step.profileName ?? null,
    };
}

const DATASET_SCENARIOS = {
    housing: {
        title: "House profiles",
        intro: [
            ["In the following pages, you will see house profiles described by ", { strong: "five attributes" }, ". ",
            "Each house is either ", { strong: "Cheap" }, " or ", { strong: "Expensive" }, ". An AI can provide the correct prediction for each profile."],
        ],
        attributes: {
            sqft_living: "Interior living area in square feet",
            bedrooms: "Number of bedrooms",
            bathrooms: "Bathroom equivalent; 1.75 means 1 full and 1 three-quarter bathroom",
            floors: "Number of floors",
            grade: "Construction and design quality from 1 to 13",
        },
    },
    loan: {
        title: "Loan application profiles",
        intro: [
            ["In the following pages, you will see loan application profiles described by ", { strong: "five attributes" }, ". ",
            "Each application is either ", { strong: "Approved" }, " or ", { strong: "Rejected" }, ". An AI can provide the correct decision for each profile."],
        ],
        attributes: {
            applicant_income: "Income reported by the applicant",
            coapplicant_income: "Income reported by the co-applicant",
            loan_amount: "Requested loan amount in thousands",
            loan_term: "Requested repayment term in months",
            credit_history: "Whether the applicant has a good or bad credit history",
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
    const dataset = String(params.get("dataset") ?? "housing").toLowerCase();
    const explanation = String(params.get("explanation") ?? "attribution").toLowerCase();
    const validDatasets = new Set(["housing", "loan"]);
    const validExplanations = new Set(["attribution", "counterfactual", "none"]);

    if (!validDatasets.has(dataset)) {
        throw new Error(`Unknown dataset '${dataset}'. Use housing or loan.`);
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

function validateDatasetLabelMapping() {
    const dataset = getDataset();
    const bundle = getDatasetBundle(dataset);
    const expectedLabels = dataset === "housing"
        ? ["Cheap", "Expensive"]
        : ["Rejected", "Approved"];
    const metadataLabels = bundle.metadata?.prediction_labels ?? [];

    if (metadataLabels.length !== 2 || metadataLabels.some((label, index) => label !== expectedLabels[index])) {
        throw new Error(`${dataset} label mapping is invalid.`);
    }

    const cases = [...(bundle.training_pool ?? []), ...(bundle.test_pool ?? [])];
    cases.forEach((payload) => {
        const labels = payload.prediction_labels ?? metadataLabels;
        const predictions = [payload.prediction, payload.counterfactual?.prediction].filter(Boolean);
        predictions.forEach((prediction) => {
            const classIndex = Number(prediction.value);
            if (labels[classIndex] !== prediction.label) {
                throw new Error(
                    `${dataset} label mapping is inconsistent for case ${payload.instance_id}: ` +
                    `class ${classIndex} is '${prediction.label}', expected '${labels[classIndex]}'.`
                );
            }
        });
    });
}

function getCopy() {
    return DATASET_COPY[getDataset()] ?? DATASET_COPY.housing;
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

const PROFILE_NAMES = [
    "Mia", "Noah", "Olivia", "Liam", "Emma", "Ava", "Ethan", "Sophia", "Lucas", "Isabella",
    "Mason", "Amelia", "Elijah", "Harper", "James", "Charlotte", "Benjamin", "Evelyn", "Logan", "Abigail",
];

function positiveHash(value) {
    let hash = 0;
    const text = String(value);
    for (let index = 0; index < text.length; index += 1) {
        hash = ((hash << 5) - hash + text.charCodeAt(index)) | 0;
    }
    return Math.abs(hash);
}

function getProfileName(payload, split) {
    const key = `${getDataset()}:${split}:${payload.instance_id}`;
    return PROFILE_NAMES[positiveHash(key) % PROFILE_NAMES.length];
}

function hasDirectionalCounterfactual(payload, targetPrediction) {
    return Number(payload.counterfactual?.prediction?.value) === Number(targetPrediction);
}

function sampleDirectionBlock(pool, count, block, usedInstanceIds, session) {
    const eligible = shuffleArray(pool.filter((payload) =>
        Number(payload.prediction?.value) === Number(block.sourcePrediction) &&
        hasDirectionalCounterfactual(payload, block.targetPrediction) &&
        !usedInstanceIds.has(Number(payload.instance_id))
    ));
    if (eligible.length < count) {
        throw new Error(
            `${session.title} needs ${count} unused ${block.id} cases, but only ${eligible.length} are available.`
        );
    }
    return eligible.slice(0, count).map((payload, index) => {
        usedInstanceIds.add(Number(payload.instance_id));
        return {
            phase: "test",
            id: `${session.id}-${block.id}-${index + 1}`,
            split: "test",
            payload,
            sessionId: session.id,
            sessionTitle: session.title,
            sessionIntro: session.intro,
            direction: block.id,
            sourcePrediction: block.sourcePrediction,
            targetPrediction: block.targetPrediction,
            profileName: getProfileName(payload, "test"),
        };
    });
}

function buildStakeholderTestCases(pool, requestedCount) {
    const sessions = shuffleArray([...(TEST_SESSIONS[getDataset()] ?? [])]);
    const usedInstanceIds = new Set();
    const cases = [];

    sessions.forEach((session) => {
        const fractionalBlocks = session.blocks.filter((block) => block.fraction !== undefined);
        session.blocks.forEach((block) => {
            const blockCount = fractionalBlocks.length > 0
                ? Math.round(requestedCount * block.fraction)
                : requestedCount;
            cases.push(...sampleDirectionBlock(
                pool,
                blockCount,
                block,
                usedInstanceIds,
                session
            ));
        });
    });
    return cases;
}

function buildCases(trainingCount, testCount) {
    const bundle = getDatasetBundle();
    const pairKeys = bundle.metadata.all_feature_pair_keys ?? [];
    const trainingPayloads = sampleBalancedTrainingPool(
        bundle.training_pool,
        Math.min(trainingCount, bundle.training_pool.length),
        pairKeys
    );
    const trainingCases = trainingPayloads.map((payload) => ({
        phase: "training",
        split: "train",
        payload,
        profileName: getProfileName(payload, "train"),
    }));
    const testCases = buildStakeholderTestCases(bundle.test_pool, testCount);
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
    if (caseItem.profileName) {
        query.set("profileName", caseItem.profileName);
    }
    if (caseItem.sessionId) {
        query.set("stakeholder", caseItem.sessionId);
    }
    if (caseItem.direction) {
        query.set("direction", caseItem.direction);
    }
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
        const values = event.data.changedDisplayedValues;
        if (iframe?.dataset.caseKey && Array.isArray(values)) {
            state.counterfactualChanges.set(iframe.dataset.caseKey, [...values]);
            if (event.data.changedRawFeatureValues) {
                state.counterfactualRawValues.set(
                    iframe.dataset.caseKey,
                    { ...event.data.changedRawFeatureValues }
                );
            }
            const step = state.cases[state.currentIndex];
            if (isRecordedPhase(step)) {
                queueSimulationEvent({
                    phase: step.phase, caseId: step.id, instanceId: step.payload.instance_id,
                    changes: event.data.changes,
                });
            }
        }
        return;
    }
    if (event.data?.type === "counterfactual-ui:screen-state") {
        const step = state.cases[state.currentIndex];
        if (state.experimentStarted && step) {
            const shownType = event.data.screenState?.explanationType;
            const eventType = shownType === "attribution"
                ? "attribution_shown"
                : shownType === "counterfactual"
                    ? "counterfactual_shown"
                    : "instance_shown";
            logStudyEvent(eventType, {
                phase: step.phase,
                screenId: step.id,
                instanceId: step.payload?.instance_id ?? step.sampleCase?.payload?.instance_id ?? null,
                ...event.data.screenState,
            });
        }
        return;
    }
    if (event.data?.type === "counterfactual-ui:simulation-feedback") {
        const step = state.cases[state.currentIndex];
        if (step?.phase === "test") {
            logStudyEvent("simulation_feedback_shown", {
                phase: step.phase,
                caseId: step.id,
                instanceId: step.payload.instance_id,
                feedback: event.data.feedback,
                prediction: event.data.prediction,
                visibleText: event.data.visibleText,
                attributeNames: event.data.attributeNames,
                instanceValues: event.data.instanceValues,
                instanceNormalizedValues: event.data.instanceNormalizedValues,
                changes: event.data.changes,
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
    return DATASET_SCENARIOS[dataset] ?? DATASET_SCENARIOS.housing;
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

        payload = pool.find(hasNumericIncreaseAndDecrease);
        payload ??= pool.find((candidate) =>
            Number(candidate.counterfactual?.prediction?.value) !== Number(candidate.prediction?.value)
        );
    }

    return getPoolCase(payload ?? pool[0])
        ?? getSampleCase(cases);
}

function getSimulationTutorialSampleCase(cases) {
    const bundle = getDatasetBundle();
    const payload = (bundle.training_pool ?? []).find((candidate) =>
        Number(candidate.prediction?.value) === 0
    );
    return getPoolCase(payload) ?? getBasicTutorialSampleCase(cases);
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
    const simulationSampleCase = getSimulationTutorialSampleCase(cases);
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
            title: "Basic Interface",
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

    steps.push({
        phase: "tutorial-simulation",
        id: "counterfactual-simulation-practice",
        title: "Practice for Final Task",
        sampleCase: simulationSampleCase,
    });

    return steps;
}

function buildPhaseSteps(caseSteps) {
    const steps = [];
    let previousSection = null;
    caseSteps.forEach((caseItem) => {
        const section = caseItem.phase === "training"
            ? "training"
            : `test:${caseItem.sessionId}`;
        if (section !== previousSection) {
            steps.push({
                phase: `${caseItem.phase}-instructions`,
                id: caseItem.phase === "training"
                    ? "training-instructions"
                    : `${caseItem.sessionId}-instructions`,
                title: caseItem.phase === "training"
                    ? "Training Session"
                    : caseItem.sessionTitle,
                sessionId: caseItem.sessionId,
                sessionIntro: caseItem.sessionIntro,
            });
            previousSection = section;
        }
        steps.push(caseItem);
    });
    if (caseSteps.some((caseItem) => caseItem.phase === "test")) {
        steps.push({
            phase: "results",
            id: "testing-results",
            title: "Testing Results",
        });
    }
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
    if (
        isTutorialStep(step) ||
        String(step.phase).endsWith("-instructions") ||
        step.phase === "results"
    ) {
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
    if (
        step.phase === "tutorial-basic" ||
        step.phase === "tutorial-explanation" ||
        step.phase === "tutorial-simulation"
    ) {
        return "Tutorial";
    }
    if (step.phase === "screening-basic" || step.phase === "screening-explanation") {
        return "Screening questions";
    }
    if (String(step.phase).endsWith("-instructions")) {
        return "Instructions";
    }
    if (step.phase === "results") {
        return "Results";
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
    "tutorial-simulation": "Final task practice",
    "training-instructions": "Training session instructions",
    training: "First training case",
    results: "Testing results",
};

function getBackdoorStepKey(step) {
    if (step?.phase === "test-instructions" || step?.phase === "test") {
        return `${step.phase}:${step.sessionId ?? step.id}`;
    }
    return step?.phase;
}

function getBackdoorStepLabel(step) {
    if (step.phase === "test-instructions") {
        return `${step.title ?? "Testing session"} - instructions`;
    }
    if (step.phase === "test") {
        return `${step.sessionTitle ?? "Testing session"} - first case`;
    }
    return BACKDOOR_LABELS[step.phase] ?? step.title ?? step.phase;
}

function updateBackdoorMenu() {
    const select = document.querySelector("#experiment_jump");
    if (!select) {
        return;
    }
    const firstIndexByStepKey = new Map();
    state.cases.forEach((step, index) => {
        const key = getBackdoorStepKey(step);
        if (key && !firstIndexByStepKey.has(key)) {
            firstIndexByStepKey.set(key, index);
        }
    });
    select.innerHTML = "";
    if (firstIndexByStepKey.size === 0) {
        select.appendChild(new Option("Start a runthrough first", ""));
        select.disabled = true;
        return;
    }
    firstIndexByStepKey.forEach((index) => {
        const step = state.cases[index];
        select.appendChild(new Option(getBackdoorStepLabel(step), String(index)));
    });
    const currentStepKey = getBackdoorStepKey(state.cases[state.currentIndex]);
    const currentStepIndex = firstIndexByStepKey.get(currentStepKey);
    select.value = String(currentStepIndex ?? state.currentIndex);
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
    body.appendChild(intro);

    const table = createElement("table", "scenario-table");
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    ["Attribute", "Description", "Value range"].forEach((header) => {
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
    const body = createElement("div", "tutorial-two-column");
    const copyPanel = createElement("div", "tutorial-copy");
    copyPanel.appendChild(createElement(
        "p",
        "",
        `Each ${getCopy().subject} profile is shown using the following interface.`
    ));
    const exampleFeature = getDataset() === "housing"
        ? "Living Area"
        : "Amount";
    const predictionText = getDataset() === "housing"
        ? `The selected box shows the AI's price ${strongText("prediction")}.`
        : `The selected box shows the AI's loan ${strongText("decision")}.`;
    appendTutorialBullets(copyPanel, [
        `The five <strong>attributes</strong> describing the ${getCopy().subject}.`,
        "The <strong>values</strong> of each attribute.",
        `Bars indicating how ${strongText("low/high")} the value is for that attribute (${getFeatureRangeExample(step.sampleCase.payload, exampleFeature)}).`,
        predictionText,
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
            choices: ["Low / High", "Attribute", "AI prediction", "Value"],
            correct: "Low / High",
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
                prompt: "What does this explanation show?",
                choices: [
                    "The two attributes that most influenced the AI's decision",
                    "The original source of the dataset",
                    "A random list of unused attributes",
                    "The participant's final answer",
                ],
                correct: "The two attributes that most influenced the AI's decision",
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
            prompt: "What does this explanation show?",
            choices: [
                "How two attribute values could change the AI's decision",
                "How each attribute contributes to the current prediction",
                "How accurate the participant's answer was",
                "The order in which profiles are sampled",
            ],
            correct: "How two attribute values could change the AI's decision",
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
            const previousAnswer = state.screeningAnswers.get(key);
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
            logStudyEvent("screening_answer_changed", {
                screenIndex: state.currentIndex,
                phase: step.phase,
                screenId: step.id,
                questionId: question.id,
                question: question.prompt,
                clickedChoice: choice,
                previousAnswer,
                answer: state.screeningAnswers.get(key),
            });
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

function joinHtmlClauses(clauses) {
    if (clauses.length <= 1) {
        return clauses[0] ?? "";
    }
    return `${clauses.slice(0, -1).join(", ")} and ${clauses[clauses.length - 1]}`;
}

function buildAttributionTutorialCopy(payload) {
    const decision = getDataset() === "housing" ? "price prediction" : "loan decision";
    const entries = getShownAttributionEntries(payload);
    const redLabel = payload.attribution?.direction_labels?.left;
    const blueLabel = payload.attribution?.direction_labels?.right;
    const influenceExamples = entries.map((entry) => {
        const sign = entry.value < 0 ? "-" : "+";
        return `${escapeHtml(entry.name)}, ${colorText(`${sign}${entry.percent}%`, entry.colorClass)}`;
    });
    const intro = createElement("div");
    const opening = createElement("p");
    opening.innerHTML = `To learn how the AI predicts, you will sometimes see an explanation for the AI's ${decision}. This explanation shows the ${strongText("two most important attributes")} for that profile.`;
    intro.appendChild(opening);
    intro.appendChild(createElement("p", "", "The explanation will show:"));
    appendTutorialList(intro, [
        `The influence of the two most important attributes (${joinHtmlClauses(influenceExamples)}). The higher the number, the stronger the influence.<ul class="tutorial-subpoints"><li>${colorText("Red bars", "tutorial-color-red")} show the attribute(s) that contribute to a ${strongText(String(redLabel).toLowerCase())} decision.</li><li>${colorText("Blue bars", "tutorial-color-blue")} show the attribute(s) that contribute to an ${strongText(String(blueLabel).toLowerCase())} decision.</li></ul>`,
        `A sentence describing the ${decision} and the influences.`,
    ], { className: "tutorial-bullets tutorial-bullets-compact", ordered: true });
    return intro;
}

function buildCounterfactualTutorialCopy(payload) {
    const decision = getDataset() === "housing" ? "price prediction" : "loan decision";
    const entries = getCounterfactualChangeEntries(payload);
    const numericDecrease = entries.find((entry) => entry.isNumeric && entry.delta < 0);
    const numericIncrease = entries.find((entry) => entry.isNumeric && entry.delta > 0);
    const categoricalChange = entries.find((entry) => !entry.isNumeric);

    const intro = createElement("div");
    const opening = createElement("p");
    opening.innerHTML = `To learn how the AI predicts, you will sometimes see an explanation for the AI's ${decision}. This explanation shows a ${strongText("counter-example")}, where ${strongText("two attributes are changed")} to alter the ${decision}.`;
    intro.appendChild(opening);
    intro.appendChild(createElement("p", "", "The explanation will show:"));
    const changeDetails = [];
    if (numericDecrease) {
        changeDetails.push(`${colorText("Red bars", "tutorial-color-red")} show decreases in value.`);
    }
    if (numericIncrease) {
        changeDetails.push(`${colorText("Blue bars", "tutorial-color-blue")} show increases in value.`);
    }
    if (categoricalChange) {
        changeDetails.push(`${colorText("Blue markers", "tutorial-color-blue")} show changed options.`);
    }
    const subpoints = (items) => `<ul class="tutorial-subpoints">${items.map((item) => `<li>${item}</li>`).join("")}</ul>`;
    appendTutorialList(intro, [
        `The ${strongText("change in the two attributes")} in the counter-example.${subpoints(changeDetails)}`,
        `A sentence describing the ${decision} and the effect of the changes.`,
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

function renderSimulationPracticePage(step) {
    const body = createElement("div", "tutorial-two-column");
    const copyPanel = createElement("div", "tutorial-copy");
    copyPanel.appendChild(createElement("p", "", "Congratulations on passing the screening questions!"));
    const previewText = createElement("p");
    previewText.innerHTML = `Before the main experiment, here is a ${strongText("preview of the main testing task")}.`;
    copyPanel.appendChild(previewText);
    const task = createElement("p");
    task.innerHTML = getDataset() === "housing"
        ? `You will be asked to ${strongText("change Cheap houses to Expensive ones or vice versa")}. Make changes in the Changes column.`
        : `You will be asked to ${strongText("change Rejected loan applications to Approved ones or vice versa")}. Make changes in the Changes column.`;
    copyPanel.appendChild(task);
    copyPanel.appendChild(createElement(
        "p",
        "",
        "Click “Reset changes” to start again. You will not be told whether your practice changes are correct or wrong."
    ));
    body.appendChild(copyPanel);

    const preview = createElement("div", "tutorial-preview");
    preview.appendChild(createIframe(step.sampleCase, {
        xaiType: "none",
        showPrediction: 1,
        counterfactualSimulation: 1,
        title: "Practice for Final Task",
    }));
    body.appendChild(preview);
    renderTutorialPage(step.title, body);
}

function renderPhaseInstructions(step) {
    const isTraining = step.phase === "training-instructions";
    const body = createElement("div", "tutorial-copy phase-instructions");

    if (isTraining) {
        const opening = createElement("p");
        opening.innerHTML = getDataset() === "housing"
            ? `You will now try to ${strongText("learn which profiles")} are cheap and expensive.`
            : `You will now try to ${strongText("learn which profiles")} are approved and rejected.`;
        body.appendChild(opening);
        const profileCount = createElement("p");
        profileCount.innerHTML = `You will see ${strongText("10 profiles")}. For each, you will:`;
        body.appendChild(profileCount);
        appendTutorialList(body, [
            `${strongText("Predict")} whether the profile is ${getDataset() === "housing" ? "cheap or expensive" : "approved or rejected"}.`,
            `${strongText("Review")} the correct answer${getExplanationType() === "none" ? "." : " and the explanation."}`,
        ], { ordered: true, className: "tutorial-bullets" });
    } else {
        (Array.isArray(step.sessionIntro) ? step.sessionIntro : [step.sessionIntro]).forEach((paragraph) => {
            const text = createElement("p");
            appendFormattedText(text, paragraph);
            body.appendChild(text);
        });
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
    if (state.experimentStarted) {
        requestAnimationFrame(() => {
            logStudyEvent("screen_viewed", {
                screenIndex: state.currentIndex,
                phase: caseItem.phase,
                screenId: caseItem.id,
                title: caseItem.title ?? getPhaseLabel(caseItem),
            });
        });
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
    if (caseItem.phase === "tutorial-simulation") {
        renderSimulationPracticePage(caseItem);
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
    if (caseItem.phase === "results") {
        renderTestingResults(caseItem);
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
    originalPanel.appendChild(createIframe(caseItem, {
        xaiType: "none",
        showPrediction: 0,
        short: true,
        title: `${copy.subject} profile`,
    }));
    layout.appendChild(originalPanel);

    const answerPanel = document.createElement("section");
    answerPanel.className = "case-panel";
    const answerTitle = document.createElement("p");
    answerTitle.className = "training-question";
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
                ...caseSnapshot(caseItem),
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
    return String(label ?? "");
}

function formatPredictionLabelForDataset(dataset, label) {
    return String(label ?? "");
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

function rawValuesDiffer(firstValue, secondValue) {
    const firstNumber = Number(firstValue);
    const secondNumber = Number(secondValue);
    if (Number.isFinite(firstNumber) && Number.isFinite(secondNumber)) {
        return Math.abs(firstNumber - secondNumber) > 1e-9;
    }
    return String(firstValue) !== String(secondValue);
}

function calculateTestingResults() {
    const testCases = state.cases.filter((step) => step.phase === "test");
    const attempts = [];

    testCases.forEach((step) => {
        const rawValues = state.counterfactualRawValues.get(caseKey(step));
        if (!rawValues) {
            return;
        }
        const changedFeatureCount = (step.payload.raw_feature_names ?? [])
            .filter((featureName, index) => rawValuesDiffer(
                rawValues[featureName],
                step.payload.raw_feature_values[index]
            )).length;
        if (changedFeatureCount === 0) {
            return;
        }

        const prediction = window.StaticModel.predictDataset(
            DATA,
            getDataset(),
            rawValues
        );
        attempts.push({
            caseId: step.id,
            instanceId: step.payload.instance_id,
            sessionId: step.sessionId,
            direction: step.direction,
            changedFeatureCount,
            prediction: compactPrediction(prediction),
            targetPrediction: Number(step.targetPrediction),
            valid: Number(prediction.value) === Number(step.targetPrediction),
        });
    });

    const validCount = attempts.filter((attempt) => attempt.valid).length;
    return {
        totalTestCount: testCases.length,
        attemptedCount: attempts.length,
        validCount,
        successRate: attempts.length > 0
            ? Math.round((validCount / attempts.length) * 100)
            : null,
        attempts,
    };
}

function appendResultMetric(container, value, label) {
    const metric = createElement("div", "results-metric");
    metric.appendChild(createElement("div", "results-metric-value", value));
    metric.appendChild(createElement("div", "results-metric-label", label));
    container.appendChild(metric);
}

function renderTestingResults(step) {
    const results = calculateTestingResults();
    const body = createElement("div", "testing-results");
    const explanation = results.attemptedCount > 0
        ? `You made changes in ${results.attemptedCount} of ${results.totalTestCount} test instances. Untouched instances are not included in the success rate.`
        : `You did not make changes in any of the ${results.totalTestCount} test instances, so no success rate was calculated.`;
    body.appendChild(createElement("p", "results-intro", explanation));

    const metrics = createElement("div", "results-metrics");
    appendResultMetric(metrics, String(results.attemptedCount), "Attempted instances");
    appendResultMetric(metrics, String(results.validCount), "Valid counterfactuals");
    appendResultMetric(
        metrics,
        results.successRate === null ? "—" : `${results.successRate}%`,
        "Success rate"
    );
    body.appendChild(metrics);
    body.appendChild(createElement(
        "p",
        "results-note",
        "A counterfactual is valid when the edited profile makes the AI reach the required target prediction for that instance."
    ));
    renderTutorialPage(step.title, body);

    if (state.experimentStarted) {
        logStudyEvent("testing_results_shown", results);
    }
}

function startRunthrough() {
    const trainingInput = document.querySelector("#experiment_training_count");
    const testInput = document.querySelector("#experiment_test_count");
    const trainingCount = clampCount(trainingInput, 10);
    const testCount = clampCount(testInput, 6);

    try {
        validateDatasetLabelMapping();
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
        state.counterfactualRawValues.clear();
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
        if (state.experimentStarted) {
            flushPendingSimulationEvent();
            logStudyEvent(delta > 0 ? "next_clicked" : "previous_clicked", {
                fromIndex: state.currentIndex,
                toIndex: nextIndex,
                fromPhase: previousStep.phase,
                fromScreenId: previousStep.id,
                toPhase: state.cases[nextIndex]?.phase,
                toScreenId: state.cases[nextIndex]?.id,
            });
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
