const urlParams = new URLSearchParams(window.location.search);
const explanationType = getExplanationType(urlParams.get("xaiType") ?? "none");
const explanationView = getExplanationView(urlParams.get("explanationView") ?? "classic");
const showPredictionPanel = urlParams.get("showPrediction") !== "0";
const datasetName = urlParams.get("appId") ?? "diabetes";
const modelName = urlParams.get("AIModel") ?? "mlp";
const xaiMethod = urlParams.get("expAlgorithm") ?? "shap";
const splitName = urlParams.get("split") ?? "test";
const instanceId = Number(urlParams.get("instanceId") ?? "0");
const explanationFeatureCount = Number(urlParams.get("k") ?? "2");
const attributeOrderSeed = urlParams.get("attributeOrderSeed");
const tutorialCalloutMode = urlParams.get("tutorialCallouts") ?? "";
const faceFiguresEnabled = urlParams.get("faceFigures") === "1";
const counterfactualSimulationEnabled = urlParams.get("counterfactualSimulation") === "1";
const counterfactualSimulationMode = getCounterfactualSimulationMode(urlParams.get("simulationMode"));
const SIMULATION_SPECIFIC_ATTRIBUTE_COUNT = 2;
const SIMULATION_BUDGET_POINTS = 10;
const PROFILE_SUBJECT_NAMES = [
    "Mia",
    "Noah",
    "Olivia",
    "Liam",
    "Emma",
    "Ava",
    "Ethan",
    "Sophia",
    "Lucas",
    "Isabella",
    "Mason",
    "Amelia",
    "Elijah",
    "Harper",
    "James",
    "Charlotte",
    "Benjamin",
    "Evelyn",
    "Logan",
    "Abigail",
];
const noneExplanationTbody = document.querySelector("#none-explanation-tbody");
const tablesWrapper = document.querySelector("#tables-wrapper");
const explanationBoxAnchor = document.querySelector("#explanation-box-anchor");
let currentExplanation = null;
let attributionChart = null;
let simulationValues = null;
let simulationAllowedAttributeIndices = null;
let simulationSpecificCandidatePending = false;
let simulationPrediction = null;
let simulationFeedback = null;

console.info("[iframe] static data mode:", Boolean(window.EXPERIMENT_DATA));

function getStaticDatasetBundle() {
    const bundle = window.EXPERIMENT_DATA?.datasets?.[datasetName];
    if (!bundle) {
        throw new Error(`Dataset '${datasetName}' is not included in the static experiment data.`);
    }
    return bundle;
}

function getStaticExplanationPayload() {
    const bundle = getStaticDatasetBundle();
    const poolName = splitName === "train" ? "training_pool" : "test_pool";
    const payload = bundle[poolName]?.find((candidate) =>
        Number(candidate.instance_id) === instanceId
    );
    if (!payload) {
        throw new Error(`Instance ${instanceId} is not included in the static ${splitName} pool.`);
    }
    return payload;
}

function getExplanationType(selectedType) {
    const normalizedType = String(selectedType ?? "none").toLowerCase();

    if (normalizedType === "counterfactual" || normalizedType === "counterfactuals") {
        return "counterfactual";
    }

    if (normalizedType === "attribution") {
        return "attribution";
    }

    return "none";
}

function getExplanationView(selectedView) {
    const normalizedView = String(selectedView ?? "classic").toLowerCase();

    if (normalizedView === "inline" || normalizedView === "inline-change") {
        return "inline";
    }

    if (normalizedView === "narrative" || normalizedView === "text") {
        return "narrative";
    }

    if (normalizedView === "persona" || normalizedView === "direct") {
        return "persona";
    }

    return "classic";
}

function getCounterfactualSimulationMode(selectedMode) {
    return "any";
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function createMeterThumb(position, label, className = "") {
    const thumb = document.createElement("span");
    thumb.className = `meter-thumb ${className}`.trim();
    thumb.style.left = `${position}%`;
    thumb.title = label;
    return thumb;
}

function createMeterDeltaLabel(attributeIndex, originalValues, updatedValues) {
    if (currentExplanation.attributeTypes[attributeIndex] === "categorical") {
        return null;
    }

    const originalValue = Number(originalValues?.[attributeIndex]);
    const updatedValue = Number(updatedValues?.[attributeIndex]);
    const delta = updatedValue - originalValue;
    if (!Number.isFinite(originalValue) || !Number.isFinite(updatedValue) || delta === 0) {
        return null;
    }

    const deltaLabel = document.createElement("span");
    deltaLabel.className = delta > 0
        ? "meter-side-delta value-delta value-delta-increase"
        : "meter-side-delta value-delta value-delta-decrease";
    deltaLabel.textContent = `${delta > 0 ? "+" : "-"} ${formatValue(Math.abs(delta))}`;
    deltaLabel.title = `Changed by ${formatValue(delta)}`;
    return deltaLabel;
}

function formatValue(value) {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) {
        return String(value);
    }

    if (Number.isInteger(numericValue) && Math.abs(numericValue) >= 100) {
        return String(numericValue);
    }

    return numericValue.toPrecision(3).replace(/\.?0+($|e)/, "$1");
}

function getAttributeName(attributeIndex) {
    return formatAttributeName(currentExplanation.attributeNames[attributeIndex]);
}

function shortenAttributeName(name) {
    return String(name ?? "").replace(/diabetes pedigree function/ig, "diabetes pedigree");
}

function formatAttributeName(name) {
    return shortenAttributeName(name)
        .replace(/[_-]+/g, " ")
        .split(/\s+/)
        .filter(Boolean)
        .map((word) => {
            if (word === word.toUpperCase()) {
                return word;
            }
            return `${word.charAt(0).toUpperCase()}${word.slice(1).toLowerCase()}`;
        })
        .join(" ");
}

function getNarrativeAttributeName(attributeIndex) {
    return formatAttributeName(currentExplanation.attributeNames[attributeIndex]);
}

function getAttributeControlHeader() {
    const attributeTypes = currentExplanation?.attributeTypes ?? [];
    const categoricalCount = attributeTypes.filter((type) => type === "categorical").length;

    if (categoricalCount === 0) {
        return "Low / High";
    }
    if (categoricalCount === attributeTypes.length) {
        return "Options";
    }
    return "Range / Options";
}

function updateAttributeControlHeader() {
    const header = document.querySelector("#none-explanation-thead .meter-scale-column-header");
    if (header) {
        header.textContent = getAttributeControlHeader();
    }
}

function createTutorialCallout(number, className) {
    const callout = document.createElement("span");
    callout.className = `tutorial-callout ${className}`;
    callout.textContent = String(number);
    callout.title = `Tutorial point ${number}`;
    callout.setAttribute("aria-hidden", "true");
    return callout;
}

function applyBasicTutorialCallouts() {
    if (tutorialCalloutMode !== "basic") {
        return;
    }

    const attributeHeader = document.querySelector("#none-explanation-thead tr:not(.case-label-row) th:nth-child(1)");
    const valueHeader = document.querySelector("#none-explanation-thead tr:not(.case-label-row) th:nth-child(2)");
    const scaleHeader = document.querySelector("#none-explanation-thead .meter-scale-column-header");
    [attributeHeader, valueHeader, scaleHeader].forEach((header, index) => {
        if (!header) {
            return;
        }
        header.classList.add("tutorial-callout-anchor");
        header.appendChild(createTutorialCallout(index + 1, "tutorial-callout-header"));
    });

    const predictionLabel = document.querySelector(".prediction-panel-label");
    if (predictionLabel) {
        predictionLabel.classList.add("tutorial-callout-anchor");
        predictionLabel.appendChild(createTutorialCallout(4, "tutorial-callout-prediction"));
    }
}

function applyExplanationTutorialCallouts() {
    if (tutorialCalloutMode !== "explanation") {
        return;
    }

    const explanationTarget = document.querySelector(".attribution-header")
        ?? document.querySelector(".counterexample-column-header")
        ?? document.querySelector("#counterfactual-table .meter-scale-column-header");
    if (explanationTarget) {
        explanationTarget.classList.add("tutorial-callout-anchor");
        explanationTarget.appendChild(createTutorialCallout(1, "tutorial-callout-header"));
    }

    const narrativePanel = document.querySelector("#narrative-panel");
    if (narrativePanel) {
        narrativePanel.classList.add("tutorial-callout-anchor");
        narrativePanel.appendChild(createTutorialCallout(2, "tutorial-callout-panel"));
    }
}

function applyTutorialCallouts() {
    if (!tutorialCalloutMode) {
        return;
    }

    document.body.classList.add("tutorial-callouts-active");
    document.querySelectorAll(".tutorial-callout").forEach((callout) => callout.remove());
    applyBasicTutorialCallouts();
    applyExplanationTutorialCallouts();
}

function renderStatusRow(message, isError = false) {
    noneExplanationTbody.innerHTML = "";
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = message;
    cell.style.padding = "1rem";
    cell.style.color = isError ? "#b42318" : "#444";
    row.appendChild(cell);
    noneExplanationTbody.appendChild(row);
}

function createPersonaMeterCell(attributeIndex, originalValues, updatedValues) {
    const meterCell = document.createElement("td");
    meterCell.className = "meter-container persona-meter-container";

    const [min, max] = currentExplanation.attributeRanges[attributeIndex];
    const originalValue = Number(originalValues[attributeIndex]);
    const updatedValue = Number(updatedValues[attributeIndex]);
    const rangeSpan = max - min || 1;
    const originalPosition = clamp(((originalValue - min) / rangeSpan) * 100, 0, 100);
    const updatedPosition = clamp(((updatedValue - min) / rangeSpan) * 100, 0, 100);
    const isIncrease = updatedValue >= originalValue;
    const blackEnd = isIncrease ? originalPosition : updatedPosition;
    const deltaStart = Math.min(originalPosition, updatedPosition);
    const deltaWidth = Math.abs(updatedPosition - originalPosition);
    const blackEndValue = min + ((blackEnd / 100) * rangeSpan);

    const meterContent = document.createElement("div");
    meterContent.className = "persona-meter-with-delta";

    const meterStack = document.createElement("div");
    meterStack.className = "persona-meter-stack";
    meterStack.title = `${formatValue(updatedValue)} (original: ${formatValue(originalValue)}, min: ${formatValue(min)}, max: ${formatValue(max)})`;

    const baseMeter = document.createElement("meter");
    baseMeter.className = "persona-meter-base";
    baseMeter.min = min;
    baseMeter.max = max;
    baseMeter.value = clamp(blackEndValue, min, max);
    meterStack.appendChild(baseMeter);

    if (deltaWidth > 0) {
        const deltaSegment = document.createElement("div");
        deltaSegment.className = "persona-meter-segment";
        deltaSegment.style.left = `${deltaStart}%`;
        deltaSegment.style.width = `${deltaWidth}%`;

        const deltaMeter = document.createElement("meter");
        deltaMeter.className = isIncrease ? "persona-meter-increase" : "persona-meter-decrease";
        deltaMeter.min = 0;
        deltaMeter.max = 1;
        deltaMeter.value = 1;
        deltaSegment.appendChild(deltaMeter);
        meterStack.appendChild(deltaSegment);
    }

    meterStack.appendChild(createMeterThumb(
        updatedPosition,
        `Current: ${formatValue(updatedValue)}`,
        "meter-thumb-current"
    ));

    meterContent.appendChild(meterStack);

    const deltaLabel = createMeterDeltaLabel(attributeIndex, originalValues, updatedValues);
    if (deltaLabel) {
        meterContent.appendChild(deltaLabel);
    } else {
        const spacer = document.createElement("span");
        spacer.className = "meter-side-delta meter-side-delta-empty";
        meterContent.appendChild(spacer);
    }

    meterCell.appendChild(meterContent);
    return meterCell;
}

function createCategoryIconsCell(options, categoryIndex, cellOptions = {}) {
    const { muted = false } = cellOptions;

    const dataCell = document.createElement("td");
    dataCell.className = "icons-container";

    for (let j = 0; j < options.length; j++) {
        const icon = document.createElement("i");
        const isSelected = j === categoryIndex;
        icon.className = isSelected ? "fas fa-circle" : "far fa-circle";
        icon.style.color = isSelected ? (muted ? "#777" : "black") : (muted ? "#c2c2c2" : "#aaa");
        icon.title = options[j];
        icon.style.cursor = "pointer";
        icon.style.margin = "0 2px";
        icon.style.fontSize = "12px";
        dataCell.appendChild(icon);
    }

    return dataCell;
}

function createMeterCell(value, min, max, cellOptions = {}) {
    const { muted = false } = cellOptions;
    const meterCell = document.createElement("td");
    meterCell.className = "meter-container";
    const rangeSpan = max - min || 1;
    const thumbPosition = clamp(((Number(value) - min) / rangeSpan) * 100, 0, 100);

    const meterStack = document.createElement("div");
    meterStack.className = "value-meter-stack";

    const meter = document.createElement("meter");
    if (muted) {
        meter.classList.add("meter-unchanged");
    }
    meter.min = min;
    meter.max = max;
    meter.value = clamp(value, min, max);
    meter.title = `${formatValue(value)} (min: ${formatValue(min)}, max: ${formatValue(max)})`;
    meterStack.appendChild(meter);
    meterStack.appendChild(createMeterThumb(
        thumbPosition,
        `Current: ${formatValue(value)}`,
        "meter-thumb-current"
    ));
    meterCell.appendChild(meterStack);

    return meterCell;
}

function createComparisonMeterCell(attributeIndex, originalValues, updatedValues, cellOptions = {}) {
    const { muted = false } = cellOptions;
    const meterCell = document.createElement("td");
    meterCell.className = "meter-container";

    const [min, max] = currentExplanation.attributeRanges[attributeIndex];
    const originalValue = Number(originalValues[attributeIndex]);
    const updatedValue = Number(updatedValues[attributeIndex]);
    const rangeSpan = max - min || 1;
    const originalPosition = clamp(((originalValue - min) / rangeSpan) * 100, 0, 100);
    const updatedPosition = clamp(((updatedValue - min) / rangeSpan) * 100, 0, 100);

    const comparisonMeter = document.createElement("div");
    comparisonMeter.className = "inline-meter-wrapper";
    if (muted) {
        comparisonMeter.style.opacity = "0.45";
    }

    const meter = document.createElement("meter");
    meter.min = min;
    meter.max = max;
    meter.value = clamp(updatedValue, min, max);
    meter.title = `${formatValue(updatedValue)} (min: ${formatValue(min)}, max: ${formatValue(max)})`;
    comparisonMeter.appendChild(meter);

    if (originalPosition !== updatedPosition) {
        const arrowHeadWidth = 8;
        const arrowTrack = document.createElement("div");
        arrowTrack.className = "comparison-arrow-track";

        const origin = document.createElement("div");
        origin.className = "comparison-arrow-origin";
        origin.style.left = `${originalPosition}%`;
        origin.title = `Original: ${formatValue(originalValue)}`;
        arrowTrack.appendChild(origin);

        const arrowLine = document.createElement("div");
        arrowLine.className = "comparison-arrow-line";
        const movesRight = updatedPosition >= originalPosition;
        const lineStart = movesRight
            ? originalPosition
            : updatedPosition;
        const lineEnd = movesRight
            ? Math.max(updatedPosition - arrowHeadWidth, originalPosition)
            : originalPosition;
        const linePercent = Math.max(lineEnd - lineStart, 0);
        const startPixelOffset = movesRight ? 1 : arrowHeadWidth;
        const widthPixelOffset = movesRight ? -1 : -arrowHeadWidth;
        arrowLine.style.left = `calc(${lineStart}% + ${startPixelOffset}px)`;
        arrowLine.style.width = linePercent > 0
            ? `calc(${linePercent}% ${widthPixelOffset < 0 ? "-" : "+"} ${Math.abs(widthPixelOffset)}px)`
            : "0";
        if (linePercent > 0) {
            arrowTrack.appendChild(arrowLine);
        }

        const arrowHead = document.createElement("div");
        const movesRightClass = updatedPosition >= originalPosition;
        arrowHead.className = movesRightClass
            ? "comparison-arrow-head comparison-arrow-head-right"
            : "comparison-arrow-head comparison-arrow-head-left";
        arrowHead.style.left = movesRightClass
            ? `calc(${updatedPosition}% - ${arrowHeadWidth}px)`
            : `${updatedPosition}%`;
        arrowHead.title = `Changed to ${formatValue(updatedValue)}`;
        arrowTrack.appendChild(arrowHead);

        comparisonMeter.appendChild(arrowTrack);
    }

    meterCell.appendChild(comparisonMeter);
    return meterCell;
}

function getCategoryIndex(attributeIndex, values) {
    const options = currentExplanation.attributeRanges[attributeIndex];
    const rawValue = values[attributeIndex];
    const numericIndex = Number(rawValue);
    if (Number.isFinite(numericIndex)) {
        return clamp(numericIndex, 0, options.length - 1);
    }

    const resolvedIndex = options.findIndex((option) => String(option) === String(rawValue));
    return resolvedIndex >= 0 ? resolvedIndex : 0;
}

function getAttributeDisplayValue(attributeIndex, values) {
    if (currentExplanation.attributeTypes[attributeIndex] === "categorical") {
        const options = currentExplanation.attributeRanges[attributeIndex];
        const categoryIndex = getCategoryIndex(attributeIndex, values);
        return options[categoryIndex] ?? String(values[attributeIndex]);
    }

    return formatValue(values[attributeIndex]);
}

function hasAttributeChanged(attributeIndex, originalValues, updatedValues) {
    if (currentExplanation.attributeTypes[attributeIndex] === "categorical") {
        return (
            getAttributeDisplayValue(attributeIndex, originalValues) !==
            getAttributeDisplayValue(attributeIndex, updatedValues)
        );
    }

    const originalValue = Number(originalValues[attributeIndex]);
    const updatedValue = Number(updatedValues[attributeIndex]);

    if (Number.isFinite(originalValue) && Number.isFinite(updatedValue)) {
        return originalValue !== updatedValue;
    }

    return String(originalValues[attributeIndex]) !== String(updatedValues[attributeIndex]);
}

function createComparisonCategoryCell(attributeIndex, originalValues, updatedValues, cellOptions = {}) {
    const { muted = false } = cellOptions;
    const options = currentExplanation.attributeRanges[attributeIndex];
    const originalIndex = getCategoryIndex(attributeIndex, originalValues);
    const updatedIndex = getCategoryIndex(attributeIndex, updatedValues);

    const dataCell = document.createElement("td");
    dataCell.className = "icons-container";

    const iconRow = document.createElement("div");
    iconRow.className = "comparison-icons";

    for (let j = 0; j < options.length; j++) {
        const icon = document.createElement("span");
        icon.className = "comparison-icon";
        if (j === updatedIndex) {
            icon.classList.add("comparison-icon-current");
        }
        if (j === originalIndex && originalIndex !== updatedIndex) {
            icon.classList.add("comparison-icon-original");
        }
        if (muted) {
            icon.classList.add("comparison-icon-muted");
        }
        icon.title = options[j];
        iconRow.appendChild(icon);
    }

    dataCell.appendChild(iconRow);
    return dataCell;
}

function createPersonaCategoryCell(attributeIndex, originalValues, updatedValues) {
    const options = currentExplanation.attributeRanges[attributeIndex];
    const originalIndex = getCategoryIndex(attributeIndex, originalValues);
    const updatedIndex = getCategoryIndex(attributeIndex, updatedValues);

    const dataCell = document.createElement("td");
    dataCell.className = "icons-container";

    const iconRow = document.createElement("div");
    iconRow.className = "comparison-icons persona-comparison-icons";

    for (let j = 0; j < options.length; j++) {
        const icon = document.createElement("span");
        icon.className = "comparison-icon";
        if (j === originalIndex && originalIndex !== updatedIndex) {
            icon.classList.add("persona-icon-original");
        }
        if (j === updatedIndex && originalIndex !== updatedIndex) {
            icon.classList.add("persona-icon-updated");
        } else if (j === updatedIndex) {
            icon.classList.add("persona-icon-current");
        }
        icon.title = options[j];
        iconRow.appendChild(icon);
    }

    const categoryContent = document.createElement("div");
    categoryContent.className = "persona-category-with-label";
    categoryContent.appendChild(iconRow);

    const updatedLabel = createCategoricalChangeLabel(
        attributeIndex,
        originalValues,
        updatedValues
    );
    if (updatedLabel) {
        categoryContent.appendChild(updatedLabel);
    }

    dataCell.appendChild(categoryContent);
    return dataCell;
}

function createCategoricalChangeLabel(attributeIndex, originalValues, updatedValues) {
    if (
        currentExplanation.attributeTypes[attributeIndex] !== "categorical" ||
        !hasAttributeChanged(attributeIndex, originalValues, updatedValues)
    ) {
        return null;
    }

    const label = document.createElement("span");
    label.className = "categorical-change-label value-delta value-delta-increase";
    label.textContent = getAttributeDisplayValue(attributeIndex, updatedValues);
    return label;
}

function createDiffCell(attributeIndex, originalValues, updatedValues) {
    const diffCell = document.createElement("td");
    diffCell.className = "diff-cell";

    const diffContent = document.createElement("div");
    diffContent.className = "diff-content";
    const isChanged = hasAttributeChanged(attributeIndex, originalValues, updatedValues);

    if (!isChanged) {
        diffCell.appendChild(diffContent);
        return diffCell;
    }

    if (currentExplanation.attributeTypes[attributeIndex] === "categorical") {
        const originalLabel = getAttributeDisplayValue(attributeIndex, originalValues);
        const updatedLabel = getAttributeDisplayValue(attributeIndex, updatedValues);

        const diffText = document.createElement("span");
        diffText.className = "diff-text";
        diffText.textContent = `${originalLabel} -> ${updatedLabel}`;
        diffText.title = `Changed from ${originalLabel} to ${updatedLabel}`;

        diffContent.appendChild(diffText);
        diffCell.appendChild(diffContent);
        return diffCell;
    }

    const originalValue = Number(originalValues[attributeIndex]);
    const updatedValue = Number(updatedValues[attributeIndex]);
    const [min, max] = currentExplanation.attributeRanges[attributeIndex];
    const rangeSpan = max - min || 1;
    const delta = updatedValue - originalValue;
    const deltaShare = clamp(Math.abs(delta) / rangeSpan, 0, 1);

    const diffTrack = document.createElement("div");
    diffTrack.className = "diff-track";

    const diffFill = document.createElement("div");
    diffFill.className = delta >= 0 ? "diff-fill diff-fill-positive" : "diff-fill diff-fill-negative";

    const fillWidth = deltaShare * 50;
    diffFill.style.width = `${fillWidth}%`;
    diffFill.style.left = delta >= 0 ? "50%" : `${50 - fillWidth}%`;
    diffTrack.appendChild(diffFill);

    diffContent.appendChild(diffTrack);
    diffCell.appendChild(diffContent);

    return diffCell;
}

function populateValueCell(valueCell, attributeIndex, values, options = {}) {
    const {
        originalValues = null,
        showNumericDelta = false,
    } = options;
    const currentLabel = getAttributeDisplayValue(attributeIndex, values);

    if (
        !showNumericDelta ||
        !originalValues
    ) {
        valueCell.textContent = currentLabel;
        return;
    }

    if (currentExplanation.attributeTypes[attributeIndex] === "categorical") {
        const originalLabel = getAttributeDisplayValue(attributeIndex, originalValues);
        if (originalLabel === currentLabel) {
            valueCell.textContent = currentLabel;
            return;
        }

        const oldValue = document.createElement("span");
        oldValue.className = "categorical-old-value-marker";
        oldValue.title = `Original: ${originalLabel}`;

        const arrow = document.createElement("span");
        arrow.className = "value-change-arrow";
        arrow.textContent = " -> ";

        const newValue = document.createElement("span");
        newValue.className = "value-delta value-delta-increase";
        newValue.textContent = currentLabel;

        valueCell.appendChild(oldValue);
        valueCell.appendChild(arrow);
        valueCell.appendChild(newValue);
        return;
    }

    const originalValue = Number(originalValues[attributeIndex]);
    const updatedValue = Number(values[attributeIndex]);
    const delta = updatedValue - originalValue;
    if (!Number.isFinite(originalValue) || !Number.isFinite(updatedValue) || delta === 0) {
        valueCell.textContent = currentLabel;
        return;
    }

    const baseValue = document.createElement("span");
    baseValue.textContent = formatValue(originalValue);

    const deltaValue = document.createElement("span");
    deltaValue.className = delta > 0 ? "value-delta value-delta-increase" : "value-delta value-delta-decrease";
    deltaValue.textContent = ` ${delta > 0 ? "+" : "-"} ${formatValue(Math.abs(delta))}`;

    valueCell.appendChild(baseValue);
    valueCell.appendChild(deltaValue);
}

function populateAttributeTable(tableBody, values, options = {}) {
    const {
        includeNames = true,
        originalValues = null,
        comparisonStyle = "classic",
        showNumericDelta = false,
    } = options;

    tableBody.innerHTML = "";

    for (let i = 0; i < currentExplanation.attributeNames.length; i++) {
        const row = document.createElement("tr");
        row.className = `attribute-row row_${i}`;
        if (i === 0) {
            row.classList.add("attribute-row-first");
        }
        if (i === currentExplanation.attributeNames.length - 1) {
            row.classList.add("attribute-row-last");
        }
        const isUnchanged = originalValues && !hasAttributeChanged(i, originalValues, values);

        if (isUnchanged && comparisonStyle !== "persona") {
            row.classList.add("counterfactual-row-unchanged");
        }

        const valueCell = document.createElement("td");
        valueCell.className = "value";

        if (currentExplanation.attributeTypes[i] === "categorical") {
            const options = currentExplanation.attributeRanges[i];
            const categoryIndex = getCategoryIndex(i, values);

            populateValueCell(valueCell, i, values, {
                originalValues,
                showNumericDelta,
            });
            if (includeNames) {
                const nameCell = document.createElement("td");
                nameCell.className = "attribute";
                nameCell.textContent = getAttributeName(i);
                row.appendChild(nameCell);
            }
            row.appendChild(valueCell);
            if (originalValues && comparisonStyle === "persona") {
                row.appendChild(createPersonaCategoryCell(i, originalValues, values));
            } else if (originalValues && comparisonStyle === "inline") {
                row.appendChild(createComparisonCategoryCell(i, originalValues, values, {
                    muted: Boolean(isUnchanged),
                }));
            } else {
                row.appendChild(createCategoryIconsCell(options, categoryIndex, {
                    muted: Boolean(isUnchanged),
                }));
            }
        } else {
            const [min, max] = currentExplanation.attributeRanges[i];
            const numericValue = Number(values[i]);

            populateValueCell(valueCell, i, values, {
                originalValues,
                showNumericDelta,
            });
            if (includeNames) {
                const nameCell = document.createElement("td");
                nameCell.className = "attribute";
                nameCell.textContent = getAttributeName(i);
                row.appendChild(nameCell);
            }
            row.appendChild(valueCell);
            if (originalValues && comparisonStyle === "persona") {
                row.appendChild(createPersonaMeterCell(i, originalValues, values));
            } else if (originalValues && comparisonStyle === "inline") {
                row.appendChild(createComparisonMeterCell(i, originalValues, values, {
                    muted: Boolean(isUnchanged),
                }));
            } else {
                row.appendChild(createMeterCell(numericValue, min, max, {
                    muted: Boolean(isUnchanged),
                }));
            }
        }

        if (originalValues && comparisonStyle === "classic") {
            row.appendChild(createDiffCell(i, originalValues, values));
        }

        tableBody.appendChild(row);
    }
}

function setCaseLabel(table, label, options = {}) {
    if (!table) {
        return;
    }

    const {
        startColumn = 0,
        columnSpan = null,
    } = options;

    const head = table.querySelector("thead");
    const referenceRow =
        head?.querySelector("tr:not(.case-label-row)") ??
        table.querySelector("tbody tr");

    if (!head || !referenceRow) {
        return;
    }

    const totalColumns = referenceRow.children.length;
    const resolvedStart = clamp(startColumn, 0, totalColumns);
    const remainingColumns = totalColumns - resolvedStart;
    const resolvedSpan = clamp(columnSpan ?? remainingColumns, 0, remainingColumns);

    let labelRow = head.querySelector(".case-label-row");
    if (!labelRow) {
        labelRow = document.createElement("tr");
        labelRow.className = "case-label-row";
        head.prepend(labelRow);
    }

    labelRow.innerHTML = "";

    if (resolvedStart > 0) {
        const spacerCell = document.createElement("th");
        spacerCell.className = "case-label-spacer";
        spacerCell.colSpan = resolvedStart;
        labelRow.appendChild(spacerCell);
    }

    if (resolvedSpan > 0) {
        const labelCell = document.createElement("th");
        labelCell.className = "case-label-cell";
        labelCell.colSpan = resolvedSpan;
        labelCell.textContent = label;
        labelRow.appendChild(labelCell);
    }

    const trailingColumns = totalColumns - resolvedStart - resolvedSpan;
    if (trailingColumns > 0) {
        const trailingSpacerCell = document.createElement("th");
        trailingSpacerCell.className = "case-label-spacer";
        trailingSpacerCell.colSpan = trailingColumns;
        labelRow.appendChild(trailingSpacerCell);
    }
}

function lockTableColumnWidths(table) {
    const referenceRow =
        table?.querySelector("thead tr:not(.case-label-row)") ??
        table?.querySelector("tbody tr");
    if (!table || !referenceRow || referenceRow.children.length === 0) {
        return;
    }

    table.querySelector("colgroup.instance-column-widths")?.remove();

    const colgroup = document.createElement("colgroup");
    colgroup.className = "instance-column-widths";
    Array.from(referenceRow.children).forEach((cell) => {
        const column = document.createElement("col");
        column.style.width = `${cell.getBoundingClientRect().width}px`;
        colgroup.appendChild(column);
    });
    table.prepend(colgroup);
    table.classList.add("fixed-instance-columns");
}

function copyTableColumnWidths(sourceTable, targetTable) {
    const sourceRow =
        sourceTable?.querySelector("thead tr:not(.case-label-row)") ??
        sourceTable?.querySelector("tbody tr");
    if (!sourceRow || !targetTable || sourceRow.children.length === 0) {
        return;
    }

    const columnWidths = Array.from(sourceRow.children).map((cell) =>
        cell.getBoundingClientRect().width
    );
    if (columnWidths.some((width) => width <= 0)) {
        return;
    }

    targetTable.querySelector("colgroup.instance-column-widths")?.remove();

    const colgroup = document.createElement("colgroup");
    colgroup.className = "instance-column-widths";
    columnWidths.forEach((width) => {
        const column = document.createElement("col");
        column.style.width = `${width}px`;
        colgroup.appendChild(column);
    });
    targetTable.prepend(colgroup);
    targetTable.classList.add("fixed-instance-columns");
}

function showAttributeValues(tableBody) {
    populateAttributeTable(tableBody, currentExplanation.attributeValues, {
        includeNames: true,
    });

    if (showPredictionPanel && !counterfactualSimulationEnabled) {
        showPrediction(tableBody, currentExplanation.prediction.value, {
            colorResult: explanationType !== "attribution" && explanationView !== "persona",
            predictionTone: explanationType === "counterfactual" ? "original" : "standard",
        });
    }
}

function showPrediction(tableBody, prediction, options = {}) {
    const {
        includeLabel = true,
        colorResult = true,
        predictionTone = "standard",
        originalPrediction = null,
    } = options;
    const existingPredictionRow = tableBody.querySelector(".prediction-row");
    if (existingPredictionRow) {
        existingPredictionRow.remove();
    }

    clearPredictionPanel();

    const predictionPanel = document.createElement("div");
    predictionPanel.id = "prediction-panel";
    predictionPanel.className = "prediction-panel";
    predictionPanel.appendChild(createPredictionChoiceGroup(prediction, {
        colorResult,
        predictionTone,
        originalPrediction,
    }));

    if (includeLabel) {
        const predictionLabel = document.createElement("div");
        predictionLabel.className = "prediction-panel-label";
        predictionLabel.textContent = "AI prediction";
        predictionLabel.title = "The AI's prediction for this instance";
        predictionPanel.appendChild(predictionLabel);
    }

    const explanationWrapper = document.querySelector("#explanation-wrapper");
    if (explanationWrapper && tablesWrapper) {
        const insertionPoint = explanationBoxAnchor?.parentElement === explanationWrapper
            ? explanationBoxAnchor.nextSibling
            : tablesWrapper.nextSibling;
        explanationWrapper.insertBefore(predictionPanel, insertionPoint);
        syncPredictionPanelWidth(predictionPanel);
    }
}

function clearPredictionPanel() {
    document.querySelector("#prediction-panel")?.remove();
}

function syncPredictionPanelWidth(predictionPanel) {
    const instanceTable = noneExplanationTbody.closest("table");
    const explanationWrapper = document.querySelector("#explanation-wrapper");
    if (!instanceTable || !explanationWrapper) {
        return;
    }

    const applyWidth = () => {
        const tableRect = instanceTable.getBoundingClientRect();
        const influenceCell = instanceTable.querySelector("#feature-attribution-chart");
        const wrapperRect = explanationWrapper.getBoundingClientRect();
        const width = influenceCell
            ? influenceCell.getBoundingClientRect().left - tableRect.left
            : tableRect.width;
        const leftOffset = tableRect.left - wrapperRect.left;

        if (width > 0) {
            predictionPanel.style.width = `${Math.round(width)}px`;
            predictionPanel.style.marginLeft = `${Math.round(leftOffset)}px`;
        }
    };

    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(applyWidth);
    } else {
        applyWidth();
    }
}

function createPredictionChoiceGroup(prediction, options = {}) {
    const {
        colorResult = true,
        predictionTone = "standard",
        originalPrediction = null,
    } = options;
    const predictionGroup = document.createElement("div");
    predictionGroup.className = "prediction-choice-group";
    if (predictionTone === "original") {
        predictionGroup.classList.add("prediction-choice-group-original");
    } else if (predictionTone === "counterfactual") {
        predictionGroup.classList.add("prediction-choice-group-counterfactual");
    }
    const selectedPrediction = Number(prediction);
    const originalPredictionValue = Number(originalPrediction);

    const labels = currentExplanation.predictionLabels.length > 0
        ? currentExplanation.predictionLabels
        : ["Class 0", "Class 1"];
    const choices = labels.map((label, index) => ({ label, index }));

    if (datasetName === "diabetes" && choices.length === 2) {
        choices.sort((a, b) => {
            const aIsNoDiabetes = String(a.label).toLowerCase().includes("no");
            const bIsNoDiabetes = String(b.label).toLowerCase().includes("no");
            return Number(aIsNoDiabetes) - Number(bIsNoDiabetes);
        });
    }

    choices.forEach(({ label, index }) => {
        const isSelected = index === selectedPrediction;
        const isOriginalPrediction = Number.isFinite(originalPredictionValue) &&
            index === originalPredictionValue;
        const choice = document.createElement("span");
        choice.className = "prediction-choice";
        choice.classList.toggle("prediction-choice-selected", isSelected);
        choice.classList.toggle(
            "prediction-choice-muted",
            !isSelected && predictionTone !== "original" && !isOriginalPrediction
        );

        if (isOriginalPrediction) {
            choice.classList.add("prediction-choice-original");
        }

        if (isSelected) {
            if (predictionTone === "counterfactual") {
                choice.classList.add("prediction-choice-counterfactual");
            } else if (predictionTone === "original") {
                choice.classList.add("prediction-choice-original");
            } else if (colorResult) {
                choice.classList.add(
                    selectedPrediction === 1 ? "prediction-choice-positive" : "prediction-choice-negative"
                );
            }
        }

        const marker = document.createElement("span");
        marker.className = "prediction-choice-marker";
        choice.appendChild(marker);

        const labelText = document.createElement("span");
        labelText.className = "prediction-choice-label";
        labelText.textContent = shortenClassLabel(label);
        choice.appendChild(labelText);

        predictionGroup.appendChild(choice);
    });

    return predictionGroup;
}

function clearCounterfactualTable() {
    const table = noneExplanationTbody.closest("table");
    const existingCounterfactualTable = tablesWrapper?.querySelector("#counterfactual-table");
    if (existingCounterfactualTable) {
        existingCounterfactualTable.remove();
    }
    const existingCounterfactualColumn = tablesWrapper?.querySelector("#counterfactual-column");
    if (existingCounterfactualColumn) {
        restoreExplanationBoxAnchor();
        existingCounterfactualColumn.remove();
    }
    const existingToggle = tablesWrapper?.querySelector("#persona-original-toggle-panel");
    if (existingToggle) {
        existingToggle.remove();
    }
    tablesWrapper?.classList.remove("counterfactual-layout");
    tablesWrapper?.classList.remove("counterfactual-persona-layout");
    if (table) {
        table.hidden = false;
        table.classList.remove("counterfactual-table-persona-original");
        table.classList.remove("fixed-instance-columns");
        table.querySelector("colgroup.instance-column-widths")?.remove();
        table.querySelector(".case-label-row")?.remove();
        clearPersonaRowHeights(table);
    }
}

function restoreExplanationBoxAnchor() {
    const explanationWrapper = document.querySelector("#explanation-wrapper");
    if (
        explanationWrapper &&
        tablesWrapper &&
        explanationBoxAnchor &&
        explanationBoxAnchor.parentElement !== explanationWrapper
    ) {
        explanationWrapper.insertBefore(explanationBoxAnchor, tablesWrapper.nextSibling);
    }
}

function clearNarrativePanel() {
    const existingNarrativePanel = explanationBoxAnchor?.querySelector("#narrative-panel");
    if (existingNarrativePanel) {
        existingNarrativePanel.remove();
    }
}

function resetAttributionChart() {
    if (attributionChart) {
        attributionChart.destroy();
        attributionChart = null;
    }
    noneExplanationTbody.closest("table")?.classList.remove("attribution-table");
}

function showAttributionChart(tableBody) {
    const attribution = currentExplanation.attribution;
    if (!attribution || !Array.isArray(attribution.values)) {
        return;
    }

    const table = tableBody.closest("table");
    const headerRow = table?.querySelector("thead tr:not(.case-label-row)");
    if (!table || !headerRow) {
        return;
    }
    table.classList.add("attribution-table");

    const existingDirectionRow = tableBody.querySelector(".influence-direction-row");
    if (existingDirectionRow) {
        existingDirectionRow.remove();
    }

    const existingHeader = headerRow.querySelector(".attribution-header");
    if (!existingHeader) {
        const attributionHeader = document.createElement("th");
        attributionHeader.className = "tooltip attribution-header";
        attributionHeader.colSpan = 1;
        attributionHeader.title = "Influence of each attribute towards the prediction";
        const attributionHeaderLabel = document.createElement("span");
        attributionHeaderLabel.className = "attribution-header-label";
        attributionHeaderLabel.textContent = "Influence";
        attributionHeader.appendChild(attributionHeaderLabel);
        headerRow.appendChild(attributionHeader);
    }

    const firstRow = tableBody.querySelector(".attribute-row");
    if (!firstRow) {
        return;
    }

    const chartPanelCell = document.createElement("td");
    chartPanelCell.id = "feature-attribution-chart";
    chartPanelCell.colSpan = 1;
    chartPanelCell.rowSpan = currentExplanation.attributeNames.length;

    const attributionPanel = document.createElement("div");
    attributionPanel.className = "feature-attribution-panel";

    const chartWrapper = document.createElement("div");
    chartWrapper.id = "feature-attribution-div";

    const canvas = document.createElement("canvas");
    canvas.id = "feature-attribution-canvas";
    canvas.width = 148;
    canvas.height = 120;

    const totalAttribution = attribution.values.reduce((sum, value) => sum + Math.abs(value), 0) || 1;
    const percentageColumn = document.createElement("div");
    percentageColumn.className = "feature-attribution-percentages";
    attribution.values.forEach((value) => {
        const percentLabel = document.createElement("span");
        percentLabel.className = value >= 0
            ? "influence-percent-label influence-percent-positive"
            : "influence-percent-label influence-percent-negative";
        percentLabel.textContent = Math.abs(value) > 0
            ? `${value >= 0 ? "+" : "-"}${Math.round((Math.abs(value) / totalAttribution) * 100)}%`
            : "";
        percentageColumn.appendChild(percentLabel);
    });

    chartWrapper.appendChild(canvas);
    attributionPanel.appendChild(chartWrapper);
    attributionPanel.appendChild(percentageColumn);
    chartPanelCell.appendChild(attributionPanel);
    firstRow.appendChild(chartPanelCell);

    const colors = attribution.values.map((value) =>
        value === 0 ? "rgba(0, 0, 0, 0)" : (value >= 0 ? "rgba(60, 136, 232)" : "rgba(234, 51, 53)")
    );

    const renderAttributionChart = () => {
        const chartHeight = Math.max(Math.round(chartPanelCell.clientHeight), 1);
        const chartWidth = Math.max(Math.round(chartWrapper.clientWidth), 148);
        const plotHeight = Math.max(chartHeight - 4, 1);
        chartWrapper.style.height = `${chartHeight}px`;
        canvas.width = chartWidth;
        canvas.height = plotHeight;
        canvas.style.width = `${chartWidth}px`;
        canvas.style.height = `${plotHeight}px`;
        percentageColumn.style.height = `${chartHeight}px`;
        directionLabels.style.width = `${chartWidth}px`;
        directionLabels.style.minWidth = `${chartWidth}px`;

        if (attributionChart) {
            attributionChart.destroy();
            attributionChart = null;
        }

        const zeroLinePlugin = {
            id: "attributionZeroLine",
            afterDraw(chart) {
                const xScale = chart.scales.x;
                const { top, bottom } = chart.chartArea;
                const zeroX = xScale.getPixelForValue(0);
                const context = chart.ctx;

                context.save();
                context.beginPath();
                context.moveTo(zeroX, top);
                context.lineTo(zeroX, bottom);
                context.lineWidth = 1.5;
                context.strokeStyle = "rgba(100, 100, 100, 0.65)";
                context.stroke();
                context.restore();
            },
        };

        attributionChart = new Chart(canvas.getContext("2d"), {
            type: "bar",
            data: {
                labels: currentExplanation.attributeNames.map((_, index) => getAttributeName(index)),
                datasets: [
                    {
                        label: "Feature Attribution",
                        data: attribution.values.map((value) => {
                            const percentage = (Math.abs(value) / totalAttribution) * 100;
                            return value < 0 ? -percentage : percentage;
                        }),
                        backgroundColor: colors,
                        borderWidth: 0,
                        barThickness: 21,
                    },
                ],
            },
            plugins: [zeroLinePlugin],
            options: {
                indexAxis: "y",
                responsive: false,
                maintainAspectRatio: false,
                animation: false,
                resizeDelay: 0,
                scales: {
                    y: {
                        beginAtZero: true,
                        display: false,
                    },
                    x: {
                        display: false,
                        min: -100,
                        max: 100,
                    },
                },
                plugins: {
                    tooltip: {
                        enabled: false,
                    },
                    legend: {
                        display: false,
                    },
                },
            },
        });
        scheduleIframeHeightPost();
    };

    const directionRow = document.createElement("tr");
    directionRow.className = "influence-direction-row";

    const columnsBeforeChart = Math.max(firstRow.children.length - 1, 1);
    const spacerCell = document.createElement("td");
    spacerCell.className = "influence-direction-spacer";
    spacerCell.colSpan = columnsBeforeChart;
    directionRow.appendChild(spacerCell);

    const directionCell = document.createElement("td");
    directionCell.className = "influence-direction-cell";
    directionCell.colSpan = 1;

    const directionLabels = document.createElement("div");
    directionLabels.className = "influence-direction-labels";

    const leftLabel = document.createElement("span");
    leftLabel.textContent = shortenClassLabel(attribution.directionLabels?.left);

    const rightLabel = document.createElement("span");
    rightLabel.textContent = shortenClassLabel(attribution.directionLabels?.right);

    directionLabels.appendChild(leftLabel);
    directionLabels.appendChild(rightLabel);
    directionCell.appendChild(directionLabels);
    directionRow.appendChild(directionCell);

    const predictionRow = tableBody.querySelector(".prediction-row");
    tableBody.insertBefore(directionRow, predictionRow);

    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(renderAttributionChart);
    } else {
        renderAttributionChart();
    }
}

function shortenClassLabel(label) {
    const labelText = String(label ?? "");
    if (datasetName === "diabetes") {
        return labelText.toLowerCase().includes("no")
            ? "Non-Diabetic"
            : "Diabetic";
    }
    if (datasetName === "safelimit") {
        return labelText;
    }
    return labelText
        .replace("No Diabetes", "No Diabetes")
        .replace("Diabetes", "Diabetes");
}

function showCounterfactualExample(tableBody) {
    const counterfactual = currentExplanation.counterfactual;
    const originalTable = tableBody.closest("table");
    if (!originalTable || !tablesWrapper || !counterfactual) {
        return;
    }

    if (explanationView !== "persona") {
        lockTableColumnWidths(originalTable);
    }

    const counterfactualTable = document.createElement("table");
    counterfactualTable.id = "counterfactual-table";
    counterfactualTable.className = "counterfactual-table";
    if (explanationView === "inline") {
        counterfactualTable.classList.add("counterfactual-table-inline");
    } else if (explanationView === "persona") {
        counterfactualTable.classList.add("counterfactual-table-persona", "counterfactual-table-persona-counterexample");
    }

    const counterfactualHead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    if (explanationView === "persona") {
        const counterexampleHeader = document.createElement("th");
        counterexampleHeader.className = "counterexample-column-header";
        counterexampleHeader.textContent = "Counter-example";
        headerRow.appendChild(counterexampleHeader);
    } else {
        const valueHeader = document.createElement("th");
        valueHeader.textContent = "Value";

        const controlHeader = document.createElement("th");
        controlHeader.className = "meter-scale-column-header";
        controlHeader.textContent = getAttributeControlHeader();

        headerRow.appendChild(valueHeader);
        headerRow.appendChild(controlHeader);
    }
    if (explanationView === "classic") {
        const diffHeader = document.createElement("th");
        diffHeader.className = "tooltip diff-column-header";
        diffHeader.title = "Difference between the original instance and the counterfactual";
        diffHeader.textContent = "Diff";
        headerRow.appendChild(diffHeader);
    }
    counterfactualHead.appendChild(headerRow);

    const counterfactualBody = document.createElement("tbody");
    counterfactualBody.id = "counterfactual-tbody";

    counterfactualTable.appendChild(counterfactualHead);
    counterfactualTable.appendChild(counterfactualBody);
    if (explanationView !== "persona") {
        setCaseLabel(counterfactualTable, "Comparable", {
            startColumn: 0,
            columnSpan: explanationView === "classic" ? 3 : 2,
        });
    }

    const counterfactualColumn = document.createElement("div");
    counterfactualColumn.id = "counterfactual-column";
    counterfactualColumn.className = "counterfactual-column";
    counterfactualColumn.appendChild(counterfactualTable);
    tablesWrapper.classList.add("counterfactual-layout");

    if (explanationView === "persona") {
        tablesWrapper.classList.add("counterfactual-persona-layout");
        originalTable.classList.add("counterfactual-table-persona-original");
        originalTable.querySelector(".case-label-row")?.remove();
    }

    tablesWrapper.appendChild(counterfactualColumn);

    if (explanationView === "persona") {
        populateCounterfactualPersonaColumn(counterfactualBody, counterfactual.feature_values);
    } else {
        populateAttributeTable(counterfactualBody, counterfactual.feature_values, {
            includeNames: false,
            originalValues: currentExplanation.attributeValues,
            comparisonStyle: explanationView,
            showNumericDelta: false,
        });
    }
    if (explanationView === "persona") {
        syncPersonaRowHeights(originalTable, counterfactualTable);
    } else {
        clearPersonaRowHeights(counterfactualTable);
    }
}

function populateCounterfactualPersonaColumn(tableBody, values) {
    tableBody.innerHTML = "";

    for (let i = 0; i < currentExplanation.attributeNames.length; i++) {
        const row = document.createElement("tr");
        row.className = `attribute-row row_${i}`;
        if (i === 0) {
            row.classList.add("attribute-row-first");
        }
        if (i === currentExplanation.attributeNames.length - 1) {
            row.classList.add("attribute-row-last");
        }

        if (currentExplanation.attributeTypes[i] === "categorical") {
            row.appendChild(createPersonaCategoryCell(i, currentExplanation.attributeValues, values));
        } else {
            row.appendChild(createPersonaMeterCell(i, currentExplanation.attributeValues, values));
        }

        tableBody.appendChild(row);
    }
}

function clearCounterfactualSimulation() {
    const existingSimulation = document.querySelector("#counterfactual-simulation");
    if (existingSimulation) {
        existingSimulation.remove();
    }
    simulationValues = null;
    simulationAllowedAttributeIndices = null;
    simulationSpecificCandidatePending = false;
    simulationPrediction = null;
    simulationFeedback = null;
}

function clearPersonaRowHeights(table) {
    table.querySelectorAll("thead tr, tbody tr").forEach((row) => {
        row.style.height = "";
    });
}

function syncPersonaRowHeights(originalTable, counterfactualTable) {
    clearPersonaRowHeights(originalTable);
    clearPersonaRowHeights(counterfactualTable);

    const originalRows = [
        ...originalTable.querySelectorAll("thead tr:not(.case-label-row), tbody tr"),
    ];
    const counterfactualRows = [
        ...counterfactualTable.querySelectorAll("thead tr:not(.case-label-row), tbody tr"),
    ];
    const rowCount = Math.min(originalRows.length, counterfactualRows.length);

    for (let i = 0; i < rowCount; i++) {
        const height = Math.max(
            originalRows[i].getBoundingClientRect().height,
            counterfactualRows[i].getBoundingClientRect().height
        );
        if (height > 0) {
            originalRows[i].style.height = `${height}px`;
            counterfactualRows[i].style.height = `${height}px`;
        }
    }
}

function createCounterfactualSimulation() {
    if (!counterfactualSimulationEnabled || !currentExplanation) {
        return;
    }

    if (!simulationValues) {
        simulationValues = [...currentExplanation.attributeValues];
    }
    simulationAllowedAttributeIndices = null;
    simulationSpecificCandidatePending = counterfactualSimulationMode === "specific";

    const simulationPanel = document.createElement("div");
    simulationPanel.id = "counterfactual-simulation";
    simulationPanel.className = "counterfactual-simulation";

    const simulationQuestion = document.createElement("p");
    simulationQuestion.className = "counterfactual-simulation-question";
    simulationQuestion.id = "counterfactual-simulation-question";
    simulationQuestion.textContent = getCounterfactualSimulationQuestion();
    simulationPanel.appendChild(simulationQuestion);

    if (counterfactualSimulationMode === "budget") {
        const budgetPanel = document.createElement("div");
        budgetPanel.id = "counterfactual-simulation-budget";
        budgetPanel.className = "counterfactual-simulation-budget";
        simulationPanel.appendChild(budgetPanel);
    }

    const simulationTables = document.createElement("div");
    simulationTables.id = "counterfactual-simulation-tables";
    simulationTables.className = "counterfactual-simulation-tables";

    const originalTable = document.createElement("table");
    originalTable.id = "counterfactual-simulation-original-table";
    originalTable.className = "counterfactual-table counterfactual-simulation-original-table";

    const originalHead = document.createElement("thead");
    const originalHeaderRow = document.createElement("tr");
    ["Attribute", "Value", getAttributeControlHeader()].forEach((label, index) => {
        const headerCell = document.createElement("th");
        headerCell.textContent = label;
        if (index === 2) {
            headerCell.className = "meter-scale-column-header";
        }
        originalHeaderRow.appendChild(headerCell);
    });
    originalHead.appendChild(originalHeaderRow);

    const originalBody = document.createElement("tbody");
    originalBody.id = "counterfactual-simulation-original-tbody";
    originalTable.appendChild(originalHead);
    originalTable.appendChild(originalBody);
    populateAttributeTable(originalBody, currentExplanation.attributeValues, {
        includeNames: true,
    });

    const changesTable = document.createElement("table");
    changesTable.id = "counterfactual-simulation-table";
    changesTable.className = "counterfactual-table counterfactual-table-persona counterfactual-simulation-table";

    const changesHead = document.createElement("thead");
    const changesHeaderRow = document.createElement("tr");
    const changesHeader = document.createElement("th");
    changesHeader.className = "simulation-changes-column-header";
    changesHeader.textContent = "Changes";
    changesHeaderRow.appendChild(changesHeader);
    changesHead.appendChild(changesHeaderRow);

    const simulationBody = document.createElement("tbody");
    simulationBody.id = "counterfactual-simulation-tbody";
    changesTable.appendChild(changesHead);
    changesTable.appendChild(simulationBody);

    const originalColumn = document.createElement("div");
    originalColumn.className = "counterfactual-simulation-original-column";
    originalColumn.appendChild(originalTable);

    if (showPredictionPanel) {
        const originalPrediction = document.createElement("div");
        originalPrediction.className = "counterfactual-simulation-original-prediction";

        const predictionLabel = document.createElement("div");
        predictionLabel.className = "prediction-panel-label";
        predictionLabel.textContent = "AI prediction";
        originalPrediction.appendChild(createPredictionChoiceGroup(
            currentExplanation.prediction.value,
            {
                colorResult: false,
                predictionTone: "original",
            }
        ));
        originalPrediction.appendChild(predictionLabel);
        originalColumn.appendChild(originalPrediction);
    }

    simulationTables.appendChild(originalColumn);
    simulationTables.appendChild(changesTable);
    simulationPanel.appendChild(simulationTables);

    const simulationActions = document.createElement("div");
    simulationActions.className = "counterfactual-simulation-actions";

    const resetButton = document.createElement("button");
    resetButton.type = "button";
    resetButton.className = "simulation-reset-button";
    resetButton.textContent = "Reset changes";
    resetButton.addEventListener("click", resetCounterfactualSimulation);
    simulationActions.appendChild(resetButton);

    simulationPanel.appendChild(simulationActions);

    document.querySelector("#explanation-wrapper")?.appendChild(simulationPanel);
    syncCounterfactualSimulationColumnWidths();
    renderCounterfactualSimulationRows();
    updateCounterfactualSimulationQuestionWidth();
    updateCounterfactualSimulationBudget();
    if (counterfactualSimulationMode === "specific") {
        initializeSpecificSimulationFeature();
    }
}

function syncCounterfactualSimulationColumnWidths() {
    const sourceTable = noneExplanationTbody.closest("table");
    const simulationOriginalTable = document.querySelector("#counterfactual-simulation-original-table");
    if (!sourceTable || !simulationOriginalTable) {
        return;
    }

    const applyWidths = () => {
        copyTableColumnWidths(sourceTable, simulationOriginalTable);
        syncCounterfactualSimulationRowHeights();
        updateCounterfactualSimulationQuestionWidth();
    };

    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(applyWidths);
    } else {
        applyWidths();
    }
}

function renderCounterfactualSimulationRows() {
    const simulationBody = document.querySelector("#counterfactual-simulation-tbody");
    if (!simulationBody || !simulationValues) {
        return;
    }

    simulationBody.innerHTML = "";
    for (let i = 0; i < currentExplanation.attributeNames.length; i++) {
        const row = document.createElement("tr");
        row.className = `attribute-row row_${i}`;
        if (i === 0) {
            row.classList.add("attribute-row-first");
        }
        if (i === currentExplanation.attributeNames.length - 1) {
            row.classList.add("attribute-row-last");
        }
        const isControlDisabled = isSimulationControlDisabled(i);

        if (currentExplanation.attributeTypes[i] === "categorical") {
            row.appendChild(createSimulationCategoryControl(i, isControlDisabled));
        } else {
            row.appendChild(createSimulationSliderControl(i, isControlDisabled));
        }

        simulationBody.appendChild(row);
    }

    updateCounterfactualSimulationBudget();
    updateCounterfactualSimulationQuestionWidth();
    syncCounterfactualSimulationRowHeights();
}

function syncCounterfactualSimulationRowHeights() {
    const originalTable = document.querySelector("#counterfactual-simulation-original-table");
    const changesTable = document.querySelector("#counterfactual-simulation-table");
    if (!originalTable || !changesTable) {
        return;
    }
    syncPersonaRowHeights(originalTable, changesTable);
}

function isSimulationControlDisabled(attributeIndex) {
    if (simulationSpecificCandidatePending) {
        return true;
    }

    if (counterfactualSimulationMode === "budget") {
        return false;
    }

    if (simulationAllowedAttributeIndices !== null) {
        return !simulationAllowedAttributeIndices.includes(attributeIndex);
    }

    return false;
}

function createSimulationSliderControl(attributeIndex, disabled) {
    const controlCell = document.createElement("td");
    controlCell.className = "meter-container simulation-control-cell";
    const [min, max] = currentExplanation.attributeRanges[attributeIndex];
    const originalValue = Number(currentExplanation.attributeValues[attributeIndex]);
    const currentValue = Number(simulationValues[attributeIndex]);

    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "simulation-slider";
    slider.min = String(min);
    slider.max = String(max);
    slider.step = Number.isInteger(min) && Number.isInteger(max) ? "1" : String((max - min) / 100 || 0.01);
    slider.value = String(clamp(
        Number.isFinite(currentValue) ? currentValue : originalValue,
        min,
        max
    ));
    slider.disabled = disabled;
    slider.addEventListener("input", () => {
        const numericValue = Number(slider.value);
        const constrainedValue = clampSimulationValueToBudget(attributeIndex, numericValue);
        slider.value = String(constrainedValue);
        simulationValues[attributeIndex] = Number.isInteger(originalValue)
            ? Math.round(constrainedValue)
            : constrainedValue;
        clearSimulationFeedback();
        refreshSimulationValueCell(attributeIndex);
        updateCounterfactualSimulationBudget();
    });
    slider.addEventListener("change", () => {
        renderCounterfactualSimulationRows();
    });

    const controlContent = document.createElement("div");
    controlContent.className = "simulation-control-content";
    controlContent.appendChild(slider);
    controlContent.appendChild(createSimulationChangeAmount(attributeIndex));
    controlCell.appendChild(controlContent);
    return controlCell;
}

function getBudgetConstrainedNumericRange(attributeIndex, min, max) {
    if (counterfactualSimulationMode !== "budget") {
        return { min, max };
    }

    const minValue = Number(min);
    const maxValue = Number(max);
    const originalValue = Number(currentExplanation.attributeValues[attributeIndex]);
    const span = maxValue - minValue;
    if (!Number.isFinite(span) || span <= 0 || !Number.isFinite(originalValue)) {
        return { min, max };
    }

    const otherCost = getSimulationSpentPoints({ ignoreAttributeIndex: attributeIndex });
    const remainingPoints = Math.max(0, SIMULATION_BUDGET_POINTS - otherCost);
    const maxNormalizedDelta = remainingPoints / 10;
    const originalPosition = (originalValue - minValue) / span;
    const constrainedMin = minValue + Math.max(0, originalPosition - maxNormalizedDelta) * span;
    const constrainedMax = minValue + Math.min(1, originalPosition + maxNormalizedDelta) * span;

    if (Number.isInteger(minValue) && Number.isInteger(maxValue)) {
        return {
            min: Math.ceil(constrainedMin),
            max: Math.floor(constrainedMax),
        };
    }

    return {
        min: Number(constrainedMin.toFixed(4)),
        max: Number(constrainedMax.toFixed(4)),
    };
}

function clampSimulationValueToBudget(attributeIndex, requestedValue) {
    if (counterfactualSimulationMode !== "budget") {
        return requestedValue;
    }

    const [min, max] = currentExplanation.attributeRanges[attributeIndex] ?? [0, 0];
    const constrainedRange = getBudgetConstrainedNumericRange(attributeIndex, min, max);
    return clamp(Number(requestedValue), constrainedRange.min, constrainedRange.max);
}

function createSimulationCategoryControl(attributeIndex, disabled) {
    const controlCell = document.createElement("td");
    controlCell.className = "icons-container simulation-control-cell";
    const options = currentExplanation.attributeRanges[attributeIndex];
    const currentIndex = getCategoryIndex(attributeIndex, simulationValues);
    const originalIndex = getCategoryIndex(attributeIndex, currentExplanation.attributeValues);
    const isChanged = hasAttributeChanged(attributeIndex, currentExplanation.attributeValues, simulationValues);

    const optionGroup = document.createElement("div");
    optionGroup.className = "simulation-category-options persona-comparison-icons";

    for (let i = 0; i < options.length; i++) {
        const optionLabel = document.createElement("label");
        optionLabel.className = "simulation-category-option";
        optionLabel.title = options[i];

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = i === currentIndex;
        checkbox.disabled = disabled || isSimulationCategoryOptionDisabled(attributeIndex, i);
        checkbox.addEventListener("change", () => {
            simulationValues[attributeIndex] = i;
            clearSimulationFeedback();
            renderCounterfactualSimulationRows();
        });

        const marker = document.createElement("span");
        marker.className = "comparison-icon";
        if (isChanged && i === originalIndex) {
            marker.classList.add("persona-icon-original");
        }
        if (i === currentIndex) {
            marker.classList.add(isChanged ? "persona-icon-updated" : "persona-icon-current");
        }

        optionLabel.appendChild(checkbox);
        optionLabel.appendChild(marker);
        optionGroup.appendChild(optionLabel);
    }

    const controlContent = document.createElement("div");
    controlContent.className = "simulation-control-content";
    controlContent.appendChild(optionGroup);
    controlContent.appendChild(createSimulationChangeAmount(attributeIndex));
    controlCell.appendChild(controlContent);
    return controlCell;
}

function createSimulationChangeAmount(attributeIndex) {
    const categoricalLabel = createCategoricalChangeLabel(
        attributeIndex,
        currentExplanation.attributeValues,
        simulationValues
    );
    if (categoricalLabel) {
        categoricalLabel.classList.add("simulation-change-amount");
        return categoricalLabel;
    }

    const deltaLabel = createMeterDeltaLabel(
        attributeIndex,
        currentExplanation.attributeValues,
        simulationValues
    );
    if (deltaLabel) {
        deltaLabel.classList.add("simulation-change-amount");
        return deltaLabel;
    }

    const spacer = document.createElement("span");
    spacer.className = "meter-side-delta meter-side-delta-empty simulation-change-amount";
    spacer.textContent = "0";
    return spacer;
}

function isSimulationCategoryOptionDisabled(attributeIndex, optionIndex) {
    if (counterfactualSimulationMode !== "budget") {
        return false;
    }

    if (optionIndex === getCategoryIndex(attributeIndex, simulationValues)) {
        return false;
    }

    const previousValue = simulationValues[attributeIndex];
    simulationValues[attributeIndex] = optionIndex;
    const wouldExceedBudget = getSimulationSpentPoints() > SIMULATION_BUDGET_POINTS;
    simulationValues[attributeIndex] = previousValue;
    return wouldExceedBudget;
}

function getSimulationChangedAttributeIndices(values = simulationValues) {
    if (!values) {
        return [];
    }

    return currentExplanation.attributeNames
        .map((_, index) => index)
        .filter((index) =>
            hasAttributeChanged(index, currentExplanation.attributeValues, values)
        );
}

function getSimulationSpentPoints(options = {}) {
    if (!simulationValues) {
        return 0;
    }

    const { ignoreAttributeIndex = null, values = simulationValues } = options;
    return currentExplanation.attributeNames.reduce((sum, _, index) => {
        if (index === ignoreAttributeIndex) {
            return sum;
        }
        return sum + getSimulationAttributeCost(index, values);
    }, 0);
}

function getSimulationAttributeCost(attributeIndex, values = simulationValues) {
    if (!hasAttributeChanged(attributeIndex, currentExplanation.attributeValues, values)) {
        return 0;
    }

    if (currentExplanation.attributeTypes[attributeIndex] === "categorical") {
        return 5;
    }

    const [min, max] = currentExplanation.attributeRanges[attributeIndex] ?? [0, 0];
    const minValue = Number(min);
    const maxValue = Number(max);
    const originalValue = Number(currentExplanation.attributeValues[attributeIndex]);
    const updatedValue = Number(values[attributeIndex]);
    const span = maxValue - minValue;
    if (
        !Number.isFinite(span) ||
        span <= 0 ||
        !Number.isFinite(originalValue) ||
        !Number.isFinite(updatedValue)
    ) {
        return 0;
    }

    return Math.abs(updatedValue - originalValue) / span * 10;
}

function updateCounterfactualSimulationBudget() {
    const budgetPanel = document.querySelector("#counterfactual-simulation-budget");
    if (!budgetPanel || counterfactualSimulationMode !== "budget") {
        return;
    }

    const spentPoints = getSimulationSpentPoints();
    const remainingPoints = Math.max(0, SIMULATION_BUDGET_POINTS - spentPoints);
    budgetPanel.textContent = `Budget: ${SIMULATION_BUDGET_POINTS} points - Remaining: ${formatSimulationPoints(remainingPoints)} - Spent: ${formatSimulationPoints(spentPoints)}`;
    budgetPanel.classList.toggle("counterfactual-simulation-budget-empty", remainingPoints <= 0);
}

function formatSimulationPoints(value) {
    const roundedValue = Math.round(Number(value) * 10) / 10;
    return Number.isInteger(roundedValue)
        ? `${roundedValue} points`
        : `${roundedValue.toFixed(1)} points`;
}

function refreshSimulationValueCell(attributeIndex) {
    const row = document.querySelector(`#counterfactual-simulation-tbody .row_${attributeIndex}`);
    const changeAmount = row?.querySelector(".simulation-change-amount");
    if (!changeAmount) {
        return;
    }

    changeAmount.replaceWith(createSimulationChangeAmount(attributeIndex));
}

function resetCounterfactualSimulation() {
    simulationValues = [...currentExplanation.attributeValues];
    clearSimulationFeedback();
    renderCounterfactualSimulationRows();
    updateCounterfactualSimulationBudget();
}

function clearSimulationFeedback() {
    simulationPrediction = null;
    simulationFeedback = null;
    renderCounterfactualSimulationFeedback();
}

function getCounterfactualSimulationQuestion() {
    const targetLabel = getOppositePredictionLabel();

    if (simulationSpecificCandidatePending) {
        if (counterfactualSimulationMode === "budget") {
            return `Finding a ${SIMULATION_BUDGET_POINTS}-point change that can change the AI prediction to ${formatSimulationOutcomeLabel(targetLabel)}...`;
        }
        return `Finding attributes that can change the AI prediction to ${formatSimulationOutcomeLabel(targetLabel)}...`;
    }

    const originalLabel = currentExplanation.prediction?.label ?? "";
    const question = getProfileCounterfactualQuestion(
        formatSimulationOutcomeLabel(originalLabel),
        formatSimulationOutcomeLabel(targetLabel)
    );

    if (counterfactualSimulationMode === "budget") {
        return `${question} You have a ${SIMULATION_BUDGET_POINTS}-point budget.`;
    }

    return question;
}

function getProfileCounterfactualQuestion(originalLabel, targetLabel) {
    const subjectName = getProfileSubjectName();

    if (datasetName === "diabetes") {
        return `${subjectName} is diagnosed as being ${originalLabel}. If ${subjectName} was to become ${targetLabel}, what minimal changes to their profile would need to occur?`;
    }

    if (datasetName === "safelimit") {
        return `${subjectName} is ${formatDrivingLimitQuestionLabel(originalLabel)} for driving. If ${subjectName} was to be ${formatDrivingLimitQuestionLabel(targetLabel)}, what minimal changes to their profile would need to occur?`;
    }

    return `This case is predicted as being ${originalLabel}. If it were to be ${targetLabel}, what minimal changes to its profile would cause this change?`;
}

function getProfileSubjectName() {
    const key = `${datasetName}:${splitName}:${Number.isFinite(instanceId) ? instanceId : 0}`;
    const index = positiveHash(key) % PROFILE_SUBJECT_NAMES.length;
    return PROFILE_SUBJECT_NAMES[index];
}

function positiveHash(value) {
    let hash = 0;
    const text = String(value);
    for (let i = 0; i < text.length; i++) {
        hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
    }
    return Math.abs(hash);
}

function formatDrivingLimitQuestionLabel(label) {
    const labelText = String(label ?? "").toLowerCase();
    return labelText.includes("below")
        ? "Below the limit"
        : "Above the limit";
}

function formatSimulationOutcomeLabel(label) {
    return shortenClassLabel(label);
}

function updateCounterfactualSimulationQuestion() {
    const question = document.querySelector("#counterfactual-simulation-question");
    if (question) {
        question.textContent = getCounterfactualSimulationQuestion();
    }
    updateCounterfactualSimulationQuestionWidth();
}

function updateCounterfactualSimulationQuestionWidth() {
    const question = document.querySelector("#counterfactual-simulation-question");
    const tables = document.querySelector("#counterfactual-simulation-tables");
    if (!question || !tables) {
        return;
    }

    const applyWidth = () => {
        const tablesWidth = tables.getBoundingClientRect().width;
        if (tablesWidth > 0) {
            question.style.maxWidth = `${Math.ceil(tablesWidth)}px`;
        }
    };

    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(applyWidth);
    } else {
        applyWidth();
    }
}

function getOppositePredictionLabel() {
    const originalPrediction = Number(currentExplanation.prediction?.value ?? 0);
    const labels = currentExplanation.predictionLabels ?? [];
    const targetIndex = originalPrediction === 0 ? 1 : 0;
    return labels[targetIndex] ?? `Class ${targetIndex}`;
}

function getSimulationTargetValue() {
    const originalPrediction = Number(currentExplanation.prediction?.value ?? 0);
    return originalPrediction === 0 ? 1 : 0;
}

function getSimulationRawValues() {
    const rawValuesByName = {};
    currentExplanation.rawAttributeValues.forEach((rawValue, index) => {
        const rawAttributeName = currentExplanation.rawAttributeNames[index];
        if (currentExplanation.attributeTypes[index] !== "categorical") {
            const numericValue = Number(simulationValues[index]);
            rawValuesByName[rawAttributeName] = Number.isFinite(numericValue)
                ? numericValue
                : rawValue;
            return;
        }

        const rawOptions = currentExplanation.rawAttributeRanges[index] ?? [];
        const displayOptions = currentExplanation.attributeRanges[index] ?? [];
        const simulatedValue = simulationValues[index];
        const numericIndex = Number(simulatedValue);

        if (Number.isFinite(numericIndex) && rawOptions.length > 0) {
            rawValuesByName[rawAttributeName] =
                rawOptions[clamp(Math.round(numericIndex), 0, rawOptions.length - 1)];
            return;
        }

        const displayIndex = displayOptions.findIndex((option) =>
            String(option) === String(simulatedValue)
        );
        if (displayIndex >= 0 && displayIndex < rawOptions.length) {
            rawValuesByName[rawAttributeName] = rawOptions[displayIndex];
            return;
        }

        rawValuesByName[rawAttributeName] = rawValue;
    });
    return rawValuesByName;
}

function getSimulationRawValuesWithCandidate(attributeIndex, candidateValue) {
    return getSimulationRawValuesWithCandidates([{
        attributeIndex,
        candidateValue,
    }]);
}

function getSimulationRawValuesWithCandidates(changes) {
    const rawValues = getSimulationRawValues();
    changes.forEach(({ attributeIndex, candidateValue }) => {
        const rawAttributeName = currentExplanation.rawAttributeNames[attributeIndex];
        if (currentExplanation.attributeTypes[attributeIndex] !== "categorical") {
            rawValues[rawAttributeName] = Number(candidateValue);
            return;
        }

        const rawOptions = currentExplanation.rawAttributeRanges[attributeIndex] ?? [];
        const displayOptions = currentExplanation.attributeRanges[attributeIndex] ?? [];
        const numericIndex = Number(candidateValue);
        if (Number.isFinite(numericIndex) && rawOptions.length > 0) {
            rawValues[rawAttributeName] = rawOptions[clamp(Math.round(numericIndex), 0, rawOptions.length - 1)];
            return;
        }

        const displayIndex = displayOptions.findIndex((option) =>
            String(option) === String(candidateValue)
        );
        if (displayIndex >= 0 && displayIndex < rawOptions.length) {
            rawValues[rawAttributeName] = rawOptions[displayIndex];
        }
    });
    return rawValues;
}

function getCandidateValuesForAttribute(attributeIndex) {
    const originalValue = currentExplanation.attributeValues[attributeIndex];
    if (currentExplanation.attributeTypes[attributeIndex] === "categorical") {
        return (currentExplanation.attributeRanges[attributeIndex] ?? [])
            .map((_, index) => index)
            .filter((index) => index !== getCategoryIndex(attributeIndex, currentExplanation.attributeValues));
    }

    const [min, max] = currentExplanation.attributeRanges[attributeIndex] ?? [0, 0];
    const minValue = Number(min);
    const maxValue = Number(max);
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue) || minValue === maxValue) {
        return [];
    }

    const candidates = new Set();
    const addCandidate = (value) => {
        const clampedValue = clamp(Number(value), minValue, maxValue);
        if (!Number.isFinite(clampedValue)) {
            return;
        }
        const roundedValue = Number.isInteger(minValue) && Number.isInteger(maxValue)
            ? Math.round(clampedValue)
            : Number(clampedValue.toFixed(4));
        if (String(roundedValue) !== String(originalValue)) {
            candidates.add(roundedValue);
        }
    };

    addCandidate(minValue);
    addCandidate(maxValue);
    for (let i = 1; i <= 9; i++) {
        addCandidate(minValue + ((maxValue - minValue) * (i / 10)));
    }

    const counterfactualValue = currentExplanation.counterfactual?.feature_values?.[attributeIndex];
    if (counterfactualValue !== undefined) {
        addCandidate(counterfactualValue);
    }

    return [...candidates];
}

function shuffledIndices(length) {
    const indices = Array.from({ length }, (_, index) => index);
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices;
}

async function predictRawValues(rawFeatureValues) {
    const counterfactual = currentExplanation?.counterfactual;
    if (!counterfactual?.raw_feature_values?.length) {
        return currentExplanation.prediction;
    }

    const progress = getStaticCounterfactualProgress(rawFeatureValues);
    return progress >= 0.8
        ? counterfactual.prediction
        : currentExplanation.prediction;
}

function getStaticCounterfactualProgress(rawFeatureValues) {
    const counterfactualValues = currentExplanation.counterfactual?.raw_feature_values;
    if (!counterfactualValues?.length) {
        return 0;
    }

    const changedIndices = currentExplanation.rawAttributeNames
        .map((_, index) => index)
        .filter((index) => valuesDiffer(
            currentExplanation.rawAttributeValues[index],
            counterfactualValues[index],
        ));
    if (changedIndices.length === 0) {
        return 0;
    }

    const progressValues = changedIndices.map((index) => {
        const originalValue = currentExplanation.rawAttributeValues[index];
        const targetValue = counterfactualValues[index];
        const currentValue = rawFeatureValues[index];

        if (currentExplanation.attributeTypes[index] === "categorical") {
            return valuesDiffer(currentValue, targetValue) ? 0 : 1;
        }

        const targetDelta = Number(targetValue) - Number(originalValue);
        if (!Number.isFinite(targetDelta) || targetDelta === 0) {
            return 0;
        }
        return clamp((Number(currentValue) - Number(originalValue)) / targetDelta, 0, 1);
    });
    return progressValues.reduce((sum, value) => sum + value, 0) / progressValues.length;
}

function valuesDiffer(firstValue, secondValue) {
    return String(firstValue) !== String(secondValue);
}

async function findSpecificSimulationCandidate() {
    return findSimulationCandidate({
        enforceBudget: false,
        preferTwoAttributeChange: true,
    });
}

async function findBudgetSimulationCandidate() {
    return findSimulationCandidate({
        enforceBudget: true,
    });
}

async function findSimulationCandidate(options = {}) {
    const { enforceBudget = false, preferTwoAttributeChange = false } = options;
    const targetValue = getSimulationTargetValue();
    const indices = shuffledIndices(currentExplanation.attributeNames.length);
    const candidateValuesByAttribute = new Map();

    for (const attributeIndex of indices) {
        candidateValuesByAttribute.set(attributeIndex, getCandidateValuesForAttribute(attributeIndex));
    }

    if (preferTwoAttributeChange) {
        const twoAttributeCandidate = await findTwoAttributeSimulationCandidate({
            indices,
            candidateValuesByAttribute,
            targetValue,
            enforceBudget,
        });
        if (twoAttributeCandidate) {
            return twoAttributeCandidate;
        }
    }

    for (const attributeIndex of indices) {
        const candidateValues = candidateValuesByAttribute.get(attributeIndex) ?? [];
        for (const candidateValue of candidateValues) {
            const changes = [{ attributeIndex, candidateValue }];
            if (enforceBudget && !isCandidateWithinSimulationBudget(changes)) {
                continue;
            }
            const prediction = await predictRawValues(
                getSimulationRawValuesWithCandidates(changes)
            );
            if (Number(prediction.value) === targetValue) {
                return {
                    changes,
                };
            }
        }
    }

    if (!preferTwoAttributeChange) {
        return findTwoAttributeSimulationCandidate({
            indices,
            candidateValuesByAttribute,
            targetValue,
            enforceBudget,
        });
    }

    return null;
}

async function findTwoAttributeSimulationCandidate(options) {
    const {
        indices,
        candidateValuesByAttribute,
        targetValue,
        enforceBudget,
    } = options;

    for (let firstIndex = 0; firstIndex < indices.length; firstIndex++) {
        for (let secondIndex = firstIndex + 1; secondIndex < indices.length; secondIndex++) {
            const firstAttributeIndex = indices[firstIndex];
            const secondAttributeIndex = indices[secondIndex];
            const firstCandidateValues = candidateValuesByAttribute.get(firstAttributeIndex) ?? [];
            const secondCandidateValues = candidateValuesByAttribute.get(secondAttributeIndex) ?? [];
            for (const firstCandidateValue of firstCandidateValues) {
                for (const secondCandidateValue of secondCandidateValues) {
                    const changes = [
                        {
                            attributeIndex: firstAttributeIndex,
                            candidateValue: firstCandidateValue,
                        },
                        {
                            attributeIndex: secondAttributeIndex,
                            candidateValue: secondCandidateValue,
                        },
                    ];
                    if (enforceBudget && !isCandidateWithinSimulationBudget(changes)) {
                        continue;
                    }
                    const prediction = await predictRawValues(
                        getSimulationRawValuesWithCandidates(changes)
                    );
                    if (Number(prediction.value) === targetValue) {
                        return {
                            changes,
                        };
                    }
                }
            }
        }
    }

    return null;
}

function isCandidateWithinSimulationBudget(changes) {
    const candidateValues = [...currentExplanation.attributeValues];
    changes.forEach(({ attributeIndex, candidateValue }) => {
        candidateValues[attributeIndex] = candidateValue;
    });
    return getSimulationSpentPoints({ values: candidateValues }) <= SIMULATION_BUDGET_POINTS;
}

function getTwoAllowedSimulationAttributeIndices(solutionAttributeIndices) {
    const allowedIndices = [];
    const addAllowedIndex = (attributeIndex) => {
        if (
            Number.isInteger(attributeIndex) &&
            attributeIndex >= 0 &&
            attributeIndex < currentExplanation.attributeNames.length &&
            !allowedIndices.includes(attributeIndex)
        ) {
            allowedIndices.push(attributeIndex);
        }
    };

    solutionAttributeIndices.forEach(addAllowedIndex);

    currentExplanation.counterfactual?.feature_values?.forEach((_, attributeIndex) => {
        if (
            allowedIndices.length < SIMULATION_SPECIFIC_ATTRIBUTE_COUNT &&
            hasAttributeChanged(
                attributeIndex,
                currentExplanation.attributeValues,
                currentExplanation.counterfactual.feature_values
            )
        ) {
            addAllowedIndex(attributeIndex);
        }
    });

    for (let attributeIndex = 0; attributeIndex < currentExplanation.attributeNames.length; attributeIndex++) {
        if (allowedIndices.length >= SIMULATION_SPECIFIC_ATTRIBUTE_COUNT) {
            break;
        }
        if (getCandidateValuesForAttribute(attributeIndex).length > 0) {
            addAllowedIndex(attributeIndex);
        }
    }

    return allowedIndices.slice(0, SIMULATION_SPECIFIC_ATTRIBUTE_COUNT);
}

async function initializeSpecificSimulationFeature() {
    try {
        const candidate = await findSpecificSimulationCandidate();
        simulationSpecificCandidatePending = false;
        if (candidate) {
            simulationAllowedAttributeIndices = getTwoAllowedSimulationAttributeIndices(
                candidate.changes.map((change) => change.attributeIndex)
            );
            simulationFeedback = {
                isCorrect: true,
                text: `Only ${formatSimulationAttributeList(simulationAllowedAttributeIndices)} can be changed in this case.`,
            };
        } else {
            simulationAllowedAttributeIndices = [];
            simulationFeedback = {
                isCorrect: false,
                text: "No two-attribute change was found that flips this case.",
            };
        }
    } catch (error) {
        simulationSpecificCandidatePending = false;
        simulationAllowedAttributeIndices = [];
        simulationFeedback = {
            isCorrect: false,
            text: String(error.message ?? error),
        };
    }

    updateCounterfactualSimulationQuestion();
    renderCounterfactualSimulationRows();
    renderCounterfactualSimulationFeedback();
}

async function initializeBudgetSimulation() {
    try {
        const candidate = await findBudgetSimulationCandidate();
        simulationSpecificCandidatePending = false;
        if (candidate) {
            simulationFeedback = {
                isCorrect: true,
                text: `A solution exists within ${SIMULATION_BUDGET_POINTS} points.`,
            };
        } else {
            simulationFeedback = {
                isCorrect: false,
                text: `No solution within ${SIMULATION_BUDGET_POINTS} points was found for this case.`,
            };
        }
    } catch (error) {
        simulationSpecificCandidatePending = false;
        simulationFeedback = {
            isCorrect: false,
            text: String(error.message ?? error),
        };
    }

    updateCounterfactualSimulationQuestion();
    renderCounterfactualSimulationRows();
    renderCounterfactualSimulationFeedback();
}

function formatSimulationAttributeList(attributeIndices) {
    return joinClauses(attributeIndices.map((attributeIndex) => getAttributeName(attributeIndex)));
}

async function checkCounterfactualSimulation() {
    const resultPanel = document.querySelector("#counterfactual-simulation-result");
    if (resultPanel) {
        resultPanel.textContent = "Checking...";
        resultPanel.className = "counterfactual-simulation-result";
    }

    try {
        const validationMessage = getSimulationValidationMessage();
        if (validationMessage) {
            simulationPrediction = null;
            simulationFeedback = {
                isCorrect: false,
                text: validationMessage,
            };
            renderCounterfactualSimulationFeedback();
            return;
        }

        simulationPrediction = await predictRawValues(getSimulationRawValues());
        const targetValue = getSimulationTargetValue();
        const isCorrect = Number(simulationPrediction.value) === targetValue;
        simulationFeedback = {
            isCorrect,
            text: isCorrect
                ? `Correct. The AI now predicts ${shortenClassLabel(simulationPrediction.label)}.`
                : `Not yet. The AI still predicts ${shortenClassLabel(simulationPrediction.label)}.`,
        };
        renderCounterfactualSimulationFeedback();
    } catch (error) {
        simulationPrediction = null;
        simulationFeedback = {
            isCorrect: false,
            text: String(error.message ?? error),
        };
        renderCounterfactualSimulationFeedback();
    }
}

function getSimulationValidationMessage() {
    const changedIndices = getSimulationChangedAttributeIndices();

    if (
        simulationAllowedAttributeIndices !== null &&
        changedIndices.some((attributeIndex) => !simulationAllowedAttributeIndices.includes(attributeIndex))
    ) {
        return `Only ${formatSimulationAttributeList(simulationAllowedAttributeIndices)} can be changed in this case.`;
    }

    if (counterfactualSimulationMode === "budget") {
        const spentPoints = getSimulationSpentPoints();
        if (spentPoints > SIMULATION_BUDGET_POINTS) {
            return `This change uses ${formatSimulationPoints(spentPoints)}, which is over the ${SIMULATION_BUDGET_POINTS}-point budget.`;
        }
    }

    return "";
}

function renderCounterfactualSimulationFeedback() {
    const resultPanel = document.querySelector("#counterfactual-simulation-result");
    if (!resultPanel) {
        return;
    }

    resultPanel.innerHTML = "";
    resultPanel.className = "counterfactual-simulation-result";

    if (!simulationPrediction && !simulationFeedback) {
        return;
    }

    if (simulationPrediction) {
        const output = document.createElement("div");
        output.className = "simulation-output";
        output.appendChild(createPredictionChoiceGroup(simulationPrediction.value, {
            colorResult: false,
            predictionTone: "counterfactual",
            originalPrediction: currentExplanation.prediction.value,
        }));
        resultPanel.appendChild(output);
    }

    if (simulationFeedback) {
        const feedback = document.createElement("div");
        feedback.className = simulationFeedback.isCorrect
            ? "simulation-feedback simulation-feedback-correct"
            : "simulation-feedback simulation-feedback-incorrect";
        feedback.textContent = simulationFeedback.text;
        resultPanel.classList.add(
            simulationFeedback.isCorrect
                ? "counterfactual-simulation-result-correct"
                : "counterfactual-simulation-result-incorrect"
        );
        resultPanel.appendChild(feedback);
    }
}

function joinClauses(clauses) {
    if (clauses.length === 0) {
        return "";
    }
    if (clauses.length === 1) {
        return clauses[0];
    }
    if (clauses.length === 2) {
        return `${clauses[0]} and ${clauses[1]}`;
    }
    return `${clauses.slice(0, -1).join(", ")}, and ${clauses[clauses.length - 1]}`;
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll("\"", "&quot;")
        .replaceAll("'", "&#39;");
}

function strongHtml(value) {
    return `<strong>${escapeHtml(value)}</strong>`;
}

function summarizeAttributionDirection(entries, label) {
    if (entries.length === 0) {
        return "";
    }

    const total = entries.reduce((sum, entry) => sum + Math.abs(entry.value), 0) || 1;
    const featureClauses = entries.slice(0, 3).map((entry) => {
        const share = Math.round((Math.abs(entry.value) / total) * 100);
        return `${escapeHtml(entry.name)} (<strong>${share}%</strong>)`;
    });
    return `${joinClauses(featureClauses)} contributed toward ${escapeHtml(label)}.`;
}

function getPatientOutcomePhrase(label) {
    const labelText = String(label ?? "").toLowerCase();

    if (datasetName === "diabetes") {
        return labelText.includes("no")
            ? "the patient would likely not develop diabetes"
            : "the patient would likely develop diabetes";
    }

    if (datasetName === "safelimit") {
        return labelText.includes("below")
            ? "the person would likely be below the limit"
            : "the person would likely be above the limit";
    }

    return `the case would likely be ${escapeHtml(label)}`;
}

function getProfileOutcomePhrase(label, options = {}) {
    const { hypothetical = false } = options;
    const labelText = String(label ?? "").toLowerCase();

    if (datasetName === "diabetes") {
        if (hypothetical) {
            return labelText.includes("no")
                ? "the person would not have diabetes"
                : "the person would have diabetes";
        }

        return labelText.includes("no")
            ? "the person does not have diabetes"
            : "the person has diabetes";
    }

    const verb = hypothetical ? "would be" : "is";

    if (datasetName === "safelimit") {
        if (hypothetical) {
            return labelText.includes("below")
                ? "the person would be below the limit"
                : "the person would be above the limit";
        }

        return labelText.includes("below")
            ? "the person is below the limit"
            : "the person is above the limit";
    }

    return `the case ${verb} ${escapeHtml(label)}`;
}

function getInfluenceDirectionLabel(value, attribution) {
    const label = value >= 0
        ? attribution.directionLabels?.right
        : attribution.directionLabels?.left;
    return shortenClassLabel(label);
}

function buildLooseAttributionInfluenceText(attribution) {
    const signedEntries = attribution.values
        .map((value, index) => ({
            name: getNarrativeAttributeName(index),
            value,
            direction: getInfluenceDirectionLabel(value, attribution),
        }))
        .filter((entry) => Math.abs(entry.value) > 0)
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 3);

    if (signedEntries.length === 0) {
        return "";
    }

    const clauses = signedEntries.map((entry) =>
        `${strongHtml(entry.name)} on being ${strongHtml(entry.direction)}`
    );

    return joinClauses(clauses);
}

function buildLooseCounterfactualChangeEntry(index, counterfactualValues) {
    const name = strongHtml(getNarrativeAttributeName(index));
    const updatedDisplay = escapeHtml(getAttributeDisplayValue(index, counterfactualValues));

    if (currentExplanation.attributeTypes[index] === "categorical") {
        return {
            name,
            phrase: `were ${updatedDisplay}`,
        };
    }

    const originalValue = Number(currentExplanation.attributeValues[index]);
    const updatedValue = Number(counterfactualValues[index]);
    if (!Number.isFinite(originalValue) || !Number.isFinite(updatedValue)) {
        return {
            name,
            phrase: `were ${updatedDisplay}`,
        };
    }

    if (updatedValue > originalValue) {
        return {
            name,
            phrase: "were higher",
        };
    }

    if (updatedValue < originalValue) {
        return {
            name,
            phrase: "were lower",
        };
    }

    return {
        name,
        phrase: "stayed about the same",
    };
}

function buildLooseCounterfactualChangeText(changedIndices, counterfactualValues) {
    const groups = [];

    changedIndices
        .map((index) => buildLooseCounterfactualChangeEntry(index, counterfactualValues))
        .forEach((entry) => {
            const existingGroup = groups.find((group) => group.phrase === entry.phrase);
            if (existingGroup) {
                existingGroup.names.push(entry.name);
                return;
            }
            groups.push({
                phrase: entry.phrase,
                names: [entry.name],
            });
        });

    return joinClauses(groups.map((group) =>
        `${joinClauses(group.names)} ${group.phrase}`
    ));
}

function buildCounterfactualChangeText(index, counterfactualValues) {
    const name = strongHtml(getNarrativeAttributeName(index));
    const originalDisplay = getAttributeDisplayValue(index, currentExplanation.attributeValues);
    const updatedDisplay = getAttributeDisplayValue(index, counterfactualValues);

    if (currentExplanation.attributeTypes[index] === "categorical") {
        return `${name} was <span class="categorical-old-value-marker" title="Original: ${escapeHtml(originalDisplay)}"></span><span class="value-change-arrow"> -> </span><span class="value-delta value-delta-increase">${escapeHtml(updatedDisplay)}</span>`;
    }

    const originalValue = Number(currentExplanation.attributeValues[index]);
    const updatedValue = Number(counterfactualValues[index]);
    const delta = updatedValue - originalValue;
    if (!Number.isFinite(originalValue) || !Number.isFinite(updatedValue) || delta === 0) {
        return `${name} was <strong>${escapeHtml(updatedDisplay)}</strong>`;
    }

    const deltaClass = delta > 0 ? "value-delta-increase" : "value-delta-decrease";
    return `${name} was ${escapeHtml(formatValue(originalValue))}<span class="value-delta ${deltaClass}"> ${delta > 0 ? "+" : "-"} ${escapeHtml(formatValue(Math.abs(delta)))}</span>`;
}

function buildNarrativeHtml() {
    if (explanationType === "counterfactual") {
        const counterfactual = currentExplanation.counterfactual;
        if (!counterfactual) {
            return "No counterfactual example was available for this instance.";
        }

        const changes = currentExplanation.attributeNames
            .map((_, index) => index)
            .filter((index) =>
                hasAttributeChanged(
                    index,
                    currentExplanation.attributeValues,
                    counterfactual.feature_values
                )
            )
            .map((index) => buildCounterfactualChangeText(index, counterfactual.feature_values));

        if (changes.length === 0) {
            return `Given this profile, the AI prediction is ${strongHtml(shortenClassLabel(currentExplanation.prediction.label))}.`;
        }

        const changedIndices = currentExplanation.attributeNames
            .map((_, index) => index)
            .filter((index) =>
                hasAttributeChanged(
                    index,
                    currentExplanation.attributeValues,
                    counterfactual.feature_values
                )
            );
        const looseChanges = buildLooseCounterfactualChangeText(
            changedIndices,
            counterfactual.feature_values
        );

        if (datasetName === "diabetes") {
            const currentDiagnosis = strongHtml(shortenClassLabel(currentExplanation.prediction.label));
            const counterfactualDiagnosis = strongHtml(shortenClassLabel(counterfactual.prediction.label));
            return `This patient is diagnosed as ${currentDiagnosis}. But, if their ${looseChanges}, then they would be diagnosed as ${counterfactualDiagnosis}.`;
        }

        if (datasetName === "safelimit") {
            const currentPrediction = strongHtml(shortenClassLabel(currentExplanation.prediction.label));
            const counterfactualPrediction = strongHtml(shortenClassLabel(counterfactual.prediction.label));
            return `This driver is predicted as ${currentPrediction}. But, if their ${looseChanges}, then they would be predicted as ${counterfactualPrediction}.`;
        }

        return `Given this profile, if ${looseChanges}, the AI prediction would be ${strongHtml(shortenClassLabel(counterfactual.prediction.label))}.`;
    }

    if (explanationType === "attribution") {
        const attribution = currentExplanation.attribution;
        if (!attribution || !Array.isArray(attribution.values)) {
            return "No attribution data was available for this instance.";
        }

        const influenceText = buildLooseAttributionInfluenceText(attribution);

        if (datasetName === "diabetes") {
            const diagnosis = strongHtml(shortenClassLabel(currentExplanation.prediction.label));
            return influenceText
                ? `This patient is diagnosed as ${diagnosis}, given the influence of ${influenceText}.`
                : `This patient is diagnosed as ${diagnosis}.`;
        }

        const predictionLabel = strongHtml(shortenClassLabel(currentExplanation.prediction.label));
        return influenceText
            ? `Given this profile, the AI prediction is ${predictionLabel}, reflecting the influence of ${influenceText}.`
            : `Given this profile, the AI prediction is ${predictionLabel}.`;
    }

    return `Given this profile, the AI prediction is ${strongHtml(shortenClassLabel(currentExplanation.prediction.label))}.`;
}

function showNarrativePanel() {
    if (!explanationBoxAnchor) {
        return;
    }

    const narrativePanel = document.createElement("div");
    narrativePanel.id = "narrative-panel";
    narrativePanel.className = "narrative-panel";

    const text = document.createElement("p");
    text.className = "narrative-panel-text";
    text.innerHTML = buildNarrativeHtml();

    narrativePanel.appendChild(text);
    explanationBoxAnchor.appendChild(narrativePanel);
    syncNarrativePanelWidth(narrativePanel);
}

function syncNarrativePanelWidth(narrativePanel) {
    const referenceElement = explanationView === "persona" && explanationType === "counterfactual"
        ? tablesWrapper
        : document.querySelector("#none-explanation-table");

    if (!referenceElement) {
        return;
    }

    const applyWidth = () => {
        const width = referenceElement.getBoundingClientRect().width;
        if (width > 0) {
            narrativePanel.style.width = `${Math.round(width)}px`;
        }
    };

    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(applyWidth);
    } else {
        applyWidth();
    }
}

function normalizeExplanationPayload(payload) {
    const attribution = payload.attribution
        ? {
            ...payload.attribution,
            directionLabels: payload.attribution.direction_labels
                ?? payload.attribution.directionLabels
                ?? {},
            shownFeatureIndices: payload.attribution.shown_feature_indices
                ?? payload.attribution.shownFeatureIndices
                ?? [],
        }
        : null;

    return applyAttributeOrder({
        attributeNames: payload.feature_names ?? [],
        rawAttributeNames: payload.raw_feature_names ?? [],
        attributeTypes: payload.feature_types ?? [],
        attributeValues: payload.feature_values ?? [],
        rawAttributeValues: payload.raw_feature_values ?? payload.feature_values ?? [],
        attributeRanges: payload.feature_ranges ?? [],
        rawAttributeRanges: payload.raw_feature_ranges ?? payload.feature_ranges ?? [],
        attribution,
        attributionMax: Math.max(payload.attribution?.max_abs_value ?? 0, 1e-9),
        explanationFeatureCount: payload.explanation_feature_count ?? explanationFeatureCount,
        prediction: payload.prediction ?? { value: 0 },
        predictionLabels: payload.prediction_labels ?? [],
        counterfactual: payload.counterfactual ?? null,
    });
}

function applyAttributeOrder(explanation) {
    if (!attributeOrderSeed || explanation.attributeNames.length <= 1) {
        return explanation;
    }

    const order = seededShuffledIndices(
        explanation.attributeNames.length,
        `${attributeOrderSeed}:${datasetName}:${modelName}`
    );
    const reorder = (values) => Array.isArray(values)
        ? order.map((index) => values[index])
        : values;
    const inverseOrder = new Map(order.map((oldIndex, newIndex) => [oldIndex, newIndex]));

    const reorderedAttribution = explanation.attribution
        ? {
            ...explanation.attribution,
            values: reorder(explanation.attribution.values),
            raw_values: reorder(explanation.attribution.raw_values),
            shownFeatureIndices: (explanation.attribution.shownFeatureIndices ?? [])
                .map((index) => inverseOrder.get(index))
                .filter((index) => index !== undefined),
        }
        : null;
    const reorderedCounterfactual = explanation.counterfactual
        ? {
            ...explanation.counterfactual,
            feature_values: reorder(explanation.counterfactual.feature_values),
            raw_feature_values: reorder(explanation.counterfactual.raw_feature_values),
        }
        : null;

    return {
        ...explanation,
        attributeNames: reorder(explanation.attributeNames),
        rawAttributeNames: reorder(explanation.rawAttributeNames),
        attributeTypes: reorder(explanation.attributeTypes),
        attributeValues: reorder(explanation.attributeValues),
        rawAttributeValues: reorder(explanation.rawAttributeValues),
        attributeRanges: reorder(explanation.attributeRanges),
        rawAttributeRanges: reorder(explanation.rawAttributeRanges),
        attribution: reorderedAttribution,
        counterfactual: reorderedCounterfactual,
    };
}

function seededShuffledIndices(length, seed) {
    const random = createSeededRandom(seed);
    const indices = Array.from({ length }, (_, index) => index);
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    return indices;
}

function createSeededRandom(seed) {
    let hash = 2166136261;
    for (let i = 0; i < seed.length; i++) {
        hash ^= seed.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
    }
    return () => {
        hash += 0x6D2B79F5;
        let value = hash;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
}

function clearFaceFigurePanel() {
    document.querySelector("#face-figure-panel")?.remove();
    tablesWrapper?.classList.remove("face-figure-table-wrapper");
    document.body.classList.remove("face-figures-active");
}

function showFaceFigurePanel() {
    if (!faceFiguresEnabled || !currentExplanation) {
        return;
    }

    assertFaceMappingAlignment();

    const wrapper = document.querySelector("#explanation-wrapper");
    if (!wrapper || !tablesWrapper) {
        return;
    }
    tablesWrapper.hidden = false;
    tablesWrapper.classList.add("face-figure-table-wrapper");
    tablesWrapper.querySelector("#none-explanation-table")?.removeAttribute("hidden");

    const panel = document.createElement("section");
    panel.id = "face-figure-panel";
    panel.className = "face-figure-panel";
    document.body.classList.add("face-figures-active");

    const cards = document.createElement("div");
    cards.className = "face-figure-cards";
    panel.appendChild(cards);

    if (explanationType === "counterfactual" && currentExplanation.counterfactual) {
        panel.classList.add("face-figure-panel-comparison");
        cards.appendChild(createFaceFigureCard({
            label: "Original",
            values: currentExplanation.attributeValues,
            prediction: currentExplanation.prediction,
            highlightMode: "counterfactual-original",
            comparisonValues: currentExplanation.counterfactual.feature_values,
        }));
        cards.appendChild(createFaceFigureCard({
            label: "Counterfactual",
            values: currentExplanation.counterfactual.feature_values,
            prediction: currentExplanation.counterfactual.prediction,
            highlightMode: "counterfactual",
            comparisonValues: currentExplanation.attributeValues,
        }));
    } else {
        cards.appendChild(createFaceFigureCard({
            label: getFaceFigureLabel(),
            values: currentExplanation.attributeValues,
            prediction: currentExplanation.prediction,
            highlightMode: explanationType === "attribution" ? "attribution" : "none",
        }));
    }

    const summary = createFaceFigureSummary();
    if (summary) {
        panel.appendChild(summary);
    }

    wrapper.insertBefore(panel, tablesWrapper);
}

function getFaceFigureLabel() {
    if (datasetName === "diabetes") {
        return "Patient";
    }
    if (datasetName === "safelimit") {
        return "Person";
    }
    return "Case";
}

function createFaceFigureCard(options) {
    const {
        label,
        values,
        prediction,
        highlightMode,
        comparisonValues = null,
    } = options;
    const card = document.createElement("div");
    card.className = "face-figure-card";

    const title = document.createElement("div");
    title.className = "face-figure-title";
    title.textContent = label;
    card.appendChild(title);

    card.appendChild(createFaceSvg(values, {
        highlightMode,
        comparisonValues,
    }));

    if (prediction && showPredictionPanel) {
        const predictionWrap = document.createElement("div");
        predictionWrap.className = "face-figure-prediction";
        predictionWrap.appendChild(createPredictionChoiceGroup(Number(prediction.value), {
            colorResult: highlightMode !== "attribution",
            predictionTone: highlightMode === "counterfactual" ? "counterfactual" : "standard",
            originalPrediction: highlightMode === "counterfactual"
                ? currentExplanation.prediction.value
                : null,
        }));
        card.appendChild(predictionWrap);
    }

    return card;
}

function assertFaceMappingAlignment() {
    const featureCount = currentExplanation.attributeNames.length;
    const alignedArrays = [
        currentExplanation.rawAttributeNames,
        currentExplanation.attributeTypes,
        currentExplanation.attributeValues,
        currentExplanation.rawAttributeValues,
        currentExplanation.attributeRanges,
        currentExplanation.rawAttributeRanges,
        currentExplanation.attribution?.values,
    ].filter(Boolean);

    const hasMismatch = alignedArrays.some((values) => values.length !== featureCount);
    const counterfactualValues = currentExplanation.counterfactual?.feature_values;
    const counterfactualMismatch = counterfactualValues &&
        counterfactualValues.length !== featureCount;
    if (hasMismatch || counterfactualMismatch) {
        throw new Error("Face figure mapping is inconsistent with the explanation feature arrays.");
    }
}

function createFaceSvg(values, options = {}) {
    const {
        highlightMode = "none",
        comparisonValues = null,
    } = options;
    const svg = createSvgElement("svg", {
        class: "face-figure-svg",
        viewBox: "0 0 240 250",
        role: "img",
    });
    const normalizedValues = currentExplanation.attributeNames.map((_, index) =>
        normalizeFaceFeatureValue(index, values)
    );
    const parameter = (index, fallback = 0.5) =>
        normalizedValues[index] === undefined ? fallback : normalizedValues[index];
    const params = {
        faceWidth: 88 + parameter(0) * 42,
        faceHeight: 136,
        jawWidth: parameter(1),
        eyeSize: 4 + parameter(2) * 7,
        browTilt: -11 + parameter(4) * 22,
        noseHeight: 18 + parameter(5) * 28,
        mouthCurve: -18 + parameter(6) * 36,
        mouthWidth: 32 + parameter(7) * 38,
        earSize: 6 + parameter(8) * 12,
        cheekRoundness: parameter(9),
    };
    params.eyeGap = Math.min(
        20 + parameter(3) * 22,
        (params.faceWidth / 2) - params.eyeSize - 10
    );
    const highlights = getFaceHighlightMap(highlightMode, values, comparisonValues);

    const faceGroup = createSvgElement("g", {
        transform: "translate(120 120)",
    });
    svg.appendChild(faceGroup);

    appendFacePart(faceGroup, [8], highlights, [
        createSvgElement("ellipse", {
            class: "face-stroke face-soft-fill",
            cx: String(-(params.faceWidth / 2 + 3)),
            cy: "0",
            rx: String(params.earSize),
            ry: String(params.earSize * 1.35),
        }),
        createSvgElement("ellipse", {
            class: "face-stroke face-soft-fill",
            cx: String(params.faceWidth / 2 + 3),
            cy: "0",
            rx: String(params.earSize),
            ry: String(params.earSize * 1.35),
        }),
    ]);

    appendFacePart(faceGroup, [0, 1, 9], highlights, [
        createSvgElement("path", {
            class: "face-stroke face-skin-fill",
            d: createFaceOutlinePath(params),
        }),
    ]);

    appendFacePart(faceGroup, [2, 3], highlights, [
        createSvgElement("circle", {
            class: "face-stroke face-dark-fill",
            cx: String(-params.eyeGap),
            cy: "-20",
            r: String(params.eyeSize),
        }),
        createSvgElement("circle", {
            class: "face-stroke face-dark-fill",
            cx: String(params.eyeGap),
            cy: "-20",
            r: String(params.eyeSize),
        }),
    ]);

    appendFacePart(faceGroup, [4], highlights, [
        createSvgElement("line", {
            class: "face-stroke",
            x1: String(-params.eyeGap - 12),
            y1: String(-39 - params.browTilt),
            x2: String(-params.eyeGap + 13),
            y2: String(-39 + params.browTilt),
        }),
        createSvgElement("line", {
            class: "face-stroke",
            x1: String(params.eyeGap - 13),
            y1: String(-39 + params.browTilt),
            x2: String(params.eyeGap + 12),
            y2: String(-39 - params.browTilt),
        }),
    ]);

    appendFacePart(faceGroup, [5], highlights, [
        createSvgElement("path", {
            class: "face-stroke face-no-fill",
            d: `M 0 -8 C ${5 + parameter(5) * 6} ${params.noseHeight / 3} ${5 + parameter(5) * 5} ${params.noseHeight - 5} 0 ${params.noseHeight}`,
        }),
        createSvgElement("path", {
            class: "face-stroke face-no-fill",
            d: `M -7 ${params.noseHeight - 3} Q 0 ${params.noseHeight + 3} 7 ${params.noseHeight - 3}`,
        }),
    ]);

    appendFacePart(faceGroup, [6, 7], highlights, [
        createSvgElement("path", {
            class: "face-stroke face-no-fill",
            d: `M ${-params.mouthWidth / 2} 55 Q 0 ${55 + params.mouthCurve} ${params.mouthWidth / 2} 55`,
        }),
    ]);

    return svg;
}

function appendFacePart(parent, featureIndices, highlights, elements) {
    const highlight = getStrongestFaceHighlight(featureIndices, highlights);
    if (elements.length === 0 && !highlight) {
        return;
    }
    const group = createSvgElement("g", {
        class: "face-feature-part",
    });
    if (highlight) {
        group.classList.add("face-feature-highlight");
        if (highlight.isTop) {
            group.classList.add("face-feature-top");
        }
        group.style.setProperty("--face-part-color", highlight.color);
        group.style.setProperty("--face-part-width", String(highlight.strokeWidth));
        group.style.setProperty("--face-part-opacity", String(highlight.opacity));
    }
    if (highlight?.label) {
        const title = createSvgElement("title");
        title.textContent = highlight.label;
        group.appendChild(title);
    }
    elements.forEach((element) => group.appendChild(element));
    parent.appendChild(group);
}

function createFaceOutlinePath(params) {
    const rx = params.faceWidth / 2;
    const ry = params.faceHeight / 2;
    const roundness = params.cheekRoundness;
    const jawWidth = params.jawWidth;
    const cheekX = rx * (0.82 + (roundness * 0.14));
    const jawX = rx * (0.28 + (jawWidth * 0.38) + (roundness * 0.08));
    const chinY = ry * (0.98 - (roundness * 0.06));
    return [
        `M 0 ${-ry}`,
        `C ${cheekX} ${-ry} ${rx} ${-ry * 0.42} ${rx * 0.94} 0`,
        `C ${rx * 0.88} ${ry * 0.42} ${jawX} ${chinY} 0 ${ry}`,
        `C ${-jawX} ${chinY} ${-rx * 0.88} ${ry * 0.42} ${-rx * 0.94} 0`,
        `C ${-rx} ${-ry * 0.42} ${-cheekX} ${-ry} 0 ${-ry}`,
        "Z",
    ].join(" ");
}

function getStrongestFaceHighlight(featureIndices, highlights) {
    return featureIndices
        .map((index) => highlights.get(index))
        .filter(Boolean)
        .sort((a, b) => b.score - a.score)[0] ?? null;
}

function getFaceHighlightMap(mode, values, comparisonValues) {
    const highlights = new Map();
    if (mode === "attribution") {
        const attribution = currentExplanation.attribution;
        const attributionValues = attribution?.values ?? [];
        const total = attributionValues.reduce((sum, value) => sum + Math.abs(value), 0) || 1;
        const topIndices = attributionValues
            .map((value, index) => ({ value: Math.abs(value), index }))
            .filter((entry) => entry.value > 0)
            .sort((a, b) => b.value - a.value)
            .slice(0, 2)
            .map((entry) => entry.index);

        attributionValues.forEach((value, index) => {
            const magnitude = Math.abs(value);
            if (magnitude <= 0) {
                return;
            }
            const share = magnitude / total;
            const isTop = topIndices.includes(index);
            highlights.set(index, {
                color: value >= 0 ? "rgba(60, 136, 232, 1)" : "rgba(234, 51, 53, 1)",
                opacity: isTop ? 1 : clamp(0.35 + Math.sqrt(share), 0.45, 0.86),
                strokeWidth: isTop ? 5.5 : 2 + (share * 8),
                isTop,
                score: magnitude,
                index,
                label: `${Math.round(share * 100)}% toward ${getInfluenceDirectionLabel(value, attribution)}`,
            });
        });
        return highlights;
    }

    if (mode === "counterfactual" || mode === "counterfactual-original") {
        if (!comparisonValues) {
            return highlights;
        }
        currentExplanation.attributeNames.forEach((_, index) => {
            if (!hasAttributeChanged(index, values, comparisonValues)) {
                return;
            }
            highlights.set(index, {
                color: "rgba(64, 143, 74, 1)",
                opacity: 1,
                strokeWidth: 4.6,
                isTop: true,
                score: getFaceChangeMagnitude(index, values, comparisonValues),
                index,
                label: mode === "counterfactual"
                    ? `${getFaceDimensionName(index)} ${getFaceChangeDirection(index, comparisonValues, values)}`
                    : `${getFaceDimensionName(index)} changed`,
            });
        });
    }

    return highlights;
}

function createFaceFigureSummary() {
    let summaryText = "";
    if (explanationType === "attribution") {
        summaryText = getFaceAttributionSummaryText();
    } else if (explanationType === "counterfactual" && currentExplanation.counterfactual) {
        summaryText = getFaceCounterfactualSummaryText();
    }
    if (!summaryText) {
        return null;
    }

    const summary = document.createElement("p");
    summary.className = "face-figure-summary";
    summary.textContent = summaryText;
    return summary;
}

function getFaceAttributionSummaryText() {
    const attribution = currentExplanation.attribution;
    const attributionValues = attribution?.values ?? [];
    const total = attributionValues.reduce((sum, value) => sum + Math.abs(value), 0) || 1;
    const entries = attributionValues
        .map((value, index) => ({
            index,
            value,
            share: Math.abs(value) / total,
        }))
        .filter((entry) => Math.abs(entry.value) > 0)
        .sort((a, b) => b.share - a.share)
        .slice(0, 2);

    if (entries.length === 0) {
        return "";
    }

    const clauses = entries.map((entry) =>
        `${getFaceDimensionName(entry.index)} (${Math.round(entry.share * 100)}% toward ${getInfluenceDirectionLabel(entry.value, attribution)})`
    );
    return `Most important face parts: ${joinClauses(clauses)}.`;
}

function getFaceCounterfactualSummaryText() {
    const counterfactualValues = currentExplanation.counterfactual?.feature_values;
    if (!counterfactualValues) {
        return "";
    }

    const changes = currentExplanation.attributeNames
        .map((_, index) => ({
            index,
            magnitude: getFaceChangeMagnitude(
                index,
                counterfactualValues,
                currentExplanation.attributeValues
            ),
            direction: getFaceChangeDirection(
                index,
                currentExplanation.attributeValues,
                counterfactualValues
            ),
        }))
        .filter((entry) => entry.magnitude > 0)
        .sort((a, b) => b.magnitude - a.magnitude)
        .slice(0, 3);

    if (changes.length === 0) {
        return "";
    }

    const clauses = changes.map((entry) =>
        `${getFaceDimensionName(entry.index)} ${entry.direction}`
    );
    return `Changed face parts: ${joinClauses(clauses)}.`;
}

function getFaceDimensionName(index) {
    const names = [
        "face width",
        "jaw width",
        "eye size",
        "eye spacing",
        "eyebrow tilt",
        "nose length",
        "mouth curve",
        "mouth width",
        "ear size",
        "face roundness",
    ];
    return names[index] ?? `face part ${index + 1}`;
}

function getFaceChangeMagnitude(index, values, comparisonValues) {
    const currentValue = normalizeFaceFeatureValue(index, values);
    const comparisonValue = normalizeFaceFeatureValue(index, comparisonValues);
    return Math.abs(currentValue - comparisonValue);
}

function getFaceChangeDirection(index, originalValues, updatedValues) {
    const originalValue = normalizeFaceFeatureValue(index, originalValues);
    const updatedValue = normalizeFaceFeatureValue(index, updatedValues);
    if (Math.abs(updatedValue - originalValue) < 1e-9) {
        return "stayed about the same";
    }

    const increased = updatedValue > originalValue;
    if (index === 4) {
        return increased ? "tilted upward" : "tilted downward";
    }
    if (index === 1) {
        return increased ? "widened" : "narrowed";
    }
    if (index === 6) {
        return increased ? "curved upward" : "curved downward";
    }
    return increased ? "increased" : "decreased";
}

function normalizeFaceFeatureValue(attributeIndex, values) {
    if (currentExplanation.attributeTypes[attributeIndex] === "categorical") {
        const options = currentExplanation.attributeRanges[attributeIndex] ?? [];
        if (options.length <= 1) {
            return 0.5;
        }
        return clamp(getCategoryIndex(attributeIndex, values) / (options.length - 1), 0, 1);
    }

    const [min, max] = currentExplanation.attributeRanges[attributeIndex] ?? [0, 0];
    const minValue = Number(min);
    const maxValue = Number(max);
    const value = Number(values?.[attributeIndex]);
    if (
        !Number.isFinite(minValue) ||
        !Number.isFinite(maxValue) ||
        !Number.isFinite(value) ||
        minValue === maxValue
    ) {
        return 0.5;
    }
    return clamp((value - minValue) / (maxValue - minValue), 0, 1);
}

function createSvgElement(tagName, attributes = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tagName);
    Object.entries(attributes).forEach(([name, value]) => {
        element.setAttribute(name, value);
    });
    return element;
}

function scheduleIframeHeightPost() {
    if (typeof window.parent?.postMessage !== "function") {
        return;
    }
    const postHeight = () => {
        const targetHeight = Math.max(
            document.body?.scrollHeight ?? 0,
            document.documentElement?.scrollHeight ?? 0,
            document.body?.getBoundingClientRect().height ?? 0
        );
        const height = Math.ceil(targetHeight + 8);
        window.parent.postMessage({
            type: "counterfactual-ui:iframe-height",
            height,
        }, "*");
    };

    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(() => requestAnimationFrame(postHeight));
    } else {
        setTimeout(postHeight, 0);
    }
}

function renderExplanation() {
    clearCounterfactualTable();
    clearNarrativePanel();
    clearPredictionPanel();
    clearCounterfactualSimulation();
    clearFaceFigurePanel();
    if (tablesWrapper) {
        tablesWrapper.hidden = false;
    }
    resetAttributionChart();

    if (counterfactualSimulationEnabled) {
        noneExplanationTbody.innerHTML = "";
        if (tablesWrapper) {
            tablesWrapper.hidden = true;
        }
    } else {
        showAttributeValues(noneExplanationTbody);
    }

    if (
        explanationType !== "none" &&
        !faceFiguresEnabled &&
        explanationView === "narrative"
    ) {
        showNarrativePanel();
        return;
    }

    if (explanationType === "attribution") {
        showAttributionChart(noneExplanationTbody);
        showNarrativePanel();
    }

    if (explanationType === "counterfactual") {
        if (explanationView !== "persona") {
            setCaseLabel(noneExplanationTbody.closest("table"), "Subject", {
                startColumn: 1,
                columnSpan: 2,
            });
        }
        showCounterfactualExample(noneExplanationTbody);
        if (explanationView === "persona") {
            showNarrativePanel();
        }
    }

    if (faceFiguresEnabled) {
        showFaceFigurePanel();
        applyTutorialCallouts();
        scheduleIframeHeightPost();
        return;
    }

    createCounterfactualSimulation();
    applyTutorialCallouts();
    scheduleIframeHeightPost();
}

async function loadExplanation() {
    renderStatusRow("Loading explanation data...");

    try {
        const payload = getStaticExplanationPayload();
        currentExplanation = normalizeExplanationPayload(payload);
        updateAttributeControlHeader();
        renderExplanation();
    } catch (error) {
        console.error("Failed to load explanation:", error);
        renderStatusRow(String(error.message ?? error), true);
    }
}

document.addEventListener("DOMContentLoaded", function () {
    loadExplanation();
});
