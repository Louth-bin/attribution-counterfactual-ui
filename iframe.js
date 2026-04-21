const urlParams = new URLSearchParams(window.location.search);
const explanationType = getExplanationType(urlParams.get("xaiType") ?? "none");
const showPredictionPanel = urlParams.get("showPrediction") !== "0";
const datasetName = urlParams.get("appId") ?? "diabetes";
const modelName = urlParams.get("AIModel") ?? "mlp";
const xaiMethod = urlParams.get("expAlgorithm") ?? "shap";
const instanceId = Number(urlParams.get("instanceId") ?? "0");
const explanationFeatureCount = Number(urlParams.get("k") ?? "3");
const apiBaseUrl = resolveApiBaseUrl(urlParams.get("apiBaseUrl"));

const noneExplanationTbody = document.querySelector("#none-explanation-tbody");
let currentExplanation = null;
let attributionChart = null;

console.info("[iframe] apiBaseUrl:", apiBaseUrl);

function buildApiUrl(path) {
    const baseUrl = apiBaseUrl.endsWith("/") ? apiBaseUrl : `${apiBaseUrl}/`;
    return new URL(path, baseUrl);
}

function appendApiPath(baseUrl) {
    const url = new URL(baseUrl, window.location.href);
    const pathParts = url.pathname.split("/").filter(Boolean);

    if (pathParts[pathParts.length - 1] !== "api") {
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

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
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

    const meter = document.createElement("meter");
    if (muted) {
        meter.classList.add("meter-unchanged");
    }
    meter.min = min;
    meter.max = max;
    meter.value = clamp(value, min, max);
    meter.title = `${formatValue(value)} (min: ${formatValue(min)}, max: ${formatValue(max)})`;
    meterCell.appendChild(meter);

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

function populateAttributeTable(tableBody, values, options = {}) {
    const {
        includeNames = true,
        originalValues = null,
    } = options;

    tableBody.innerHTML = "";

    for (let i = 0; i < currentExplanation.attributeNames.length; i++) {
        const row = document.createElement("tr");
        row.className = `attribute-row row_${i}`;
        const isUnchanged = originalValues && !hasAttributeChanged(i, originalValues, values);

        if (isUnchanged) {
            row.classList.add("counterfactual-row-unchanged");
        }

        const valueCell = document.createElement("td");
        valueCell.className = "value";

        if (currentExplanation.attributeTypes[i] === "categorical") {
            const options = currentExplanation.attributeRanges[i];
            const categoryIndex = getCategoryIndex(i, values);

            valueCell.textContent = options[categoryIndex] ?? String(values[i]);
            if (includeNames) {
                const nameCell = document.createElement("td");
                nameCell.className = "attribute";
                nameCell.textContent = currentExplanation.attributeNames[i];
                row.appendChild(nameCell);
            }
            row.appendChild(valueCell);
            row.appendChild(createCategoryIconsCell(options, categoryIndex, {
                muted: Boolean(isUnchanged),
            }));
        } else {
            const [min, max] = currentExplanation.attributeRanges[i];
            const numericValue = Number(values[i]);

            valueCell.textContent = formatValue(values[i]);
            if (includeNames) {
                const nameCell = document.createElement("td");
                nameCell.className = "attribute";
                nameCell.textContent = currentExplanation.attributeNames[i];
                row.appendChild(nameCell);
            }
            row.appendChild(valueCell);
            row.appendChild(createMeterCell(numericValue, min, max, {
                muted: Boolean(isUnchanged),
            }));
        }

        if (originalValues) {
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

function showAttributeValues(tableBody) {
    populateAttributeTable(tableBody, currentExplanation.attributeValues, {
        includeNames: true,
    });

    if (showPredictionPanel) {
        showPrediction(tableBody, currentExplanation.prediction.value);
    }
}

function showPrediction(tableBody, prediction, options = {}) {
    const { includeLabel = true } = options;
    const firstRow = tableBody.querySelector(".attribute-row");
    if (!firstRow) {
        return;
    }

    const existingPredictionRow = tableBody.querySelector(".prediction-row");
    if (existingPredictionRow) {
        existingPredictionRow.remove();
    }

    const predictionRow = document.createElement("tr");
    predictionRow.className = "prediction-row";

    const predictionValueCell = document.createElement("td");
    predictionValueCell.className = "prediction-row-value";
    predictionValueCell.colSpan = includeLabel
        ? firstRow.children.length - 1
        : firstRow.children.length;

    const predictionValue = document.createElement("span");
    predictionValue.classList.add("prediction-result");
    predictionValue.classList.add(
        prediction === 1 ? "prediction-result-positive" : "prediction-result-negative"
    );
    predictionValue.innerText =
        currentExplanation.predictionLabels[prediction] ?? String(prediction);

    predictionValueCell.appendChild(predictionValue);
    if (includeLabel) {
        const predictionLabelCell = document.createElement("td");
        predictionLabelCell.className = "attribute prediction-row-label";
        predictionLabelCell.innerText = "AI prediction";
        predictionLabelCell.title = "The AI's prediction for this instance";
        predictionRow.appendChild(predictionLabelCell);
    }
    predictionRow.appendChild(predictionValueCell);

    tableBody.appendChild(predictionRow);
}

function clearCounterfactualTable() {
    const table = noneExplanationTbody.closest("table");
    const wrapper = table?.parentElement;
    const existingCounterfactualTable = wrapper?.querySelector("#counterfactual-table");
    if (existingCounterfactualTable) {
        existingCounterfactualTable.remove();
    }
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

    const existingDirectionRow = tableBody.querySelector(".influence-direction-row");
    if (existingDirectionRow) {
        existingDirectionRow.remove();
    }

    const existingHeader = headerRow.querySelector(".attribution-header");
    if (!existingHeader) {
        const attributionHeader = document.createElement("th");
        attributionHeader.className = "tooltip attribution-header";
        attributionHeader.colSpan = 2;
        attributionHeader.title = "Influence of each attribute towards the prediction";
        attributionHeader.textContent = "Influence";
        headerRow.appendChild(attributionHeader);
    }

    const firstRow = tableBody.querySelector(".attribute-row");
    if (!firstRow) {
        return;
    }

    const chartPanelCell = document.createElement("td");
    chartPanelCell.id = "feature-attribution-chart";
    chartPanelCell.colSpan = 2;
    chartPanelCell.rowSpan = currentExplanation.attributeNames.length;

    const chartWrapper = document.createElement("div");
    chartWrapper.id = "feature-attribution-div";

    const canvas = document.createElement("canvas");
    canvas.id = "feature-attribution-canvas";
    canvas.width = 120;
    canvas.height = 120;

    chartWrapper.appendChild(canvas);
    chartPanelCell.appendChild(chartWrapper);
    firstRow.appendChild(chartPanelCell);

    const colors = attribution.values.map((value) =>
        value >= 0 ? "rgba(60, 136, 232)" : "rgba(234, 51, 53)"
    );

    const renderAttributionChart = () => {
        const chartHeight = Math.max(Math.round(chartPanelCell.clientHeight), 1);
        canvas.height = chartHeight;
        canvas.style.height = `${chartHeight}px`;

        if (attributionChart) {
            attributionChart.destroy();
            attributionChart = null;
        }

        attributionChart = new Chart(canvas.getContext("2d"), {
            type: "bar",
            data: {
                labels: currentExplanation.attributeNames,
                datasets: [
                    {
                        label: "Feature Attribution",
                        data: attribution.values.map((value) =>
                            clamp((value / currentExplanation.attributionMax) * 100, -100, 100)
                        ),
                        backgroundColor: colors,
                        barThickness: 21,
                    },
                ],
            },
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
    };

    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(renderAttributionChart);
    } else {
        renderAttributionChart();
    }

    const directionRow = document.createElement("tr");
    directionRow.className = "influence-direction-row";

    const columnsBeforeChart = Math.max(firstRow.children.length - 1, 1);
    const spacerCell = document.createElement("td");
    spacerCell.className = "influence-direction-spacer";
    spacerCell.colSpan = columnsBeforeChart;
    directionRow.appendChild(spacerCell);

    const directionCell = document.createElement("td");
    directionCell.className = "influence-direction-cell";
    directionCell.colSpan = 2;

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
}

function shortenClassLabel(label) {
    const labelText = String(label ?? "");
    return labelText
        .replace(" Credit Risk", "")
        .replace("No Diabetes", "No Diabetes")
        .replace("Diabetes", "Diabetes");
}

function showCounterfactualExample(tableBody) {
    const counterfactual = currentExplanation.counterfactual;
    const originalTable = tableBody.closest("table");
    const wrapper = originalTable?.parentElement;
    if (!originalTable || !wrapper || !counterfactual) {
        return;
    }

    const counterfactualTable = document.createElement("table");
    counterfactualTable.id = "counterfactual-table";
    counterfactualTable.className = "counterfactual-table";

    const counterfactualHead = document.createElement("thead");
    const headerRow = document.createElement("tr");

    const valueHeader = document.createElement("th");
    valueHeader.textContent = "Value";

    const controlHeader = document.createElement("th");
    controlHeader.textContent = "";

    const diffHeader = document.createElement("th");
    diffHeader.className = "tooltip diff-column-header";
    diffHeader.title = "Difference between the original instance and the counterfactual";
    diffHeader.textContent = "Diff";

    headerRow.appendChild(valueHeader);
    headerRow.appendChild(controlHeader);
    headerRow.appendChild(diffHeader);
    counterfactualHead.appendChild(headerRow);

    const counterfactualBody = document.createElement("tbody");
    counterfactualBody.id = "counterfactual-tbody";

    counterfactualTable.appendChild(counterfactualHead);
    counterfactualTable.appendChild(counterfactualBody);
    setCaseLabel(counterfactualTable, "Comparable", {
        startColumn: 0,
        columnSpan: 3,
    });
    wrapper.appendChild(counterfactualTable);

    populateAttributeTable(counterfactualBody, counterfactual.feature_values, {
        includeNames: false,
        originalValues: currentExplanation.attributeValues,
    });
    if (showPredictionPanel) {
        showPrediction(counterfactualBody, counterfactual.prediction.value, {
            includeLabel: false,
        });
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

    return {
        attributeNames: payload.feature_names ?? [],
        attributeTypes: payload.feature_types ?? [],
        attributeValues: payload.feature_values ?? [],
        attributeRanges: payload.feature_ranges ?? [],
        attribution,
        attributionMax: Math.max(payload.attribution?.max_abs_value ?? 0, 1e-9),
        explanationFeatureCount: payload.explanation_feature_count ?? explanationFeatureCount,
        prediction: payload.prediction ?? { value: 0 },
        predictionLabels: payload.prediction_labels ?? [],
        counterfactual: payload.counterfactual ?? null,
    };
}

function renderExplanation() {
    clearCounterfactualTable();
    showAttributeValues(noneExplanationTbody);

    if (explanationType === "attribution") {
        showAttributionChart(noneExplanationTbody);
    }

    if (explanationType === "counterfactual") {
        setCaseLabel(noneExplanationTbody.closest("table"), "Subject", {
            startColumn: 1,
            columnSpan: 2,
        });
        showCounterfactualExample(noneExplanationTbody);
    }
}

async function loadExplanation() {
    renderStatusRow("Loading explanation data...");

    try {
        const endpoint = buildApiUrl("explanations");
        endpoint.searchParams.set("dataset", datasetName);
        endpoint.searchParams.set("model", modelName);
        endpoint.searchParams.set("xaiMethod", xaiMethod);
        endpoint.searchParams.set("instanceId", String(instanceId));
        endpoint.searchParams.set("xaiType", explanationType);
        endpoint.searchParams.set("k", String(explanationFeatureCount));

        console.log("[iframe] explanation request:", endpoint.toString());
        const response = await fetch(endpoint);
        console.info("[iframe] explanation response:", {
            status: response.status,
            ok: response.ok,
            contentType: response.headers.get("content-type"),
            body: await response.clone().text(),
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error ?? `Request failed with ${response.status}`);
        }

        currentExplanation = normalizeExplanationPayload(payload);
        renderExplanation();
    } catch (error) {
        console.error("Failed to load explanation:", error);
        renderStatusRow(String(error.message ?? error), true);
    }
}

document.addEventListener("DOMContentLoaded", function () {
    loadExplanation();
});
