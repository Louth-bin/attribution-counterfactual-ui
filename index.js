const urlParams = new URLSearchParams(window.location.search);
const fallbackApiBaseUrl = "http://127.0.0.1:5000";
const apiBaseUrl = resolveApiBaseUrl(urlParams.get("apiBaseUrl"));
const DATASET_DEFAULT_FEATURE_COUNT = {
    diabetes: 2,
    ceramic: 2,
    safelimit: 2,
};

let iframeUrl = "";

console.info("[index] apiBaseUrl:", apiBaseUrl);

function buildApiUrl(path) {
    const baseUrl = apiBaseUrl.endsWith("/") ? apiBaseUrl : `${apiBaseUrl}/`;
    return new URL(path, baseUrl);
}

function appendApiPath(baseUrl) {
    const url = new URL(baseUrl, window.location.href);
    const isLocal = isLocalHost(url.hostname);
    const pathParts = url.pathname.split("/").filter(Boolean);

    if (isLocal) {
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

function isLocalHost(hostname) {
    return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

function resolveApiBaseUrl(configuredApiBaseUrl) {
    if (configuredApiBaseUrl) {
        return appendApiPath(configuredApiBaseUrl);
    }

    if (window.location.origin && window.location.origin !== "null") {
        return appendApiPath(window.location.origin);
    }

    return appendApiPath(fallbackApiBaseUrl);
}

function buildIframeUrl() {
    const dataset = document.querySelector("#input_appId").value;
    const instanceId = document.querySelector("#input_instanceId").value;
    const aiModel = document.querySelector("#input_AIModel").value;
    const expAlgorithm = document.querySelector("#input_exp_algorithm").value;
    const xaiType = document.querySelector("#input_xaiType").value;
    const explanationView = document.querySelector("#input_explanationView").value;
    const explanationFeatureCount = document.querySelector("#input_explanationFeatureCount").value;
    const counterfactualSimulation = document.querySelector("#input_counterfactualSimulation").checked
        ? 1
        : 0;
    const showPrediction = document.querySelector("#input_showPrediction").checked ? 1 : 0;

    const iframeQuery = new URLSearchParams({
        appId: dataset,
        AIModel: aiModel,
        xaiType,
        explanationView,
        expAlgorithm,
        instanceId,
        k: explanationFeatureCount,
        showPrediction: String(showPrediction),
        counterfactualSimulation: String(counterfactualSimulation),
        apiBaseUrl,
    });

    return `iframe.html?${iframeQuery.toString()}`;
}

function updateIframeUrl() {
    const iframe = document.querySelector("#interactiveIframe");
    iframeUrl = buildIframeUrl();
    iframe.src = iframeUrl;
    iframe.height = 900;
}

function clampInstanceId() {
    const input = document.querySelector("#input_instanceId");
    const minValue = Number(input.min || 0);
    const maxValue = Number(input.max || 0);
    let instanceId = Number(input.value);

    if (!Number.isFinite(instanceId)) {
        instanceId = minValue;
    }

    instanceId = Math.max(instanceId, minValue);
    instanceId = Math.min(instanceId, maxValue);
    input.value = instanceId;
    return instanceId;
}

async function refreshDatasetMetadata() {
    const dataset = document.querySelector("#input_appId").value;
    const instanceInput = document.querySelector("#input_instanceId");
    const explanationFeatureCountInput = document.querySelector("#input_explanationFeatureCount");
    const defaultFeatureCount = DATASET_DEFAULT_FEATURE_COUNT[dataset];

    if (defaultFeatureCount) {
        explanationFeatureCountInput.value = String(defaultFeatureCount);
    }

    try {
        const metadataUrl = buildApiUrl("metadata");
        metadataUrl.searchParams.set("dataset", dataset);

        console.info("[index] metadata request:", metadataUrl.toString());
        const response = await fetch(metadataUrl);
        console.info("[index] metadata response:", {
            status: response.status,
            ok: response.ok,
            contentType: response.headers.get("content-type"),
            body: await response.clone().text(),
        });
        if (!response.ok) {
            throw new Error(`Metadata request failed with ${response.status}`);
        }

        const payload = await response.json();
        const maxInstanceId = Math.max((payload.available_instance_count ?? 1) - 1, 0);
        instanceInput.max = String(maxInstanceId);
        clampInstanceId();
    } catch (error) {
        console.error("Failed to refresh dataset metadata:", error);
        instanceInput.max = "10000";
    }
}

async function syncControlsAndIframe() {
    await refreshDatasetMetadata();
    updateIframeUrl();
}

function updateInstanceId() {
    clampInstanceId();
    updateIframeUrl();
}

function updateExplanationFeatureCount() {
    const input = document.querySelector("#input_explanationFeatureCount");
    const minValue = Number(input.min || 1);
    const maxValue = Number(input.max || 20);
    let value = Number(input.value);

    if (!Number.isFinite(value)) {
        value = minValue;
    }

    input.value = Math.min(Math.max(Math.round(value), minValue), maxValue);
    updateIframeUrl();
}

function updateInstanceIdTyping(event) {
    if (
        event.key.length === 1 &&
        !/[0-9]/.test(event.key) &&
        !["Backspace", "Delete"].includes(event.key)
    ) {
        event.preventDefault();
    }
}

document.querySelector("#copyButton").onclick = function () {
    const absoluteIframeUrl = new URL(iframeUrl, window.location.href).href;
    navigator.clipboard
        .writeText(absoluteIframeUrl)
        .then(() => {
            const button = document.querySelector("#copyButton");
            const originalText = button.innerHTML;
            button.innerHTML = "Copied!";
            setTimeout(() => {
                button.innerHTML = originalText;
            }, 2000);
        })
        .catch((error) => {
            console.error("Failed to copy:", error);
        });
};

window.addEventListener("message", function (event) {
    console.log(event.data);
});

document.querySelector("#input_appId").onchange = syncControlsAndIframe;
document.querySelector("#input_AIModel").onchange = updateIframeUrl;
document.querySelector("#input_xaiType").onchange = updateIframeUrl;
document.querySelector("#input_explanationView").onchange = updateIframeUrl;
document.querySelector("#input_exp_algorithm").onchange = updateIframeUrl;
document.querySelector("#input_instanceId").onchange = updateInstanceId;
document.querySelector("#input_instanceId").onkeypress = updateInstanceIdTyping;
document.querySelector("#input_explanationFeatureCount").onchange = updateExplanationFeatureCount;
document.querySelector("#input_explanationFeatureCount").onkeypress = updateInstanceIdTyping;
document.querySelector("#input_showPrediction").onchange = updateIframeUrl;
document.querySelector("#input_counterfactualSimulation").onchange = updateIframeUrl;

syncControlsAndIframe();
