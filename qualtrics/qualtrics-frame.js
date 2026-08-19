/*
 * Shared JavaScript for the training question and both testing questions.
 *
 * Required question HTML:
 *   Training: <div id="cf-study-root" data-phase="training">
 *   Testing:  <div id="cf-study-root" data-phase="test" data-test-label="0|1">
 *
 * Loop & Merge field 1 is 0-9 for training and each testing block.
 */
Qualtrics.SurveyEngine.addOnload(function () {
    "use strict";

    var question = this;
    var root = document.getElementById("cf-study-root");
    if (!root) return;

    var CASE_IDS = {
        housing: {
            training: [14693, 13350, 2968, 15034, 4666, 8187, 6557, 2898, 2864, 12633],
            test: {
                0: [3174, 2275, 675, 102, 577, 2255, 2987, 614, 1760, 692],
                1: [2447, 1751, 553, 172, 2728, 2462, 1155, 456, 2767, 1657]
            }
        },
        safelimit: {
            training: [610, 1025, 1230, 960, 239, 124, 143, 131, 213, 322],
            test: {
                0: [87, 203, 106, 143, 300, 211, 262, 69, 251, 177],
                1: [98, 261, 4, 209, 245, 281, 148, 265, 25, 117]
            }
        },
        diabetes: {
            training: [36, 190, 225, 115, 1, 2, 118, 7, 74, 60],
            test: {
                0: [32, 30, 14, 53, 35, 42, 26, 7, 56, 33],
                1: [59, 0, 12, 20, 15, 52, 58, 11, 31, 51]
            }
        }
    };
    var LABELS = {
        housing: ["Cheap", "Expensive"],
        safelimit: ["Above Limit", "Below Limit"],
        diabetes: ["Diabetes", "No Diabetes"]
    };

    function getEmbeddedData(name) {
        return String(Qualtrics.SurveyEngine.getEmbeddedData(name) || "").trim();
    }

    function saveRecord(field, record) {
        var records;
        try {
            records = JSON.parse(getEmbeddedData(field) || "[]");
            if (!Array.isArray(records)) records = [];
        } catch (_error) {
            records = [];
        }

        var existingIndex = records.findIndex(function (item) {
            return Number(item.instanceId) === Number(record.instanceId);
        });
        if (existingIndex >= 0) records[existingIndex] = record;
        else records.push(record);

        Qualtrics.SurveyEngine.setEmbeddedData(field, JSON.stringify(records));
    }

    function setNextEnabled(enabled) {
        if (enabled) question.enableNextButton();
        else question.disableNextButton();
    }

    var domain = getEmbeddedData("appId").toLowerCase();
    if (!CASE_IDS[domain]) domain = "housing";
    Array.prototype.forEach.call(root.querySelectorAll("[data-domain]"), function (item) {
        item.hidden = item.getAttribute("data-domain") !== domain;
    });

    var phase = root.getAttribute("data-phase") === "test" ? "test" : "training";
    var testLabel = Number(root.getAttribute("data-test-label"));
    var loopIndex = parseInt("${lm://Field/1}", 10);
    if (!Number.isFinite(loopIndex)) loopIndex = 0;
    var presentationPosition = parseInt("${lm://CurrentLoopNumber}", 10);
    if (!Number.isFinite(presentationPosition)) presentationPosition = loopIndex + 1;

    var caseList = phase === "training"
        ? CASE_IDS[domain].training
        : CASE_IDS[domain].test[testLabel];
    var instanceId = caseList && caseList[loopIndex];

    var iframe = document.getElementById("cf-case-frame");
    var title = document.getElementById("cf-case-title");
    var status = document.getElementById("cf-status");
    var answerPanel = document.getElementById("cf-training-answer");
    var explanation = getEmbeddedData("xaiType") || "attribution";
    var startedAt = Date.now();
    var prediction = null;
    var selectedPrediction = null;
    var latestChanges = [];
    var latestRawValues = null;
    var latestFeedback = null;

    if (instanceId === undefined) {
        status.textContent = "The Loop & Merge rows are not configured correctly.";
        return;
    }

    function makeIframeUrl(xaiType, showPrediction, simulation) {
        var base = getEmbeddedData("ui_base_url") ||
            "https://louth-bin.github.io/attribution-counterfactual-ui/iframe.html";
        var parameters = new URLSearchParams({
            appId: domain,
            xaiType: xaiType,
            split: phase === "training" ? "train" : "test",
            instanceId: String(instanceId),
            showPrediction: showPrediction ? "1" : "0",
            counterfactualSimulation: simulation ? "1" : "0"
        });
        return base + (base.indexOf("?") >= 0 ? "&" : "?") + parameters.toString();
    }

    function trainingRecord() {
        return {
            domain: domain,
            caseNumber: presentationPosition,
            casePoolPosition: loopIndex + 1,
            instanceId: instanceId,
            explanation: explanation,
            selectedPrediction: selectedPrediction,
            correctPrediction: prediction ? Number(prediction.value) : null,
            correct: prediction && selectedPrediction !== null
                ? Number(prediction.value) === selectedPrediction
                : null,
            responseMs: Date.now() - startedAt
        };
    }

    function testingRecord() {
        return {
            domain: domain,
            direction: LABELS[domain][testLabel] + " to " + LABELS[domain][1 - testLabel],
            caseNumberWithinDirection: presentationPosition,
            casePoolPositionWithinDirection: loopIndex + 1,
            instanceId: instanceId,
            explanation: explanation,
            originalPrediction: prediction,
            changes: latestChanges,
            changedRawFeatureValues: latestRawValues,
            feedback: latestFeedback,
            responseMs: Date.now() - startedAt
        };
    }

    function showTrainingChoices() {
        if (!answerPanel || !prediction || selectedPrediction !== null) return;
        answerPanel.innerHTML = "";
        LABELS[domain].forEach(function (label, value) {
            var button = document.createElement("button");
            button.type = "button";
            button.className = "cf-answer-button";
            button.textContent = label;
            button.addEventListener("click", function () {
                selectedPrediction = value;
                var correct = Number(prediction.value) === value;
                status.textContent = correct
                    ? "Correct. Review the AI answer and explanation, then continue."
                    : "Not quite. Review the AI answer and explanation, then continue.";
                status.className = correct ? "cf-status cf-correct" : "cf-status cf-incorrect";
                Array.prototype.forEach.call(answerPanel.querySelectorAll("button"), function (item) {
                    item.disabled = true;
                });
                saveRecord("training_log_json", trainingRecord());
                iframe.src = makeIframeUrl(explanation, true, false);
                setNextEnabled(true);
            });
            answerPanel.appendChild(button);
        });
        answerPanel.hidden = false;
        status.textContent = "Select your answer below the profile.";
    }

    window.addEventListener("message", function (event) {
        if (event.source !== iframe.contentWindow || !event.data) return;
        var data = event.data;

        if (data.type === "counterfactual-ui:iframe-height") {
            iframe.style.height = Math.max(320, Math.min(900, Number(data.height) || 0)) + "px";
        } else if (data.type === "counterfactual-ui:screen-state") {
            if (data.screenState && data.screenState.prediction) {
                prediction = data.screenState.prediction;
            }
            if (phase === "training") showTrainingChoices();
        } else if (phase === "test" && data.type === "counterfactual-ui:simulation-change") {
            latestChanges = Array.isArray(data.changes) ? data.changes : [];
            latestRawValues = data.changedRawFeatureValues || null;
            if (latestChanges.length > 0) {
                status.textContent = "Your changes are saved. You may continue or revise them.";
                saveRecord("testing_log_json", testingRecord());
                setNextEnabled(true);
            } else {
                status.textContent = "Make at least one change before continuing.";
                setNextEnabled(false);
            }
        } else if (phase === "test" && data.type === "counterfactual-ui:simulation-feedback") {
            latestFeedback = {
                feedback: data.feedback || null,
                prediction: data.prediction || null
            };
            saveRecord("testing_log_json", testingRecord());
        }
    });

    if (phase === "training") {
        title.textContent = "Training case " + presentationPosition + " of 10";
        status.textContent = "Loading the profile...";
        iframe.src = makeIframeUrl("none", false, false);
    } else {
        title.textContent = LABELS[domain][testLabel] + " to " +
            LABELS[domain][1 - testLabel] + ": case " + presentationPosition + " of 10";
        status.textContent = "Make at least one change before continuing.";
        iframe.src = makeIframeUrl("none", true, true);
    }
    setNextEnabled(false);

    var nextButton = document.getElementById("NextButton");
    if (nextButton) {
        nextButton.addEventListener("click", function () {
            saveRecord(
                phase === "training" ? "training_log_json" : "testing_log_json",
                phase === "training" ? trainingRecord() : testingRecord()
            );
        }, true);
    }
});
