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
            training: [130100, 130101, 130102, 130103, 130104, 130105, 130106, 130107, 130108, 130109, 130110, 130111],
            test: {
                0: [130200, 130201, 130202, 130203, 130204, 130205, 130206, 130207, 130208, 130209],
                1: [130300, 130301, 130302, 130303, 130304, 130305, 130306, 130307, 130308, 130309]
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
    if (explanation === "counterfactuals") explanation = "counterfactual";
    var TRAINING_EXPLANATION_DELAY_MS = 8000;
    var startedAt = Date.now();
    var prediction = null;
    var selectedPrediction = null;
    var latestChanges = [];
    var latestRawValues = null;
    var latestFeedback = null;
    var latestScreenState = null;
    var testingAttempts = [];
    var currentAttempt = null;
    var attemptSequence = 0;
    var trainingReviewUnlockAt = null;
    var trainingReviewTimer = null;
    var trainingFeedbackText = "";

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
        if (domain === "diabetes" && phase === "test" && simulation) {
            parameters.set("immutableFeatures", "Glucose");
            parameters.set("maxChangedFeatures", "1");
        }
        return base + (base.indexOf("?") >= 0 ? "&" : "?") + parameters.toString();
    }

    function getOrCreateCheckPanel(id) {
        var panel = document.getElementById(id);
        if (panel) return panel;
        panel = document.createElement("div");
        panel.id = id;
        panel.className = "cf-check-panel";
        panel.style.cssText = "margin:16px 0;padding:14px;border:1px solid #d4dae2;border-radius:8px;background:#f8fafc";
        status.parentNode.insertBefore(panel, status);
        return panel;
    }

    function normalizeChangeDirection(change) {
        var before = Number(change.originalValue);
        var after = Number(change.newValue);
        if (Number.isFinite(before) && Number.isFinite(after)) {
            if (after > before) return "increased";
            if (after < before) return "decreased";
        }
        return "changed";
    }

    function describeChanges(changes) {
        if (!Array.isArray(changes) || changes.length === 0) return "No attributes were changed";
        return changes.map(function (change) {
            return change.attributeName + " " + normalizeChangeDirection(change);
        }).join("; ");
    }

    function changedAttributeNames(changes) {
        if (!Array.isArray(changes)) return [];
        return changes.map(function (change) {
            return String(change.attributeName);
        }).sort();
    }

    function makeAttributePairDistractors(correctNames, attributeNames) {
        var correctKey = correctNames.join("|");
        var pairs = [];
        for (var i = 0; i < attributeNames.length; i += 1) {
            for (var j = i + 1; j < attributeNames.length; j += 1) {
                var pair = [attributeNames[i], attributeNames[j]].sort();
                if (pair.join("|") !== correctKey) pairs.push(pair.join(" and "));
            }
        }
        return pairs.slice(0, 3);
    }

    function createMcq(name, prompt, choices, correctValue, onChange) {
        var wrapper = document.createElement("fieldset");
        wrapper.style.cssText = "border:0;margin:0 0 14px;padding:0";
        var legend = document.createElement("legend");
        legend.textContent = prompt;
        legend.style.cssText = "font-weight:700;margin-bottom:8px";
        wrapper.appendChild(legend);
        choices.forEach(function (choice) {
            var label = document.createElement("label");
            label.style.cssText = "display:block;margin:6px 0";
            var input = document.createElement("input");
            input.type = "radio";
            input.name = name;
            input.value = choice.value;
            input.style.marginRight = "8px";
            input.addEventListener("change", function () {
                onChange({
                    value: choice.value,
                    text: choice.text,
                    correctValue: correctValue,
                    correct: choice.value === correctValue
                });
            });
            label.appendChild(input);
            label.appendChild(document.createTextNode(choice.text));
            wrapper.appendChild(label);
        });
        return wrapper;
    }

    function allCheckAnswersPresent(answers) {
        return Boolean(answers && answers.target && answers.change);
    }

    function trainingRecord() {
        return {
            domain: domain,
            caseNumber: loopIndex + 1,
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
            attempts: testingAttempts,
            responseMs: Date.now() - startedAt
        };
    }

    function finishTrainingReview() {
        if (!latestScreenState || selectedPrediction === null) return;
        if (trainingReviewUnlockAt && Date.now() < trainingReviewUnlockAt) {
            var secondsRemaining = Math.max(1, Math.ceil((trainingReviewUnlockAt - Date.now()) / 1000));
            status.textContent = trainingFeedbackText + " Review the explanation carefully (" + secondsRemaining + "s).";
            if (trainingReviewTimer) window.clearTimeout(trainingReviewTimer);
            trainingReviewTimer = window.setTimeout(finishTrainingReview, 250);
            return;
        }
        if (trainingReviewTimer) {
            window.clearTimeout(trainingReviewTimer);
            trainingReviewTimer = null;
        }
        status.textContent = trainingFeedbackText + " You may continue.";
        setNextEnabled(true);
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
                trainingFeedbackText = correct ? "Correct." : "Not quite.";
                status.textContent = trainingFeedbackText + " Loading the AI answer and explanation...";
                status.className = correct ? "cf-status cf-correct" : "cf-status cf-incorrect";
                Array.prototype.forEach.call(answerPanel.querySelectorAll("button"), function (item) {
                    item.disabled = true;
                });
                saveRecord("training_log_json", trainingRecord());
                iframe.src = makeIframeUrl(explanation, true, false);
                trainingReviewUnlockAt = null;
                setNextEnabled(false);
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
            latestScreenState = data.screenState || null;
            if (data.screenState && data.screenState.prediction) {
                prediction = data.screenState.prediction;
            }
            if (phase === "training") showTrainingChoices();
            if (
                phase === "training" &&
                selectedPrediction !== null &&
                data.screenState &&
                data.screenState.explanationType === explanation
            ) {
                if (!trainingReviewUnlockAt) {
                    trainingReviewUnlockAt = Date.now() + TRAINING_EXPLANATION_DELAY_MS;
                }
                finishTrainingReview();
            }
        } else if (phase === "test" && data.type === "counterfactual-ui:simulation-change") {
            latestChanges = Array.isArray(data.changes) ? data.changes : [];
            latestRawValues = data.changedRawFeatureValues || null;
            if (latestChanges.length > 0) {
                if (
                    !currentAttempt ||
                    currentAttempt.feedback
                ) {
                    currentAttempt = {
                        attemptNumber: ++attemptSequence,
                        atMs: Date.now() - startedAt,
                        changes: latestChanges,
                        changedRawFeatureValues: latestRawValues,
                        feedback: null
                    };
                    testingAttempts.push(currentAttempt);
                } else {
                    currentAttempt.atMs = Date.now() - startedAt;
                    currentAttempt.changes = latestChanges;
                    currentAttempt.changedRawFeatureValues = latestRawValues;
                    currentAttempt.feedback = null;
                }
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
            if (currentAttempt) {
                currentAttempt.feedback = latestFeedback;
            }
            saveRecord("testing_log_json", testingRecord());
        }
    });

    if (phase === "training") {
        title.textContent = "Training case";
        status.textContent = "Loading the profile...";
        iframe.src = makeIframeUrl("none", false, false);
    } else {
        title.textContent = LABELS[domain][testLabel] + " to " +
            LABELS[domain][1 - testLabel] + ": case " + presentationPosition + " of " + caseList.length;
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
