"use strict";

const fs = require("fs");
const vm = require("vm");

global.window = global;
vm.runInThisContext(fs.readFileSync("static/experiment-data.js", "utf8"));
vm.runInThisContext(fs.readFileSync("static-model.js", "utf8"));

for (const [datasetName, bundle] of Object.entries(EXPERIMENT_DATA.datasets)) {
    const cases = [...bundle.training_pool, ...bundle.test_pool];
    for (const payload of cases) {
        const prediction = StaticModel.predictDataset(
            EXPERIMENT_DATA,
            datasetName,
            payload.raw_feature_values
        );
        if (prediction.value !== payload.prediction.value) {
            throw new Error(
                `${datasetName}:${payload.instance_id} expected ` +
                `${payload.prediction.value}, got ${prediction.value}`
            );
        }
    }
    console.log(`${datasetName}: ${cases.length}/${cases.length} browser predictions match`);
}

const qsf = JSON.parse(
    fs.readFileSync("qualtrics/UPLOAD_THIS_Qualtrics_Starter.qsf", "utf8")
);
let scriptCount = 0;
for (const element of qsf.SurveyElements) {
    if (element.Element !== "SQ" || !element.Payload?.QuestionJS) continue;
    new Function(element.Payload.QuestionJS);
    scriptCount += 1;
}
console.log(`qualtrics: ${scriptCount} question scripts parse successfully`);
