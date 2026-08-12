(function initializeStaticModel(global) {
    "use strict";

    function stableSigmoid(value) {
        if (value >= 0) {
            return 1 / (1 + Math.exp(-value));
        }
        const exponential = Math.exp(value);
        return exponential / (1 + exponential);
    }

    function getOrderedFeatureValues(model, rawFeatureValues) {
        const values = Array.isArray(rawFeatureValues)
            ? rawFeatureValues
            : model.feature_names.map((name) => rawFeatureValues?.[name]);
        if (values.length !== model.feature_names.length) {
            throw new Error(
                `Model expected ${model.feature_names.length} features, got ${values.length}.`
            );
        }
        return values;
    }

    function preprocessValues(model, rawValues) {
        const preprocessing = model.preprocessing ?? {};
        if (preprocessing.type === "standard-scaler") {
            return rawValues.map((value, index) => {
                const numericValue = Number(value);
                if (!Number.isFinite(numericValue)) {
                    throw new Error(`Feature '${model.feature_names[index]}' is not numerical.`);
                }
                return (numericValue - Number(preprocessing.mean[index])) /
                    Number(preprocessing.scale[index]);
            });
        }
        if (preprocessing.type !== "column-transformer-v1") {
            throw new Error(`Unsupported preprocessing '${preprocessing.type}'.`);
        }

        const valueByName = Object.fromEntries(
            model.feature_names.map((name, index) => [name, rawValues[index]])
        );
        const numeric = preprocessing.numeric ?? {};
        const transformed = (numeric.feature_names ?? []).map((name, index) => {
            const value = Number(valueByName[name]);
            if (!Number.isFinite(value)) {
                throw new Error(`Feature '${name}' is not numerical.`);
            }
            return (value - Number(numeric.mean[index])) / Number(numeric.scale[index]);
        });

        const categorical = preprocessing.categorical ?? {};
        (categorical.feature_names ?? []).forEach((name, featureIndex) => {
            const value = String(valueByName[name]);
            const categories = categorical.categories?.[featureIndex] ?? [];
            categories.forEach((category) => {
                transformed.push(String(category) === value ? 1 : 0);
            });
        });
        return transformed;
    }

    function applyDenseLayer(inputs, layer, activation) {
        const outputs = layer.biases.map((bias, outputIndex) => {
            return inputs.reduce(
                (sum, input, inputIndex) =>
                    sum + input * Number(layer.weights[inputIndex][outputIndex]),
                Number(bias)
            );
        });
        if (activation === "relu") {
            return outputs.map((value) => Math.max(0, value));
        }
        if (activation === "identity") {
            return outputs;
        }
        throw new Error(`Unsupported hidden activation '${activation}'.`);
    }

    function predict(model, rawFeatureValues) {
        if (!model || model.format !== "sklearn-mlp-binary-v1") {
            throw new Error("This dataset does not include a supported browser model.");
        }

        const rawValues = getOrderedFeatureValues(model, rawFeatureValues);
        let values = preprocessValues(model, rawValues);

        model.layers.forEach((layer, index) => {
            const isOutputLayer = index === model.layers.length - 1;
            values = applyDenseLayer(
                values,
                layer,
                isOutputLayer ? "identity" : model.hidden_activation
            );
        });

        if (values.length !== 1 || model.output_activation !== "logistic") {
            throw new Error("Only binary logistic MLP outputs are supported.");
        }
        const positiveProbability = stableSigmoid(values[0]);
        const predictionValue = positiveProbability > 0.5
            ? Number(model.classes[1])
            : Number(model.classes[0]);
        const labelIndex = model.classes.findIndex(
            (classValue) => Number(classValue) === predictionValue
        );
        return {
            value: predictionValue,
            label: model.class_labels[labelIndex] ?? String(predictionValue),
            probabilities: [1 - positiveProbability, positiveProbability],
        };
    }

    function predictDataset(experimentData, datasetName, rawFeatureValues) {
        const dataset = experimentData?.datasets?.[datasetName];
        if (!dataset) {
            throw new Error(`Dataset '${datasetName}' is not included in the static data.`);
        }
        return predict(dataset.browser_model, rawFeatureValues);
    }

    global.StaticModel = Object.freeze({ predict, predictDataset });
})(window);
