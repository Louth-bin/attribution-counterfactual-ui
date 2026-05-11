from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, jsonify, request

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.ai_models import list_model_names
    from src.pipeline import ExplanationPipeline
    from src.runtime_config import DATASET_RUNTIME_CONFIGS
    from src.xai_methods import list_xai_methods
else:
    from .ai_models import list_model_names
    from .pipeline import ExplanationPipeline
    from .runtime_config import DATASET_RUNTIME_CONFIGS
    from .xai_methods import list_xai_methods


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
pipeline = ExplanationPipeline()
LOGGER = logging.getLogger("counterfactual.server")


def _get_bool_arg(name: str, default: bool = False) -> bool:
    raw_value = request.args.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_arg(name: str, default: int) -> int:
    raw_value = request.args.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"Query parameter '{name}' must be an integer.") from error


def _coerce_feature_value(value: Any, dtype: Any) -> Any:
    if pd.api.types.is_integer_dtype(dtype):
        return int(round(float(value)))
    if pd.api.types.is_float_dtype(dtype):
        return float(value)
    return value


def _prediction_payload(
    prediction_value: int,
    probabilities: list[float] | None,
    class_labels: list[str],
) -> dict[str, Any]:
    return {
        "prediction": {
            "value": prediction_value,
            "label": class_labels[prediction_value]
            if prediction_value < len(class_labels)
            else str(prediction_value),
            "probabilities": [
                {
                    "label": class_labels[index]
                    if index < len(class_labels)
                    else f"Class {index}",
                    "value": float(probability),
                }
                for index, probability in enumerate(probabilities or [])
            ],
        }
    }


def create_app() -> Flask:
    app = Flask(__name__)
    app.logger.handlers.clear()
    app.logger.propagate = True

    @app.before_request
    def log_request():
        LOGGER.info(
            "Incoming request: method=%s path=%s args=%s",
            request.method,
            request.path,
            dict(request.args),
        )

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        LOGGER.info(
            "Completed request: method=%s path=%s status=%s",
            request.method,
            request.path,
            response.status_code,
        )
        return response

    @app.get("/health")
    def healthcheck():
        LOGGER.info("Healthcheck requested")
        return jsonify({"status": "ok"})

    @app.get("/metadata")
    def metadata():
        dataset_name = request.args.get("dataset", "diabetes")
        force_resplit = _get_bool_arg("forceResplit")
        LOGGER.info(
            "Building metadata payload: dataset=%s force_resplit=%s",
            dataset_name,
            force_resplit,
        )
        payload = pipeline.get_metadata(
            dataset_name=dataset_name,
            force_resplit=force_resplit,
        )
        LOGGER.info(
            "Metadata ready: dataset=%s feature_count=%s available_instances=%s",
            dataset_name,
            len(payload.get("feature_names", [])),
            payload.get("available_instance_count"),
        )
        return jsonify(payload)

    @app.get("/explanations")
    def explain_instance():
        dataset_name = request.args.get("dataset", "diabetes")
        model_name = request.args.get("model", "mlp")
        xai_method_name = request.args.get("xaiMethod", "shap")
        xai_type = request.args.get("xaiType", "attribution")
        instance_id = int(request.args.get("instanceId", "0"))
        explanation_feature_count = max(_get_int_arg("k", 3), 0)
        counterfactual_mode = request.args.get("counterfactualMode", "minimal")
        controllable_only = _get_bool_arg("controllableOnly")
        force_resplit = _get_bool_arg("forceResplit")
        force_retrain = _get_bool_arg("forceRetrain")
        LOGGER.info(
            "Building explanation payload: dataset=%s model=%s xai_method=%s xai_type=%s instance_id=%s k=%s counterfactual_mode=%s controllable_only=%s force_resplit=%s force_retrain=%s",
            dataset_name,
            model_name,
            xai_method_name,
            xai_type,
            instance_id,
            explanation_feature_count,
            counterfactual_mode,
            controllable_only,
            force_resplit,
            force_retrain,
        )

        payload = pipeline.get_instance_payload(
            dataset_name=dataset_name,
            model_name=model_name,
            xai_method_name=xai_method_name,
            instance_id=instance_id,
            xai_type=xai_type,
            explanation_feature_count=explanation_feature_count,
            counterfactual_mode=counterfactual_mode,
            controllable_only=controllable_only,
            force_resplit=force_resplit,
            force_retrain=force_retrain,
        )
        LOGGER.info(
            "Explanation ready: dataset=%s model=%s feature_count=%s prediction=%s",
            dataset_name,
            model_name,
            len(payload.get("feature_names", [])),
            payload.get("prediction", {}).get("label"),
        )
        return jsonify(payload)

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict():
        if request.method == "OPTIONS":
            return ("", 204)

        payload = request.get_json(silent=True) or {}
        dataset_name = payload.get("dataset") or payload.get("appId") or "diabetes"
        model_name = payload.get("model") or payload.get("AIModel") or "mlp"
        force_resplit = bool(payload.get("forceResplit", False))
        force_retrain = bool(payload.get("forceRetrain", False))
        prepared_assets = pipeline.prepare_assets(
            dataset_name=dataset_name,
            model_name=model_name,
            force_resplit=force_resplit,
            force_retrain=force_retrain,
        )
        dataset_bundle = prepared_assets.dataset
        estimator = prepared_assets.model_artifact.estimator

        feature_payload = (
            payload.get("raw_feature_values")
            or payload.get("feature_values")
            or payload.get("features")
            or payload.get("instance")
        )
        if feature_payload is None:
            raise ValueError(
                "Prediction request must include raw_feature_values, feature_values, features, or instance."
            )

        if isinstance(feature_payload, dict):
            feature_values = [
                feature_payload.get(feature_name, feature_payload.get(display_name))
                for feature_name, display_name in zip(
                    dataset_bundle.feature_names,
                    dataset_bundle.feature_display_names,
                )
            ]
        else:
            feature_values = list(feature_payload)

        if len(feature_values) != len(dataset_bundle.feature_names):
            raise ValueError(
                "Prediction request feature count does not match the dataset. "
                f"Expected {len(dataset_bundle.feature_names)}, got {len(feature_values)}."
            )

        coerced_feature_values = [
            _coerce_feature_value(
                value,
                dataset_bundle.train_df[feature_name].dtype,
            )
            for feature_name, value in zip(dataset_bundle.feature_names, feature_values)
        ]
        feature_frame = pd.DataFrame(
            [coerced_feature_values],
            columns=dataset_bundle.feature_names,
        )
        prediction_value = int(estimator.predict(feature_frame)[0])
        probabilities = None
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(feature_frame)[0].tolist()

        return jsonify(
            _prediction_payload(
                prediction_value=prediction_value,
                probabilities=probabilities,
                class_labels=dataset_bundle.class_labels,
            )
        )

    @app.errorhandler(Exception)
    def handle_error(error):
        status_code = 500
        if isinstance(error, ValueError):
            status_code = 400
        elif isinstance(error, IndexError):
            status_code = 404

        LOGGER.exception("Request failed with status %s: %s", status_code, error)
        return jsonify({"error": str(error)}), status_code

    return app


app = create_app()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def log_startup_summary() -> None:
    LOGGER.info("Starting Counterfactual UI backend")
    LOGGER.info("Repo root: %s", REPO_ROOT)
    LOGGER.info("Backend API healthcheck: http://%s:%s/health", DEFAULT_HOST, DEFAULT_PORT)
    LOGGER.info("Available models: %s", ", ".join(list_model_names()))
    LOGGER.info("Available XAI methods: %s", ", ".join(list_xai_methods()))
    LOGGER.info("Configured datasets: %s", ", ".join(sorted(DATASET_RUNTIME_CONFIGS)))

    for dataset_name, config in sorted(DATASET_RUNTIME_CONFIGS.items()):
        LOGGER.info(
            "Runtime config for %s: excluded=%s controllable=%s feature_selection_enabled=%s top_n=%s",
            dataset_name,
            list(config.excluded_feature_names),
            list(config.controllable_feature_names),
            config.feature_selection.enabled,
            config.feature_selection.top_n,
        )
        LOGGER.info(
            "Friendly labels for %s: %s",
            dataset_name,
            config.friendly_feature_names,
        )
        try:
            metadata = pipeline.get_metadata(dataset_name)
            LOGGER.info(
                "Dataset preflight ok: dataset=%s feature_count=%s available_instances=%s prediction_labels=%s",
                dataset_name,
                len(metadata.get("feature_names", [])),
                metadata.get("available_instance_count"),
                metadata.get("prediction_labels"),
            )
        except Exception as error:
            LOGGER.exception("Dataset preflight failed for %s: %s", dataset_name, error)


if __name__ == "__main__":
    configure_logging()
    log_startup_summary()
    LOGGER.info(
        "Binding Flask backend API to http://%s:%s with debug=False",
        DEFAULT_HOST,
        DEFAULT_PORT,
    )
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=False, use_reloader=False)
