from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

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


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(REPO_ROOT), static_url_path="")
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
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        LOGGER.info(
            "Completed request: method=%s path=%s status=%s",
            request.method,
            request.path,
            response.status_code,
        )
        return response

    @app.get("/api/health")
    def healthcheck():
        LOGGER.info("Healthcheck requested")
        return jsonify({"status": "ok"})

    @app.get("/api/metadata")
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

    @app.get("/api/explanations")
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

    @app.errorhandler(Exception)
    def handle_error(error):
        status_code = 500
        if isinstance(error, ValueError):
            status_code = 400
        elif isinstance(error, IndexError):
            status_code = 404

        LOGGER.exception("Request failed with status %s: %s", status_code, error)
        return jsonify({"error": str(error)}), status_code

    @app.get("/")
    def serve_index():
        LOGGER.info("Serving index.html")
        # return jsonify({"message": "Nothing to see here."})
        return send_from_directory(REPO_ROOT, "index.html")

    @app.get("/<path:asset_path>")
    def serve_asset(asset_path: str):
        LOGGER.info("Serving asset: %s", asset_path)
        return send_from_directory(REPO_ROOT, asset_path)

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
    LOGGER.info("Open this URL in your browser: http://127.0.0.1:5000/")
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
    LOGGER.info("Binding Flask server to http://127.0.0.1:5000 with debug=False")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
