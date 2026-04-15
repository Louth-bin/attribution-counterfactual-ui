from __future__ import annotations

from .shap_method import ShapXAI


XAI_METHOD_REGISTRY = {
    "shap": ShapXAI(),
}


def get_xai_method(method_name: str) -> ShapXAI:
    normalized_name = method_name.lower()
    if normalized_name not in XAI_METHOD_REGISTRY:
        available_methods = ", ".join(sorted(XAI_METHOD_REGISTRY))
        raise ValueError(
            f"Unsupported XAI method '{method_name}'. Available methods: {available_methods}."
        )

    return XAI_METHOD_REGISTRY[normalized_name]


def list_xai_methods() -> list[str]:
    return sorted(XAI_METHOD_REGISTRY)
