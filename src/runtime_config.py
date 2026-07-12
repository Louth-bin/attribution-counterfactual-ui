from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSelectionConfig:
    enabled: bool = False
    method: str = "permutation"
    top_n: int | None = None
    n_repeats: int = 8
    scoring: str = "accuracy"
    random_state: int = 42
    force_recompute: bool = False


@dataclass(frozen=True)
class DatasetRuntimeConfig:
    label: str | None = None
    friendly_feature_names: dict[str, str] = field(default_factory=dict)
    friendly_category_names: dict[str, dict[str, str]] = field(default_factory=dict)
    category_orders: dict[str, tuple[str, ...]] = field(default_factory=dict)
    excluded_feature_names: tuple[str, ...] = ()
    controllable_feature_names: tuple[str, ...] = ()
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)


# Edit this file before starting the server if you want to:
# - rename attributes shown in the UI
# - rename categorical values shown in the UI
# - set intuitive categorical value ordering
# - hard-exclude attributes so they are never used downstream
# - mark which attributes are allowed to change when "controllable only" is enabled
# - keep only the top-n features based on permutation importance
DATASET_RUNTIME_CONFIGS: dict[str, DatasetRuntimeConfig] = {
    "diabetes": DatasetRuntimeConfig(
        label="Diabetes",
        friendly_feature_names={
            "glucose": "Glucose",
            "blood_pressure": "Blood Pressure",
            "skin_thickness": "Skin Thickness",
            "insulin": "Insulin",
            "bmi": "BMI",
            "age": "Age",
        },
        friendly_category_names={},
        category_orders={},
        excluded_feature_names=(
            "skin_thickness",
        ),
        controllable_feature_names=(
            "glucose",
            "blood_pressure",
            "insulin",
            "bmi",
        ),
        feature_selection=FeatureSelectionConfig(
            enabled=False,
            top_n=None,
        ),
    ),
    "ceramic": DatasetRuntimeConfig(
        label="Technical Ceramic Firing",
        friendly_feature_names={
            "peak_temperature_c": "Peak Firing Temperature (C)",
            "heating_rate_c_min": "Heating Rate (C/min)",
            "soak_time_min": "Soak Time (min)",
            "cooling_rate_c_min": "Cooling Rate (C/min)",
            "body_moisture_pct": "Green Body Moisture (%)",
            "pressing_pressure_mpa": "Pressing Pressure (MPa)",
            "green_density_g_cm3": "Green Density (g/cm3)",
            "particle_d50_um": "Particle D50 (um)",
            "feldspar_pct": "Feldspar Content (%)",
        },
        controllable_feature_names=(
            "peak_temperature_c",
            "heating_rate_c_min",
            "soak_time_min",
            "cooling_rate_c_min",
            "body_moisture_pct",
            "pressing_pressure_mpa",
            "green_density_g_cm3",
            "particle_d50_um",
            "feldspar_pct",
        ),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            top_n=8,
        ),
    ),
    "safelimit": DatasetRuntimeConfig(
        label="SafeLimit",
        friendly_feature_names={
            "units": "Alcohol Units",
            "weight": "Weight",
            "duration": "Drinking Duration",
            "gender": "Gender",
            "stomach_fullness": "Stomach Fullness",
        },
        friendly_category_names={
            "gender": {
                "female": "Female",
                "male": "Male",
            },
            "stomach_fullness": {
                "empty": "Empty",
                "full": "Full",
            },
        },
        category_orders={
            "gender": ("female", "male"),
            "stomach_fullness": ("empty", "full"),
        },
        controllable_feature_names=(
            "units",
            "duration",
            "stomach_fullness",
        ),
        feature_selection=FeatureSelectionConfig(
            enabled=False,
            top_n=None,
        ),
    ),
}


def get_dataset_runtime_config(dataset_name: str) -> DatasetRuntimeConfig:
    return DATASET_RUNTIME_CONFIGS.get(dataset_name.lower(), DatasetRuntimeConfig())
