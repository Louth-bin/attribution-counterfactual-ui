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
            "pregnancies": "Pregnancies",
            "glucose": "Glucose",
            "blood_pressure": "Blood Pressure",
            "skin_thickness": "Skin Thickness",
            "insulin": "Insulin",
            "bmi": "BMI",
            "diabetes_pedigree_function": "Diabetes Pedigree Function",
            "age": "Age",
        },
        friendly_category_names={},
        category_orders={},
        excluded_feature_names=(
            # "insulin",
        ),
        controllable_feature_names=(
            "glucose",
            "blood_pressure",
            "insulin",
            "bmi",
        ),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            top_n=6,
        ),
    ),
    "german_credit": DatasetRuntimeConfig(
        label="German Credit",
        friendly_feature_names={
            "checking_status": "Checking Account Status",
            "duration": "Loan Duration",
            "credit_history": "Credit History",
            "purpose": "Loan Purpose",
            "credit_amount": "Credit Amount",
            "savings_status": "Savings Status",
            "employment": "Employment Length",
            "installment_commitment": "Installment Rate",
            "personal_status": "Personal Status",
            "other_parties": "Other Parties",
            "residence_since": "Residence Since",
            "property_magnitude": "Property",
            "age": "Age",
            "other_payment_plans": "Other Payment Plans",
            "housing": "Housing",
            "existing_credits": "Existing Credits",
            "job": "Job",
            "num_dependents": "Dependents",
            "own_telephone": "Own Telephone",
            "foreign_worker": "Foreign Worker",
        },
        friendly_category_names={
            "checking_status": {
                "<0": "Negative",
                "0<=X<200": "Low",
                ">=200": "High",
                "no checking": "None",
            },
            "credit_history": {
                "critical/other existing credit": "Critical",
                "existing paid": "Paid",
                "delayed previously": "Delayed",
                "no credits/all paid": "No Credits",
                "all paid": "All Paid",
            },
            "savings_status": {
                "<100": "Low",
                "100<=X<500": "Medium",
                "500<=X<1000": "High",
                ">=1000": "Very High",
                "no known savings": "None",
            },
            "employment": {
                "unemployed": "Unemployed",
                "<1": "<1 Year",
                "1<=X<4": "1-3 Years",
                "4<=X<7": "4-6 Years",
                ">=7": "7+ Years",
            },
            "personal_status": {
                "female div/dep/mar": "Female",
                "male div/sep": "Male Div/Sep",
                "male mar/wid": "Male Mar/Wid",
                "male single": "Male Single",
            },
            "other_parties": {
                "none": "None",
                "co applicant": "Co-applicant",
                "guarantor": "Guarantor",
            },
            "property_magnitude": {
                "real estate": "Real Estate",
                "life insurance": "Life Insurance",
                "car": "Car",
                "no known property": "None",
            },
            "other_payment_plans": {
                "none": "None",
                "bank": "Bank",
                "stores": "Store",
            },
            "housing": {
                "own": "Own",
                "rent": "Rent",
                "for free": "Free",
            },
            "job": {
                "unemp/unskilled non res": "Unemployed",
                "unskilled resident": "Unskilled",
                "skilled": "Skilled",
                "high qualif/self emp/mgmt": "Highly Skilled",
            },
            "own_telephone": {
                "none": "No",
                "yes": "Yes",
            },
            "foreign_worker": {
                "no": "No",
                "yes": "Yes",
            },
        },
        category_orders={
            "checking_status": ("no checking", "<0", "0<=X<200", ">=200"),
            "credit_history": (
                "no credits/all paid",
                "all paid",
                "existing paid",
                "delayed previously",
                "critical/other existing credit",
            ),
            "savings_status": (
                "no known savings",
                "<100",
                "100<=X<500",
                "500<=X<1000",
                ">=1000",
            ),
            "employment": ("unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"),
            "other_parties": ("none", "co applicant", "guarantor"),
            "property_magnitude": (
                "no known property",
                "car",
                "life insurance",
                "real estate",
            ),
            "other_payment_plans": ("none", "stores", "bank"),
            "housing": ("for free", "rent", "own"),
            "job": (
                "unemp/unskilled non res",
                "unskilled resident",
                "skilled",
                "high qualif/self emp/mgmt",
            ),
            "own_telephone": ("none", "yes"),
            "foreign_worker": ("no", "yes"),
        },
        excluded_feature_names=(
            "purpose",
        ),
        controllable_feature_names=(
            "checking_status",
            "duration",
            "credit_history",
            "credit_amount",
            "savings_status",
            "employment",
            "installment_commitment",
            "other_parties",
            "residence_since",
            "property_magnitude",
            "other_payment_plans",
            "housing",
            "existing_credits",
            "job",
            "num_dependents",
            "own_telephone",
        ),
        feature_selection=FeatureSelectionConfig(
            enabled=True,
            top_n=6,
        ),
    ),
}


def get_dataset_runtime_config(dataset_name: str) -> DatasetRuntimeConfig:
    return DATASET_RUNTIME_CONFIGS.get(dataset_name.lower(), DatasetRuntimeConfig())
