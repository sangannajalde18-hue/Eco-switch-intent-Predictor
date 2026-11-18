"""
Core prediction logic: DataFrame construction, pipeline inference, and SHAP computation.
"""

import logging
import traceback
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name mapping: Python-friendly alias → exact pipeline column name
# ---------------------------------------------------------------------------
ALIAS_TO_COLUMN: dict[str, str] = {
    "Age": "Age",
    "Income": "Income",
    "No.of products used for a cycle": "No.of products used for a cycle",
    "years_using_product": "years_using_product",
    "Intensity of flow": "Intensity of flow",
    "Single day changing frequency": "Single day changing frequency",
    "Duration of single product per day": "Duration of single product per day",
    "Monthly cost spend": "Monthly cost spend",
    "Manufacturer should produce eco friendly products": "Manufacturer should produce eco friendly products",
    "Occupation": "Occupation",
    "Area": "Area",
    "Menstrual product": "Menstrual product",
    "Disposable/Reusable product": "Disposable/Reusable product",
    "Method of disposing": "Method of disposing",
    "Availability of disposable machines": "Availability of disposable machines",
    # NOTE: pipeline was trained WITH trailing space — must preserve it
    "Awareness about impact of menstrual waste": "Awareness about impact of menstrual waste ",
    "Promoting Gov policies": "Promoting Gov policies",
    "Any discomfort": "Any discomfort",
    "Aware about Gov policies": "Aware about Gov policies",
    "Shift to other product": "Shift to other product",
    "reason_given": "reason_given",
    "product_reason_given": "product_reason_given",
}

# Ordered list of columns exactly as the pipeline expects them
PIPELINE_COLUMNS = list(ALIAS_TO_COLUMN.values())


def build_dataframe(request_data: dict[str, Any]) -> pd.DataFrame:
    """
    Convert the API request dict (keyed by alias/original names) into a
    single-row pandas DataFrame with columns in the exact order the pipeline
    was trained on.
    """
    row: dict[str, Any] = {}
    for alias, col_name in ALIAS_TO_COLUMN.items():
        # request_data may use either the alias or the original column name
        value = request_data.get(alias) if request_data.get(alias) is not None else request_data.get(col_name)
        row[col_name] = value

    df = pd.DataFrame([row], columns=PIPELINE_COLUMNS)
    return df


def build_explainer(pipeline: Any) -> Optional[Any]:
    """
    Build an appropriate SHAP explainer for the final estimator in the pipeline.
    Returns None if the model type is unsupported or if shap import fails.
    """
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        try:
            from xgboost import XGBClassifier
            xgb_types = (XGBClassifier,)
        except ImportError:
            xgb_types = ()

        clf = pipeline.named_steps.get("clf")
        if clf is None:
            # Try last step generically
            clf = pipeline[-1]

        if isinstance(clf, (RandomForestClassifier, GradientBoostingClassifier) + xgb_types):
            logger.info("Building TreeExplainer for %s", type(clf).__name__)
            return shap.TreeExplainer(clf)

        if isinstance(clf, LogisticRegression):
            logger.info("Building LinearExplainer for LogisticRegression")
            # Use a zero background with the correct number of features
            n_features = clf.coef_.shape[1]
            background = np.zeros((1, n_features))
            return shap.LinearExplainer(
                clf, masker=shap.maskers.Independent(data=background)
            )

        logger.warning("Unknown estimator type %s — SHAP disabled", type(clf).__name__)
        return None

    except Exception as exc:
        logger.warning("Could not build SHAP explainer: %s", exc)
        return None


def run_prediction(
    pipeline: Any,
    explainer: Optional[Any],
    request_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the full prediction pipeline:
      1. Build the feature DataFrame
      2. Get switch_intent (0/1) and probability
      3. Compute SHAP values (non-fatal)

    Returns a dict matching PredictionResponse fields.
    """
    df = build_dataframe(request_data)

    switch_intent = int(pipeline.predict(df)[0])
    probability = round(float(pipeline.predict_proba(df)[0][1]), 4)
    label = "🌿 Likely to Switch" if switch_intent == 1 else "🌱 Not Yet Ready to Switch"

    shap_values: Optional[dict[str, float]] = None
    if explainer is not None:
        try:
            # Transform with all steps except the final classifier
            preprocessor = pipeline[:-1]
            X_transformed = preprocessor.transform(df)

            # Get feature names from the ColumnTransformer
            feature_names = list(preprocessor.get_feature_names_out())

            raw_shap = explainer.shap_values(X_transformed)

            # For binary classifiers, shap_values returns a list [class0, class1]
            if isinstance(raw_shap, list):
                values = raw_shap[1][0]
            else:
                values = raw_shap[0]

            shap_values = {
                name: round(float(val), 6)
                for name, val in zip(feature_names, values)
            }
        except Exception as exc:
            logger.warning("SHAP computation failed (non-fatal): %s\n%s", exc, traceback.format_exc())
            shap_values = None

    return {
        "switch_intent": switch_intent,
        "probability": probability,
        "label": label,
        "shap_values": shap_values,
    }
