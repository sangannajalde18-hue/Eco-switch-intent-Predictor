"""
Pydantic v2 request and response schemas for the Eco-Switch Intent prediction API.

Column names use aliases that match the exact column names the sklearn pipeline
was trained on. The frontend/API consumers use the alias names (original survey strings).
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """All 22 feature-vector fields required by the trained pipeline."""

    # --- Numeric features ---
    Age: float = Field(..., ge=17, le=120, description="Respondent age in years")
    Income: float = Field(..., ge=0, description="Monthly household income (₹)")

    # alias matches exact pipeline column name
    products_per_cycle: float = Field(
        ..., ge=1, le=100,
        alias="No.of products used for a cycle",
        description="Number of menstrual products used per cycle"
    )
    years_using_product: float = Field(
        ..., ge=0, le=50,
        description="Years using current menstrual product"
    )

    # --- Ordinal features ---
    intensity_of_flow: Literal["Low", "Medium", "High"] = Field(
        ..., alias="Intensity of flow"
    )
    single_day_changing_frequency: Literal[
        "Less than 3 times", "3-5 times", "More than 5 times"
    ] = Field(..., alias="Single day changing frequency")
    duration_of_single_product: Literal[
        "1-3 hours", "4-6 hours", "More than 6 hours"
    ] = Field(..., alias="Duration of single product per day")
    monthly_cost: Literal[
        "Less than 100 Rs", "100-150 Rs", "150-200 Rs", "More than 200 Rs"
    ] = Field(..., alias="Monthly cost spend")
    manufacturer_eco: Literal[
        "Disagree", "Neutral", "Agree", "Strongly agree"
    ] = Field(..., alias="Manufacturer should produce eco friendly products")

    # --- Nominal features ---
    Occupation: Literal["Student", "Service", "Business", "Homemaker", "Others"]
    Area: Literal["Urban", "Rural"]
    menstrual_product: Literal[
        "Sanitary napkins", "Organic pads", "Menstrual cups", "Tampons", "Cloth"
    ] = Field(..., alias="Menstrual product")
    disposable_reusable: Literal["Disposable", "Reusable", "Both"] = Field(
        ..., alias="Disposable/Reusable product"
    )
    method_of_disposing: Literal[
        "General household waste",
        "Separated dustbins for sanitary napkins",
        "Burns it",
        "Other",
    ] = Field(..., alias="Method of disposing")
    availability_of_machines: Literal["Yes", "No"] = Field(
        ..., alias="Availability of disposable machines"
    )
    awareness_impact: Literal["Yes", "No"] = Field(
        ..., alias="Awareness about impact of menstrual waste"
    )
    promoting_gov_policies: Literal["Yes", "No"] = Field(
        ..., alias="Promoting Gov policies"
    )
    any_discomfort: Literal["Yes", "No"] = Field(..., alias="Any discomfort")
    aware_gov_policies: Literal["Yes", "No"] = Field(
        ..., alias="Aware about Gov policies"
    )
    shift_to_other: Literal["Yes", "No"] = Field(..., alias="Shift to other product")

    # --- Binary passthrough flags ---
    reason_given: Literal[0, 1] = Field(
        ..., description="1 if respondent gave a reason for switching"
    )
    product_reason_given: Literal[0, 1] = Field(
        ..., description="1 if respondent gave a reason for their current product"
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    switch_intent: int = Field(..., description="0 = Not Ready, 1 = Likely to Switch")
    probability: float = Field(..., description="Probability of switching (class 1), 0–1")
    label: str = Field(..., description="Human-readable prediction label")
    shap_values: Optional[dict[str, float]] = Field(
        None, description="SHAP feature contributions (top features)"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
