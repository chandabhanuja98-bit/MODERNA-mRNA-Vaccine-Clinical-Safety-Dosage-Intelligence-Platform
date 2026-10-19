from pydantic import BaseModel, Field
from typing import List


class ClinicalSafetyReport(BaseModel):
    Eligible_For_Vaccine: bool = Field(
        ...,
        description="Whether the report marks the patient eligible."
    )

    Recommended_Dose_mg: float = Field(
        ...,
        description="Dose value in milligrams, subject to guideline checks."
    )

    Reasoning_Steps: List[str] = Field(
        ...,
        description="Ordered reasoning steps explaining the output."
    )

    FDA_Safety_Warnings_Cited: List[str] = Field(
        ...,
        description="Exact relevant excerpts from the provided guidelines."
    )

    Disclaimer: str = Field(
        default=(
            "Student engineering project output — "
            "not real clinical guidance."
        ),
        description="Required educational-project disclaimer."
    )