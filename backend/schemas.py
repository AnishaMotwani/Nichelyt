from pydantic import BaseModel, Field
from typing import Optional


class AnalyzeRequest(BaseModel):
    niche: str = Field(..., min_length=1)
    goal: str = Field(..., min_length=1)
    post_idea: str = Field(..., min_length=1)

    followers: float = Field(..., gt=0)

    post_type: str = Field(..., pattern="^(Photo|Album|Video)$")

    caption: str = Field(..., min_length=1)

    post_date: str
    post_time: str

    tone: Optional[str] = "Educational"

    target_audience: Optional[str] = "General audience"

    language: Optional[str] = "English"

    run_counterfactual: bool = True

    run_shap: bool = True

    run_gemini: bool = True