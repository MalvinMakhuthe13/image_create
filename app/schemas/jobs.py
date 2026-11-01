from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


GenerationMode = Literal["verified", "experimental", "both"]


class TargetSettings(BaseModel):
    coverage: Literal["maximum_observable"] = "maximum_observable"
    include_top: bool = True
    include_underside: bool = True
    include_low_angles: bool = True


class BudgetSettings(BaseModel):
    maximum_frames: int = Field(default=200, ge=1, le=2000)
    maximum_gpu_seconds: int = Field(default=1800, ge=0)


class CreateJobRequest(BaseModel):
    nif_id: UUID | None = None
    reconstruction_job_id: UUID | None = None
    generation_mode: GenerationMode = "both"
    target: TargetSettings = Field(default_factory=TargetSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)


class JobResponse(BaseModel):
    id: UUID
    status: Literal["queued", "analysing", "failed", "complete"] = "queued"
    generation_mode: GenerationMode
    message: str
