from dataclasses import dataclass


@dataclass(frozen=True)
class ViewCandidate:
    candidate_id: str
    visibility_gain: float
    information_gain: float
    redundancy: float
    estimated_cost: float


@dataclass(frozen=True)
class PlannerConfig:
    max_frames: int = 200
    min_information_gain: float = 0.01
    max_redundancy: float = 0.95
    cost_weight: float = 0.05
