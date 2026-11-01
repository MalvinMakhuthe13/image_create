from app.engine.planning.models import PlannerConfig, ViewCandidate
from app.engine.planning.optimizer import select_viewpoints


def test_planner_filters_redundant_and_low_gain_candidates():
    candidates = [
        ViewCandidate("good", 0.7, 0.8, 0.1, 0.1),
        ViewCandidate("duplicate", 0.1, 0.4, 0.99, 0.1),
        ViewCandidate("low-gain", 0.1, 0.001, 0.1, 0.1),
    ]
    selected = select_viewpoints(candidates, PlannerConfig(max_frames=10))
    assert [item.candidate_id for item in selected] == ["good"]
