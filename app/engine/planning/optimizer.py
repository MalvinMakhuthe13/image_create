from app.engine.planning.models import PlannerConfig, ViewCandidate


def utility(candidate: ViewCandidate, config: PlannerConfig) -> float:
    """Rank candidates by expected value; values must be calibrated on benchmark data."""
    return (
        candidate.visibility_gain
        + candidate.information_gain
        - candidate.redundancy
        - config.cost_weight * candidate.estimated_cost
    )


def select_viewpoints(
    candidates: list[ViewCandidate],
    config: PlannerConfig | None = None,
) -> list[ViewCandidate]:
    """Select useful candidates and stop when marginal value is too low.

    This is a deterministic baseline planner, not yet a geometry visibility solver.
    Production planning must supply candidates from reconstructed geometry and account
    for occlusion, camera frustum, surface confidence, and spatial coverage.
    """
    config = config or PlannerConfig()
    ranked = sorted(candidates, key=lambda item: utility(item, config), reverse=True)
    selected: list[ViewCandidate] = []

    for candidate in ranked:
        if len(selected) >= config.max_frames:
            break
        if candidate.redundancy > config.max_redundancy:
            continue
        if candidate.information_gain < config.min_information_gain:
            continue
        if utility(candidate, config) <= 0:
            continue
        selected.append(candidate)

    return selected
