from __future__ import annotations


def recommend_capture_improvements(
    image_features: list[dict],
    pair_results: list[dict],
    connectivity: dict,
) -> list[dict]:
    """Generate conservative advice tied to measured pair/image evidence only."""
    recommendations = []
    for image in image_features:
        index = image["image_index"]
        if image["feature_count"] < 80:
            recommendations.append({
                "image_indices": [index],
                "priority": "high",
                "recommendation": (
                    "Retake this view with more visible, well-lit surface texture and a steadier camera."
                ),
                "evidence": {
                    "feature_count": image["feature_count"],
                    "threshold": 80,
                },
                "confidence": "heuristic",
            })

    for pair in pair_results:
        left, right = pair["image_indices"]
        if not pair["verified"]:
            recommendations.append({
                "image_indices": [left, right],
                "priority": "high",
                "recommendation": (
                    "Capture an additional intermediate viewpoint showing shared distinctive "
                    "details from both views; increase overlap and avoid large viewpoint jumps."
                ),
                "evidence": {
                    "pair_type": pair["pair_type"],
                    "reason": pair["reason"],
                    "tentative_matches": pair["tentative_matches"],
                    "inlier_matches": pair["inlier_matches"],
                },
                "confidence": "pairwise_geometry",
            })
        elif pair.get("spatial_distribution", {}).get("warning"):
            recommendations.append({
                "image_indices": [left, right],
                "priority": "medium",
                "recommendation": (
                    "Retake or add a wider view that spreads distinctive object details "
                    "across more of the frame, while keeping both views overlapping."
                ),
                "evidence": {
                    "warning": pair["spatial_distribution"]["warning"],
                    "left_spread": pair["spatial_distribution"]["left_image"],
                    "right_spread": pair["spatial_distribution"]["right_image"],
                },
                "confidence": "2d_spatial_heuristic",
            })

    for component in connectivity["components"]:
        if len(component) == 1 and connectivity["image_count"] > 1:
            index = component[0]
            recommendations.append({
                "image_indices": [index],
                "priority": "high",
                "recommendation": (
                    "Add one or more bridge photographs between this isolated view and "
                    "the main capture sequence, preserving visible overlap."
                ),
                "evidence": {
                    "connectivity_component": component,
                    "isolated": True,
                },
                "confidence": "connectivity_graph",
            })

    # Stable deduplication avoids repeating the same advice for the same image pair.
    unique = {}
    for item in recommendations:
        key = (tuple(item["image_indices"]), item["recommendation"])
        unique.setdefault(key, item)
    return list(unique.values())
