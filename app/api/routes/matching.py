from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.engine.matching import (
    build_connectivity_graph,
    extract_features,
    match_descriptors,
    verify_geometry,
)

router = APIRouter(prefix="/analysis", tags=["geometric matching"])
STORED_IMAGE_PATTERN = re.compile(r"^\d{3}_[0-9a-f]{12}\.(jpg|png|webp|tif|bmp)$", re.IGNORECASE)


def _candidate_pairs(image_count: int, maximum: int) -> list[tuple[int, int]]:
    """Prioritise neighbouring capture frames, then sample remaining pairs evenly."""
    if image_count < 2:
        return []
    minimum_chain = image_count - 1
    if maximum < minimum_chain:
        raise ValueError(
            f"max_matching_pair_checks must be at least {minimum_chain} "
            "to preserve adjacent-frame checks"
        )

    selected = [(index, index + 1) for index in range(image_count - 1)]
    if len(selected) >= maximum:
        return selected[:maximum]

    selected_set = set(selected)
    remaining = [
        pair for pair in combinations(range(image_count), 2)
        if pair not in selected_set
    ]
    slots = maximum - len(selected)
    if len(remaining) <= slots:
        selected.extend(remaining)
    elif slots:
        # Evenly distribute the additional comparisons over the pair list.
        last = len(remaining) - 1
        chosen_indices = sorted({round(i * last / max(1, slots - 1)) for i in range(slots)})
        selected.extend(remaining[index] for index in chosen_indices)
    return selected


@router.post("/{analysis_id}/matching")
def match_uploaded_analysis(analysis_id: UUID) -> dict:
    """Run bounded CPU geometric matching against images from a completed upload."""
    folder = Path(settings.upload_dir) / str(analysis_id)
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="Analysis upload was not found")

    image_paths = sorted(
        path for path in folder.iterdir()
        if path.is_file() and STORED_IMAGE_PATTERN.fullmatch(path.name)
    )
    if not image_paths:
        raise HTTPException(status_code=404, detail="No stored analysis images were found")

    try:
        pairs = _candidate_pairs(len(image_paths), settings.max_matching_pair_checks)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    features = []
    try:
        for index, path in enumerate(image_paths):
            features.append(
                extract_features(
                    path,
                    index,
                    max_dimension=settings.matching_max_dimension,
                    max_features=settings.matching_max_features,
                )
            )
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract features from stored image: {exc}",
        ) from exc

    verified_pairs = []
    pair_results = []
    for left_index, right_index in pairs:
        matches = match_descriptors(features[left_index], features[right_index])
        result = verify_geometry(features[left_index], features[right_index], matches)
        verified_pairs.append((left_index, right_index, result))
        pair_results.append({
            "image_indices": [left_index, right_index],
            "filenames": [image_paths[left_index].name, image_paths[right_index].name],
            **result.to_dict(),
        })

    graph = build_connectivity_graph(len(image_paths), verified_pairs)
    return {
        "analysis_id": str(analysis_id),
        "status": "complete",
        "processing": {
            "mode": "cpu_local",
            "gpu_required": False,
            "feature_extractor": "SIFT",
            "pair_selection": "adjacent_frames_plus_evenly_sampled_pairs",
        },
        "summary": {
            "images_processed": len(image_paths),
            "pairs_evaluated": len(pairs),
            "pair_check_limit": settings.max_matching_pair_checks,
            "verified_pairs": graph["verified_edge_count"],
            "connected_components": len(graph["components"]),
            "isolated_images": len(graph["isolated_image_indices"]),
        },
        "images": [
            {
                "image_index": index,
                "filename": path.name,
                "feature_count": features[index].feature_count,
            }
            for index, path in enumerate(image_paths)
        ],
        "pair_results": pair_results,
        "connectivity": graph,
        "coverage_assessment": {
            "status": "not_assessed",
            "reason": (
                "Pairwise geometric connectivity does not estimate camera poses or prove "
                "complete 3D surface coverage. Unchecked image pairs may contain additional links."
            ),
        },
        "limitations": [
            "This stage performs CPU feature matching and pairwise geometric verification only.",
            "A verified pair is not a calibrated camera pose or a 3D reconstruction.",
            "The bounded pair schedule can omit valid links between images that were not compared.",
            "Disconnected images may be caused by blur, lighting, viewpoint changes, or missing overlap.",
        ],
    }
