from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageFeatures:
    image_index: int
    keypoints: tuple[tuple[float, float], ...]
    descriptors: np.ndarray | None
    image_width: int
    image_height: int
    scale_x: float
    scale_y: float

    @property
    def feature_count(self) -> int:
        return len(self.keypoints)


@dataclass(frozen=True)
class DescriptorMatch:
    query_index: int
    train_index: int
    distance: float


@dataclass(frozen=True)
class GeometryResult:
    verified: bool
    model: str
    tentative_matches: int
    inlier_matches: int
    inlier_ratio: float
    median_reprojection_error_px: float | None
    reason: str
    inlier_match_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
