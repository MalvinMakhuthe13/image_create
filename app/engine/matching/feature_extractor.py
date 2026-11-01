from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.engine.matching.models import ImageFeatures


def extract_features(image_path: Path, image_index: int, *, max_dimension: int = 1600,
                     max_features: int = 6000) -> ImageFeatures:
    if max_dimension < 128 or max_features < 32:
        raise ValueError("max_dimension must be >=128 and max_features must be >=32")
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        source_width, source_height = image.size
        focal_35mm = source.getexif().get(41989)
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=max_features).detectAndCompute(gray, None)
    scale_x, scale_y = source_width / image.width, source_height / image.height
    points = tuple((float(k.pt[0] * scale_x), float(k.pt[1] * scale_y)) for k in keypoints)
    if descriptors is not None:
        descriptors = np.asarray(descriptors, dtype=np.float32)
    camera_matrix = None
    intrinsics_source = None
    try:
        focal_35mm = float(focal_35mm) if focal_35mm is not None else None
    except (TypeError, ValueError):
        focal_35mm = None
    if focal_35mm is not None and focal_35mm > 0:
        # Approximate pixel focal length from EXIF 35mm-equivalent focal length.
        # This does not provide calibrated distortion coefficients.
        diagonal_px = float(np.hypot(source_width, source_height))
        focal_px = focal_35mm * diagonal_px / 43.2666
        camera_matrix = np.array([
            [focal_px, 0.0, source_width / 2.0],
            [0.0, focal_px, source_height / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        intrinsics_source = "exif_35mm_equivalent_approximation"
    return ImageFeatures(
        image_index, points, descriptors, source_width, source_height,
        scale_x, scale_y, camera_matrix, intrinsics_source,
    )
