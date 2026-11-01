from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.engine.matching.models import ImageFeatures


def extract_features(image_path: Path, image_index: int, *, max_dimension: int = 1600,
                     max_features: int = 6000, mask: np.ndarray | None = None,
                     feature_profile: str = "baseline") -> ImageFeatures:
    if max_dimension < 128 or max_features < 32:
        raise ValueError("max_dimension must be >=128 and max_features must be >=32")
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        source_width, source_height = image.size
        focal_35mm = source.getexif().get(41989)
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector_mask = None
    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.ndim != 2 or mask_array.shape != (source_height, source_width):
            raise ValueError("feature mask must match the EXIF-oriented source image dimensions")
        detector_mask = cv2.resize(
            (mask_array > 0).astype(np.uint8) * 255,
            (image.width, image.height),
            interpolation=cv2.INTER_NEAREST,
        )
        # Ignore the uncertain segmentation edge: SIFT descriptors straddling
        # object/background boundaries are a common source of false matches.
        detector_mask = cv2.erode(
            detector_mask, np.ones((3, 3), dtype=np.uint8), iterations=1
        )
    if feature_profile not in {"baseline", "object"}:
        raise ValueError("feature_profile must be 'baseline' or 'object'")
    if feature_profile == "object":
        # Lower contrast threshold recovers subtle fabric/stitching details.
        # CLAHE is applied only to the detector input; source pixels are untouched.
        detector_image = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        ).apply(gray)
        detector = cv2.SIFT_create(
            nfeatures=max_features,
            contrastThreshold=0.012,
            edgeThreshold=12,
            sigma=1.2,
        )
    else:
        detector_image = gray
        detector = cv2.SIFT_create(nfeatures=max_features)
    keypoints, descriptors = detector.detectAndCompute(detector_image, detector_mask)
    scale_x, scale_y = source_width / image.width, source_height / image.height
    points = tuple((float(k.pt[0] * scale_x), float(k.pt[1] * scale_y)) for k in keypoints)
    if descriptors is not None:
        descriptors = np.asarray(descriptors, dtype=np.float32)
    camera_matrix = None
    intrinsics_source = None
    distortion_coefficients = None
    calibration_path = image_path.with_name(image_path.name + ".calibration.json")
    if calibration_path.is_file():
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
            distortion = np.asarray(calibration.get("distortion_coefficients", []), dtype=np.float64)
            if (matrix.shape == (3, 3) and np.isfinite(matrix).all() and matrix[0, 0] > 0 and matrix[1, 1] > 0 and calibration.get("image_width") == source_width and calibration.get("image_height") == source_height):
                camera_matrix = matrix
                distortion_coefficients = distortion.reshape(-1, 1) if distortion.size else None
                intrinsics_source = "user_supplied_calibration"
        except (OSError, ValueError, KeyError, TypeError):
            pass
    try:
        focal_35mm = float(focal_35mm) if focal_35mm is not None else None
    except (TypeError, ValueError):
        focal_35mm = None
    if camera_matrix is None and focal_35mm is not None and focal_35mm > 0:
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
        scale_x, scale_y, camera_matrix, intrinsics_source, distortion_coefficients,
    )
