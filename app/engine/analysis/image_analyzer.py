from __future__ import annotations

import warnings

import cv2
import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, ImageStat

FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif", "BMP": ".bmp"}


@dataclass(frozen=True)
class ImageAnalysis:
    width: int
    height: int
    format: str
    mode: str
    file_size_bytes: int
    sharpness_score: float
    laplacian_variance: float
    edge_directionality: float
    clipped_fraction: float
    histogram_entropy_bits: float
    brightness_mean: float
    underexposed_fraction: float
    overexposed_fraction: float
    quality_flags: list[str]
    difference_hash: str
    captured_at: str | None
    camera_make: str | None
    camera_model: str | None
    focal_length_mm: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def dhash(image: Image.Image) -> str:
    small = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = small.tobytes()
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{bits:016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def _exif_text(exif, key: int) -> str | None:
    value = exif.get(key)
    if value is None:
        return None
    return str(value)


def analyze_image(path: Path, max_pixels: int) -> tuple[ImageAnalysis, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as source:
            image_format = (source.format or "").upper()
            if image_format not in FORMAT_EXTENSIONS:
                raise ValueError(f"Unsupported image format: {image_format or 'unknown'}")
            source_width, source_height = source.size
            if source_width < 1 or source_height < 1 or source_width * source_height > max_pixels:
                raise ValueError(
                    f"Image dimensions {source_width}x{source_height} exceed the pixel limit"
                )
            exif = source.getexif()
            captured_at = _exif_text(exif, 36867) or _exif_text(exif, 306)
            camera_make = _exif_text(exif, 271)
            camera_model = _exif_text(exif, 272)
            focal = exif.get(37386)
            try:
                focal_length = float(focal) if focal is not None else None
            except (TypeError, ValueError, ZeroDivisionError):
                focal_length = None
            mode = source.mode
            image = ImageOps.exif_transpose(source).convert("RGB")
            # Report the same orientation and dimensions used by matching,
            # masks and calibration; the original file bytes remain untouched.
            width, height = image.size
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(image)
    brightness = float(ImageStat.Stat(gray).mean[0])
    sharpness = float(ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).var[0])
    histogram = gray.histogram()
    total = max(1, sum(histogram))
    under = sum(histogram[:12]) / total
    over = sum(histogram[244:]) / total
    gray_array = np.asarray(gray, dtype=np.uint8)
    laplacian_variance = float(cv2.Laplacian(gray_array, cv2.CV_64F).var())
    gradient_x = cv2.Sobel(gray_array, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_array, cv2.CV_64F, 0, 1, ksize=3)
    gradient_x_variance = float(gradient_x.var())
    gradient_y_variance = float(gradient_y.var())
    edge_directionality = max(gradient_x_variance, gradient_y_variance) / max(
        min(gradient_x_variance, gradient_y_variance), 1e-9
    )
    clipped_fraction = (sum(histogram[:3]) + sum(histogram[253:])) / total
    probabilities = np.asarray(histogram, dtype=np.float64) / total
    nonzero = probabilities[probabilities > 0]
    histogram_entropy = float(-(nonzero * np.log2(nonzero)).sum())
    flags = []
    if min(width, height) < 640:
        flags.append("low_resolution")
    if sharpness < 35 or laplacian_variance < 80:
        flags.append("possibly_blurry")
    if laplacian_variance < 120 and edge_directionality > 4.0:
        flags.append("possible_directional_motion_blur")
    if clipped_fraction > 0.20:
        flags.append("significant_highlight_or_shadow_clipping")
    if brightness < 45 or under > 0.25:
        flags.append("possibly_underexposed")
    if brightness > 210 or over > 0.25:
        flags.append("possibly_overexposed")

    result = ImageAnalysis(
        width, height, image_format, mode, path.stat().st_size,
        round(sharpness, 3), round(laplacian_variance, 3),
        round(edge_directionality, 3), round(clipped_fraction, 4),
        round(histogram_entropy, 3), round(brightness, 3), round(under, 4), round(over, 4),
        flags, dhash(image), captured_at, camera_make, camera_model, focal_length,
    )
    return result, FORMAT_EXTENSIONS[image_format]
