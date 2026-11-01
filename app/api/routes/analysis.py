from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.engine.analysis.image_analyzer import analyze_image, hamming_distance

router = APIRouter(prefix="/analysis", tags=["image analysis"])
CHUNK_SIZE = 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"}


@router.post("/images")
async def analyse_images(files: list[UploadFile] = File(...)) -> dict:  # noqa: B008
    """Analyse uploaded photographs locally using CPU-only image processing."""
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one image")
    if len(files) > settings.max_images_per_analysis:
        raise HTTPException(status_code=413, detail=f"Maximum {settings.max_images_per_analysis} images per analysis")

    analysis_id = uuid4()
    folder = Path(settings.upload_dir) / str(analysis_id)
    folder.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    total_bytes = 0
    assets = []

    try:
        for index, upload in enumerate(files, start=1):
            if upload.content_type not in ALLOWED_CONTENT_TYPES:
                raise HTTPException(status_code=415, detail=f"{upload.filename or 'File'}: unsupported content type")
            temporary = folder / f"{index:03d}.upload"
            digest = hashlib.sha256()
            size = 0
            with temporary.open("wb") as output:
                while chunk := await upload.read(CHUNK_SIZE):
                    size += len(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise HTTPException(status_code=413, detail=f"Combined upload exceeds {settings.max_upload_mb} MB")
                    digest.update(chunk)
                    output.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="Empty files are not allowed")
            try:
                metrics, extension = analyze_image(temporary, settings.max_image_pixels)
            except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, OSError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"{upload.filename or 'File'} is not a supported, valid image: {exc}") from exc
            sha = digest.hexdigest()
            stored_name = f"{index:03d}_{sha[:12]}{extension}"
            final_path = folder / stored_name
            temporary.replace(final_path)
            assets.append({
                "asset_id": str(uuid4()),
                "original_filename": Path(upload.filename or f"image-{index}").name[:200],
                "stored_filename": stored_name,
                "sha256": sha,
                "storage_path": str(final_path),
                "metrics": metrics.to_dict(),
            })

        exact: dict[str, list[int]] = {}
        for i, asset in enumerate(assets):
            exact.setdefault(asset["sha256"], []).append(i)
        exact_groups = [
            {"image_indices": indices, "filenames": [assets[i]["original_filename"] for i in indices]}
            for indices in exact.values() if len(indices) > 1
        ]
        near = []
        for left in range(len(assets)):
            for right in range(left + 1, len(assets)):
                if assets[left]["sha256"] == assets[right]["sha256"]:
                    continue
                distance = hamming_distance(assets[left]["metrics"]["difference_hash"], assets[right]["metrics"]["difference_hash"])
                if distance <= settings.near_duplicate_hash_distance:
                    near.append({"image_indices": [left, right], "filenames": [assets[left]["original_filename"], assets[right]["original_filename"]], "hash_distance": distance, "similarity_estimate": round(1 - distance / 64, 4)})

        flag_counts: dict[str, int] = {}
        for asset in assets:
            for flag in asset["metrics"]["quality_flags"]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        return {
            "analysis_id": str(analysis_id),
            "status": "complete",
            "processing": {"mode": "cpu_local", "gpu_required": False},
            "summary": {"images_received": len(assets), "total_bytes": total_bytes, "exact_duplicate_groups": len(exact_groups), "near_duplicate_pairs": len(near), "quality_flag_counts": flag_counts},
            "images": assets,
            "exact_duplicates": exact_groups,
            "near_duplicates": near,
            "coverage_assessment": {"status": "not_assessed", "reason": "Image similarity and EXIF metadata do not establish camera poses, geometric overlap, or 3D surface coverage."},
            "limitations": [
                "Sharpness and exposure flags are heuristic indicators, not definitive judgments.",
                "Perceptual-hash similarity is not a geometric feature match.",
                "Camera pose, overlap, missing viewpoints, and 3D coverage are not yet estimated.",
                "Original uploaded bytes are preserved unchanged in the local uploads directory.",
            ],
        }
    except Exception:
        # Make each multi-image upload atomic; remove partial files on failure.
        shutil.rmtree(folder, ignore_errors=True)
        raise
