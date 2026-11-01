# image_create — FUMOCA Frame Intelligence

Standalone CPU-first image-capture analysis and geometric matching application. The current
implementation is an image-quality and sparse pairwise-geometry foundation, not a complete
photogrammetry reconstruction engine.

## Non-negotiable rules
- Original photographs are immutable.
- Synthetic/AI images and virtual views are never silently mixed into production photo datasets.
- Pairwise matching is not proof of a camera trajectory or complete 3D coverage.
- Unknown surfaces remain unknown; do not invent coverage percentages.
- Every diagnostic must preserve its evidence and limitations.

## Quick start (Python 3.11+)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn app.main:app --reload
pytest -q
```
API docs: http://127.0.0.1:8000/docs

## Current CPU pipeline

1. Upload and preserve original JPEG, PNG, WebP, TIFF or BMP photographs.
2. Inspect resolution, EXIF, sharpness, Laplacian variance, directional edge response,
   exposure/clipping, histogram entropy, exact hashes and perceptual near-duplicates.
3. Extract SIFT features and perform mutual ratio-tested descriptor matching.
4. Evaluate a bounded adjacent-first set of image pairs using RANSAC fundamental and
   homography models.
5. Report inlier residuals, model competition, possible planar/low-parallax ambiguity,
   spatial concentration, graph connectivity and evidence-linked capture suggestions.
6. When EXIF 35mm-equivalent focal length exists for both images, estimate approximate
   intrinsics, relative rotation/translation direction and pairwise sparse triangulation.

## Manual object masks

The inspection workspace lets the user paint same-size binary masks. Object-only geometric
verification requires valid manual masks for both images; automatic GrabCut proposals are
advisory and never replace a missing manual mask. The all-scene connectivity graph remains
separate from object-only verification.

## Camera geometry and coverage limits

EXIF-derived intrinsics are approximate, not a true lens calibration. Pairwise translation
has unknown monocular scale. Sparse pairwise triangulation is not a globally fused point cloud.
There is not yet global bundle adjustment, multi-view track fusion, dense reconstruction,
mesh generation, occlusion-aware surface visibility or a defensible 3D coverage percentage.
The API therefore keeps `coverage_assessment.status = not_assessed`.

See:
- `docs/GEOMETRY_AND_COVERAGE.md`
- `docs/OBJECT_ONLY_MATCHING.md`
- `docs/SPATIAL_DIAGNOSTICS.md`

## Deployment security

Job metadata is stored in a local SQLite database, so queued records survive a single-host process restart. This is not a distributed worker queue; job execution, retries and multi-host coordination still need a worker service.\n\nDevelopment mode is intended for localhost. Outside development, the API fails closed unless
`IMAGE_CREATE_API_KEY` is configured. Requests to `/api/v1/*` must include
`X-API-Key: <configured value>`.

This is a shared API key, not per-user authentication or tenant isolation. Before public
multi-user deployment, add user identity and authorization, analysis ownership, rate limiting,
quotas, persistent object storage, background workers, retention/cleanup policies and audit
logging. Do not expose local upload folders directly to the public internet.
