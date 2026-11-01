# image_create — CPU-first photogrammetry intelligence

`image_create` is a local-first image-capture analysis and multi-view geometry foundation. It is **not yet a complete dense photogrammetry or production reconstruction application**. The current implementation provides image intake, capture diagnostics, geometric matching, explicit camera calibration, and an initial sparse multi-view reconstruction prototype.

## Non-negotiable data and evidence rules

- Original photographs are immutable source evidence.
- Synthetic/AI images and virtual views are never silently mixed into the original capture set.
- A descriptor match is not a geometric inlier; a geometric inlier is not a camera pose; sparse landmarks are not a dense surface.
- Unknown, occluded, weakly observed and unobserved surfaces remain unknown. Never invent coverage percentages.
- Every result must retain its source-image indices, method, confidence/quality indicators and limitations.
- Monocular reconstruction has arbitrary scale unless a trusted scale reference or metric sensor input is supplied.

## Quick start (Python 3.11+)

```powershell
.\scripts\run_local.ps1
```

For manual setup, create a Python 3.11+ virtual environment and install `-e ".[dev]"`.

Browser Studio: http://127.0.0.1:8000/  
API docs: http://127.0.0.1:8000/docs

## Implemented foundation

1. Browser capture entry point for multi-image upload, image-quality summary and handoff to visual correspondence/mask inspection. Upload and preserve JPEG, PNG, WebP, TIFF and BMP source photographs with validation, hashes and image-quality metrics.
2. Inspect resolution, EXIF, sharpness, blur, directional edge response, exposure/clipping, histogram entropy and exact/near duplicates.
3. Extract CPU SIFT features and perform mutual ratio-tested descriptor matching.
4. Evaluate a bounded adjacent-first set of image pairs with fundamental-matrix and homography RANSAC, residuals, planar/low-parallax safeguards and image-space support diagnostics.
5. Keep all-scene connectivity separate from object-only analysis. Object-only geometric verification requires valid user-painted masks on both images; automatic subject proposals remain advisory.
6. Produce evidence-linked capture recommendations from measured image and pair diagnostics.
7. Accept explicit per-image camera calibration (camera matrix and supported OpenCV distortion coefficients), checking calibration dimensions. EXIF-derived intrinsics are approximate only.
8. Estimate relative camera poses where supported, triangulate pairwise sparse points, fuse feature observations into multi-view tracks, initialize connected camera components and run a bounded robust SciPy bundle-adjustment prototype.
9. Persist job metadata in SQLite/WAL and protect non-development API routes with a configured shared API key.

## What is not complete

The current sparse reconstruction is a prototype, not a mature SfM implementation. It does not yet provide a validated global pose-graph optimizer with loop closure, production-grade track management, dense multi-view stereo, dense depth fusion, surface normals, occlusion-aware visibility, point-cloud fusion/cleanup, meshing, texturing, watertightness checks or defensible object-surface coverage.

The job API currently persists queued job metadata; it does **not** execute a production processing queue. There are no worker leases, retries, cancellation, checkpoints or distributed scheduling. A shared API key is not per-user authentication, authorization or tenant isolation. The complete reconstruction Studio workflow, persistent object storage, benchmark suite and production deployment hardening also remain.

The API must therefore continue to return `coverage_assessment.status = not_assessed` until a tested visibility method can support a coverage claim. Do not infer surface coverage from match counts, camera count or sparse landmarks.

## Recommended implementation order

1. Stabilize and benchmark sparse camera geometry, coordinate conventions, track quality and bundle-adjustment diagnostics.
2. Add a supported dense reconstruction backend with explicit CPU/GPU capability detection, resource limits and failure reporting.
3. Fuse depth maps into a confidence-aware point cloud; reject inconsistent, occluded and low-confidence observations.
4. Add surface reconstruction, normals, texture provenance, mesh cleanup and watertightness diagnostics.
5. Compute visibility/coverage only from registered cameras and measured surface evidence; report unknown regions explicitly.
6. Implement real durable workers, progress, retries, cancellation, checkpoint/restart and resource management.
7. Add per-user identity/authorization, analysis ownership, quotas, rate limits, storage retention/deletion, monitoring and backups.
8. Complete the capture-to-export Studio workflow and benchmark it on varied real datasets (small products, fabric, rocks, vehicles, buildings/interiors, reflective objects, low light and incomplete captures).

## Security

Development mode is intended for localhost. Outside development, the API fails closed unless `IMAGE_CREATE_API_KEY` is configured. API clients use `X-API-Key`; the browser Studio exchanges it for an HttpOnly, Secure, SameSite=Strict session cookie so image and inspection requests can be authenticated without putting the key in URLs. The browser session requires HTTPS outside development. This is a shared service key only, not per-user identity or tenant isolation. Do not expose local upload folders directly to the public internet.

See `docs/TESTING_GUIDE.md`, `docs/GEOMETRY_AND_COVERAGE.md`, `docs/OBJECT_ONLY_MATCHING.md`, `docs/SPATIAL_DIAGNOSTICS.md` and `docs/ROADMAP.md` and `docs/RECONSTRUCTION_BACKEND_DECISION.md` for the test workflow, technical contracts, backend licensing assessment and remaining work.
