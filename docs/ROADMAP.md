# Engineering roadmap and acceptance gates

This roadmap distinguishes implemented code from capabilities that still need implementation and measured validation. A feature is not complete because an endpoint returns a successful response or a preview looks plausible.

## Status at this branch

### Browser workflow — initial entry point implemented
- [x] Browser upload page, capture-set quality summary and handoff to visual inspection.
- [ ] Complete reconstruction Studio with camera calibration UI, progress-aware reconstruction, 3D review, surface validation, repair and exports.

### Foundation — partially implemented
- [x] FastAPI image upload and validation, hashes, EXIF and image-quality analysis.
- [x] CPU SIFT extraction and bounded adjacent-first pair matching.
- [x] Fundamental/homography geometric verification and degeneracy warnings.
- [x] Manual-mask-only object geometric verification, kept separate from all-scene connectivity.
- [x] Explicit per-image camera calibration and distortion-aware pose/reprojection handling.
- [x] Initial multi-view track fusion and sparse bundle-adjustment prototype.
- [x] SQLite/WAL job metadata persistence and non-development shared API-key gate.
- [ ] Per-user identity, ownership and authorization.
- [ ] Actual durable processing workers, leases, retries, cancellation, progress and restart recovery.
- [ ] Immutable source registry with retention/deletion policy and durable object storage.

### Camera geometry — prototype, not production-complete
- [x] Relative pose estimates from supported calibrated image pairs.
- [x] Sparse triangulation and multi-view feature-track construction.
- [x] Connected-component initialization and bounded sparse bundle adjustment.
- [ ] Independently validated camera-coordinate convention and transform-composition tests across arbitrary camera graphs.
- [ ] Robust global pose-graph optimization, loop closure and drift correction.
- [ ] Track outlier lifecycle, re-triangulation, track quality scoring and scalable incremental reconstruction.
- [ ] Benchmark-driven acceptance against a mature SfM reference implementation.
- [ ] Metric scale from explicit scale bars, known dimensions or trusted sensor priors.

### Dense geometry — not implemented
- [x] Initial backend/licensing research: COLMAP CPU dense limitation and OpenMVS CPU SGM/AGPL trade-off documented.
- [ ] Select and approve a backend/license path; do not bundle AGPL software without an explicit compliance decision.
- [ ] Backend interface and capability discovery for a mature dense MVS engine.
- [ ] CPU/GPU selection, VRAM/RAM estimates, timeouts and clear unsupported-hardware reporting.
- [ ] Per-view dense depth and confidence maps.
- [ ] Geometric consistency checks, occlusion rejection and depth fusion.
- [ ] Confidence-aware point-cloud filtering, normals and outlier cleanup.
- [ ] Surface reconstruction, mesh decimation, hole/watertightness diagnostics and texture generation.
- [ ] Provenance from each dense point/face/texture region back to contributing source views.

### Surface visibility and capture intelligence — not implemented
- [ ] Camera frustum and surface visibility computation on reconstructed geometry.
- [ ] Occlusion-aware observed/unobserved/uncertain surface labels.
- [ ] Coverage heatmaps and percentages with a documented denominator and confidence interval.
- [ ] Viewpoint-gap detection and next-best-photo recommendations with evidence and predicted gain.
- [ ] Object motion, reflective/transparent material and lighting-change warnings.
- [ ] Object-specific capture profiles (vehicles, rocks, small products, buildings/interiors).

### Product workflow and operations — not implemented
- [ ] End-to-end Studio: import, inspect, mask, calibrate, capture advice, reconstruct, review, repair and export.
- [ ] Durable job execution, progress events, retries, cancellation, checkpoints and restart recovery.
- [ ] Resource quotas, bounded concurrency, structured logs, metrics, backups and operational alerts.
- [ ] Per-user auth, authorization, tenant isolation, rate limits, quotas and audit logs.
- [ ] Persistent object storage, retention/deletion and secure download permissions.
- [ ] Export profiles and integration API with complete provenance.

### Verification and benchmark programme — not complete
- [ ] Curated real datasets: cap/shoe, textured product, rock, vehicle, building/interior, reflective object, low light and incomplete capture.
- [ ] Ground-truth or reference comparisons for camera poses, reprojection error, depth accuracy, completeness and mesh quality.
- [ ] Regression thresholds and performance/memory budgets for CPU-only runs.
- [ ] Failure tests: planar scenes, low parallax, repeated texture, moving objects, blur, exposure shifts, disconnected views and incorrect calibration.

## Delivery sequence

1. **Geometry correctness gate:** coordinate conventions, pose composition, gauge constraints, track validation, optimizer termination and residual diagnostics.
2. **Dense reconstruction gate:** integrate a mature supported engine; do not silently substitute a fake depth/mesh result when unavailable.
3. **Fusion and surface gate:** depth confidence, multi-view consistency, point-cloud fusion, normals, meshing and texture provenance.
4. **Visibility gate:** only calculate surface coverage when registered camera geometry and a measurable surface denominator exist.
5. **Worker gate:** execute long-running stages outside request handlers with durable state, retries, cancellation and recovery.
6. **Security/operations gate:** per-user access controls, quotas, durable storage, retention, monitoring and backups.
7. **Studio/export gate:** make the entire workflow understandable and auditable from original photographs to final deliverables.
8. **Benchmark gate:** prove results on diverse real captures before calling the product production-ready.

## Release rule

Do not mark a stage complete until tests cover its failure modes, benchmark results are recorded, provenance is preserved, and the API clearly distinguishes unavailable, partial, failed and validated outputs. Unknown surface area must remain unknown; synthetic or AI-generated content must remain separately labelled and opt-in.
