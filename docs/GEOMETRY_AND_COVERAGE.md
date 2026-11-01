# Geometry, camera pose and coverage: current contract

This document describes the current implementation on the development branch. It separates measured image evidence, pairwise geometry, sparse reconstruction and dense surface visibility. These are different levels of evidence and must not be conflated.

## Pairwise geometric verification

The CPU matcher extracts SIFT descriptors and applies mutual Lowe-ratio matching. A bounded adjacent-first scheduler evaluates candidate pairs with RANSAC fundamental-matrix and homography models. The response reports model support, residuals, planar/low-parallax ambiguity and 4x4 image-space support diagnostics.

Homography-dominant or planar-ambiguous pairs are deliberately prevented from seeding the global sparse camera pose graph. Relative pose is attempted from eligible fundamental-matrix inliers only. This is a conservative safeguard, not a complete degeneracy classifier. The pair schedule may omit valid non-adjacent links.

## Camera intrinsics and distortion

Explicit per-image calibration can be supplied with:

`PUT /api/v1/analysis/{analysis_id}/calibration/{image_index}`

Example body:

```json
{
  "camera_matrix": [
    [1200.0, 0.0, 960.0],
    [0.0, 1200.0, 540.0],
    [0.0, 0.0, 1.0]
  ],
  "distortion_coefficients": [-0.12, 0.03, 0.0, 0.0, 0.0]
}
```

Calibration dimensions must match the original uploaded image. The camera matrix and distortion coefficients must come from a real calibration for the same camera/lens and image configuration. The feature extractor prefers this explicit calibration. When explicit calibration is absent, EXIF 35mm-equivalent focal length may be used to derive approximate intrinsics with a centred-principal-point assumption; this is not a substitute for calibration.

Distortion coefficients are passed to OpenCV point undistortion and bundle-adjustment projection where present. A calibration file is stored as a sidecar next to its source image; preserve the sidecar with the source analysis.

## Sparse multi-view reconstruction

When eligible calibrated pairwise pose edges exist, the current CPU prototype:
1. Builds connected camera components.
2. Initializes component camera transforms from pairwise relative poses.
3. Joins pairwise inlier observations into multi-view feature tracks, rejecting tracks that contain conflicting detections from the same image.
4. Triangulates sparse landmarks.
5. Runs robust SciPy least-squares bundle adjustment with a sparse Jacobian, a fixed reference camera and a soft baseline constraint to remove gauge freedom.

The response reports component status, camera and landmark counts, observation counts, optimizer status/message and median reprojection error. Each disconnected component has an independent arbitrary monocular coordinate frame and scale. The baseline constraint fixes a gauge; it does not create metric scale.

This is an initial bounded CPU prototype, not a mature production SfM engine. Track association can still contain mismatches, pairwise pose errors can contaminate initialization, optimizer termination does not prove global correctness, and a low reprojection error does not by itself prove correct scene geometry. Results require benchmark validation and review.

## Dense reconstruction and surface coverage

Dense multi-view stereo, per-view depth confidence, geometric depth consistency, occlusion rejection, depth fusion, dense point-cloud cleanup, surface normals, meshing, texturing, watertightness analysis and occlusion-aware visibility are **not implemented**.

Therefore:
- `coverage_assessment.status` must remain `not_assessed`.
- Sparse landmarks must never be converted into a surface-coverage percentage.
- Camera count, feature count, pairwise inlier count and image-space grid occupancy are not 3D surface coverage.
- An unseen or occluded region must be reported as unknown, not filled with invented geometry.
- A disconnected camera component must not be merged into another component without a validated registration.

A future coverage calculation needs a reconstructed surface, registered camera poses, per-camera visibility/occlusion tests, a documented surface-area denominator, and confidence/unknown labels.

## Capture advice contract

Current recommendations are evidence-linked to image quality, pairwise geometric support and graph connectivity. They can identify blur, exposure problems, near duplicates, weak pair links and missing connectivity. They cannot yet assert that a physical underside, rear panel, rock face or interior region is missing. That requires a validated global camera model and surface visibility.

Future next-best-photo suggestions should include the affected surface region, supporting camera views, expected visibility gain, uncertainty and the reason a new capture is needed.

## Job execution and deployment boundary

Job metadata is persisted in SQLite/WAL and survives a process restart when the database is on persistent disk. The current job endpoint creates and retrieves queued metadata; it does not execute a production processing pipeline. Durable workers, leases, progress, retries, cancellation, checkpoints, multi-host coordination and shared object storage remain to be implemented.

Outside development, API-prefix routes require `X-API-Key` matching `IMAGE_CREATE_API_KEY`; the service fails closed if the key is missing. This is a shared service key, not per-user authentication or tenant isolation. Public multi-user deployment still requires identity, ownership/authorization, quotas, rate limits, retention/deletion, audit logs, monitoring and backups.
