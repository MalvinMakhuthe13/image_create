# Geometry, camera pose and coverage: current contract

## Pairwise geometric verification

The CPU matcher extracts SIFT descriptors and applies mutual Lowe-ratio matching. Each candidate pair is evaluated with both a RANSAC fundamental matrix and a RANSAC homography. The response reports each model's inlier count, the selected model, a Sampson residual for a selected fundamental matrix or symmetric transfer residual for a selected homography, and 4x4 image-space support diagnostics.

A homography-dominant result is flagged as possibly planar or low-parallax. Concentrated inliers are rejected as a verified connectivity edge. These are conservative diagnostics, not a complete degeneracy classifier. A homography can also fit a rotating camera, a distant scene, or a small field of view; a fundamental matrix can be poorly conditioned. Treat the warnings as evidence to review, not a semantic scene label.

## Relative camera pose

When both images contain EXIF 35mm-equivalent focal length, the feature extractor derives an approximate pixel focal length from the image diagonal and assumes a centred principal point. This is an approximation, not a camera calibration: lens distortion, crop, digital zoom, sensor geometry and manufacturer metadata can make it inaccurate.

When intrinsics are available, the system estimates an essential matrix and recovers pairwise relative rotation and translation direction. Translation magnitude is unknown from a monocular pair. If EXIF intrinsics were used, the response labels calibration quality as approximate.

When intrinsics are missing, relative pose is explicitly returned as unavailable. The system does not silently invent a focal length.

## Sparse triangulation

If relative pose is recoverable, inlier correspondences are triangulated into a pairwise sparse point set. The response reports positive-depth point count and median reprojection error. These points have arbitrary monocular scale and are not fused into a global reconstruction.

## What is not yet 3D coverage

A pairwise sparse point cloud is not a dense surface, mesh, watertight model, or visibility-complete scene. The current system does not yet:
- optimize all camera poses jointly with bundle adjustment;
- reconcile scale and drift across a multi-image pose graph;
- fuse tracks observed in three or more images into globally consistent 3D landmarks;
- estimate dense depth, surface normals, occlusion or unseen surface regions;
- calculate object-surface coverage percentages or certify underside/rear visibility.

Accordingly, the API must continue to report `coverage_assessment.status = not_assessed`. Do not translate pairwise match counts or sparse triangulation counts into a percentage of object coverage.

## Capture advice contract

The current API can identify weak pairwise links, blur/exposure concerns, near duplicates and missing geometric support. It must not claim that a particular physical surface is missing until global camera geometry and surface visibility have been computed. Future capture advice should cite the supporting views, predicted visibility gain, and uncertainty.

## Deployment safety

Development mode remains convenient for localhost use. Outside development, all API-prefix routes require `X-API-Key` matching the `IMAGE_CREATE_API_KEY` environment variable. The service fails closed with HTTP 503 if the environment is non-development and the key is absent.

This is a shared service key, not user authentication or tenant isolation. Before public multi-user deployment, add per-user identity, authorization and analysis ownership, quotas, rate limiting, persistent object storage, background workers, retention policies, and audit logs. Do not expose the local upload directory as a public static directory.

## Job persistence boundary

Job metadata is persisted in a local SQLite database using WAL mode and a busy timeout. This survives API process restarts on a single host with persistent disk. It does not yet provide distributed locking, worker leases, retry scheduling, progress events, or multi-host coordination. Production processing still requires a dedicated worker queue and durable shared storage.


## Explicit camera calibration and sparse global optimization

A per-image calibration can be supplied with:
`PUT /api/v1/analysis/{analysis_id}/calibration/{image_index}`

Example JSON:
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
The matrix and distortion coefficients must come from a real calibration for the same image dimensions and camera/lens configuration. The endpoint validates dimensions, focal lengths, principal point and coefficient count. Re-run matching after supplying calibration.

When calibrated pairwise pose edges are available, the CPU sparse reconstruction stage:
1. Builds connected camera components.
2. Initializes component camera transforms from pairwise relative poses.
3. Joins pairwise inlier observations into multi-view feature tracks.
4. Triangulates sparse landmarks.
5. Runs robust SciPy least-squares bundle adjustment over camera rotations, translations and 3D landmarks, fixing a reference camera and constraining one baseline to remove gauge freedom.

This is an initial sparse global optimization, not a mature production SfM system. Component scales remain arbitrary; approximate EXIF intrinsics remain approximate; track association can contain errors; the bounded optimizer may stop before convergence. Inspect component status and reprojection residuals before trusting camera poses. No dense reconstruction or surface-coverage percentage is inferred.


Pose-seeding safeguard: homography-selected or homography-dominant/planar-ambiguous pairs do not seed the global camera pose graph. Relative pose is attempted from fundamental-matrix inliers only. This intentionally reduces the number of usable pose edges rather than allowing planar or low-parallax ambiguity to contaminate the global reconstruction.
