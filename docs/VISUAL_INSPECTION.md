# Visual correspondence inspector and manual object masks

Open `http://127.0.0.1:8000/api/v1/analysis/{analysis_id}/inspection` after starting the local API.

The browser workspace provides:

- A side-by-side correspondence viewer with green RANSAC inliers and red sampled rejected candidates.
- A manual object-mask editor with paint and erase tools.
- PNG mask persistence under `data/masks/{analysis_id}/{image_index:03d}.png`.
- Manual masks take precedence over automatic GrabCut proposals during the next matching run.

Mask convention: white (>=128) is object foreground; black is background. The mask must have exactly the same pixel dimensions as its source image. The editor preserves the original image and saves a separate mask file.

The mask support is diagnostic only. It does not change SIFT descriptors, RANSAC thresholds, geometric verification, or connectivity edges. A mask is only as accurate as the user's painting. Coverage remains `not_assessed`; 2D correspondences do not prove complete 3D surface coverage.

Everything runs locally on CPU. No GPU or external service is required.
