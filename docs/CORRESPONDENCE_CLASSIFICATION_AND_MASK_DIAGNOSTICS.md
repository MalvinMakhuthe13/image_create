# Correspondence classification and manual-mask diagnostics

## Correspondence states

Every diagnostic correspondence now includes `match_status`:

- `inlier`: included by the selected RANSAC model.
- `outlier`: evaluated by a fitted RANSAC model and excluded by its inlier mask.
- `unverified`: no usable geometric model was fitted (`model == "none"`).

Pair-level verification remains separate from correspondence-level classification. A pair can fail its minimum-inlier or inlier-ratio threshold while still having individually classified RANSAC inliers and outliers.

## Manual-mask diagnostics

For each pair with two valid saved manual masks, `object_only.mask_filter_diagnostics` reports:

- total descriptor matches;
- candidates admitted independently by the left and right masks;
- candidates admitted by both masks;
- left-inside/right-outside, left-outside/right-inside, and outside-both counts;
- foreground fractions for both masks.

The counts use the existing exact keypoint-centre test in original-image pixel coordinates. They do not expand masks, alter descriptor matching, change RANSAC thresholds, or establish 3D surface coverage.

Each image also reports `manual_mask.status`, `foreground_fraction`, `foreground_pixels`, and `total_pixels`. Missing or dimension-invalid masks are reported as `missing_or_invalid`.

## Preserved behaviour

- All-scene matching and its connectivity graph are unchanged.
- Object-only RANSAC remains independent and requires valid manual masks for both images.
- Automatic GrabCut proposals are not substituted for missing manual masks.
- Existing RANSAC thresholds are unchanged.
