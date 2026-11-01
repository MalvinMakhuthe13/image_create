# Subject-region diagnostics (CPU-only)

The matching API now adds `subject_region_diagnostics` and a
`subject_region_support` object to each pair result.

## Method

- A low-resolution OpenCV GrabCut proposal is generated per image.
- SIFT keypoints are classified as inside/outside that proposal.
- Verified pair correspondences are counted as both-foreground, both-background,
  or mixed-region.
- A warning is emitted when at least eight geometric inliers exist and more
  verified correspondences fall outside the proposed foreground in both images
  or across mixed regions than inside both proposed foreground regions.

## Important limitations

This is **not semantic object segmentation**. GrabCut may include a tabletop,
stand, floor-adjacent region, or other connected background as foreground. It
may also omit parts of the object. The output is a review cue only.

It does not:
- alter SIFT descriptors, RANSAC thresholds, geometric verification, or graph edges;
- establish that matches lie on the intended physical object;
- estimate camera poses or prove 3D surface coverage;
- replace a manually supplied subject mask.

For reliable object-only scoring, the next stage should support user-provided
binary masks (or a reviewed/edited mask) and render correspondence overlays
against those masks. Until then, interpret the automatic region score as
foreground-proposal support, not object support.

## Validation

The full test suite passed locally in the build environment: 21 tests.
The sample seven-image matching API request completed in CPU mode in
approximately 24 seconds. RANSAC settings were not changed.
