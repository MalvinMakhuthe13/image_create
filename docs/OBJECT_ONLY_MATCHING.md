# Object-only masked matching

The visual inspection workspace supports two separate correspondence modes:

- **All-scene baseline**: the original SIFT descriptor matching and geometric verification. The existing connectivity graph continues to use this result.
- **Object-only (manual-mask)**: descriptor matches are first generated using the existing all-scene feature matching process. A match is eligible for object-only geometric verification only when BOTH original-image keypoint endpoints fall inside their respective saved manual foreground masks.

## Manual masks

Manual masks use white for object foreground and black for background. Each saved mask must match its source image dimensions.

A valid saved manual mask is required for both images in a pair. If either mask is missing or invalid, object-only verification reports `missing_manual_mask`. There is no automatic GrabCut fallback in this workflow.

## Mask boundary-distance diagnostics

For pairs with two valid manual masks, the system also measures how far excluded descriptor-match endpoints lie from the nearest foreground pixel.

Distances are calculated independently for the left and right images using OpenCV's precise Euclidean distance transform. The correspondence-level histogram counts a correspondence once, using the smaller measurable distance of its two endpoints.

Distance bands:

- 0?2 pixels
- Over 2?5 pixels
- Over 5?10 pixels
- Over 10?20 pixels
- Over 20 pixels

Out-of-image endpoints, invalid keypoint indices and masks with no foreground pixels are reported separately where applicable.

These measurements are advisory only. They do not expand masks, change endpoint eligibility, admit excluded matches into RANSAC, relax geometric thresholds or change connectivity.

## Interpretation and limitations

The object-only RANSAC pass receives only original descriptor matches for which both endpoints fall inside their saved manual masks.

The all-scene baseline remains the source for graph connectivity. Do not interpret pair verification or boundary-distance measurements as calibrated camera poses, a reconstructed 3D point cloud, or proof of complete object-surface coverage.

The boundary-distance diagnostic explains how close excluded matches are to the manually defined foreground. It does not determine whether those matches are genuine object correspondences.
