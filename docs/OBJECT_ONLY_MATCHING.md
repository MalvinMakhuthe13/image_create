# Object-only masked matching

The visual inspection workspace supports two separate correspondence modes:

- **All-scene baseline**: the original SIFT descriptor matching and geometric verification. The existing connectivity graph continues to use this result.
- **Object-only (automatic subject mask)**: masked CLAHE-SIFT features are extracted independently inside each accepted U2-Net automatic mask, unless a user-supplied manual correction exists. A match is eligible only when BOTH original-image keypoint endpoints fall inside their respective accepted masks.

## Automatic masks and optional manual corrections

Manual masks use white for object foreground and black for background. Each saved mask must match its source image dimensions.

Install the optional segmentation dependencies with `python -m pip install -e ".[dev,segmentation]"`. U2-Net runs through ONNX Runtime's CPU provider and generated masks are cached separately from source images. Manual masks override generated masks. If the model is unavailable, GrabCut is reported as a low-confidence heuristic and is not accepted for object-only RANSAC. If either image has no accepted automatic or manual mask, object-only verification reports `missing_automatic_or_manual_mask`. See `docs/AUTOMATIC_SUBJECT_SEGMENTATION.md` for setup, quality diagnostics and limitations.

## Mask boundary-distance diagnostics

For pairs with two valid accepted masks, the system also measures how far excluded descriptor-match endpoints lie from the nearest foreground pixel.

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
