# Object-only masked matching

The visual inspection workspace supports two separate correspondence modes:

- **All-scene baseline**: the original SIFT descriptor matching and RANSAC pass. The existing connectivity graph continues to use this result.
- **Object-only (masked)**: a second pass that first removes every SIFT keypoint outside each image's white foreground mask, then performs descriptor matching and RANSAC using only the retained descriptors.

Manual masks use white for object and black for background. A saved mask must match the source image dimensions. If a mask is absent, the current CPU GrabCut foreground proposal is used and is explicitly heuristic; it may include the table/floor or omit parts of the object.

The inspection page has a correspondence-mode selector. Select **Object-only (masked)** to review whether verified lines remain on the masked object. If the object-only pass has too few features or inliers, it reports an unverified result rather than silently falling back to background matches.

The all-scene baseline remains the source for graph connectivity for now. Do not interpret object-only pair verification as a calibrated camera pose, 3D point cloud, or proof of complete surface coverage. The RANSAC thresholds have not been relaxed.
