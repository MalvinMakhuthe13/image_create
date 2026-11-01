# Spatial diagnostics and matching safeguards

## Correspondence uniqueness

Raw RANSAC inliers are preserved and reported. A second, advisory unique-location count retains the best descriptor-distance correspondence when another correspondence is within 5 pixels in BOTH images. The exact 4D distance check is accelerated with a 4D spatial hash. The radius and inclusive threshold remain 5.0 pixels.

The raw inlier spatial concentration warning is calculated before deduplication. This prevents repeated detections from collapsing the sample below the warning threshold and hiding a concentrated cluster. Unique-location grids are reported separately. Object-only manual-mask results receive their own spatial report, computed only from the filtered object-only matches and their own RANSAC inliers.

## Per-image duplicate-location count

For each image, `near_duplicate_location_groups_5px` counts occupied 5-pixel buckets containing multiple inlier locations. This is a coarse bucket statistic, not an exact Euclidean cluster count.

## Synthetic equivalence/performance check

A local randomized comparison over 500 generated correspondence sets produced identical retained correspondence indices for the original pairwise implementation and the spatial-hash implementation. A separate synthetic 4,000-correspondence run retained the same 2,381 unique correspondences and measured approximately 0.68 s for pairwise comparison versus 0.05 s for spatial hashing (about 14x faster) in that local environment.

These are synthetic microbenchmarks, not a guarantee for every capture set or machine. Real-world performance and matching precision must still be benchmarked on representative photographs.

## Interpretation

Spatial distribution is 2D image-space evidence only. It does not prove 3D surface coverage, calibrated pose, or reconstruction quality. Neither raw nor unique inlier counts are allowed to change geometric verification or all-scene graph connectivity.
