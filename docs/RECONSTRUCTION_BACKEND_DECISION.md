# Dense reconstruction backend decision

**Status:** research completed; implementation is deliberately gated on licensing and benchmark evidence.

## Requirements

A backend for this product must support:

- Sparse camera registration and robust pose optimization.
- CPU-first dense depth estimation and depth-map fusion.
- Surface reconstruction, mesh refinement and texturing.
- Explicit hardware capability detection and bounded resource use.
- A license compatible with the intended distribution and hosted-service model.
- Reproducible outputs and source-image provenance.

A backend that only returns sparse landmarks does not satisfy the dense reconstruction requirement.

## COLMAP

COLMAP is a strong reference implementation for structure-from-motion and multi-view stereo. Its project license is BSD; third-party dependency licenses still need to be reviewed for a particular build.

Its CPU mode supports feature extraction, matching and sparse mapping. Its documented dense PatchMatch stereo pipeline requires a CUDA-capable build/GPU; current releases also describe ROCm/HIP acceleration on supported AMD GPUs. This does not provide a general CPU-only dense path.

References:
- https://colmap.github.io/cli.html
- https://colmap.github.io/faq.html
- https://github.com/colmap/colmap/blob/main/doc/license.rst
- https://github.com/colmap/colmap/blob/main/CHANGELOG.rst

## OpenMVS

OpenMVS provides a broad pipeline: native SfM, dense point clouds, mesh reconstruction, mesh refinement and texturing. Its usage documentation describes a CPU Semi-Global Matching route for dense point-cloud generation using `DensifyPointCloud --fusion-mode -2`. CUDA acceleration is optional in its build architecture.

However, the current OpenMVS repository identifies its license as **AGPL-3.0**. That is a material distribution and hosted-service consideration. It must not be silently bundled into a closed product or deployed as a network service without a deliberate compliance decision and legal review. The app does not copy or vendor OpenMVS code.

References:
- https://github.com/cdcseacave/openMVS
- https://github.com/cdcseacave/openMVS/blob/develop/COPYRIGHT.md
- https://github.com/cdcseacave/openMVS/blob/develop/docs/wiki/Usage.md
- https://github.com/cdcseacave/openMVS/blob/develop/docs/architecture.md

## Current decision

1. Keep the in-repository CPU analysis and sparse geometry foundation independent of third-party engine licensing.
2. Use COLMAP as a benchmark/reference for camera registration and sparse reconstruction; do not represent it as a CPU dense solution.
3. Keep OpenMVS as a candidate CPU dense backend, but do not vendor, download or automatically execute it until the product's AGPL compliance path is explicitly approved.
4. Do not generate synthetic depth, point clouds, meshes or coverage values as placeholders.
5. Keep dense reconstruction and surface coverage marked unavailable/not assessed until an engine is selected, integrated, tested and benchmarked.

## Acceptance tests before selecting a backend

- Camera registration success and failure rates on known datasets.
- Reprojection error and camera trajectory consistency.
- Dense depth accuracy and completeness against ground truth.
- Point-cloud outlier rate and surface completeness.
- Mesh topology, holes, non-manifold edges and texture reprojection.
- CPU runtime, peak RAM, disk footprint and restart behaviour.
- License and third-party dependency review for both desktop and hosted deployment.
