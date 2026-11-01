# Roadmap

## Milestone 1 — Foundation
- API and schemas
- authentication and persistent job storage
- resumable upload and immutable source registry
- image validation, hashes, EXIF and quality metrics

## Milestone 2 — Geometry and evidence
- COLMAP adapter
- camera convention tests
- geometry adapter interfaces
- surface evidence/confidence representation

## Milestone 3 — Adaptive viewpoint planning
- spherical candidate sampling
- visibility and occlusion scoring
- information-gain ranking
- adaptive refinement and stopping

## Milestone 4 — Verified renderer
- camera-accurate geometry rendering
- texture provenance
- per-pixel confidence/evidence masks

## Milestone 5 — Experimental and hybrid generation
- model registry and benchmark harness
- geometry-conditioned generation
- synthetic-region detection and review
- separate provenance and approval

## Milestone 6 — Studio and export
- coverage heatmap and camera preview
- side-by-side frame review
- RealityScan-safe export profiles
- FUMOCA integration API

No milestone is complete merely because a model produces an image. Acceptance requires
repeatable benchmark results and preserved provenance.
