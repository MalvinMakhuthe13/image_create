# image_create — FUMOCA Frame Intelligence

Standalone application for sparse-image analysis, geometry-backed viewpoint rendering,
experimental novel-view generation, validation, and controlled dataset export.

## Non-negotiable rules
- Original photographs are immutable.
- Verified renders are virtual views of existing geometry, not new physical observations.
- Experimental AI views are synthetic and excluded from RealityScan production exports by default.
- Unknown surfaces remain unknown; no confidence inflation.
- Every asset records provenance, hashes, camera metadata, and model versions where applicable.

## Quick start (Python 3.11+)
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
pytest
```
API docs: http://127.0.0.1:8000/docs

## CPU Image Analysis v1
`POST /api/v1/analysis/images` accepts multiple JPEG, PNG, WebP, TIFF, and BMP photographs.
It preserves original bytes locally and reports dimensions, basic EXIF metadata, heuristic
sharpness/exposure metrics, SHA-256 exact duplicates, and perceptual-hash near-duplicates.

This stage runs on CPU and does not require CUDA or a GPU. It does not yet estimate camera
poses, geometric feature matches, camera overlap, missing viewpoints, or true 3D coverage.
The upload endpoint is unauthenticated and intended for localhost development only.
See `README_CPU_ANALYSIS.md` for setup and test instructions.
