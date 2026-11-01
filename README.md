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
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
pytest
```
API docs: http://127.0.0.1:8000/docs

## Initial scope
This starter implements the API foundation, typed job contracts, adaptive planner interface,
health endpoint, and unit tests. It deliberately does not pretend that geometry reconstruction
or image generation is already implemented.
