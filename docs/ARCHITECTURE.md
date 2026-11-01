# Architecture — v0.1

## Components
- FastAPI control plane: authentication integration, job creation/status, signed upload URLs.
- Ingestion: immutable source registration, file validation, hashes, EXIF and quality analysis.
- Geometry: COLMAP and learned reconstruction adapters; canonical camera convention.
- Coverage: observed / inferred / unknown surface evidence and visibility maps.
- Planner: adaptive camera sampling, expected information gain, redundancy control and stopping.
- Verified renderer: render from an existing reconstruction only.
- Experimental generator: AI-generated views, explicitly synthetic.
- Validator: identity, camera, geometry, texture, provenance and quality checks.
- Export: original-only default; controlled experiments in separate folders.

## Evidence classes
1. `observed`: supported by one or more original photographs.
2. `reconstructed`: estimated by the geometry pipeline from observed evidence.
3. `inferred`: model-derived estimate with uncertainty.
4. `unknown`: unsupported or unobserved.

Do not collapse these classes into one coverage percentage.

## Camera convention
Use a versioned, explicit coordinate convention. Store camera intrinsics separately from
camera-to-world extrinsics. Convert explicitly at COLMAP/RealityScan boundaries. Never
assume metric scale without calibration.

## Job lifecycle
`queued -> analysing -> geometry -> planning -> generating -> validating -> exporting -> complete`
Terminal alternatives: `partial`, `failed`, `cancelled`.

Workers must use leases, heartbeats, bounded retries, idempotent task keys and immutable
artifact outputs. A failed experimental stage must not invalidate verified outputs.

## Storage
Store object keys, checksums, MIME types, dimensions, source lineage, model version, seed,
camera metadata and validation status in the asset manifest. Keep original image bytes
immutable. Do not put secrets or permanent public URLs in manifests.
