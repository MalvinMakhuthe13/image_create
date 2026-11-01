# CPU geometric matching integration

This patch adds a separate endpoint and does not change the existing image-upload route.

## Endpoint

After uploading images with `POST /api/v1/analysis/images`, pass the returned
`analysis_id` to:

`POST /api/v1/analysis/{analysis_id}/matching`

The endpoint reads the preserved local upload files, extracts SIFT features once per
image, performs bounded pair comparisons, verifies candidate matches with RANSAC,
and returns pair results plus a connectivity graph.

## CPU limits

Settings in `app/core/config.py`:

- `max_matching_pair_checks`: default 250
- `matching_max_dimension`: default 1200
- `matching_max_features`: default 3000

Adjacent pairs are prioritised, followed by evenly sampled remaining pairs. The
configured pair limit must allow at least `image_count - 1` comparisons so every
adjacent pair can be checked.

## Interpretation

A verified pair is evidence of geometric consistency between two photographs. It
does not provide calibrated camera poses, metric scale, dense reconstruction, or
proof of complete 3D surface coverage. Pair selection is bounded, so unchecked
pairs may contain additional valid links.

## Test

Run:

```powershell
ruff check app tests
pytest -q
```
