# Automatic subject segmentation

## Intended behaviour

Object-only matching first uses a valid user correction, when one exists. Otherwise it attempts a cached U2-Net mask generated with rembg and ONNX Runtime's CPU execution provider. The mask is stored separately under `data/automatic_masks/<analysis-id>/`; original uploaded photographs are never rewritten.

Install the optional model dependencies:

```powershell
python -m pip install -e ".[dev,segmentation]"
```

The first model inference may download U2-Net weights and can take longer than subsequent images. The inference session is reused within the worker process. CPU inference is supported; no CUDA/GPU is required.

## Safety and fallback rules

- A U2-Net output is a model prediction, not ground truth. It may miss thin details, transparent/reflective parts, or include nearby background.
- Basic plausibility diagnostics include foreground fraction, connected-component count and largest-component fraction. These are sanity checks, not a learned confidence score or a guarantee of a complete silhouette.
- If the optional model is absent or fails, GrabCut may produce a diagnostic fallback, but that fallback is labelled low confidence and is **not** accepted for object-only RANSAC.
- If neither an accepted automatic model mask nor a manual correction exists for either image, object-only matching reports that the mask is unavailable. All-scene matching and connectivity continue independently.
- Manual corrections override automatic masks. They are saved separately and do not alter the cached automatic mask or original photo.
- Cached masks are tied to the immutable uploaded image set. Re-uploading creates a new analysis ID and new cache namespace.

## Review workflow

1. Install the segmentation extra.
2. Upload a capture set and run matching once. This generates and caches model masks.
3. Open the inspection workspace and choose **Automatic subject mask review**.
4. Load the detected mask to inspect the model result. Paint/erase and save only if a correction is needed.
5. Re-run matching to use the manual correction; otherwise the accepted automatic mask is used automatically.

## Current limitations

This is a single-image U2-Net foreground model applied independently to each frame. It does not yet perform cross-frame object identity tracking, temporal mask propagation, multi-view silhouette consistency, or automatic recovery of a subject when U2-Net fails. Those require a separately validated sequence-level stage. The current quality metrics do not certify that every accessory or narrow protrusion is retained.
