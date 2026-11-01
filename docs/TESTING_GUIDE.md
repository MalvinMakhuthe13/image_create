# Local testing guide

This guide tests the capabilities implemented today. It does not imply that dense reconstruction, 3D coverage or final mesh export is complete.

## 1. Start the application

**Windows PowerShell** (from the repository root):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\run_local.ps1
```

**Linux / macOS**:

```bash
chmod +x scripts/run_local.sh
./scripts/run_local.sh
```

Open the Capture Studio at http://127.0.0.1:8000/ and API reference at http://127.0.0.1:8000/docs.

## 2. Prepare a useful first capture

Use one stationary, textured object and stable lighting.

- Capture a complete circuit with substantial overlap between neighbouring photographs.
- Add higher and lower camera elevations; one horizontal ring does not establish top or underside visibility.
- Keep the object still. Do not rotate it between shots unless the capture method accounts for that.
- Avoid motion blur, changing zoom, exposure shifts and moving shadows.
- Glossy, transparent, reflective, plain and repetitive surfaces are difficult; test them as separate stress cases.
- Preserve original images. Do not mix AI-generated views, renders or edited composites into the source set.

These are capture suggestions, not guarantees.

## 3. Test the implemented workflow

1. Select original photographs in the browser Capture Studio.
2. Upload and review dimensions, image-quality flags, exact duplicates and near-duplicate candidates.
3. Open the visual inspection workspace.
4. Run geometric matching.
5. Review connected components, verified pairs, correspondence spread and sparse reconstruction diagnostics.
6. Paint and save manual object masks on both images in a pair.
7. Switch to object-only matching. Only matches with both endpoints inside the saved masks should enter object-only RANSAC.
8. If you have genuine camera calibration for the same camera, lens and image configuration, enter it. Do not guess calibration values.
9. Repeat with difficult captures: blur, repeated textures, planar scenes, low parallax, missing transitions and disconnected viewpoints.

## 4. Interpret results correctly

- Image similarity is not geometric verification.
- A RANSAC inlier is not a camera pose.
- A camera pose is not a dense surface.
- Sparse landmarks are not a coverage percentage.
- EXIF-derived focal estimates are approximate, not calibrated intrinsics.
- Disconnected camera components have independent arbitrary coordinate frames and scales.
- The API must report `coverage_assessment.status = not_assessed`.

## 5. Automated verification

```powershell
.\.venv\Scripts\python.exe -m ruff check app tests --select E4,E7,E9,F
.\.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions also tests Python 3.11 and 3.12.

## 6. Not yet available for real-world acceptance

The app does not yet run a production dense multi-view stereo pipeline, fuse dense depth maps, generate and validate a final textured mesh, compute occlusion-aware surface coverage, or execute durable background reconstruction jobs. It is suitable for testing image intake, matching, masking, calibration and sparse-geometry foundations—not for treating its output as a finished photogrammetry deliverable.
