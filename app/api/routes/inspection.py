from __future__ import annotations

import io
import json
import re
import warnings
from pathlib import Path
from uuid import UUID

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image, ImageOps

from app.core.config import settings

router = APIRouter(prefix="/analysis", tags=["visual inspection"])
STORED_IMAGE_PATTERN = re.compile(
    r"^\d{3}_[0-9a-f]{12}\.(jpg|png|webp|tif|bmp)$", re.IGNORECASE
)


def _images(analysis_id: UUID) -> list[Path]:
    folder = Path(settings.upload_dir) / str(analysis_id)
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail="Analysis upload was not found")
    paths = sorted(
        path for path in folder.iterdir()
        if path.is_file() and STORED_IMAGE_PATTERN.fullmatch(path.name)
    )
    if not paths:
        raise HTTPException(status_code=404, detail="No stored analysis images were found")
    return paths


def _mask_path(analysis_id: UUID, image_index: int) -> Path:
    return Path(settings.upload_dir).parent / "masks" / str(analysis_id) / f"{image_index:03d}.png"


@router.get("/{analysis_id}/inspection", response_class=HTMLResponse)
def inspection_workspace(analysis_id: UUID) -> HTMLResponse:
    _images(analysis_id)
    html = INSPECTION_HTML.replace("__ANALYSIS_ID__", str(analysis_id))
    return HTMLResponse(html)


@router.get("/{analysis_id}/images/{image_index}")
def get_inspection_image(analysis_id: UUID, image_index: int) -> FileResponse:
    paths = _images(analysis_id)
    if image_index < 0 or image_index >= len(paths):
        raise HTTPException(status_code=404, detail="Image index was not found")
    return FileResponse(paths[image_index])



@router.put("/{analysis_id}/calibration/{image_index}")
def save_camera_calibration(analysis_id: UUID, image_index: int, calibration: dict = Body(...)) -> dict:
    """Save explicit per-image pinhole intrinsics and optional OpenCV distortion coefficients."""
    paths = _images(analysis_id)
    if image_index < 0 or image_index >= len(paths):
        raise HTTPException(status_code=404, detail="Image index was not found")
    try:
        matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(calibration.get("distortion_coefficients", []), dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="camera_matrix must be a numeric 3x3 matrix") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise HTTPException(status_code=422, detail="camera_matrix must be finite and exactly 3x3")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or abs(matrix[2, 2] - 1.0) > 1e-6:
        raise HTTPException(status_code=422, detail="Camera focal lengths must be positive and K[2,2] must equal 1")
    if distortion.size not in {0, 4, 5, 8, 12, 14} or not np.isfinite(distortion).all():
        raise HTTPException(status_code=422, detail="OpenCV distortion coefficients must contain 0, 4, 5, 8, 12 or 14 finite values")
    with Image.open(paths[image_index]) as source:
        width, height = ImageOps.exif_transpose(source).size
    if not (0 <= matrix[0, 2] < width and 0 <= matrix[1, 2] < height):
        raise HTTPException(status_code=422, detail="Principal point must fall within the source image")
    payload = {
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "image_width": width,
        "image_height": height,
        "calibration_type": "user_supplied_pinhole",
    }
    target = paths[image_index].with_name(paths[image_index].name + ".calibration.json")
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "status": "saved", "image_index": image_index,
        "dimensions": {"width": width, "height": height},
        "intrinsics_source": "user_supplied_calibration",
        "distortion_coefficient_count": int(distortion.size),
        "note": "Rerun geometric matching to use this calibration.",
    }

@router.get("/{analysis_id}/masks/{image_index}")
def get_manual_mask(analysis_id: UUID, image_index: int) -> Response:
    paths = _images(analysis_id)
    if image_index < 0 or image_index >= len(paths):
        raise HTTPException(status_code=404, detail="Image index was not found")
    target = _mask_path(analysis_id, image_index)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="No manual mask has been saved for this image")
    return FileResponse(target, media_type="image/png")


@router.get("/{analysis_id}/automatic-masks/{image_index}")
def get_automatic_mask(analysis_id: UUID, image_index: int) -> Response:
    """Return the cached model-generated mask; never create a heuristic mask here."""
    paths = _images(analysis_id)
    if image_index < 0 or image_index >= len(paths):
        raise HTTPException(status_code=404, detail="Image index was not found")
    target = Path(settings.upload_dir).parent / "automatic_masks" / str(analysis_id) / f"{image_index:03d}.png"
    if not target.is_file():
        raise HTTPException(status_code=404, detail="No accepted automatic model mask is available; run matching with the segmentation extra installed")
    return FileResponse(target, media_type="image/png")


@router.put("/{analysis_id}/masks/{image_index}")
async def save_manual_mask(
    analysis_id: UUID,
    image_index: int,
    file: UploadFile = File(...),
) -> dict:
    paths = _images(analysis_id)
    if image_index < 0 or image_index >= len(paths):
        raise HTTPException(status_code=404, detail="Image index was not found")
    if file.content_type not in {"image/png", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Mask must be uploaded as PNG")
    raw = await file.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Mask exceeds 25 MB")
    with Image.open(paths[image_index]) as source:
        expected_size = ImageOps.exif_transpose(source).size
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                # Reject mismatched dimensions before decoding pixel data.
                if source.size != expected_size:
                    raise HTTPException(
                        status_code=422,
                        detail="Mask dimensions do not match the oriented source image",
                    )
                source.load()
                mask = source.convert("L")
    except HTTPException:
        raise
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=422, detail="Uploaded mask is not a valid safe image") from exc
    if mask.size != expected_size:
        raise HTTPException(
            status_code=422,
            detail=f"Mask dimensions {mask.size} do not match source image {expected_size}",
        )
    # Normalize antialiased brush edges to a binary foreground/background mask.
    binary = mask.point(lambda value: 255 if value >= 128 else 0, mode="L")
    target = _mask_path(analysis_id, image_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    binary.save(target, format="PNG", optimize=True)
    foreground_fraction = float(np.asarray(binary, dtype=np.uint8).mean() / 255.0)
    return {
        "status": "saved",
        "image_index": image_index,
        "filename": paths[image_index].name,
        "width": expected_size[0],
        "height": expected_size[1],
        "foreground_fraction": round(foreground_fraction, 4),
        "mask_convention": "white=object_foreground, black=background",
    }


INSPECTION_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Frame Intelligence — Visual Inspection</title>
<style>
:root{color-scheme:dark;font-family:Inter,Segoe UI,Arial,sans-serif;background:#10151d;color:#e9eef5}*{box-sizing:border-box}body{margin:0}header{padding:18px 24px;background:#182230;border-bottom:1px solid #344253}h1{font-size:20px;margin:0 0 5px}p{color:#aebdce;margin:4px 0 12px}.wrap{max-width:1500px;margin:auto;padding:20px}.tabs,.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}button,select,input{background:#263548;color:#f3f6fa;border:1px solid #53677e;border-radius:7px;padding:9px 12px}button{cursor:pointer}button.primary{background:#2878d0;border-color:#2878d0}button.active{outline:2px solid #68b5ff}label{font-size:13px;color:#bdcadd}.panel{background:#182230;border:1px solid #344253;border-radius:12px;padding:16px;margin:14px 0}.canvasbox{background:#0b1017;border:1px solid #3a4a5c;border-radius:8px;overflow:auto;max-height:72vh}canvas{display:block;max-width:100%;height:auto}.hint{font-size:13px;color:#aebdce}.status{padding:9px 12px;border-radius:7px;background:#223449;color:#cfe4ff;min-height:38px}.result-summary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:12px 0}.result-card{background:#111a25;border:1px solid #3a4a5c;border-radius:8px;padding:12px}.result-card h3{font-size:14px;margin:0 0 8px}.result-card p{font-size:13px;margin:4px 0;color:#c4d0dd}.result-card b{color:#f3f6fa}@media(max-width:650px){.result-summary{grid-template-columns:1fr}}.legend{display:flex;gap:16px;font-size:13px;color:#c4d0dd}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.hidden{display:none!important}.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,340px);gap:16px}@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Frame Intelligence · Visual Inspection</h1><p>CPU-only correspondence review and automatic subject masks</p></header>
<div class="wrap"><div class="status" id="status">Loading image set…</div>
<div class="tabs"><button class="active" id="tabMatches">Correspondence viewer</button><button id="tabMask">Subject mask</button></div>
<section id="matchesPanel" class="panel"><h2>Geometric correspondences</h2><div class="toolbar"><button class="primary" id="runMatching">Run / refresh matching</button><label>Image pair <select id="pairSelect"><option>Run matching first</option></select></label><label>Correspondences <select id="correspondenceMode"><option value="all">All-scene baseline</option><option value="object" selected>Object-only (manual masks)</option></select></label><label><input type="checkbox" id="showOutliers" checked> Show non-inliers</label></div>
<div id="resultSummary" class="result-summary"></div><div class="legend"><span><i class="dot" style="background:#39df8c"></i>RANSAC inlier</span><span><i class="dot" style="background:#ff6464"></i>RANSAC-tested outlier</span><span><i class="dot" style="background:#f2b84b"></i>Unverified candidate</span></div>
<div class="canvasbox"><canvas id="pairCanvas"></canvas></div><p class="hint">Green = RANSAC inlier; red = candidate tested and rejected by a fitted RANSAC model; amber = unverified because no usable model was fitted. Object-only mode independently reruns RANSAC using descriptor matches whose two endpoints are inside accepted automatic subject masks or optional manual corrections. The all-scene baseline and its connectivity graph remain separate.</p></section>
<section id="maskPanel" class="panel hidden"><h2>Automatic subject mask review · optional manual correction</h2><div class="grid"><div><div class="toolbar"><label>Image <select id="maskImage"></select></label><button id="loadMask">Load detected mask</button><button id="clearMask">Clear mask</button></div><div class="canvasbox"><canvas id="maskCanvas"></canvas></div></div><aside><div class="toolbar"><button class="active" id="paintTool">Paint object</button><button id="eraseTool">Erase</button></div><label>Brush size <input id="brushSize" type="range" min="3" max="120" value="28"></label><p class="hint">Review the automatically generated subject mask first. Paint or erase only when a correction is genuinely needed. You can zoom the browser to work on details. White = object; black = background.</p><button class="primary" id="saveMask">Save optional correction</button><p class="hint">Manual corrections are stored separately from the automatic mask and original photographs. Save at the source image's exact dimensions.</p></aside></div></section>
</div>
<script>
const id='__ANALYSIS_ID__', api='/api/v1/analysis/'+id;
const $=s=>document.querySelector(s); const status=m=>$('#status').textContent=m;
let report=null, imgs=[], selectedPair=null, activeTool='paint', drawing=false, maskDirty=false;
const pairCanvas=$('#pairCanvas'), pairCtx=pairCanvas.getContext('2d');
const maskCanvas=$('#maskCanvas'), maskCtx=maskCanvas.getContext('2d');
let maskImage=null, maskBase=null, maskOverlay=null;
async function loadImages(){const r=await fetch(api+'/matching',{method:'POST'});if(!r.ok)throw new Error('Matching request failed: '+r.status+' '+await r.text());report=await r.json();imgs=await Promise.all(report.images.map(async im=>{const image=new Image();image.src=api+'/images/'+im.image_index;await image.decode();return image}));
 const sel=$('#pairSelect');sel.innerHTML='';report.pair_results.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${p.filenames[0]} ↔ ${p.filenames[1]} · ALL-SCENE ${p.verified?'VERIFIED':'NOT VERIFIED'} · ${p.inlier_matches} inliers`;sel.append(o)});
 const m=$('#maskImage');m.innerHTML='';report.images.forEach(im=>{const o=document.createElement('option');o.value=im.image_index;o.textContent=im.filename;m.append(o)});
 if(report.pair_results.length){sel.value='0';drawPair()}await loadMaskEditor();status(`Matching complete · ${report.summary.verified_pairs} all-scene verified pairs · ${report.summary.connected_components} components · ${report.summary.isolated_images} isolated images`);}
function drawPair(){
 if(!report)return;const p=report.pair_results[Number($('#pairSelect').value)];if(!p)return;selectedPair=p;
 const objectMode=$('#correspondenceMode').value==='object';const data=objectMode?p.object_only:p;const correspondences=objectMode?(data?.diagnostic_correspondences||[]):p.diagnostic_correspondences;
 const a=imgs[p.image_indices[0]],b=imgs[p.image_indices[1]];const maxW=560,scale=Math.min(1,maxW/Math.max(a.naturalWidth,b.naturalWidth));
 const aw=Math.round(a.naturalWidth*scale),ah=Math.round(a.naturalHeight*scale),bw=Math.round(b.naturalWidth*scale),bh=Math.round(b.naturalHeight*scale),gap=36,head=28;
 pairCanvas.width=aw+bw+gap;pairCanvas.height=Math.max(ah,bh)+head;pairCtx.fillStyle='#0b1017';pairCtx.fillRect(0,0,pairCanvas.width,pairCanvas.height);
 pairCtx.fillStyle='#dce7f3';pairCtx.font='14px Segoe UI';pairCtx.fillText(p.filenames[0],0,19);pairCtx.fillText(p.filenames[1],aw+gap,19);pairCtx.drawImage(a,0,head,aw,ah);pairCtx.drawImage(b,aw+gap,head,bw,bh);
 const showOut=$('#showOutliers').checked;for(const c of correspondences){const s=c.match_status||(c.is_inlier?'inlier':(data?.model&&data.model!=='none'?'outlier':'unverified'));if(s!=='inlier'&&!showOut)continue;const x1=c.left.x*scale,y1=head+c.left.y*scale,x2=aw+gap+c.right.x*scale,y2=head+c.right.y*scale;const color=s==='inlier'?'rgba(57,223,140,.88)':s==='outlier'?'rgba(255,100,100,.62)':'rgba(242,184,75,.88)';pairCtx.strokeStyle=color;pairCtx.lineWidth=s==='inlier'?1.7:1.2;pairCtx.beginPath();pairCtx.moveTo(x1,y1);pairCtx.lineTo(x2,y2);pairCtx.stroke();pairCtx.fillStyle=color;pairCtx.beginPath();pairCtx.arc(x1,y1,2.5,0,Math.PI*2);pairCtx.fill();pairCtx.beginPath();pairCtx.arc(x2,y2,2.5,0,Math.PI*2);pairCtx.fill();}
 if(objectMode&&data?.status==='missing_automatic_or_manual_mask'){status(p.filenames.join(' ↔ ')+' · OBJECT-ONLY UNAVAILABLE · valid saved accepted automatic or manual masks required for both images.');$('#resultSummary').innerHTML='<div class="result-card"><h3>AUTOMATIC / MANUAL SUBJECT-MASK OBJECT-ONLY</h3><p>Install the segmentation extra or save an optional manual correction for each image, then refresh matching.</p></div>';return;}
 const fm=data?.feature_mask_diagnostics,base=data?.baseline_mask_comparison||data?.mask_filter_diagnostics;
 const objectCard=objectMode?'<div class="result-card"><h3>SUBJECT-MASK OBJECT-ONLY</h3><p>Masked SIFT features: <b>'+(data.feature_extraction?.left_object_features??0)+' left / '+(data.feature_extraction?.right_object_features??0)+' right</b></p><p>Raw descriptor matches: <b>'+(data.raw_descriptor_match_count??data.filtered_match_count??0)+'</b></p><p>Both endpoints inside masks: <b>'+(fm?.matches_with_both_endpoints_inside_masks??data.filtered_match_count??0)+'</b></p><p>RANSAC input: <b>'+(fm?.ransac_input_matches??data.tentative_matches??0)+'</b> · Inliers: <b>'+(data.inlier_matches??0)+'</b></p><p>Spatial support: <b>'+(data.spatial_distribution?.spatial_support||'not available')+'</b></p><p>Rejected by endpoint mask gate: <b>'+(fm?.rejected_for_mask_violation??0)+'</b></p></div>':'<div class="result-card"><h3>MANUAL-MASK OBJECT-ONLY</h3><p>Select Object-only to inspect independent masked matching.</p></div>';
 const baselineCard='<div class="result-card"><h3>ALL-SCENE BASELINE (CONNECTIVITY)</h3><p>Descriptor matches: <b>'+(p.tentative_matches??0)+'</b></p><p>RANSAC inliers: <b>'+(p.inlier_matches??0)+'</b></p><p>Pair status: <b>'+(p.verified?'VERIFIED':'NOT VERIFIED')+'</b></p><p>Baseline endpoints inside both masks: <b>'+(base?.eligible_by_both_masks??'n/a')+'/'+(base?.total_descriptor_matches??'n/a')+'</b></p><p>These mask counts describe baseline descriptors, not object-only features.</p></div>';
 $('#resultSummary').innerHTML=objectMode?objectCard+baselineCard:baselineCard+objectCard;
 const reason=data?.reason==='insufficient_matches'?' · not enough candidates for RANSAC':data?.reason==='ransac_failed'?' · RANSAC could not fit a model':data?.reason==='insufficient_inliers'?' · too few inliers':data?.reason==='low_inlier_ratio'?' · low inlier ratio':'';
 status(p.filenames.join(' ↔ ')+' · '+(objectMode?'OBJECT-ONLY MANUAL-MASK RANSAC':'ALL-SCENE BASELINE')+' · '+(data?.verified?'VERIFIED':'NOT VERIFIED')+' · '+(data?.tentative_matches||0)+' RANSAC-input matches · '+(data?.inlier_matches||0)+' inliers'+reason);
}
function renderMask(){if(!maskImage)return;const maxW=900,scale=Math.min(1,maxW/maskImage.naturalWidth);const w=Math.round(maskImage.naturalWidth*scale),h=Math.round(maskImage.naturalHeight*scale);if(maskCanvas.width!==w||maskCanvas.height!==h){maskCanvas.width=w;maskCanvas.height=h;}maskCtx.clearRect(0,0,w,h);maskCtx.drawImage(maskImage,0,0,w,h);maskCtx.drawImage(maskOverlay,0,0,w,h);}
async function loadMaskEditor(){const idx=Number($('#maskImage').value||0);maskImage=imgs[idx];maskOverlay=document.createElement('canvas');maskOverlay.width=maskImage.naturalWidth;maskOverlay.height=maskImage.naturalHeight;const ctx=maskOverlay.getContext('2d');ctx.clearRect(0,0,maskOverlay.width,maskOverlay.height);
 try{let r=await fetch(api+'/masks/'+idx);let sourceLabel='manual correction';if(!r.ok){r=await fetch(api+'/automatic-masks/'+idx);sourceLabel='automatic subject mask'}if(r.ok){const saved=new Image();saved.src=URL.createObjectURL(await r.blob());await saved.decode();const temp=document.createElement('canvas');temp.width=maskOverlay.width;temp.height=maskOverlay.height;const tc=temp.getContext('2d');tc.drawImage(saved,0,0);const pixels=tc.getImageData(0,0,temp.width,temp.height);for(let i=0;i<pixels.data.length;i+=4){const fg=pixels.data[i]>=128;pixels.data[i]=255;pixels.data[i+1]=255;pixels.data[i+2]=255;pixels.data[i+3]=fg?180:0;}ctx.putImageData(pixels,0,0);status('Loaded '+sourceLabel+'.')}}catch(e){}maskDirty=false;renderMask();}
function paintAt(e){const rect=maskCanvas.getBoundingClientRect(),sx=maskOverlay.width/rect.width,sy=maskOverlay.height/rect.height,x=(e.clientX-rect.left)*sx,y=(e.clientY-rect.top)*sy,r=Number($('#brushSize').value)*sx/2;const c=maskOverlay.getContext('2d');c.save();c.globalCompositeOperation=activeTool==='paint'?'source-over':'destination-out';c.fillStyle='rgba(255,255,255,1)';c.beginPath();c.arc(x,y,r,0,Math.PI*2);c.fill();c.restore();maskDirty=true;renderMask();}
maskCanvas.addEventListener('pointerdown',e=>{drawing=true;maskCanvas.setPointerCapture(e.pointerId);paintAt(e)});maskCanvas.addEventListener('pointermove',e=>{if(drawing)paintAt(e)});maskCanvas.addEventListener('pointerup',()=>drawing=false);maskCanvas.addEventListener('pointercancel',()=>drawing=false);
$('#runMatching').onclick=async()=>{try{status('Running CPU feature matching…');await loadImages()}catch(e){status(e.message)}};
$('#pairSelect').onchange=drawPair;$('#correspondenceMode').onchange=drawPair;$('#showOutliers').onchange=drawPair;$('#maskImage').onchange=loadMaskEditor;
$('#paintTool').onclick=()=>{activeTool='paint';$('#paintTool').classList.add('active');$('#eraseTool').classList.remove('active')};$('#eraseTool').onclick=()=>{activeTool='erase';$('#eraseTool').classList.add('active');$('#paintTool').classList.remove('active')};
$('#clearMask').onclick=()=>{maskOverlay.getContext('2d').clearRect(0,0,maskOverlay.width,maskOverlay.height);maskDirty=true;renderMask()};
$('#loadMask').onclick=loadMaskEditor;
$('#saveMask').onclick=async()=>{const idx=Number($('#maskImage').value);const out=document.createElement('canvas');out.width=maskOverlay.width;out.height=maskOverlay.height;const c=out.getContext('2d');c.fillStyle='#000';c.fillRect(0,0,out.width,out.height);c.drawImage(maskOverlay,0,0);const blob=await new Promise(resolve=>out.toBlob(resolve,'image/png'));const fd=new FormData();fd.append('file',blob,`mask_${idx}.png`);const r=await fetch(api+'/masks/'+idx,{method:'PUT',body:fd});const data=await r.json();if(!r.ok){status(data.detail||'Mask save failed');return}maskDirty=false;status(`Saved manual mask for ${data.filename} · ${Math.round(data.foreground_fraction*100)}% foreground. Run matching again to use it.`)};
$('#tabMatches').onclick=()=>{$('#matchesPanel').classList.remove('hidden');$('#maskPanel').classList.add('hidden');$('#tabMatches').classList.add('active');$('#tabMask').classList.remove('active')};$('#tabMask').onclick=()=>{$('#maskPanel').classList.remove('hidden');$('#matchesPanel').classList.add('hidden');$('#tabMask').classList.add('active');$('#tabMatches').classList.remove('active');renderMask()};
loadImages().catch(e=>status(e.message));
</script></body></html>'''
