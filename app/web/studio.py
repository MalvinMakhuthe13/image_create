from __future__ import annotations

STUDIO_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>image_create · Capture Studio</title>
<style>
:root{color-scheme:dark;font-family:Inter,Segoe UI,Arial,sans-serif;background:#0b1119;color:#edf3fb;--panel:#141f2c;--line:#2a3a4d;--muted:#a7b7ca;--accent:#55a6ff}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(ellipse at 12% 0%,#1c3148 0,transparent 42%),#0b1119}
header{border-bottom:1px solid var(--line);background:#101a26d9;padding:22px max(22px,calc((100vw - 1180px)/2));display:flex;align-items:center;justify-content:space-between;gap:16px}
.brand{font-size:20px;font-weight:750;letter-spacing:.2px}.brand span{color:var(--accent)}.tag{font-size:12px;color:#b8c9dc;border:1px solid #3b5068;border-radius:30px;padding:6px 10px}
main{max-width:1180px;margin:0 auto;padding:38px 22px 70px}.intro{max-width:760px;margin-bottom:28px}.eyebrow{color:#79baff;text-transform:uppercase;letter-spacing:1.8px;font-size:11px;font-weight:700}h1{font-size:clamp(30px,5vw,48px);line-height:1.08;margin:12px 0}p{color:var(--muted);line-height:1.6}.panel{background:linear-gradient(145deg,#182536,#121c28);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:0 18px 48px #0003}
.drop{border:1.5px dashed #54708d;border-radius:13px;padding:38px 18px;text-align:center;background:#0e1925;transition:.15s}.drop.drag{border-color:var(--accent);background:#142a40}.drop strong{display:block;font-size:18px;margin:5px 0 8px}.muted{font-size:13px;color:var(--muted)}button,.file-button{display:inline-flex;align-items:center;justify-content:center;border:0;border-radius:9px;padding:11px 16px;font-weight:700;background:var(--accent);color:#071321;cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}.secondary{background:#26394e;color:#e5effa;border:1px solid #405873}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:center;margin-top:16px}input[type=file]{display:none}.keyrow{display:flex;gap:10px;align-items:center;justify-content:flex-end;margin-top:16px}.keyrow label{font-size:12px;color:var(--muted)}input[type=password]{background:#0b1520;color:#edf3fb;border:1px solid var(--line);border-radius:8px;padding:9px 11px;width:min(290px,60vw)}
#selected{margin-top:18px;display:grid;gap:8px}.fileline{display:flex;justify-content:space-between;gap:12px;padding:9px 11px;border-radius:8px;background:#0d1722;font-size:13px}.fileline span:last-child{color:var(--muted);white-space:nowrap}
#status{margin-top:16px;min-height:22px;color:#b9d9ff;font-size:14px}.results{margin-top:24px;display:none}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:16px 0}.stat{background:#0e1925;border:1px solid var(--line);border-radius:10px;padding:13px}.stat b{display:block;font-size:23px}.stat span{font-size:11px;color:var(--muted)}.images{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}.image-card{border:1px solid var(--line);background:#0e1925;border-radius:10px;padding:12px;min-width:0}.image-card b{display:block;overflow-wrap:anywhere;font-size:13px}.image-card small{display:block;color:var(--muted);margin-top:6px}.flag{display:inline-block;margin:7px 4px 0 0;border-radius:20px;background:#44351d;color:#ffd58c;padding:4px 7px;font-size:10px}.good{background:#163d34;color:#9ce4c8}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}.notice{margin-top:18px;padding:12px 14px;border-left:3px solid #d2a34c;background:#2a241a;color:#e8d5b4;font-size:12px;line-height:1.5}
@media(max-width:650px){header{padding:16px 18px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.keyrow{justify-content:flex-start;flex-wrap:wrap}}
</style>
</head>
<body>
<header><div class="brand">image<span>_create</span></div><div class="tag">CPU-FIRST CAPTURE INTELLIGENCE</div></header>
<main>
<section class="intro"><div class="eyebrow">Photogrammetry workspace</div><h1>Start with your photographs.<br>Know what they can prove.</h1><p>Upload an image set to inspect image quality, duplicates, camera metadata and capture readiness. Then open the inspection workspace to review geometric correspondences and create precise object masks.</p></section>
<section class="panel">
<div class="drop" id="drop"><strong>Drop your capture photographs here</strong><div class="muted">JPEG · PNG · WebP · TIFF · BMP</div><div class="row"><label class="file-button" for="files">Choose photographs</label><button class="secondary" id="clear" type="button">Clear selection</button></div><input id="files" type="file" accept="image/jpeg,image/png,image/webp,image/tiff,image/bmp" multiple></div>
<div id="selected"><div class="muted">No photographs selected yet.</div></div>
<div class="keyrow"><label for="apiKey">API key (only required outside development)</label><input id="apiKey" type="password" autocomplete="off" placeholder="IMAGE_CREATE_API_KEY"></div>
<div class="row"><button id="analyse" type="button" disabled>Analyse capture set</button></div>
<div id="status" role="status" aria-live="polite"></div>
</section>
<section class="results panel" id="results">
<div class="eyebrow">Analysis complete</div><h2>Capture-set overview</h2><div class="stats" id="stats"></div>
<h3>Photographs</h3><div class="images" id="images"></div>
<div class="actions"><button id="inspect" type="button">Open visual inspection workspace</button><button class="secondary" id="newSet" type="button">Analyse another capture set</button></div>
<div class="notice">Important: image quality and duplicate checks are not 3D surface coverage. The current application reports coverage as not assessed until dense geometry and visibility validation are available.</div>
</section>
</main>
<script>
const $=s=>document.querySelector(s), filesInput=$('#files'), drop=$('#drop');
let selected=[],lastAnalysis=null;
const bytes=n=>n<1024*1024?(n/1024).toFixed(1)+' KB':(n/1024/1024).toFixed(1)+' MB';
function renderFiles(){const box=$('#selected');box.replaceChildren();if(!selected.length){const p=document.createElement('div');p.className='muted';p.textContent='No photographs selected yet.';box.append(p)}else selected.forEach((f,i)=>{const row=document.createElement('div');row.className='fileline';const name=document.createElement('span');name.textContent=(i+1)+'. '+f.name;const size=document.createElement('span');size.textContent=bytes(f.size);row.append(name,size);box.append(row)});$('#analyse').disabled=!selected.length}
function addFiles(list){const incoming=Array.from(list||[]);const allowed=new Set(['image/jpeg','image/png','image/webp','image/tiff','image/bmp']);const invalid=incoming.filter(f=>!allowed.has(f.type));selected=selected.concat(incoming.filter(f=>allowed.has(f.type)));renderFiles();$('#status').textContent=invalid.length?invalid.length+' unsupported file(s) ignored.':' '}
filesInput.addEventListener('change',e=>{addFiles(e.target.files);filesInput.value=''});
['dragenter','dragover'].forEach(name=>drop.addEventListener(name,e=>{e.preventDefault();drop.classList.add('drag')}));
['dragleave','drop'].forEach(name=>drop.addEventListener(name,e=>{e.preventDefault();drop.classList.remove('drag')}));
drop.addEventListener('drop',e=>addFiles(e.dataTransfer.files));
$('#clear').onclick=()=>{selected=[];renderFiles();$('#status').textContent='Selection cleared.'};
$('#analyse').onclick=async()=>{if(!selected.length)return;const button=$('#analyse');button.disabled=true;$('#status').textContent='Uploading originals and analysing image quality…';const form=new FormData();selected.forEach(f=>form.append('files',f));const headers={};if($('#apiKey').value){headers['X-API-Key']=$('#apiKey').value;const session=await fetch('/api/v1/session',{method:'POST',headers:{'X-API-Key':$('#apiKey').value},credentials:'same-origin'});if(!session.ok){const failure=await session.json();throw new Error(failure.detail||'Browser authentication failed.')}}try{const response=await fetch('/api/v1/analysis/images',{method:'POST',headers,body:form,credentials:'same-origin'});const data=await response.json();if(!response.ok)throw new Error(data.detail||'Analysis failed ('+response.status+')');lastAnalysis=data;renderReport(data);$('#status').textContent='Analysis complete. Original photographs were preserved unchanged.'}catch(error){$('#status').textContent=error.message||String(error)}finally{button.disabled=!selected.length}};
function renderReport(data){$('#results').style.display='block';const summary=data.summary;const metrics=[['Photographs',summary.images_received],['Exact duplicate groups',summary.exact_duplicate_groups],['Near-duplicate pairs',summary.near_duplicate_pairs],['Quality flags',Object.values(summary.quality_flag_counts||{}).reduce((a,b)=>a+b,0)]];$('#stats').replaceChildren();metrics.forEach(([label,value])=>{const box=document.createElement('div');box.className='stat';const number=document.createElement('b');number.textContent=value;const caption=document.createElement('span');caption.textContent=label;box.append(number,caption);$('#stats').append(box)});$('#images').replaceChildren();data.images.forEach(image=>{const card=document.createElement('div');card.className='image-card';const title=document.createElement('b');title.textContent=image.original_filename;const detail=document.createElement('small');detail.textContent=image.metrics.width+' × '+image.metrics.height+' · '+bytes(image.metrics.file_size_bytes);card.append(title,detail);const flags=image.metrics.quality_flags||[];if(!flags.length){const flag=document.createElement('span');flag.className='flag good';flag.textContent='No heuristic flags';card.append(flag)}else flags.forEach(value=>{const flag=document.createElement('span');flag.className='flag';flag.textContent=value.replaceAll('_',' ');card.append(flag)});$('#images').append(card)});$('#results').scrollIntoView({behavior:'smooth',block:'start'})}
$('#inspect').onclick=()=>{if(lastAnalysis)location.href='/api/v1/analysis/'+lastAnalysis.analysis_id+'/inspection'};
$('#newSet').onclick=()=>{lastAnalysis=null;selected=[];renderFiles();$('#results').style.display='none';$('#status').textContent='Ready for another capture set.';window.scrollTo({top:0,behavior:'smooth'})};
</script>
</body></html>"""
