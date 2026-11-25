"""Write a standalone Canvas viewer for small experimental tetrahedral meshes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import tet_quality


def write_viewer(path, vertices, tets, boundary_faces, report):
    """Write self-contained HTML; viewing it requires no server or CDN.

    The section control selects tetrahedra by centroid, not by exact geometric
    clipping.  Drawing their complete faces exposes the actual interior cells.
    """
    points = np.asarray(vertices, dtype=float)
    cells = np.asarray(tets, dtype=np.int64)
    boundary = np.asarray(boundary_faces, dtype=np.int64)
    payload = {
        "vertices": points.tolist(), "tets": cells.tolist(),
        "boundary": boundary.tolist(), "report": report,
        "quality": [tet_quality(points[tet]) for tet in cells],
    }

    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot encode {type(value).__name__}")

    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=encode)
    # Prevent report strings from terminating the inert JSON script element.
    data = data.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_HTML.replace("__MESH_DATA__", data), encoding="utf-8")
    return target


_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAFT 3D · 四面体网格查看器</title>
<style>
:root{color-scheme:dark;font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;color:#e6eef7;background:#0c111b}
*{box-sizing:border-box}body{margin:0;padding:28px;max-width:1600px;margin:auto}header{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:22px}.eyebrow{font-size:11px;letter-spacing:2px;color:#5be0cb;text-transform:uppercase;margin-bottom:8px}h1{font-size:27px;font-weight:650;margin:0 0 9px}p{margin:0;color:#a3b1c5;font-size:13px;line-height:1.6}.badge{font-size:12px;border:1px solid #314457;border-radius:24px;padding:9px 14px;white-space:nowrap}.pass{color:#74e6bc;background:#14332f}.fail{color:#ffc086;background:#3c2b20}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}.stat{background:#131c2b;border:1px solid #263246;border-radius:12px;padding:15px 19px}.stat small{display:block;color:#92a3bb;font-size:11px;letter-spacing:.5px;margin-bottom:7px}.stat strong{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}.stat span{font-size:11px;color:#8b9cb5;margin-left:7px}
.workspace{display:grid;grid-template-columns:minmax(0,1fr) 264px;gap:16px}.stage{position:relative;min-width:0;height:64vh;min-height:460px;border-radius:16px;border:1px solid #273247;background:radial-gradient(ellipse at 45% 35%,#1d2a40,#101724 65%);overflow:hidden}canvas{display:block;width:100%;height:100%;touch-action:none;cursor:grab}canvas:active{cursor:grabbing}.corner{position:absolute;top:17px;left:19px;pointer-events:none;font-size:11px;color:#8ea5c0;line-height:1.7}.legend{position:absolute;bottom:18px;left:19px;pointer-events:none;font-size:10px;color:#9bacc1}.gradient{display:inline-block;vertical-align:middle;width:120px;height:6px;border-radius:5px;background:linear-gradient(90deg,hsl(15,76%,55%),hsl(125,76%,55%),hsl(205,76%,55%));margin:0 7px}.help{position:absolute;bottom:17px;right:19px;font-size:11px;color:#788da8;pointer-events:none}
aside{background:#131c2b;border:1px solid #263246;border-radius:16px;padding:21px}h2{font-size:14px;margin:0 0 20px;font-weight:600}.control{margin-bottom:23px}.control label{display:flex;align-items:center;justify-content:space-between;font-size:12px;color:#bdcadb;margin-bottom:12px}.control output{color:#66dec9;font-variant-numeric:tabular-nums}input[type=range]{width:100%;accent-color:#65dfca;cursor:pointer}select,button{font:inherit;font-size:12px;color:#dbe6f4;background:#1c2a3e;border:1px solid #374860;border-radius:7px;padding:9px 10px}select{width:100%}button{cursor:pointer}button:hover{background:#2b3d55}.buttons{display:flex;gap:8px}.buttons button{flex:1}.check{display:flex;gap:8px;align-items:center;font-size:12px;color:#b9c8dc;margin:15px 0}.check input{accent-color:#65dfca}.note{border-top:1px solid #2b374a;padding-top:17px;margin-top:22px;font-size:11px;color:#8fa1ba;line-height:1.8}.foot{display:flex;justify-content:space-between;gap:20px;margin-top:15px;font-size:11px;color:#697f99}.foot span:last-child{text-align:right}details{margin-top:18px;font-size:12px;color:#a7b8ce}summary{cursor:pointer}pre{max-height:330px;overflow:auto;background:#131c2b;padding:15px;border-radius:8px;font-size:11px;line-height:1.5}
@media(max-width:850px){body{padding:18px}.workspace{grid-template-columns:1fr}.stage{height:55vh;min-height:350px}aside{display:grid;grid-template-columns:1fr 1fr;gap:0 25px}aside h2,.note{grid-column:1/-1}.stats{grid-template-columns:repeat(2,1fr)}h1{font-size:23px}.help{display:none}.foot{display:block}.foot span{display:block;margin-bottom:5px}.foot span:last-child{text-align:left}}
</style>
</head>
<body>
<header><div><div class="eyebrow">SAFT-inspired · experimental volume meshing</div><h1>四面体网格 / 内部剖视</h1><p>旋转模型、移动剖切位置，查看体网格中的每一个单元。</p><p id="contribution"></p></div><div class="badge" id="status">检查报告</div></header>
<section class="stats"><div class="stat"><small>四面体总数</small><strong id="total">—</strong><span id="visible"></span></div><div class="stat"><small>顶点数</small><strong id="verts">—</strong></div><div class="stat"><small>最小质量 · MEAN RATIO</small><strong id="quality">—</strong><span id="median"></span></div><div class="stat"><small>相对体积误差</small><strong id="volume">—</strong></div></section>
<div class="workspace"><div class="stage"><canvas id="view" aria-label="可旋转和剖切的三维四面体网格"></canvas><div class="corner" id="caption"></div><div class="legend">质量低<span class="gradient"></span>高</div><div class="help">拖动旋转 · 滚轮缩放</div></div>
<aside><h2>查看控制</h2><div class="control"><label for="axis">剖切方向</label><select id="axis"><option value="0">X 方向</option><option value="1">Y 方向</option><option value="2">Z 方向</option></select></div><div class="control"><label for="cut">保留位置 <output id="cutValue">50%</output></label><input id="cut" type="range" min="0" max="100" value="50"></div><div class="control"><label for="explode">单元分离 <output id="explodeValue">6%</output></label><input id="explode" type="range" min="0" max="35" value="6"></div><div class="control"><label for="opacity">面透明度 <output id="opacityValue">82%</output></label><input id="opacity" type="range" min="10" max="100" value="82"></div><label class="check"><input id="boundary" type="checkbox" checked>显示原始边界轮廓</label><label class="check"><input id="wire" type="checkbox" checked>显示所有单元边线</label><div class="buttons"><button id="reset">重置视角</button><button id="spin">自动旋转</button></div><p class="note">按四面体<strong>质心位置</strong>保留单元，显示保留单元的完整面。剖面呈阶梯状，未对四面体做几何裁剪。<br>质量 1 为正四面体；接近 0 表示扁薄单元。</p></aside></div>
<div class="foot"><span id="algorithm"></span><span>浮点数值验证 · 推进前沿研究原型 · 非工业认证网格器</span></div>
<details><summary>查看完整验证和生成报告</summary><pre id="report"></pre></details>
<script type="application/json" id="mesh-data">__MESH_DATA__</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("mesh-data").textContent), $=id=>document.getElementById(id);
const report=D.report||{}, validation=report.validation||report.validation_report||report, generation=report.run||report.meshing||report.generation||report;
const number=x=>Number.isFinite(x)?x.toLocaleString():"—", decimal=x=>Number.isFinite(x)?x.toFixed(3):"—";
$("total").textContent=number(D.tets.length);$("verts").textContent=number(D.vertices.length);
const qualities=D.quality.slice().sort((a,b)=>a-b), mid=Math.floor(qualities.length/2);
const computedMedian=qualities.length?(qualities.length%2?qualities[mid]:(qualities[mid-1]+qualities[mid])/2):undefined;
const qMin=Number.isFinite(validation.quality?.min)?validation.quality.min:qualities[0];
const qMedian=Number.isFinite(validation.quality?.median)?validation.quality.median:computedMedian;
$("quality").textContent=decimal(qMin);$("median").textContent="中位 "+decimal(qMedian);
$("volume").textContent=Number.isFinite(validation.relative_volume_error)?validation.relative_volume_error.toExponential(2):"未提供";
const checked=validation.valid;
$("status").textContent=checked===true?"✓ 已通过数值验证":checked===false?"存在验证问题":"未附验证结论";
$("status").classList.add(checked===true?"pass":"fail");
$("algorithm").textContent=generation.algorithm||"SAFT-inspired star-kernel advancing front";
if(generation.cpu_completion){
 const fraction=Number.isFinite(generation.aft_volume_fraction)?"（体积 "+(100*generation.aft_volume_fraction).toFixed(1)+"%）":"";
 $("contribution").textContent="推进前沿生成 "+generation.growth_tets+" 个单元"+fraction+"，CPU 补全 "+generation.closure_tets+" 个单元。";
}else if(generation.algorithm?.includes("local 3D")){
 $("contribution").textContent="由局部推进前沿完成；未调用 CPU 体网格补全。";
}
$("report").textContent=JSON.stringify(report,null,2);
const bounds=[0,1,2].map(k=>[Math.min(...D.vertices.map(v=>v[k])),Math.max(...D.vertices.map(v=>v[k]))]);
const center=bounds.map(b=>(b[0]+b[1])/2), scale=Math.max(...bounds.map(b=>b[1]-b[0]))||1;
const vertices=D.vertices.map(v=>v.map((x,k)=>2*(x-center[k])/scale));
const cells=D.tets.map((ids,i)=>({ids,quality:D.quality[i],center:[0,1,2].map(k=>ids.reduce((s,id)=>s+vertices[id][k],0)/4)}));
const faces=[[1,2,3],[0,3,2],[0,1,3],[0,2,1]];
const boundaryEdges=[];const edgeKeys=new Set();for(const f of D.boundary){for(const [i,j] of [[0,1],[1,2],[2,0]]){const a=f[i],b=f[j],key=[Math.min(a,b),Math.max(a,b)].join(',');if(!edgeKeys.has(key)){edgeKeys.add(key);boundaryEdges.push([a,b]);}}}
const canvas=$("view"), ctx=canvas.getContext("2d");let width=0,height=0,yaw=-.65,pitch=.35,zoom=1,drag=null,spin=false,dirty=true;
function rotate(v){const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x=v[0]*cy+v[2]*sy,z=-v[0]*sy+v[2]*cy;return[x,v[1]*cp-z*sp,v[1]*sp+z*cp];}
function project(v){const r=rotate(v),p=4.8/(4.8-r[2]),s=Math.min(width,height)*.32*zoom;return[width/2+r[0]*s*p,height/2-r[1]*s*p,r[2]];}
function path(points){ctx.beginPath();ctx.moveTo(points[0][0],points[0][1]);for(let i=1;i<points.length;i++)ctx.lineTo(points[i][0],points[i][1]);ctx.closePath();}
function draw(){
 dirty=false;ctx.clearRect(0,0,width,height);const axis=+$('axis').value,cut=+$('cut').value/100,shrink=1-+$('explode').value/100,opacity=+$('opacity').value/100;
 const lo=2*(bounds[axis][0]-center[axis])/scale,hi=2*(bounds[axis][1]-center[axis])/scale,plane=lo+cut*(hi-lo);
 const selected=cells.filter(c=>cut===1||c.center[axis]<=plane),drawFaces=[];
 $('visible').textContent='显示 '+number(selected.length);$('caption').textContent=['X','Y','Z'][axis]+' ≤ '+(bounds[axis][0]+cut*(bounds[axis][1]-bounds[axis][0])).toPrecision(4)+'\n'+selected.length+' / '+cells.length+' tetrahedra';
 for(const cell of selected){const points=cell.ids.map(id=>vertices[id].map((x,k)=>cell.center[k]+(x-cell.center[k])*shrink)),projected=points.map(project);for(const f of faces){const points2=f.map(i=>projected[i]);drawFaces.push({p:points2,z:points2.reduce((s,p)=>s+p[2],0)/3,q:cell.quality});}}
 drawFaces.sort((a,b)=>a.z-b.z);
 for(const face of drawFaces){path(face.p);const hue=15+190*Math.min(1,Math.max(0,face.q)),light=42+9*(face.z+1)/2;ctx.fillStyle=`hsla(${hue},65%,${light}%,${opacity})`;ctx.fill();if($('wire').checked){ctx.strokeStyle='rgba(9,20,36,.80)';ctx.lineWidth=.9;ctx.stroke();}}
 if($('boundary').checked){ctx.strokeStyle='rgba(162,193,224,.23)';ctx.lineWidth=.7;ctx.beginPath();for(const [a,b] of boundaryEdges){const p=project(vertices[a]),q=project(vertices[b]);ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);}ctx.stroke();}
 // Orientation triad stays readable as the object rotates.
 const origin=[width-53,48],names=['X','Y','Z'],colors=['#ee8e8e','#8fe4b3','#89bfff'];for(let i=0;i<3;i++){const a=[0,0,0];a[i]=1;const r=rotate(a);ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(origin[0]+r[0]*26,origin[1]-r[1]*26);ctx.strokeStyle=colors[i];ctx.lineWidth=1.5;ctx.stroke();ctx.fillStyle=colors[i];ctx.font='10px Segoe UI';ctx.fillText(names[i],origin[0]+r[0]*34-3,origin[1]-r[1]*34+3);}
}
function resize(){const b=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2);width=b.width;height=b.height;canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);dirty=true;}
new ResizeObserver(resize).observe(canvas);
for(const id of ['axis','cut','explode','opacity','boundary','wire'])$(id).addEventListener('input',()=>{for(const x of ['cut','explode','opacity'])$(x+'Value').textContent=$(x).value+'%';dirty=true;});
canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);drag=[e.clientX,e.clientY];});canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];dirty=true;});for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>drag=null);
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.min(3,Math.max(.3,zoom*Math.exp(-e.deltaY*.001)));dirty=true;},{passive:false});
$('reset').onclick=()=>{yaw=-.65;pitch=.35;zoom=1;dirty=true;};$('spin').onclick=()=>{spin=!spin;$('spin').textContent=spin?'暂停旋转':'自动旋转';};
let last=0;function frame(time){if(spin){yaw+=Math.min(time-last,40)*.00015;dirty=true;}last=time;if(dirty)draw();requestAnimationFrame(frame);}resize();requestAnimationFrame(frame);
</script></body></html>'''
