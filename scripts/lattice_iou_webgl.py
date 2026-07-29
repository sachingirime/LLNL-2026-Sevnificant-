#!/usr/bin/env python
"""Interactive 3D viewer for the lattice IoU tables, as one self-contained .html file.

Draws every strut as a line segment in the CT's own coordinates, coloured by its
agreement with the nominal design, with the classification thresholds on live sliders.
That combination is the point: the distribution plots let you pick a cut numerically,
but only moving the cut while watching the specimen tells you whether the struts it
selects are a scattered defect population or a coherent artefact -- a face, a slab, a
gradient. Classification happens in the vertex shader off a per-vertex (IoU, weakest
station) attribute, so dragging a slider reclassifies all 18k struts per frame.

Lines rather than meshes on purpose: 18,468 struts is 37k vertices, about 750 KB, so
the whole specimen loads instantly and you can see through it to the interior. A
tube mesh would be millions of triangles and would hide exactly what you came to see.

Self-contained: no ES modules, no external requests, works from a file:// URL.

    python scripts/lattice_iou_webgl.py --dir outputs/lattice_iou
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html,body { margin:0; height:100%; overflow:hidden; background:#0b0e14; color:#e6edf3;
              font:13px/1.5 ui-sans-serif,system-ui,sans-serif; }
  canvas#v { display:block; width:100vw; height:100vh; touch-action:none; cursor:grab; }
  canvas#v.drag { cursor:grabbing; }
  #hud { position:fixed; top:12px; left:12px; padding:13px 15px; border:1px solid #30363d;
         border-radius:8px; background:rgba(13,17,23,.88); backdrop-filter:blur(6px);
         width:296px; max-height:calc(100vh - 24px); overflow:auto; }
  #hud h1 { margin:0 0 2px; font-size:14px; font-weight:600; }
  #sub { color:#8b949e; font-size:11.5px; margin-bottom:10px; }
  .row { display:flex; align-items:center; gap:8px; padding:3px 0; cursor:pointer;
         user-select:none; }
  .row input { accent-color:#58a6ff; margin:0; }
  .sw { width:11px; height:11px; border-radius:2px; flex:0 0 auto; }
  .nm { flex:1; font-weight:500; }
  .ct { color:#8b949e; font-variant-numeric:tabular-nums; font-size:12px; }
  .rule { color:#6e7681; font-size:10.5px; margin:-3px 0 4px 34px; }
  .sec { margin-top:11px; padding-top:9px; border-top:1px solid #30363d; }
  .sl { display:block; margin:7px 0 2px; color:#8b949e; font-size:11.5px; }
  .sl b { color:#e6edf3; font-variant-numeric:tabular-nums; }
  input[type=range] { width:100%; accent-color:#58a6ff; margin:0; }
  #hist { display:block; width:100%; height:62px; margin-top:6px; border-radius:3px; }
  #note { margin-top:10px; padding-top:9px; border-top:1px solid #30363d; color:#8b949e;
          font-size:11.5px; }
  #hint { position:fixed; right:12px; bottom:11px; color:#8b949e; font-size:12px;
          pointer-events:none; }
  #err { position:fixed; inset:0; display:none; place-items:center; padding:24px;
         background:#160b0b; color:#ff9c9c; font:13px/1.5 ui-monospace,monospace;
         white-space:pre-wrap; overflow:auto; }
</style>
</head>
<body>
<canvas id="v"></canvas>
<div id="hud">
  <h1>__TITLE__</h1>
  <div id="sub">__SUB__</div>
  <div id="legend"></div>
  <div class="sec">
    <label class="sl">missing if IoU &lt; <b id="lc1">0.06</b></label>
    <input type="range" id="s1" min="0" max="0.5" step="0.002" value="0.06">
    <label class="sl">severed if a station &lt; <b id="lc2">0.06</b></label>
    <input type="range" id="s2" min="0" max="0.5" step="0.002" value="0.06">
    <canvas id="hist"></canvas>
    <div class="rule" style="margin:3px 0 0 0">IoU distribution (sqrt counts); the line is the current cut</div>
  </div>
  <div class="sec">
    <label class="row"><input type="checkbox" id="ramp"><span class="nm">shade the rest by IoU</span></label>
  </div>
  <div id="note">__NOTE__</div>
</div>
<div id="hint">drag rotate · shift-drag pan · scroll zoom</div>
<pre id="err"></pre>
<script>
const META = __META__;
function fail(m){const e=document.getElementById('err');e.style.display='grid';
  e.textContent='WebGL viewer error:\\n\\n'+m;}
function b64(s){const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u.buffer;}
try{
  const pos=new Float32Array(b64(META.pos));      // 2 verts per strut, xyz
  const met=new Float32Array(b64(META.met));      // 2 verts per strut, (iou, pmin)
  const sIou=new Float32Array(b64(META.siou));    // per strut, for counting
  const sPmin=new Float32Array(b64(META.spmin));
  const N=sIou.length;

  const cv=document.getElementById('v');
  const gl=cv.getContext('webgl2',{antialias:true});
  if(!gl) throw new Error('WebGL2 unavailable in this browser.');

  const VS=`#version 300 es
  layout(location=0) in vec3 aPos;
  layout(location=1) in vec2 aM;            // x = strut IoU, y = weakest station fill
  uniform mat4 uMVP; uniform float uCut, uSta, uRamp;
  uniform vec3 uShow;                        // visibility per class: ok, severed, missing
  out vec4 vC;
  void main(){
    float cls = (aM.x < uCut) ? 2.0 : ((aM.y < uSta) ? 1.0 : 0.0);
    vec3 col; float a; float show;
    if(cls > 1.5)      { col=vec3(0.890,0.286,0.282); a=1.00; show=uShow.z; }
    else if(cls > 0.5) { col=vec3(0.922,0.408,0.204); a=0.92; show=uShow.y; }
    else               { col=vec3(0.722,0.718,0.698); a=0.13; show=uShow.x;
      if(uRamp > 0.5){
        float t = clamp((aM.x - 0.30) / 0.40, 0.0, 1.0);
        col = mix(vec3(0.804,0.886,0.984), vec3(0.051,0.212,0.420), t);
        a = 0.42;
      }
    }
    vC = vec4(col, a*show);
    gl_Position = uMVP*vec4(aPos,1.0);
  }`;
  const FS=`#version 300 es
  precision highp float; in vec4 vC; out vec4 f;
  void main(){ if(vC.a < 0.004) discard; f=vC; }`;
  function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(o));return o;}
  const pr=gl.createProgram();
  gl.attachShader(pr,sh(gl.VERTEX_SHADER,VS)); gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,FS));
  gl.linkProgram(pr);
  if(!gl.getProgramParameter(pr,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(pr));
  gl.useProgram(pr);

  const vao=gl.createVertexArray(); gl.bindVertexArray(vao);
  const vb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,vb);
  gl.bufferData(gl.ARRAY_BUFFER,pos,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  const mb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,mb);
  gl.bufferData(gl.ARRAY_BUFFER,met,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,2,gl.FLOAT,false,0,0);
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);

  const U=n=>gl.getUniformLocation(pr,n);
  const uMVP=U('uMVP'), uCut=U('uCut'), uSta=U('uSta'), uShow=U('uShow'), uRamp=U('uRamp');

  // ---- legend, driven by the live cuts
  const CLASSES=[
    {name:'intact',  color:'#b8b7b2', rule:'above both cuts'},
    {name:'severed', color:'#eb6834', rule:'a station along the edge is empty'},
    {name:'missing', color:'#e34948', rule:'no material in the nominal cylinder'},
  ];
  const shown=[true,true,true];
  const L=document.getElementById('legend');
  const cts=[];
  CLASSES.forEach((c,i)=>{
    const row=document.createElement('label'); row.className='row';
    row.innerHTML=`<input type="checkbox" checked>`+
      `<span class="sw" style="background:${c.color}"></span>`+
      `<span class="nm">${c.name}</span><span class="ct"></span>`;
    row.querySelector('input').addEventListener('change',e=>{shown[i]=e.target.checked;draw();});
    L.appendChild(row);
    cts.push(row.querySelector('.ct'));
    const rl=document.createElement('div'); rl.className='rule'; rl.textContent=c.rule;
    L.appendChild(rl);
  });

  let cut=0.06, sta=0.06, ramp=false;
  function recount(){
    let m=0,s=0;
    for(let i=0;i<N;i++){ if(sIou[i]<cut) m++; else if(sPmin[i]<sta) s++; }
    const o=N-m-s;
    const f=(v)=>`${v} · ${(100*v/N).toFixed(2)}%`;
    cts[0].textContent=f(o); cts[1].textContent=f(s); cts[2].textContent=f(m);
  }

  // ---- inline histogram of the IoU distribution with the cut marked
  const hc=document.getElementById('hist'), hx=hc.getContext('2d');
  const NB=72, bins=new Float32Array(NB);
  for(let i=0;i<N;i++){ let b=Math.min(NB-1,Math.max(0,Math.floor(sIou[i]*NB))); bins[b]++; }
  let bmax=0; for(let i=0;i<NB;i++) bmax=Math.max(bmax,bins[i]);
  function hist(){
    const dpr=Math.min(devicePixelRatio||1,2);
    const w=hc.clientWidth, h=62;
    if(hc.width!==w*dpr){ hc.width=w*dpr; hc.height=h*dpr; }
    hx.setTransform(dpr,0,0,dpr,0,0);
    hx.clearRect(0,0,w,h);
    hx.fillStyle='#161b22'; hx.fillRect(0,0,w,h);
    hx.fillStyle='#3987e5';
    for(let i=0;i<NB;i++){
      // sqrt so the 0.5% defect spike stays visible beside the main mode
      const bh=Math.sqrt(bins[i]/bmax)*(h-6);
      hx.fillRect(i*w/NB, h-bh, Math.max(w/NB-0.6,0.8), bh);
    }
    hx.strokeStyle='#e34948'; hx.lineWidth=1.5;
    hx.beginPath(); hx.moveTo(cut*w,0); hx.lineTo(cut*w,h); hx.stroke();
  }

  const s1=document.getElementById('s1'), s2=document.getElementById('s2');
  const l1=document.getElementById('lc1'), l2=document.getElementById('lc2');
  s1.addEventListener('input',e=>{cut=+e.target.value; l1.textContent=cut.toFixed(3);
    recount(); hist(); draw();});
  s2.addEventListener('input',e=>{sta=+e.target.value; l2.textContent=sta.toFixed(3);
    recount(); draw();});
  document.getElementById('ramp').addEventListener('change',e=>{ramp=e.target.checked;draw();});

  // ---- camera
  const mul=(a,b)=>{const o=new Float32Array(16);
    for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;
      for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}return o;};
  const persp=(f,ar,n,fa)=>{const t=1/Math.tan(f/2),o=new Float32Array(16);
    o[0]=t/ar;o[5]=t;o[10]=(fa+n)/(n-fa);o[11]=-1;o[14]=2*fa*n/(n-fa);return o;};
  function view(rx,ry,d,px,py){
    const cx=Math.cos(rx),sx=Math.sin(rx),cy=Math.cos(ry),sy=Math.sin(ry);
    const R=new Float32Array([cy,sy*sx,-sy*cx,0, 0,cx,sx,0, sy,-cy*sx,cy*cx,0, 0,0,0,1]);
    const T=new Float32Array(16); T[0]=T[5]=T[10]=T[15]=1; T[12]=px;T[13]=py;T[14]=-d;
    return mul(T,R);
  }
  let rx=-0.42, ry=0.72, dist=META.radius*3.0, px=0, py=0, drag=false, lx=0, ly=0;
  cv.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;
    cv.classList.add('drag');cv.setPointerCapture(e.pointerId);});
  cv.addEventListener('pointerup',()=>{drag=false;cv.classList.remove('drag');});
  cv.addEventListener('pointermove',e=>{if(!drag)return;
    const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
    if(e.shiftKey){px+=dx*dist*0.0016;py-=dy*dist*0.0016;}
    else{ry+=dx*0.0075;rx+=dy*0.0075;} draw();});
  cv.addEventListener('wheel',e=>{e.preventDefault();dist*=Math.exp(e.deltaY*0.0012);draw();},
    {passive:false});

  function draw(){
    const dpr=Math.min(devicePixelRatio||1,2);
    const w=Math.floor(cv.clientWidth*dpr),h=Math.floor(cv.clientHeight*dpr);
    if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}
    gl.viewport(0,0,w,h);
    gl.clearColor(0.043,0.055,0.078,1); gl.clear(gl.COLOR_BUFFER_BIT);
    const MV=view(rx,ry,dist,px,py);
    const P=persp(0.85,w/h,Math.max(0.05,dist-META.radius*3),dist+META.radius*3);
    gl.uniformMatrix4fv(uMVP,false,mul(P,MV));
    gl.uniform1f(uCut,cut); gl.uniform1f(uSta,sta); gl.uniform1f(uRamp,ramp?1:0);
    gl.uniform3f(uShow,shown[0]?1:0,shown[1]?1:0,shown[2]?1:0);
    // intact first so the flagged struts draw over it
    gl.drawArrays(gl.LINES,0,N*2);
  }
  addEventListener('resize',()=>{draw();hist();});
  recount(); hist(); draw();
}catch(e){fail((e&&e.message)||String(e));}
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="outputs/lattice_iou")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--keep-boundary", action="store_true",
                   help="include the boundary-cap and plate-embedded struts")
    a = p.parse_args()

    root = Path(a.dir)
    out = Path(a.out) if a.out else root / "lattice_iou_3d.html"
    summary = json.loads((root / "summary.json").read_text())
    um = summary["geometry"]["um_per_voxel"]

    d = np.genfromtxt(root / "struts.csv", delimiter=",", names=True)
    keep = np.ones(len(d["iou"]), bool) if a.keep_boundary else \
        (d["is_boundary"] == 0) & (d["embedded"] == 0)

    # (x, y, z) in mm, centred, so the default camera framing is scale-free
    p0 = np.column_stack([d["x0"], d["y0"], d["z0"]])[keep] * um / 1000.0
    p1 = np.column_stack([d["x1"], d["y1"], d["z1"]])[keep] * um / 1000.0
    centre = 0.5 * (np.minimum(p0.min(0), p1.min(0)) + np.maximum(p0.max(0), p1.max(0)))
    p0, p1 = p0 - centre, p1 - centre
    radius = float(np.abs(np.vstack([p0, p1])).max())

    verts = np.empty((len(p0) * 2, 3), np.float32)
    verts[0::2], verts[1::2] = p0, p1

    iou = d["iou"][keep].astype(np.float32)
    pmin = d["profile_min"][keep].astype(np.float32)
    met = np.empty((len(iou) * 2, 2), np.float32)
    met[0::2, 0] = met[1::2, 0] = iou
    met[0::2, 1] = met[1::2, 1] = pmin

    enc = lambda arr: base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()
    meta = {"pos": enc(verts), "met": enc(met), "siou": enc(iou), "spmin": enc(pmin),
            "radius": radius}

    note = (f"{len(iou)} struts · nominal {summary['geometry']['nominal_strut_diameter_um']:.0f} "
            f"um cylinder · {um:.2f} um/voxel · registration correction "
            f"{'applied' if summary['correction_applied'] else 'NOT applied'}")
    html = (HTML.replace("__TITLE__", "Strut IoU vs nominal design")
                .replace("__SUB__", "drag the cuts; watch what they select")
                .replace("__NOTE__", note)
                .replace("__META__", json.dumps(meta)))
    out.write_text(html)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(iou)} struts)")


if __name__ == "__main__":
    main()
