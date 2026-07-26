#!/usr/bin/env python
"""Render a per-strut classification as an interactive WebGL2 line graph (one .html).

Takes the measure_struts .npz table plus the registered design JSON and draws every
strut as a line segment coloured by its class, with per-class toggles. Lines rather
than meshes: 18468 struts is 37k vertices (~450 KB) instead of millions of triangles,
so the whole specimen loads instantly and you can see through it.

Self-contained: no ES modules, no external requests, works from a file:// URL.

    python scripts/graph_webgl.py table.npz design.json -o out.html
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

# categorical slots 1-3 of the validated default palette (all-pairs safe), plus a
# recessive grey for the nominal population, which is context rather than identity
# name, colour, and the rule that defines the class -- shown in the legend so the
# viewer is self-documenting and no class is identified by colour alone
CLASSES = [
    ("nominal",    "#b8b7b2", "coverage > 0.999"),
    ("partial",    "#1baf7a", "0.9 - 0.999"),
    ("broken/gap", "#2a78d6", "0.01 - 0.9"),
    ("missing",    "#eb6834", "coverage < 0.01"),
]

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
  canvas { display:block; width:100vw; height:100vh; touch-action:none; cursor:grab; }
  canvas.drag { cursor:grabbing; }
  #hud { position:fixed; top:12px; left:12px; padding:12px 14px; border:1px solid #30363d;
         border-radius:8px; background:rgba(13,17,23,.86); backdrop-filter:blur(6px);
         min-width:240px; }
  #hud h1 { margin:0 0 8px; font-size:14px; font-weight:600; }
  .row { display:flex; align-items:center; gap:8px; padding:3px 0; cursor:pointer;
         user-select:none; }
  .row input { accent-color:#58a6ff; margin:0; }
  .sw { width:11px; height:11px; border-radius:2px; flex:0 0 auto; }
  .nm { flex:1; font-weight:500; }
  .ct { color:#8b949e; font-variant-numeric:tabular-nums; }
  .rule { color:#6e7681; font-size:10.5px; margin:-2px 0 3px 34px; }
  #note { margin-top:9px; padding-top:8px; border-top:1px solid #30363d; color:#8b949e;
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
<div id="hud"><h1>__TITLE__</h1><div id="legend"></div><div id="note">__NOTE__</div></div>
<div id="hint">drag rotate · shift-drag pan · scroll zoom</div>
<pre id="err"></pre>
<script>
const META = __META__;
function fail(m){const e=document.getElementById('err');e.style.display='grid';
  e.textContent='WebGL viewer error:\\n\\n'+m;}
function b64(s){const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u.buffer;}
try{
  const pos=new Float32Array(b64(META.pos));
  const cv=document.getElementById('v');
  const gl=cv.getContext('webgl2',{antialias:true});
  if(!gl) throw new Error('WebGL2 unavailable in this browser.');
  const VS=`#version 300 es
  layout(location=0) in vec3 aPos;
  uniform mat4 uMVP; uniform vec3 uColor; out vec3 vC;
  void main(){ vC=uColor; gl_Position=uMVP*vec4(aPos,1.0); }`;
  const FS=`#version 300 es
  precision highp float; in vec3 vC; out vec4 f; uniform float uAlpha;
  void main(){ f=vec4(vC,uAlpha); }`;
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
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  const uMVP=gl.getUniformLocation(pr,'uMVP'),uCol=gl.getUniformLocation(pr,'uColor'),
        uA=gl.getUniformLocation(pr,'uAlpha');

  // legend + toggles; ranges are contiguous because segments are sorted by class
  const shown=META.classes.map(c=>c.on);
  const L=document.getElementById('legend');
  META.classes.forEach((c,i)=>{
    const row=document.createElement('label'); row.className='row';
    row.innerHTML=`<input type="checkbox" ${c.on?'checked':''}>`+
      `<span class="sw" style="background:${c.color}"></span>`+
      `<span class="nm">${c.name}</span><span class="ct">${c.count} · ${c.pct}%</span>`;
    row.querySelector('input').addEventListener('change',e=>{shown[i]=e.target.checked;draw();});
    L.appendChild(row);
    const rl=document.createElement('div'); rl.className='rule';
    rl.textContent='material '+c.rule; L.appendChild(rl);
  });

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
  let rx=-0.4, ry=0.7, dist=META.radius*3.0, px=0, py=0, drag=false, lx=0, ly=0;
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
    META.classes.forEach((c,i)=>{
      if(!shown[i]||c.count===0) return;
      gl.uniform3f(uCol,c.rgb[0],c.rgb[1],c.rgb[2]);
      gl.uniform1f(uA,c.alpha);
      gl.drawArrays(gl.LINES,c.start,c.count*2);
    });
  }
  addEventListener('resize',draw); draw();
}catch(e){fail((e&&e.message)||String(e));}
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("table", help="measure_struts .npz")
    p.add_argument("design_json")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--title", default="Per-strut classification")
    p.add_argument("--note", default="")
    p.add_argument("--drop-caps", action="store_true", default=True,
                   help="exclude boundary-cap struts (about half are never printed)")
    args = p.parse_args()

    d = np.load(args.table)
    mf = d["material_fraction"]
    g = json.load(open(args.design_json))
    pos = np.array([j["position"] for j in g["junctions"]], float)[:, [2, 1, 0]]
    pairs = d["pairs"]
    edge = np.array([s.get("unit_cell_edge_idx", -1) for s in g["struts"]])
    v, c = np.unique(edge, return_counts=True)
    caps = set(v[c < c.max() * 0.5].tolist())
    iscap = np.array([e in caps for e in edge])
    keep = ~iscap if args.drop_caps else np.ones(len(pairs), bool)

    # class per strut, by coverage of material along the centreline
    cls = np.zeros(len(pairs), dtype=np.int32)
    cls[(mf >= 0.9) & (mf <= 0.999)] = 1
    cls[(mf >= 0.01) & (mf < 0.9)] = 2
    cls[mf < 0.01] = 3

    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
    centre = (pos.max(0) + pos.min(0)) / 2

    segs, meta_classes, cursor = [], [], 0
    n_keep = int(keep.sum())
    for ci, (name, hexcol, rule) in enumerate(CLASSES):
        sel = keep & (cls == ci)
        seg = np.stack([a[sel] - centre, b[sel] - centre], 1).astype(np.float32)
        segs.append(seg.reshape(-1, 3))
        rgb = [int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        meta_classes.append(dict(name=name, color=hexcol, rgb=rgb, rule=rule,
                                 start=cursor, count=int(sel.sum()),
                                 pct=round(100 * sel.sum() / max(1, n_keep), 2),
                                 on=True, alpha=0.28 if ci == 0 else 0.95))
        cursor += int(sel.sum()) * 2

    allseg = np.concatenate(segs, 0) if segs else np.zeros((0, 3), np.float32)
    radius = float(np.linalg.norm(allseg, axis=1).max()) if len(allseg) else 1.0
    meta = dict(pos=base64.b64encode(np.ascontiguousarray(allseg, np.float32)).decode(),
                classes=meta_classes, radius=radius)
    html = (HTML.replace("__META__", json.dumps(meta))
                .replace("__TITLE__", args.title)
                .replace("__NOTE__", args.note))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size/1e6:.2f} MB, "
          f"{n_keep} struts, {len(allseg)} vertices)")
    for m in meta_classes:
        print(f"  {m['name']:12s} {m['count']:6d}  {m['pct']:5.2f}%")


if __name__ == "__main__":
    main()
