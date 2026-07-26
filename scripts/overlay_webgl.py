#!/usr/bin/env python
"""Overlay a per-strut classification on the as-built material surface (one WebGL2 .html).

Draws the segmentation mask as a dim translucent isosurface and the classified design
struts as coloured lines in the SAME voxel frame, so you can see whether a strut flagged
"missing" really has no metal there or whether the design line is simply sitting a couple
of voxels off it -- the distinction that decides defect vs registration drift.

    python scripts/overlay_webgl.py mask.tif table.npz design.json -o out.html
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

CLASSES = [
    ("nominal",    "#b8b7b2", "coverage > 0.999", 0.35),
    ("partial",    "#1baf7a", "0.9 - 0.999",      0.95),
    ("broken/gap", "#2a78d6", "0.01 - 0.9",       0.95),
    ("missing",    "#eb6834", "coverage < 0.01",  0.95),
]


def mask_surface(path, max_dim, level):
    """Isosurface of the mask, with vertices mapped back to ORIGINAL voxel coordinates."""
    import tifffile
    from skimage.measure import block_reduce, marching_cubes
    vol = tifffile.imread(path) if str(path).endswith((".tif", ".tiff")) \
        else np.load(path, mmap_mode="r")
    vol = np.asarray(vol)
    f = max(1, int(np.ceil(max(vol.shape) / max_dim)))
    small = block_reduce(vol.astype(np.float32), (f, f, f), np.mean) if f > 1 else \
        vol.astype(np.float32)
    verts, faces, _, _ = marching_cubes(np.pad(small, 1), level=level)
    # undo the pad (1 cell) and the block reduction: block i covers originals
    # [i*f, (i+1)*f), centred at i*f + (f-1)/2
    verts = (verts - 1.0) * f + (f - 1) / 2.0
    return verts.astype(np.float32), faces.astype(np.uint32), f


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
         border-radius:8px; background:rgba(13,17,23,.88); backdrop-filter:blur(6px);
         min-width:262px; max-height:92vh; overflow:auto; }
  #hud h1 { margin:0 0 8px; font-size:14px; font-weight:600; }
  .row { display:flex; align-items:center; gap:8px; padding:3px 0; cursor:pointer;
         user-select:none; }
  .row input[type=checkbox] { accent-color:#58a6ff; margin:0; }
  .sw { width:11px; height:11px; border-radius:2px; flex:0 0 auto; }
  .nm { flex:1; font-weight:500; } .ct { color:#8b949e; font-variant-numeric:tabular-nums; }
  .rule { color:#6e7681; font-size:10.5px; margin:-2px 0 3px 34px; }
  .sec { margin-top:9px; padding-top:8px; border-top:1px solid #30363d; }
  .slab { display:block; margin-top:4px; color:#8b949e; font-size:11.5px; }
  .slab input { width:100%; accent-color:#58a6ff; }
  #note { color:#8b949e; font-size:11.5px; }
  #hint { position:fixed; right:12px; bottom:11px; color:#8b949e; font-size:12px;
          pointer-events:none; }
  #err { position:fixed; inset:0; display:none; place-items:center; padding:24px;
         background:#160b0b; color:#ff9c9c; font:13px/1.5 ui-monospace,monospace;
         white-space:pre-wrap; overflow:auto; }
</style>
</head>
<body>
<canvas id="v"></canvas>
<div id="hud"><h1>__TITLE__</h1><div id="legend"></div>
  <div class="sec"><label class="row"><input type="checkbox" id="mk" checked>
    <span class="sw" style="background:#8ab4f8"></span>
    <span class="nm">as-built mask</span><span class="ct" id="mkct"></span></label>
    <label class="slab">opacity <span id="ov"></span><input type="range" id="op"
      min="10" max="100" value="100"></label>
    <label class="slab">clip x <span id="cv"></span><input type="range" id="clip"
      min="0" max="100" value="100"></label></div>
  <div class="sec" id="note">__NOTE__</div></div>
<div id="hint">drag rotate · shift-drag pan · scroll zoom</div>
<pre id="err"></pre>
<script>
const META = __META__;
function fail(m){const e=document.getElementById('err');e.style.display='grid';
  e.textContent='WebGL viewer error:\\n\\n'+m;}
function b64(s){const b=atob(s),u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return u.buffer;}
try{
  const lp=new Float32Array(b64(META.lines));
  const mp=new Float32Array(b64(META.mverts));
  const mi=new Uint32Array(b64(META.mfaces));
  const cv=document.getElementById('v');
  const gl=cv.getContext('webgl2',{antialias:true});
  if(!gl) throw new Error('WebGL2 unavailable in this browser.');
  const mk=(vs,fs)=>{
    const sh=(t,s)=>{const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
      if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(o));return o;};
    const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,vs));
    gl.attachShader(p,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));
    return p;};
  // flat colour program (lines)
  const PL=mk(`#version 300 es
    layout(location=0) in vec3 aPos; uniform mat4 uMVP; uniform float uClip;
    out float vX;
    void main(){ vX=aPos.z; gl_Position=uMVP*vec4(aPos,1.0); }`,
   `#version 300 es
    precision highp float; in float vX; out vec4 f;
    uniform vec3 uColor; uniform float uAlpha, uClip;
    void main(){ if(vX>uClip) discard; f=vec4(uColor,uAlpha); }`);
  // shaded surface program (mask); normals from screen-space derivatives
  const PS=mk(`#version 300 es
    layout(location=0) in vec3 aPos; uniform mat4 uMVP,uMV; out vec3 vV; out float vX;
    void main(){ vV=(uMV*vec4(aPos,1.0)).xyz; vX=aPos.z; gl_Position=uMVP*vec4(aPos,1.0); }`,
   `#version 300 es
    precision highp float; in vec3 vV; in float vX; out vec4 f;
    uniform float uAlpha,uClip;
    void main(){ if(vX>uClip) discard;
      vec3 n=normalize(cross(dFdx(vV),dFdy(vV))); if(n.z<0.0) n=-n;
      vec3 L1=normalize(vec3(0.4,0.7,1.0)), L2=normalize(vec3(-0.6,-0.3,0.5));
      float d=0.78*max(dot(n,L1),0.0)+0.32*max(dot(n,L2),0.0)+0.16;
      float rim=pow(1.0-max(n.z,0.0),2.5)*0.28;
      f=vec4(vec3(0.54,0.70,0.97)*d+rim,uAlpha); }`);
  const lvao=gl.createVertexArray(); gl.bindVertexArray(lvao);
  const lb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,lb);
  gl.bufferData(gl.ARRAY_BUFFER,lp,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  const svao=gl.createVertexArray(); gl.bindVertexArray(svao);
  const sb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,sb);
  gl.bufferData(gl.ARRAY_BUFFER,mp,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  const ib=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,mi,gl.STATIC_DRAW);
  gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);

  const shown=META.classes.map(c=>c.on); let showMask=true;
  const L=document.getElementById('legend');
  META.classes.forEach((c,i)=>{
    const row=document.createElement('label'); row.className='row';
    row.innerHTML=`<input type="checkbox" ${c.on?'checked':''}>`+
      `<span class="sw" style="background:${c.color}"></span>`+
      `<span class="nm">${c.name}</span><span class="ct">${c.count} · ${c.pct}%</span>`;
    row.querySelector('input').addEventListener('change',e=>{shown[i]=e.target.checked;draw();});
    L.appendChild(row);
    const r=document.createElement('div'); r.className='rule';
    r.textContent='material '+c.rule; L.appendChild(r);
  });
  document.getElementById('mkct').textContent=(META.mtris/1000).toFixed(0)+'k tris';
  document.getElementById('mk').addEventListener('change',e=>{showMask=e.target.checked;draw();});
  let mAlpha=META.malpha;
  const op=document.getElementById('op'), oval=document.getElementById('ov');
  oval.textContent=Math.round(mAlpha*100)+'%';
  op.value=Math.round(mAlpha*100);
  op.addEventListener('input',()=>{mAlpha=+op.value/100;
    oval.textContent=op.value+'%'; draw();});
  const clip=document.getElementById('clip'), cval=document.getElementById('cv');
  let clipX=1e9;
  clip.addEventListener('input',()=>{
    const f=+clip.value/100;
    clipX = f>=1 ? 1e9 : META.zmin + f*(META.zmax-META.zmin);
    cval.textContent = f>=1 ? '(off)' : Math.round(clipX-META.zmin)+' vox';
    draw();});

  const mul=(a,b)=>{const o=new Float32Array(16);
    for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;
      for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s;}return o;};
  const persp=(fv,ar,n,fa)=>{const t=1/Math.tan(fv/2),o=new Float32Array(16);
    o[0]=t/ar;o[5]=t;o[10]=(fa+n)/(n-fa);o[11]=-1;o[14]=2*fa*n/(n-fa);return o;};
  function view(rx,ry,d,px,py){
    const cx=Math.cos(rx),sx=Math.sin(rx),cy=Math.cos(ry),sy=Math.sin(ry);
    const R=new Float32Array([cy,sy*sx,-sy*cx,0, 0,cx,sx,0, sy,-cy*sx,cy*cx,0, 0,0,0,1]);
    const T=new Float32Array(16);T[0]=T[5]=T[10]=T[15]=1;T[12]=px;T[13]=py;T[14]=-d;
    return mul(T,R);}
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
    gl.clearColor(0.043,0.055,0.078,1);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    const MV=view(rx,ry,dist,px,py);
    const MVP=mul(persp(0.85,w/h,Math.max(0.05,dist-META.radius*3),dist+META.radius*3),MV);
    // Solid mask FIRST (opaque, writes depth), then the strut lines with the depth test
    // off so the classification stays readable against an opaque body. Drop the opacity
    // slider below 100% to see struts sitting inside the metal.
    if(showMask){
      gl.useProgram(PS); gl.bindVertexArray(svao);
      gl.enable(gl.DEPTH_TEST); gl.depthMask(true);
      gl.uniformMatrix4fv(gl.getUniformLocation(PS,'uMVP'),false,MVP);
      gl.uniformMatrix4fv(gl.getUniformLocation(PS,'uMV'),false,MV);
      gl.uniform1f(gl.getUniformLocation(PS,'uAlpha'),mAlpha);
      gl.uniform1f(gl.getUniformLocation(PS,'uClip'),clipX);
      gl.drawElements(gl.TRIANGLES,mi.length,gl.UNSIGNED_INT,0);}
    gl.useProgram(PL); gl.bindVertexArray(lvao);
    gl.disable(gl.DEPTH_TEST);
    gl.uniformMatrix4fv(gl.getUniformLocation(PL,'uMVP'),false,MVP);
    gl.uniform1f(gl.getUniformLocation(PL,'uClip'),clipX);
    META.classes.forEach((c,i)=>{
      if(!shown[i]||c.count===0)return;
      gl.uniform3f(gl.getUniformLocation(PL,'uColor'),c.rgb[0],c.rgb[1],c.rgb[2]);
      gl.uniform1f(gl.getUniformLocation(PL,'uAlpha'),c.alpha);
      gl.drawArrays(gl.LINES,c.start,c.count*2);});
    gl.enable(gl.DEPTH_TEST);
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
    p.add_argument("mask")
    p.add_argument("design_json")
    p.add_argument("--table", default=None,
                   help="optional measure_struts .npz; without it every strut is drawn in "
                        "one colour and no defect classes are shown")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--max-dim", type=int, default=150,
                   help="mask mesh resolution; it is context, so keep it coarse")
    p.add_argument("--level", type=float, default=0.5)
    p.add_argument("--mask-alpha", type=float, default=1.0,
                   help="1.0 = solid, like webgl_mask_full.html; adjustable in the UI")
    p.add_argument("--title", default="Strut classes on the as-built mask")
    p.add_argument("--note", default="")
    args = p.parse_args()

    mv, mf_, fac = mask_surface(args.mask, args.max_dim, args.level)
    print(f"mask mesh: {len(mv)} verts, {len(mf_)} tris (downsample x{fac})")

    g = json.load(open(args.design_json))
    pos = np.array([j["position"] for j in g["junctions"]], float)[:, [2, 1, 0]]
    edge = np.array([s.get("unit_cell_edge_idx", -1) for s in g["struts"]])
    v, c = np.unique(edge, return_counts=True)
    caps = set(v[c < c.max() * 0.5].tolist())
    keep = ~np.array([e in caps for e in edge])

    if args.table:
        d = np.load(args.table)
        pairs = d["pairs"]
        if "verdict" in d.files:
            # Tables written by compare_defect_methods.py carry explicit, mutually
            # exclusive verdicts.  Preserve them instead of reclassifying coverage.
            labels = [str(x) for x in d["verdict"]]
            order = ["nominal", "missing", "disconnected", "uncertain", "thin", "dross", "bent"]
            colours = {"nominal": "#b8b7b2", "missing": "#eb6834", "disconnected": "#2a78d6",
                       "uncertain": "#e9c46a", "thin": "#9b5de5", "dross": "#43aa8b", "bent": "#f72585"}
            present = [x for x in order if x in labels]
            cls = np.array([present.index(x) for x in labels], np.int32)
            classes = [(x, colours[x], "method verdict", .95 if x != "nominal" else .35) for x in present]
        else:
            cov = d["material_fraction"]
            cls = np.zeros(len(pairs), np.int32)
            cls[(cov >= 0.9) & (cov <= 0.999)] = 1
            cls[(cov >= 0.01) & (cov < 0.9)] = 2
            cls[cov < 0.01] = 3
            classes = CLASSES
    else:
        # design only: one colour, no classification claimed
        pairs = np.array([[s["junction0"], s["junction1"]] for s in g["struts"]], np.int64)
        cls = np.zeros(len(pairs), np.int32)
        classes = [("design struts", "#eb6834", "from the registered JSON", 0.9)]

    a, b = pos[pairs[:, 0]], pos[pairs[:, 1]]
    # ONE shared centre for both geometries, or they will not overlay
    centre = (mv.max(0) + mv.min(0)) / 2
    print(f"shared centre {centre.round(1)} | design bbox {pos.min(0).round(0)}..{pos.max(0).round(0)}"
          f" | mask bbox {mv.min(0).round(0)}..{mv.max(0).round(0)}")

    segs, meta_cls, cur = [], [], 0
    n_keep = int(keep.sum())
    for ci, (name, hexcol, rule, alpha) in enumerate(classes):
        sel = keep & (cls == ci)
        seg = np.stack([a[sel] - centre, b[sel] - centre], 1).astype(np.float32)
        segs.append(seg.reshape(-1, 3))
        meta_cls.append(dict(name=name, color=hexcol, rule=rule, alpha=alpha,
                             rgb=[int(hexcol[i:i+2], 16)/255 for i in (1, 3, 5)],
                             start=cur, count=int(sel.sum()),
                             pct=round(100*sel.sum()/max(1, n_keep), 2), on=True))
        cur += int(sel.sum()) * 2
    lines = np.concatenate(segs, 0)
    mvc = (mv - centre).astype(np.float32)
    radius = float(max(np.linalg.norm(mvc, axis=1).max(),
                       np.linalg.norm(lines, axis=1).max()))

    meta = dict(lines=base64.b64encode(np.ascontiguousarray(lines, np.float32)).decode(),
                mverts=base64.b64encode(np.ascontiguousarray(mvc, np.float32)).decode(),
                mfaces=base64.b64encode(np.ascontiguousarray(mf_, np.uint32)).decode(),
                mtris=int(len(mf_)), malpha=args.mask_alpha, classes=meta_cls,
                radius=radius, zmin=float(mvc[:, 2].min()), zmax=float(mvc[:, 2].max()))
    html = (HTML.replace("__META__", json.dumps(meta))
                .replace("__TITLE__", args.title).replace("__NOTE__", args.note))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB)")
    for m in meta_cls:
        print(f"  {m['name']:12s} {m['count']:6d}  {m['pct']:5.2f}%")


if __name__ == "__main__":
    main()
