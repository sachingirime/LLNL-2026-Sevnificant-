#!/usr/bin/env python
"""Turn a binary 3D volume into a self-contained WebGL2 mesh viewer (one .html).

Written because pyvista's ``export_html`` emits a vtk.js "OfflineLocalView" page whose
loader throws on the embedded scene and silently falls back to Kitware's drop-file
placeholder. This writes plain WebGL2 with no library, no ES modules and no external
requests, so it works straight off a file:// URL.

Payload is positions + indices only; per-face normals are derived in the fragment shader
from screen-space derivatives, which halves the data and gives faceted shading.

    python scripts/mesh_webgl.py mask.tif -o out.html --level 0.5 --max-dim 192
    python scripts/mesh_webgl.py frangi.tif -o out.html --threshold 0.05 --decimate 0.6
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import tifffile


def load(path):
    if str(path).endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return tifffile.imread(path)


def downsample(vol, max_dim, pool="mean"):
    """Block-reduce so max(shape) <= max_dim.

    Pool BEFORE thresholding, and with ``mean`` by default. Max-pooling a binary volume
    dilates it -- one suprathreshold voxel fills the whole block -- which inflated a 4.2%
    foreground to 19.0% at factor 5 and flattened every thickness difference. Mean-pooling
    the float response keeps relative thickness (1.6x inflation instead of 4.6x). ``max``
    is still available: it never loses a thin strut, at the cost of fattening everything.
    """
    f = max(1, int(np.ceil(max(vol.shape) / max_dim)))
    if f == 1:
        return np.asarray(vol), 1
    from skimage.measure import block_reduce
    op = np.max if pool == "max" else np.mean
    return block_reduce(np.asarray(vol, dtype=np.float32), (f, f, f), op), f


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html,body { margin:0; height:100%; overflow:hidden; background:__BG__;
              color:#e6edf3; font:13px/1.45 ui-sans-serif,system-ui,sans-serif; }
  canvas { display:block; width:100vw; height:100vh; touch-action:none; cursor:grab; }
  canvas.drag { cursor:grabbing; }
  #hud { position:fixed; top:12px; left:12px; padding:10px 13px; border:1px solid #30363d;
         border-radius:8px; background:rgba(13,17,23,.82); backdrop-filter:blur(6px); }
  #hud h1 { margin:0 0 4px; font-size:14px; font-weight:600; }
  #hud p { margin:0; color:#8b949e; font-size:12px; }
  #err { position:fixed; inset:0; display:none; place-items:center; padding:24px;
         background:#160b0b; color:#ff9c9c; font:13px/1.5 ui-monospace,monospace;
         white-space:pre-wrap; overflow:auto; }
  #hint { position:fixed; right:12px; bottom:11px; color:#8b949e; font-size:12px;
          pointer-events:none; }
</style>
</head>
<body>
<canvas id="v"></canvas>
<div id="hud"><h1>__TITLE__</h1><p id="stats"></p></div>
<div id="hint">drag rotate · shift-drag pan · scroll zoom</div>
<pre id="err"></pre>
<script>
const META = __META__;
function fail(m){ const e=document.getElementById('err');
  e.style.display='grid'; e.textContent='WebGL viewer error:\\n\\n'+m; }
function b64(s){ const b=atob(s), u=new Uint8Array(b.length);
  for(let i=0;i<b.length;i++) u[i]=b.charCodeAt(i); return u.buffer; }
try{
  const pos = new Float32Array(b64(META.pos));
  const idx = new Uint32Array(b64(META.idx));
  const cv = document.getElementById('v');
  const gl = cv.getContext('webgl2', {antialias:true, depth:true});
  if(!gl) throw new Error('WebGL2 unavailable in this browser.');
  document.getElementById('stats').textContent =
    (idx.length/3).toLocaleString()+' triangles · '+(pos.length/3).toLocaleString()+' verts';

  const VS = `#version 300 es
  layout(location=0) in vec3 aPos;
  uniform mat4 uMVP; uniform mat4 uMV;
  out vec3 vView;
  void main(){ vec4 p=uMV*vec4(aPos,1.0); vView=p.xyz; gl_Position=uMVP*vec4(aPos,1.0); }`;
  const FS = `#version 300 es
  precision highp float;
  in vec3 vView; out vec4 frag;
  uniform vec3 uColor;
  void main(){
    vec3 n = normalize(cross(dFdx(vView), dFdy(vView)));
    if(n.z < 0.0) n = -n;                       // face the camera
    vec3 L1 = normalize(vec3(0.4,0.7,1.0));
    vec3 L2 = normalize(vec3(-0.6,-0.3,0.5));
    float d = 0.78*max(dot(n,L1),0.0) + 0.32*max(dot(n,L2),0.0) + 0.16;
    float rim = pow(1.0-max(n.z,0.0), 2.5)*0.28;
    frag = vec4(uColor*d + rim, 1.0);
  }`;
  function sh(t,src){ const s=gl.createShader(t); gl.shaderSource(s,src); gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s; }
  const pr = gl.createProgram();
  gl.attachShader(pr, sh(gl.VERTEX_SHADER,VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER,FS));
  gl.linkProgram(pr);
  if(!gl.getProgramParameter(pr,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr));
  gl.useProgram(pr);

  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb);
  gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  gl.enable(gl.DEPTH_TEST);   // no face culling: marching-cubes winding is not guaranteed
  const C = META.color;
  gl.uniform3f(gl.getUniformLocation(pr,'uColor'), C[0], C[1], C[2]);
  const uMVP = gl.getUniformLocation(pr,'uMVP'), uMV = gl.getUniformLocation(pr,'uMV');

  // --- minimal mat4 ---
  const mul=(a,b)=>{const o=new Float32Array(16);
    for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;
      for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k]; o[i*4+j]=s;} return o;};
  const persp=(f,ar,n,fa)=>{const t=1/Math.tan(f/2),o=new Float32Array(16);
    o[0]=t/ar; o[5]=t; o[10]=(fa+n)/(n-fa); o[11]=-1; o[14]=2*fa*n/(n-fa); return o;};
  function view(rx,ry,dist,px,py){
    const cx=Math.cos(rx),sx=Math.sin(rx),cy=Math.cos(ry),sy=Math.sin(ry);
    const R=new Float32Array([ cy,sy*sx,-sy*cx,0, 0,cx,sx,0, sy,-cy*sx,cy*cx,0, 0,0,0,1 ]);
    const T=new Float32Array(16); T[0]=T[5]=T[10]=T[15]=1;
    T[12]=px; T[13]=py; T[14]=-dist;
    return mul(T,R);
  }
  let rx=-0.42, ry=0.72, dist=META.radius*3.1, px=0, py=0;
  let drag=false, shift=false, lx=0, ly=0;
  cv.addEventListener('pointerdown',e=>{drag=true;shift=e.shiftKey;lx=e.clientX;ly=e.clientY;
    cv.classList.add('drag');cv.setPointerCapture(e.pointerId);});
  cv.addEventListener('pointerup',e=>{drag=false;cv.classList.remove('drag');});
  cv.addEventListener('pointermove',e=>{ if(!drag) return;
    const dx=e.clientX-lx, dy=e.clientY-ly; lx=e.clientX; ly=e.clientY;
    if(shift||e.shiftKey){ px+=dx*dist*0.0016; py-=dy*dist*0.0016; }
    else { ry+=dx*0.0075; rx+=dy*0.0075; }
    draw(); });
  cv.addEventListener('wheel',e=>{ e.preventDefault();
    dist*=Math.exp(e.deltaY*0.0012); draw(); },{passive:false});

  function draw(){
    const dpr=Math.min(devicePixelRatio||1,2);
    const w=Math.floor(cv.clientWidth*dpr), h=Math.floor(cv.clientHeight*dpr);
    if(cv.width!==w||cv.height!==h){ cv.width=w; cv.height=h; }
    gl.viewport(0,0,w,h);
    gl.clearColor(META.bg[0],META.bg[1],META.bg[2],1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    const MV=view(rx,ry,dist,px,py);
    const P=persp(0.85, w/h, Math.max(0.05,dist-META.radius*3), dist+META.radius*3);
    gl.uniformMatrix4fv(uMV,false,MV); gl.uniformMatrix4fv(uMVP,false,mul(P,MV));
    gl.drawElements(gl.TRIANGLES, idx.length, gl.UNSIGNED_INT, 0);
  }
  addEventListener('resize',draw); draw();
}catch(e){ fail((e&&e.message)||String(e)); }
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("volume")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--threshold", type=float, default=None,
                   help="threshold a float volume (e.g. a Frangi response) before meshing")
    p.add_argument("--level", type=float, default=0.5, help="marching-cubes level")
    p.add_argument("--max-dim", type=int, default=256, help="downsample so max(shape)<=this")
    p.add_argument("--pool", choices=["mean", "max"], default="mean",
                   help="block-reduce operator (default mean; see downsample())")
    p.add_argument("--decimate", type=float, default=0.0,
                   help="fraction of triangles to REMOVE via pyvista (0.6 = keep 40%%)")
    p.add_argument("--color", default="#ff7043")
    p.add_argument("--bg", default="#0b0e14")
    p.add_argument("--title", default="")
    args = p.parse_args()

    vol = load(args.volume)
    # Downsample FIRST (on the continuous data), threshold SECOND -- doing it the other way
    # round max-pools a binary volume, which dilates struts and destroys thickness variation.
    vol, f = downsample(vol, args.max_dim, args.pool)
    if args.threshold is not None:
        vol = (vol > args.threshold).astype(np.uint8)
    # report the fraction that will actually end up inside the isosurface, not (vol>0),
    # which is 100% for a raw CT and tells you nothing
    inside = 100 * float((vol > args.level).mean())
    print(f"volume {vol.shape} (downsample x{f}, {args.pool}-pool), "
          f"inside isosurface (>{args.level:g}) {inside:.1f}%")

    from skimage.measure import marching_cubes
    # Pad with one empty layer: without it, structures touching the volume boundary are
    # left as OPEN tubes (marching cubes does not cap them), so a cropped block shows
    # hollow strut ends instead of flat cut faces.
    verts, faces, _, _ = marching_cubes(np.pad(vol, 1).astype(np.float32),
                                        level=args.level)
    print(f"marching cubes: {len(verts)} verts, {len(faces)} tris (padded, closed)")

    if args.decimate > 0:
        import pyvista as pv
        m = pv.PolyData(verts, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
        m = m.decimate(args.decimate)
        verts = np.asarray(m.points)
        faces = m.faces.reshape(-1, 4)[:, 1:]
        print(f"decimated to {len(verts)} verts, {len(faces)} tris")

    centre = (verts.max(0) + verts.min(0)) / 2
    verts = (verts - centre).astype(np.float32)
    radius = float(np.linalg.norm(verts, axis=1).max())

    def rgb(h):
        h = h.lstrip("#")
        return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]

    meta = {
        "pos": base64.b64encode(np.ascontiguousarray(verts, np.float32)).decode(),
        "idx": base64.b64encode(np.ascontiguousarray(faces, np.uint32)).decode(),
        "color": rgb(args.color), "bg": rgb(args.bg), "radius": radius,
    }
    title = args.title or Path(args.volume).stem
    html = (HTML.replace("__META__", json.dumps(meta))
                .replace("__TITLE__", title)
                .replace("__BG__", args.bg))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html)
    print(f"wrote {args.out}  ({Path(args.out).stat().st_size/1e6:.1f} MB, "
          f"{len(faces)} tris) — open directly, no server needed")


if __name__ == "__main__":
    main()
