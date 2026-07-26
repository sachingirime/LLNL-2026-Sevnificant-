#!/usr/bin/env node
/** Generate a portable, interactive canvas viewer for an octet lattice JSON. */
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const input = path.join(root, "data", "missing_struts", "octet_truss_9x9x9.json");
const output = path.join(root, "outputs", "octet_truss_9x9x9_interactive.html");
const lattice = JSON.parse(fs.readFileSync(input, "utf8"));

const positions = lattice.junctions.map(({ position }) => position);
const families = { xy: 0, xz: 1, yz: 2 };
const edges = lattice.struts.map(({ junction0, junction1 }) => {
  const a = positions[junction0], b = positions[junction1];
  const family = [0, 1, 2].filter(i => Math.abs(b[i] - a[i]) > 1e-9).map(i => "xyz"[i]).join("");
  return [junction0, junction1, families[family]];
});
const data = JSON.stringify({ positions, edges });

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Octet truss 9 × 9 × 9 — interactive viewer</title>
<style>
  :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
  * { box-sizing: border-box; }
  body { margin: 0; overflow: hidden; background: #08111d; color: #eef6ff; }
  canvas { display: block; width: 100vw; height: 100vh; touch-action: none; cursor: grab; }
  canvas.dragging { cursor: grabbing; }
  #panel { position: fixed; top: 16px; left: 16px; max-width: 330px; padding: 14px 16px; border: 1px solid #38516d; border-radius: 10px; background: rgba(9, 21, 35, .87); box-shadow: 0 8px 30px rgba(0,0,0,.28); backdrop-filter: blur(8px); }
  h1 { margin: 0 0 5px; font-size: 16px; } p, #stats { margin: 0; color: #b9cadc; font-size: 12px; line-height: 1.45; }
  #legend { display: flex; flex-wrap: wrap; gap: 5px 10px; margin-top: 9px; color: #dce9f6; font-size: 11px; } .key { display: inline-flex; align-items: center; gap: 4px; } .swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .controls { display: grid; grid-template-columns: 1fr auto; gap: 8px 12px; align-items: center; margin-top: 12px; font-size: 13px; }
  label { display: contents; } input[type=range] { width: 145px; accent-color: #63c7ff; } button { grid-column: 1 / -1; padding: 7px; border: 1px solid #52718f; border-radius: 6px; color: #eef6ff; background: #17334d; cursor: pointer; } button:hover { background: #204766; }
  #hint { position: fixed; right: 16px; bottom: 14px; color: #bdd2e7; font-size: 12px; text-align: right; pointer-events: none; }
</style>
</head>
<body>
<canvas id="view" aria-label="Interactive 3D octet lattice"></canvas>
<section id="panel">
  <h1>Octet truss · 9 × 9 × 9</h1>
  <p id="stats"></p>
  <div id="legend"><span class="key"><i class="swatch" style="background:#37b9ff"></i>XY struts</span><span class="key"><i class="swatch" style="background:#b174ff"></i>XZ struts</span><span class="key"><i class="swatch" style="background:#37d6a3"></i>YZ struts</span><span class="key"><i class="swatch" style="background:#ffba5c"></i>nodes</span></div>
  <div class="controls">
    <label>Strut brightness <input id="brightness" type="range" min="15" max="100" value="68"></label>
    <label>Node size <input id="nodes" type="range" min="0" max="100" value="20"></label>
    <button id="reset" type="button">Reset view</button>
  </div>
</section>
<div id="hint">Drag to rotate · Right-drag / Shift-drag to pan · Scroll to zoom</div>
<script>
// Compact geometry extracted from data/missing_struts/octet_truss_9x9x9.json.
const DATA = ${data};
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const stats = document.getElementById('stats');
const brightness = document.getElementById('brightness');
const nodeSize = document.getElementById('nodes');
const positions = DATA.positions, edges = DATA.edges;
const extent = positions.reduce((a,p) => ({ min: [Math.min(a.min[0],p[0]),Math.min(a.min[1],p[1]),Math.min(a.min[2],p[2])], max: [Math.max(a.max[0],p[0]),Math.max(a.max[1],p[1]),Math.max(a.max[2],p[2])] }), {min:[Infinity,Infinity,Infinity],max:[-Infinity,-Infinity,-Infinity]});
const center = extent.min.map((v,i) => (v + extent.max[i]) / 2);
const size = Math.max(...extent.max.map((v,i) => v - extent.min[i]));
stats.textContent = positions.length.toLocaleString() + ' junctions · ' + edges.length.toLocaleString() + ' struts · dimensions ' + extent.max.map((v,i) => v - extent.min[i]).join(' × ');

let width = 0, height = 0, dpr = 1, yaw = -0.72, pitch = 0.48, zoom = 1.0, panX = 0, panY = 0;
let pointer = null;
function resize() { dpr = Math.min(devicePixelRatio || 1, 2); width = innerWidth; height = innerHeight; canvas.width = width*dpr; canvas.height = height*dpr; ctx.setTransform(dpr,0,0,dpr,0,0); render(); }
function project(p) {
  const x = p[0]-center[0], y = p[1]-center[1], z = p[2]-center[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const rx = cy*x + sy*z, rz = -sy*x + cy*z;
  return [rx, cp*y-sp*rz, sp*y+cp*rz];
}
function render() {
  ctx.clearRect(0,0,width,height);
  const scale = Math.min(width,height) / (size * 1.52) * zoom;
  const screen = p => [width/2 + panX + p[0]*scale, height/2 + panY - p[1]*scale, p[2]];
  const transformed = positions.map(p => screen(project(p)));
  const drawEdges = edges.map(e => [transformed[e[0]], transformed[e[1]], e[2]]);
  drawEdges.sort((a,b) => ((a[0][2]+a[1][2]) - (b[0][2]+b[1][2])));
  const level = Number(brightness.value) / 100;
  ctx.lineWidth = Math.max(0.7, Math.min(2.4, scale * .045));
  const colors = [[55,185,255], [177,116,255], [55,214,163]];
  for (const [a,b,family] of drawEdges) {
    const depth = (a[2]+b[2])/(size || 1);
    const alpha = Math.max(.12, Math.min(1, level * (.55 + .22*depth)));
    const color = colors[family];
    ctx.strokeStyle = 'rgba(' + color[0] + ',' + color[1] + ',' + color[2] + ',' + alpha + ')';
    ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
  }
  const node = Number(nodeSize.value) / 100 * Math.max(.5, Math.min(2.4, scale*.055));
  if (node > .08) {
    ctx.fillStyle = 'rgba(255,186,92,.86)';
    for (const p of transformed) { ctx.beginPath(); ctx.arc(p[0],p[1],node,0,Math.PI*2); ctx.fill(); }
  }
  ctx.fillStyle = '#9fc4e2'; ctx.font = '12px system-ui'; ctx.fillText('x', width/2 + panX + scale*size*.58, height/2 + panY + 4);
}
window.addEventListener('resize', resize);
canvas.addEventListener('contextmenu', e => e.preventDefault());
canvas.addEventListener('pointerdown', e => { pointer = {x:e.clientX,y:e.clientY, pan:e.button===2 || e.shiftKey}; canvas.setPointerCapture(e.pointerId); canvas.classList.add('dragging'); });
canvas.addEventListener('pointermove', e => { if (!pointer) return; const dx=e.clientX-pointer.x, dy=e.clientY-pointer.y; pointer.x=e.clientX; pointer.y=e.clientY; if (pointer.pan) { panX += dx; panY += dy; } else { yaw += dx*.008; pitch = Math.max(-1.54, Math.min(1.54, pitch + dy*.008)); } render(); });
canvas.addEventListener('pointerup', () => { pointer=null; canvas.classList.remove('dragging'); });
canvas.addEventListener('pointercancel', () => { pointer=null; canvas.classList.remove('dragging'); });
canvas.addEventListener('wheel', e => { e.preventDefault(); const factor = Math.exp(-e.deltaY*.001); zoom = Math.max(.18, Math.min(8, zoom*factor)); render(); }, {passive:false});
brightness.addEventListener('input', render); nodeSize.addEventListener('input', render);
document.getElementById('reset').addEventListener('click', () => { yaw=-.72; pitch=.48; zoom=1; panX=panY=0; render(); });
resize();
</script>
</body>
</html>`;

fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, html);
console.log(output);
console.log(`${positions.length} junctions, ${edges.length} struts`);
