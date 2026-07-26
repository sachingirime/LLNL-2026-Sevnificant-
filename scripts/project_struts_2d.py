#!/usr/bin/env python
"""Inspect registered lattice struts in a strut-aligned 2-D projection space.

For each design strut, the binary CT mask is resampled into a local coordinate frame.
The local volume is maximum-projected through a 7-voxel depth, producing a 2-D image
whose axes are longitudinal distance and one transverse coordinate.  Unlike an axial
CT slice, this projection is aligned to the strut and therefore does not turn a healthy
oblique strut into a changing ellipse.

The output is a screening experiment, not a replacement for 3-D topology: projection
can hide a gap behind material along its projection direction.  Its x-rate plot is the
same registration-drift falsification used by the volumetric comparison.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile


def load_graph(path):
    g = json.loads(Path(path).read_text())
    p = np.asarray([j["position"] for j in g["junctions"]], float)[:, [2, 1, 0]]
    e = np.asarray([[s["junction0"], s["junction1"]] for s in g["struts"]], int)
    edge = np.asarray([s.get("unit_cell_edge_idx", -1) for s in g["struts"]])
    vals, cnt = np.unique(edge, return_counts=True); rare = set(vals[cnt < cnt.max()*.5])
    return p, e, np.asarray([v in rare for v in edge])


def bases(a, b):
    """Return orthonormal (longitudinal, transverse-1, transverse-2) per strut."""
    e = b - a; e /= np.linalg.norm(e, axis=1)[:, None]
    ref = np.tile([1., 0., 0.], (len(e), 1))
    ref[np.abs(e[:, 0]) > .9] = [0., 1., 0.]
    n1 = np.cross(e, ref); n1 /= np.linalg.norm(n1, axis=1)[:, None]
    n2 = np.cross(e, n1)
    return e, n1, n2


def resample_max_projection(mask, a, b, width=3, depth=3, n_s=33):
    """Return (strut, longitudinal, transverse) 2-D max projections."""
    e, n1, n2 = bases(a, b)
    ts = np.linspace(.15, .85, n_s); u = np.arange(-width, width+1); v = np.arange(-depth, depth+1)
    centres = a[:, None, :] + (b-a)[:, None, :] * ts[None, :, None]
    # (N,S,U,V,3), only ~5.6M points; avoid a full rotated volume.
    pts = (centres[:, :, None, None, :] + u[None, None, :, None, None]*n1[:, None, None, None, :]
           + v[None, None, None, :, None]*n2[:, None, None, None, :])
    q = np.rint(pts).astype(np.int32); shape = np.asarray(mask.shape)
    good = ((q >= 0) & (q < shape)).all(-1); q = np.clip(q, 0, shape-1)
    sample = mask[q[..., 0], q[..., 1], q[..., 2]] & good
    return sample.max(3), (e, n1, n2)


def longest_false(x):
    cur = best = 0
    for y in x:
        cur = cur+1 if not y else 0; best = max(best, cur)
    return best


def classify(project, eligible):
    profile = project.any(2); coverage = profile.mean(1)
    gap = np.asarray([longest_false(x) for x in profile])
    v = np.full(len(coverage), "uncertain", dtype="U12")
    missing = eligible & (coverage <= .12)
    broken = eligible & ~missing & profile[:, 0] & profile[:, -1] & (gap >= 4)
    nominal = eligible & ~missing & ~broken & (coverage >= .97)
    v[missing] = "missing"; v[broken] = "disconnected"; v[nominal] = "nominal"
    confidence = np.clip(np.maximum(np.abs(coverage-.12)/.12, np.abs(coverage-.97)/.03), 0, 1)
    return v, confidence, coverage, gap, profile


def write_csv(path, caps, plates, mid, v, conf, cov, gap, family):
    with path.open("w", newline="") as f:
        w=csv.writer(f); w.writerow(["strut_id","is_boundary_cap","in_plate_region","midpoint_z","midpoint_y","midpoint_x","verdict","confidence","projected_coverage","longest_empty_run_samples","orientation_family"])
        for i in range(len(v)):
            w.writerow([i,int(caps[i]),int(plates[i]),*[f"{x:.3f}" for x in mid[i]],v[i],f"{conf[i]:.3f}",f"{cov[i]:.5f}",int(gap[i]),int(family[i])])


def visuals(out, project, v, cov, mid, eligible, e, n1):
    # Gallery contains low- and high-x flags so a reviewer can see the drift failure.
    choice=[]
    for lab in ["missing", "disconnected"]:
        for region in [eligible & (mid[:,2] <= 300), eligible & (mid[:,2] > 500)]:
            inds=np.where((v==lab)&region)[0]
            choice.extend(inds[np.argsort(cov[inds])[:6]].tolist())
    choice=choice[:24]
    fig, axes=plt.subplots(4,6,figsize=(13,8),squeeze=False)
    for ax, i in zip(axes.ravel(),choice):
        ax.imshow(project[i].T, cmap="gray", origin="lower", aspect="auto", vmin=0,vmax=1)
        ax.set_title(f"#{i} {v[i]}\nx={mid[i,2]:.0f}, c={cov[i]:.2f}",fontsize=7); ax.set_xticks([]);ax.set_yticks([])
    for ax in axes.ravel()[len(choice):]: ax.axis("off")
    fig.suptitle("Strut-aligned 2-D maximum projections (longitudinal × transverse)",fontsize=12); fig.tight_layout()
    fig.savefig(out/"projection_gallery.png",dpi=170);plt.close(fig)
    # Derive the six signed design orientation families, then project each strut midpoint
    # into that family's normal plane.  This is a 2-D location map, not a radiograph.
    signed=np.sign(e); signed[signed==0]=1; keys=np.unique(signed,axis=0)
    fam=np.array([np.where((keys==q).all(1))[0][0] for q in signed])
    ncol=4; nrow=int(np.ceil(len(keys)/ncol)); fig, axes=plt.subplots(nrow,ncol,figsize=(16,4*nrow),squeeze=False)
    for k,ax in enumerate(axes.ravel()):
        if k >= len(keys): ax.axis("off"); continue
        i0=np.where(fam==k)[0][0]
        sel=eligible&(fam==k); u=(mid@n1[i0]); n2=np.cross(e[i0],n1[i0]); w=mid@n2
        ax.scatter(u[sel],w[sel],s=1,c="#d0d5db",label="nominal")
        for lab,col in [("missing","#e76f51"),("disconnected","#457b9d"),("uncertain","#e9c46a")]:
            q=sel&(v==lab); ax.scatter(u[q],w[q],s=8,c=col,label=lab)
        ax.set(title=f"family {k}: direction {keys[k].astype(int)}",xlabel="local u [vox]",ylabel="local v [vox]"); ax.legend(markerscale=2,fontsize=7)
    fig.suptitle("2-D normal-plane maps of projection verdicts",fontsize=12);fig.tight_layout();fig.savefig(out/"orientation_projection_maps.png",dpi=170);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].hist(cov[eligible],bins=40,color="#718096");ax[0].set(title="Strut-aligned 2-D projected coverage",xlabel="longitudinal coverage",ylabel="struts")
    x=mid[:,2]; bins=np.linspace(x[eligible].min(),x[eligible].max(),19); ctr=(bins[:-1]+bins[1:])/2
    for lab,col in [("missing","#e76f51"),("disconnected","#457b9d"),("uncertain","#e9c46a")]:
        y=[np.mean(v[eligible&(x>=lo)&(x<hi)]==lab) for lo,hi in zip(bins[:-1],bins[1:])]
        ax[1].plot(ctr,y,marker="o",ms=3,label=lab,color=col)
    ax[1].set(title="Projection verdict rate versus x",xlabel="midpoint x [vox]",ylabel="fraction");ax[1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/"projection_diagnostics.png",dpi=170);plt.close(fig)
    return fam


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mask",default="data/9x9x9_octet_lattice/segmentation/mask.tif"); ap.add_argument("--design",default="data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"); ap.add_argument("--out",default="outputs/method_comparison/strut_aligned_2d_projection"); args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True); pos,pairs,caps=load_graph(args.design); a,b=pos[pairs[:,0]],pos[pairs[:,1]];mid=(a+b)/2;plates=(mid[:,0]<95)|(mid[:,0]>665);eligible=~caps&~plates
    print("reading mask"); mask=tifffile.imread(args.mask).astype(bool); project,(e,n1,_)=resample_max_projection(mask,a,b);del mask
    v,conf,cov,gap,profile=classify(project,eligible);fam=visuals(out,project,v,cov,mid,eligible,e,n1);write_csv(out/"struts.csv",caps,plates,mid,v,conf,cov,gap,fam)
    np.savez_compressed(out/"raw_measurements.npz",pairs=pairs,verdict=v,confidence=conf,projected_coverage=cov,longest_empty_run_samples=gap,profile=profile,eligible=eligible,midpoint=mid,orientation_family=fam)
    counts={x:int(((v==x)&eligible).sum()) for x in np.unique(v)}; (out/"metadata.json").write_text(json.dumps({"projection":"maximum through +/-3 voxels in strut-local transverse direction","image_shape":[33,7],"eligible":int(eligible.sum()),"counts":counts},indent=2))
    print(counts)
if __name__=="__main__": main()
