#!/usr/bin/env python
"""Render raw-CT and binary-mask audits for strut-aligned 2-D projection calls.

Each candidate is shown twice: the maximum raw-intensity projection through the
strut-local slab and the matching binary maximum projection.  Low-x and high-x examples
are deliberately separated so registration drift can be visually distinguished from a
credible missing/disconnected strut candidate.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from project_struts_2d import bases


def raw_project(vol, a, b, width=3, depth=3, n_s=33):
    e,n1,n2=bases(a[None],b[None]); ts=np.linspace(.15,.85,n_s);u=np.arange(-width,width+1);w=np.arange(-depth,depth+1)
    c=a[None,None,:]+(b-a)[None,None,:]*ts[None,:,None]
    p=c[:,:,None,None,:]+u[None,None,:,None,None]*n1[:,None,None,None,:]+w[None,None,None,:,None]*n2[:,None,None,None,:]
    q=np.rint(p).astype(int);shape=np.asarray(vol.shape);good=((q>=0)&(q<shape)).all(-1);q=np.clip(q,0,shape-1)
    x=vol[q[...,0],q[...,1],q[...,2]].astype(float);x[~good]=0
    return x.max(3)[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--raw",default="data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif");ap.add_argument("--design",default="data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json");ap.add_argument("--results",default="outputs/method_comparison/strut_aligned_2d_projection/raw_measurements.npz");ap.add_argument("--out",default="outputs/method_comparison/strut_aligned_2d_projection/candidate_raw_ct_audit.png");args=ap.parse_args()
    d=np.load(args.results);v=d["verdict"].astype(str);mid=d["midpoint"];cov=d["projected_coverage"];eligible=d["eligible"]
    g=json.loads(Path(args.design).read_text());pos=np.asarray([j["position"] for j in g["junctions"]],float)[:,[2,1,0]];pairs=d["pairs"];a,b=pos[pairs[:,0]],pos[pairs[:,1]]
    # 3 low-drift and 3 high-drift examples for each call type, ordered by coverage.
    chosen=[]
    for lab in ["missing","disconnected"]:
        for name,region in [("low x",eligible&(mid[:,2]<=300)),("high x",eligible&(mid[:,2]>500))]:
            ind=np.where((v==lab)&region)[0]; chosen.append((lab,name,ind[np.argsort(cov[ind])[:3]]))
    print("loading raw CT for",sum(len(x[2]) for x in chosen),"audit candidates")
    vol=tifffile.imread(args.raw)
    fig,axes=plt.subplots(4,6,figsize=(14,9),squeeze=False)
    for row,(lab,region,ids) in enumerate(chosen):
        for j,i in enumerate(ids):
            raw=raw_project(vol,a[i],b[i]); lo,hi=np.percentile(raw,[1,99.7]);
            ax=axes[row,2*j];ax.imshow(raw.T,cmap="gray",origin="lower",aspect="auto",vmin=lo,vmax=max(hi,lo+1));ax.set_title(f"raw #{i}\n{lab}, {region}",fontsize=8)
            ax=axes[row,2*j+1];ax.imshow((raw>40127).T,cmap="gray",origin="lower",aspect="auto",vmin=0,vmax=1);ax.set_title(f"thresholded projection\nx={mid[i,2]:.0f}, c={cov[i]:.2f}",fontsize=8)
            for aa in axes[row,2*j:2*j+2]:aa.set_xticks([]);aa.set_yticks([])
    fig.suptitle("Strut-aligned 2-D audit: raw CT max projection (left) vs thresholded projection (right)",fontsize=13)
    fig.text(.5,.01,"Rows: missing low-x, missing high-x, disconnected low-x, disconnected high-x.  High-x examples test registration-drift confounding.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.96));Path(args.out).parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.out,dpi=180);print("wrote",args.out)

if __name__=="__main__":main()
