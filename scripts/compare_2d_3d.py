#!/usr/bin/env python
"""Create direct agreement visuals for strut-aligned 2-D and 3-D tube methods."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--three-d",default="outputs/method_comparison/comparison_arrays.npz");ap.add_argument("--two-d",default="outputs/method_comparison/strut_aligned_2d_projection/raw_measurements.npz");ap.add_argument("--out",default="outputs/method_comparison/2d_vs_3d");args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    a=np.load(args.three_d);b=np.load(args.two_d);eligible=a["eligible"]&b["eligible"];v3=a["tube_verdict"].astype(str);v2=b["verdict"].astype(str);x=a["midpoint"][:,2]
    labels=["nominal","missing","disconnected","uncertain"]
    cm=np.array([[(eligible&(v2==r)&(v3==c)).sum() for c in labels] for r in labels])
    np.savetxt(out/"confusion_matrix.csv",cm,fmt="%d",delimiter=",",header="2D\\3D,"+",".join(labels),comments="")
    fig,ax=plt.subplots(1,2,figsize=(12,4.7))
    im=ax[0].imshow(cm,cmap="Blues");ax[0].set(xticks=range(4),yticks=range(4),xticklabels=labels,yticklabels=labels,xlabel="3-D tube verdict",ylabel="2-D projection verdict",title="Eligible-strut verdict agreement")
    for i in range(4):
        for j in range(4):ax[0].text(j,i,f"{cm[i,j]:,}",ha="center",va="center",fontsize=9,color="white" if cm[i,j]>cm.max()*.5 else "black")
    fig.colorbar(im,ax=ax[0],fraction=.046,pad=.04,label="struts")
    bins=np.linspace(x[eligible].min(),x[eligible].max(),19);ctr=(bins[:-1]+bins[1:])/2
    for name,v,style in [("2-D max projection",v2,"-"),("3-D tube",v3,"--")]:
        for lab,col in [("missing","#e76f51"),("disconnected","#457b9d")]:
            y=[np.mean(v[eligible&(x>=lo)&(x<hi)]==lab) for lo,hi in zip(bins[:-1],bins[1:])]
            ax[1].plot(ctr,y,style,color=col,label=f"{name}: {lab}")
    ax[1].set(title="Shared registration-drift diagnostic",xlabel="midpoint x [vox]",ylabel="flag fraction");ax[1].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(out/"agreement_and_x_rates.png",dpi=180);plt.close(fig)
    same=(v2==v3)&eligible; summary={"eligible":int(eligible.sum()),"same":int(same.sum()),"agreement_fraction":float(same.sum()/eligible.sum()),"matrix":cm.tolist(),"labels":labels}
    (out/"summary.json").write_text(__import__("json").dumps(summary,indent=2));print(summary)
if __name__=="__main__":main()
