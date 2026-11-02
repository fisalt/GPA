"""Redraw exported diagnostic arrays without importing torch or using a GPU."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument("arrays");p.add_argument("--output",required=True)
    args=p.parse_args();data=json.loads(Path(args.arrays).read_text())
    out=Path(args.output);count=data["levels"]
    matrices=[np.asarray(v) for v in data["relations"]]
    scale=max((float(np.abs(v).max()) for v in matrices),default=1.) or 1.
    fig,axes=plt.subplots(count,count,figsize=(max(6,2*count),max(5,2*count)),
                          squeeze=False,layout="constrained")
    images=iter(matrices);im=None
    for i in range(count):
        for j in range(count):
            ax=axes[i,j];ax.set_title(f"R {i} → {j}");ax.set_xticks([]);ax.set_yticks([])
            if i!=j:
                im=ax.imshow(next(images),aspect="auto",cmap="RdBu_r",vmin=-scale,vmax=scale)
            else:
                ax.text(.5,.5,"Self edge omitted",ha="center",va="center",transform=ax.transAxes)
                ax.set_axis_off()
    if im is not None:fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,label="Signed state value")
    precision=data.get('protocol',{}).get('dtype','unspecified precision')
    fig.suptitle(f"Dynamic R: last LM block, example 0\nHead mean; {precision}")
    fig.savefig(out.with_suffix('.png'),dpi=150,bbox_inches="tight");plt.close(fig)
    if data.get('routes') is not None:
        routes=np.asarray(data['routes'])
        fig,axes=plt.subplots(count,1,figsize=(10,max(3,2*count)),squeeze=False,layout="constrained")
        for i in range(count):
            axes[i,0].imshow(routes[i].T,aspect="auto",origin="lower",cmap="magma",vmin=0)
            axes[i,0].set_ylabel(f"level {i} slot")
        axes[-1,0].set_xlabel("token position")
        fig.suptitle("Normalized |route after MLP|\nLast LM block, example 0, head mean")
        fig.savefig(out.with_suffix('.routes.png'),dpi=150,bbox_inches="tight");plt.close(fig)
    if data.get('gates') is not None:
        fig,ax=plt.subplots(figsize=(6,3),layout="constrained")
        im=ax.imshow(data['gates'],aspect="auto",origin="lower",cmap="viridis")
        ax.set_xlabel("head");ax.set_ylabel("answer token order")
        ax.set_title("Read gate: mean absolute channel magnitude")
        fig.colorbar(im,ax=ax);fig.savefig(out.with_suffix('.gates.png'),dpi=150,bbox_inches="tight")
        plt.close(fig)


if __name__=='__main__':main()
