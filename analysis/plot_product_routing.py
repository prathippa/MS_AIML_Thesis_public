"""Render the thesis figure from the complete, stored extension result JSON."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('results/extension_v8/product_routing_full/results.json'))
    ap.add_argument('--output',type=Path,default=Path('../thesis_source/img/fig4_5_product_routing.png'))
    a=ap.parse_args();data=json.loads(a.input.read_text())
    fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True,sharey=True)
    styles={'confidence':('#333333','-'),'agreement':('#0072B2','--'),'schedule':('#D55E00','-.'),'combined':('#009E73',':')}
    for ax,(study,m,title) in zip(axes.flat,[('main_v1','A','ATLAS: Claude'),('main_v1','B','ATLAS: GPT-5'),('hscc_v1','A','HSCodeComp: Claude'),('hscc_v1','B','HSCodeComp: GPT-5')]):
        for policy,rows in data[f'{study}/test/{m}']['curves'].items():
            color,style=styles[policy]
            ax.plot([r['coverage'] for r in rows],[r['selective_risk'] for r in rows],label=policy.capitalize(),color=color,linestyle=style)
        ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.set_ylim(0,1);ax.set_xlim(.1,1)
    for ax in axes[-1]:ax.set_xlabel('Product coverage')
    for ax in axes[:,0]:ax.set_ylabel('Exact-code selective risk')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
    fig.tight_layout(rect=(0,.06,1,1));a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=200);plt.close(fig)

if __name__=='__main__':main()
