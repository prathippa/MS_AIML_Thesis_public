"""Plot the frozen, verified follow-up outputs without paid calls."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]
def main():
 data=json.loads((BASE/'results/prospective_v8a/product_analysis/results.json').read_text())
 fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
 styles={'confidence':('#333333','-'),'agreement':('#0072B2','--'),'schedule':('#009E73','-.'),'combined':('#D55E00','-')}
 for ax,m,name in zip(axes,'AB',['Claude designated','GPT-5 designated']):
  for policy,(color,style) in styles.items():
   values=data[m]['curves'][policy];ax.plot([r['coverage'] for r in values],[r['selective_risk'] for r in values],label=policy.capitalize(),color=color,ls=style,lw=1.8)
  ax.axvline(.5,color='#777777',ls=':',lw=1);ax.set(title=name,xlabel='Share selected (coverage)',xlim=(.1,1),ylim=(0,1));ax.grid(alpha=.18)
 axes[0].set_ylabel('Exact-code error rate among selected products')
 axes[1].legend(frameon=False,fontsize=8,loc='lower right');fig.tight_layout()
 out=BASE.parent/'thesis_source/img/fig4_6_prospective.png';fig.savefig(out,dpi=220,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
