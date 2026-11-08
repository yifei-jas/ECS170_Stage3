"""Generate figures and a comparison table from completed experiment records."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'result/stage_3_result/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    rows=json.loads((args.run_dir/'comparison.json').read_text())
    with (args.run_dir/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:
        path=args.run_dir/row['dataset']/row['variant']
        with (path/'history.csv').open(newline='') as f:history=list(csv.DictReader(f))
        fig,ax=plt.subplots(figsize=(6,3.2))
        for key,label in [('train_loss','Training'),('validation_loss','Validation')]:
            ax.plot([int(r['epoch']) for r in history],[float(r[key]) for r in history],label=label)
        ax.set(xlabel='Epoch',ylabel='Cross-entropy loss',title=f'{row["dataset"]}: {row["variant"]}')
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(path/'loss_curve.png',dpi=180)
        plt.close(fig)
    lines=['# Stage 3 experimental results','',
           '| Dataset | Configuration | Selected | Validation accuracy | Test accuracy | Macro F1 |',
           '|---|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["dataset"]} | {r["variant"]} | {r["selected"]} | '
                     f'{r["validation_accuracy"]:.2%} | {r["test_accuracy"]:.2%} | {r["test_f1_macro"]:.2%} |')
    lines+=['','All selection.json files are written before the corresponding dataset test scores.',
            'Models are initialized independently. The split, seed and epoch budget are fixed within each dataset.',
            'ORL labels are shifted from 1-40 to 0-39; use the stored label_offset to interpret predictions.',
            'ORL has 40 test images: one error changes accuracy by 2.5 percentage points.',
            'One seed only: these results do not establish statistical superiority.',
            'Training metrics include dropout and, for CIFAR, random crops/flips. Validation disables both.']
    (args.run_dir/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
