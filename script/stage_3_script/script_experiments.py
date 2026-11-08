"""Run predefined CNN comparisons, lock selection, then evaluate test images."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'result/stage_3_result/.matplotlib'))
import numpy as np
import torch
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Setting_Image_Classification import (
    Setting_Image_Classification,MODELS,prepare_images,measure)
from local_code.stage_3_code.Evaluate_Classification import Evaluate_Classification


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=['MNIST','ORL','CIFAR'],default=['MNIST','ORL','CIFAR'])
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--mnist-epochs',type=int,default=8)
    parser.add_argument('--orl-epochs',type=int,default=80)
    parser.add_argument('--cifar-epochs',type=int,default=40)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume',action='store_true',help='Reuse complete validation runs in the named output directory')
    args=parser.parse_args()
    output=args.output_dir or ROOT/'result/stage_3_result'/datetime.now().strftime('experiments_%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=args.resume)
    plans={name:{'dataset':name,'epochs':getattr(args,f'{name.lower()}_epochs'),
                 'batch_size':32 if name=='ORL' else 128,'learning_rate':.001,
                 'weight_decay':1e-4 if name=='CIFAR' else 0.,'seed':42,'device':args.device}
           for name in args.datasets}
    (output/'experiment_plan.json').write_text(json.dumps(plans,indent=2))
    summary=[]
    for name in args.datasets:
        print(f'\nLOADING {name}',flush=True)
        data=Dataset_Loader(ROOT/'stage_3_data',name).load()
        with (ROOT/'stage_3_data'/name).open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        reports={}
        for variant in ('baseline','deeper'):
            config=dict(plans[name],variant=variant)
            destination=output/name/variant
            if args.resume and (destination/'metrics.json').exists():
                report=json.loads((destination/'metrics.json').read_text())
                if report['config']!=config: raise ValueError(f'Configuration changed: {destination}')
            else:
                experiment=Setting_Image_Classification(config,data,destination)
                report=experiment.load_run_save_evaluate()
            reports[variant]=report
        with np.load(output/name/'baseline/split_indices.npz') as a,np.load(output/name/'deeper/split_indices.npz') as b:
            np.testing.assert_array_equal(a['train'],b['train'])
            np.testing.assert_array_equal(a['validation'],b['validation'])
        selected=max(reports,key=lambda v:(reports[v]['validation']['accuracy'],-reports[v]['validation_loss']))
        selection={'dataset':name,'selected_variant':selected,'dataset_sha256':digest,
                   'selection_rule':'Highest validation accuracy, then lower validation loss',
                   'test_not_used_for_selection':True}
        (output/name/'selection.json').write_text(json.dumps(selection,indent=2))
        device=torch.device(args.device)
        tx=prepare_images(data['test']['X'],name,device)
        ty=torch.as_tensor(data['test']['y'],device=device)
        for variant,report in reports.items():
            destination=output/name/variant
            checkpoint=torch.load(destination/'model.pt',map_location=device,weights_only=True)
            model=MODELS[name](variant).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            loss,acc,pred=measure(model,tx,ty,checkpoint['mean'],checkpoint['std'])
            evaluator=Evaluate_Classification(40 if name=='ORL' else 10)
            evaluator.data={'true_y':data['test']['y'],'pred_y':pred}
            metrics=evaluator.evaluate()
            metrics['loss']=loss
            (destination/'test_metrics.json').write_text(json.dumps(metrics,indent=2))
            np.savetxt(destination/'test_predictions.csv',np.column_stack([data['test']['y'],pred]),
                       delimiter=',',fmt='%d',header='true_label,predicted_label',comments='')
            summary.append({'dataset':name,'variant':variant,'selected':variant==selected,
                            'validation_accuracy':report['validation']['accuracy'],
                            'best_epoch':report['best_epoch'],'parameters':report['parameters'],
                            'training_seconds':report['training_seconds'],
                            **{f'test_{k}':v for k,v in metrics.items() if isinstance(v,(int,float))}})
            print(f'TEST {name}/{variant}: {acc:.2%}; selected={variant==selected}',flush=True)
        (output/'comparison.json').write_text(json.dumps(summary,indent=2))
        del data,tx,ty,model
        if device.type=='cuda': torch.cuda.empty_cache()
    print(f'COMPLETE {output}',flush=True)


if __name__=='__main__':
    main()
