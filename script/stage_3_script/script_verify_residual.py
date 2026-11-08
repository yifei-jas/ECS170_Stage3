"""Verify the residual checkpoint and report a final comparison without retraining."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from script_cifar_residual import evaluate
from script_refine_cifar import evaluate_views
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR_Residual import Method_CNN_CIFAR_Residual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    checkpoint=torch.load(args.run_dir/'model.pt',map_location='cuda',weights_only=True)
    cfg=checkpoint['config']
    model=Method_CNN_CIFAR_Residual(cfg['depth'],cfg['width']).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint['state_dict'])
    data=Dataset_Loader(ROOT/'stage_3_data','CIFAR').load()
    tx=torch.as_tensor(data['test']['X'],device='cuda');ty=torch.as_tensor(data['test']['y'],device='cuda')
    if 'inference_policy' in checkpoint:
        _,accuracy,pred=evaluate_views(model,tx,ty,checkpoint['mean'],checkpoint['std'],checkpoint['inference_policy'])
    else:
        _,accuracy,pred=evaluate(model,tx,ty,checkpoint['mean'],checkpoint['std'])
    saved=np.loadtxt(args.run_dir/'test_predictions.csv',delimiter=',',skiprows=1,dtype=int)
    np.testing.assert_array_equal(saved[:,0],data['test']['y'])
    np.testing.assert_array_equal(saved[:,1],pred)
    scores=json.loads((args.run_dir/'test_metrics.json').read_text())
    np.testing.assert_allclose(accuracy,scores['accuracy'],rtol=0,atol=1e-12)
    report=json.loads((args.run_dir/'metrics.json').read_text())
    baseline=json.loads((ROOT/'result/stage_3_result/comparison_v2/comparison.json').read_text())
    for r in baseline:
        if r['dataset']=='CIFAR':r['selected']=False
    baseline.append({'dataset':'CIFAR','variant':f'WRN-{cfg["depth"]}-{cfg["width"]}',
        'selected':True,'best_epoch':report['best_epoch'],'parameters':report['parameters'],
        'validation_accuracy':report['validation_accuracy'],
        **{f'test_{k}':v for k,v in scores.items() if isinstance(v,(int,float))}})
    (ROOT/'result/stage_3_result/final_comparison.json').write_text(json.dumps(baseline,indent=2))
    print(f'VERIFIED residual model: all {len(pred)} predictions match; test accuracy={accuracy:.2%}',flush=True)


if __name__=='__main__':main()
