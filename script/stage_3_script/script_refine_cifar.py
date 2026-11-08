"""Select BatchNorm calibration and deterministic inference views on validation only."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR_Residual import Method_CNN_CIFAR_Residual
from local_code.stage_3_code.Evaluate_Classification import Evaluate_Classification


@torch.inference_mode()
def evaluate_views(model,x,y,mean,std,policy,batch_size=128):
    model.eval();loss=0.;correct=0;predictions=[]
    for start in range(0,len(y),batch_size):
        raw=x[start:start+batch_size].float().div(255)
        labels=y[start:start+batch_size]
        padded=F.pad(raw,(2,2,2,2),mode='reflect') if policy=='ten_crop' else None
        positions=[(2,2),(0,0),(0,4),(4,0),(4,4)] if padded is not None else [(0,0)]
        total=None;count=0
        for yy,xx in positions:
            view=padded[:,:,yy:yy+32,xx:xx+32] if padded is not None else raw
            view=((view-mean)/std).contiguous(memory_format=torch.channels_last)
            for flip in (False,True):
                with torch.amp.autocast('cuda',dtype=torch.float16):
                    logits=model(view.flip(-1) if flip else view).float()
                total=logits if total is None else total+logits
                count+=1
        logits=total/count
        loss+=F.cross_entropy(logits,labels,reduction='sum').item()
        pred=logits.argmax(1);correct+=(pred==labels).sum().item()
        predictions.append(pred.cpu().numpy())
    return loss/len(y),correct/len(y),np.concatenate(predictions)


@torch.no_grad()
def calibrate_bn(model,images,mean,std):
    modules=[m for m in model.modules() if isinstance(m,nn.BatchNorm2d)]
    momenta=[m.momentum for m in modules]
    for m in modules:m.reset_running_stats();m.momentum=None
    model.train()
    for raw in images.split(256):
        x=((raw.float().div(255)-mean)/std).contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast('cuda',dtype=torch.float16):model(x)
    for m,momentum in zip(modules,momenta):m.momentum=momentum
    model.eval()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--evaluate-test',action='store_true')
    parser.add_argument('--quick',action='store_true',help='Validation-only probe: last epoch, fit-calibrated BN, flip inference')
    args=parser.parse_args()
    if args.quick and args.evaluate_test:parser.error('--quick is validation-only')
    output=args.output_dir or args.run_dir/'refined'
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    best=torch.load(args.run_dir/'model.pt',map_location='cpu',weights_only=True)
    latest=torch.load(args.run_dir/'latest.pt',map_location='cpu',weights_only=True)
    cfg=best['config']
    data=Dataset_Loader(ROOT/'stage_3_data','CIFAR').load()
    with np.load(args.run_dir/'split_indices.npz') as split:ti,vi=split['train'],split['validation']
    xt=torch.as_tensor(data['train']['X'][ti],device='cuda')
    xv=torch.as_tensor(data['train']['X'][vi],device='cuda');yv=torch.as_tensor(data['train']['y'][vi],device='cuda')
    mean,std=best['mean'].cuda(),best['std'].cuda()
    model=Method_CNN_CIFAR_Residual(cfg['depth'],cfg['width']).cuda().to(memory_format=torch.channels_last)
    candidates=[];winner=None;best_score=(-1.,-float('inf'))
    policies=['flip'] if args.quick else ['flip','ten_crop']
    sources=[('last_epoch',latest,latest['epoch'])] if args.quick else [
        ('training_best',best,best['best_epoch']),('last_epoch',latest,latest['epoch'])]
    if not args.quick:
        averaged={key:(value.float()+latest['state_dict'][key].float())*.5
                  if value.is_floating_point() else latest['state_dict'][key]
                  for key,value in best['state_dict'].items()}
        sources.append(('weight_average',{'state_dict':averaged},latest['epoch']))
    calibrated_options=(True,) if args.quick else (False,True)
    plan={'weight_candidates':[v[0] for v in sources],
          'batchnorm_candidates':['fit_only_calibration'] if args.quick else ['as_trained','fit_only_calibration'],
          'inference_candidates':policies,'calibration_samples':len(ti),
          'weight_average_requires_calibration':True,
          'selection_rule':'Validation accuracy then lower validation loss; test excluded'}
    (output/'refinement_plan.json').write_text(json.dumps(plan,indent=2))
    for source,checkpoint,epoch in sources:
        for calibrated in calibrated_options:
            if source=='weight_average' and not calibrated:continue
            model.load_state_dict(checkpoint['state_dict'])
            if calibrated:calibrate_bn(model,xt,mean,std)
            for policy in policies:
                loss,accuracy,pred=evaluate_views(model,xv,yv,mean,std,policy)
                entry={'source':source,'epoch':epoch,'batchnorm_calibrated':calibrated,
                       'inference_policy':policy,'validation_accuracy':accuracy,'validation_loss':loss}
                if source=='weight_average':entry['averaged_epochs']=[best['best_epoch'],latest['epoch']]
                candidates.append(entry)
                print('VALIDATION CANDIDATE',entry,flush=True)
                if (accuracy,-loss)>best_score:
                    best_score=(accuracy,-loss);winner=entry
                    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    torch.save({'state_dict':state,'config':cfg,'mean':best['mean'],'std':best['std'],
                                'best_epoch':epoch,'inference_policy':policy,
                                'batchnorm_calibrated':calibrated},output/'model.pt')
                    np.savetxt(output/'validation_predictions.csv',np.column_stack([vi,data['train']['y'][vi],pred]),
                               delimiter=',',fmt='%d',header='train_row,true_label,predicted_label',comments='')
    (output/'validation_candidates.json').write_text(json.dumps(candidates,indent=2))
    (output/'selection.json').write_text(json.dumps(winner,indent=2))
    report={'config':cfg,'best_epoch':winner['epoch'],'validation_accuracy':winner['validation_accuracy'],
            'validation_loss':winner['validation_loss'],'fit_samples':len(ti),'validation_samples':len(vi),
            'parameters':sum(p.numel() for p in model.parameters()),'epochs_completed':latest['epoch'],
            'postprocessing':winner,'normalization_mean':mean.flatten().cpu().tolist(),
            'normalization_std':std.flatten().cpu().tolist()}
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    shutil.copyfile(args.run_dir/'history.csv',output/'history.csv')
    shutil.copyfile(args.run_dir/'split_indices.npz',output/'split_indices.npz')
    if args.evaluate_test:
        selected=torch.load(output/'model.pt',map_location='cuda',weights_only=True)
        model.load_state_dict(selected['state_dict'])
        tx=torch.as_tensor(data['test']['X'],device='cuda');ty=torch.as_tensor(data['test']['y'],device='cuda')
        loss,accuracy,pred=evaluate_views(model,tx,ty,mean,std,selected['inference_policy'])
        evaluator=Evaluate_Classification(10);evaluator.data={'true_y':data['test']['y'],'pred_y':pred}
        scores=evaluator.evaluate();scores['loss']=loss
        (output/'test_metrics.json').write_text(json.dumps(scores,indent=2))
        np.savetxt(output/'test_predictions.csv',np.column_stack([data['test']['y'],pred]),delimiter=',',fmt='%d',
                   header='true_label,predicted_label',comments='')
        print(f'LOCKED TEST ACCURACY {accuracy:.2%}',flush=True)
    print('REFINEMENT COMPLETE',output,flush=True)


if __name__=='__main__':main()
