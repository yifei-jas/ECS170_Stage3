"""Train a stronger CIFAR model; use validation only until explicit test evaluation."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Setting_Image_Classification import augment_cifar
from local_code.stage_3_code.Method_CNN_CIFAR_Residual import Method_CNN_CIFAR_Residual
from local_code.stage_3_code.Evaluate_Classification import Evaluate_Classification


@torch.inference_mode()
def evaluate(model,x,y,mean,std,tta=True,batch_size=128):
    model.eval()
    loss,correct=0.,0
    predictions=[]
    for start in range(0,len(y),batch_size):
        xb=x[start:start+batch_size].float().div(255)
        xb=((xb-mean)/std).contiguous(memory_format=torch.channels_last)
        yb=y[start:start+batch_size]
        with torch.amp.autocast('cuda',dtype=torch.float16):
            logits=model(xb)
            if tta:logits=(logits+model(xb.flip(-1)))*.5
        logits=logits.float()
        loss+=F.cross_entropy(logits,yb,reduction='sum').item()
        pred=logits.argmax(1)
        correct+=(pred==yb).sum().item()
        predictions.append(pred.cpu().numpy())
    return loss/len(y),correct/len(y),np.concatenate(predictions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'result/stage_3_result/cifar_residual_v1')
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--width',type=int,default=4)
    parser.add_argument('--depth',type=int,default=28)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--evaluate-test',action='store_true')
    args=parser.parse_args()
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=args.resume or args.evaluate_test)
    cfg={'depth':args.depth,'width':args.width,'epochs':args.epochs,'batch_size':args.batch_size,
         'seed':42,'optimizer':'SGD','learning_rate':.1,'momentum':.9,'nesterov':True,
         'weight_decay':.0005,'warmup_epochs':5,'mixup_alpha':.2,'mixup_probability':.5,
         'cutout_size':16,'horizontal_flip_tta':True,'amp':True,'device':'cuda'}
    if not torch.cuda.is_available():raise RuntimeError('This training recipe requires a CUDA GPU')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    data=Dataset_Loader(ROOT/'stage_3_data','CIFAR').load()
    base=ROOT/'result/stage_3_result/comparison_v2/CIFAR/baseline'
    with np.load(base/'split_indices.npz') as split:
        ti,vi=split['train'],split['validation']
    np.savez(output/'split_indices.npz',train=ti,validation=vi)
    base_metrics=json.loads((base/'metrics.json').read_text())
    mean=torch.tensor(base_metrics['normalization_mean'],device='cuda').view(1,3,1,1)
    std=torch.tensor(base_metrics['normalization_std'],device='cuda').view(1,3,1,1)
    xt=torch.as_tensor(data['train']['X'][ti],device='cuda')
    yt=torch.as_tensor(data['train']['y'][ti],device='cuda')
    xv=torch.as_tensor(data['train']['X'][vi],device='cuda')
    yv=torch.as_tensor(data['train']['y'][vi],device='cuda')
    model=Method_CNN_CIFAR_Residual(args.depth,args.width).cuda().to(memory_format=torch.channels_last)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,nesterov=True,weight_decay=.0005)
    scaler=torch.amp.GradScaler('cuda')
    history=[];first_epoch=1;best_accuracy=-1.;best_loss=float('inf');best_epoch=0
    if args.resume and (output/'latest.pt').exists():
        latest=torch.load(output/'latest.pt',map_location='cuda',weights_only=True)
        if latest['config']!=cfg:raise ValueError('Resume configuration mismatch')
        model.load_state_dict(latest['state_dict']);optimizer.load_state_dict(latest['optimizer'])
        scaler.load_state_dict(latest['scaler']);history=latest['history'];first_epoch=latest['epoch']+1
        best_accuracy=latest['best_accuracy'];best_loss=latest['best_loss'];best_epoch=latest['best_epoch']
        torch.set_rng_state(latest['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([r.cpu() for r in latest['cuda_rng']])
        random.setstate(latest['python_rng'])
        s=latest['numpy_rng'];np.random.set_state((s[0],np.array(s[1],dtype=np.uint32),s[2],s[3],s[4]))
        print(f'RESUMED epoch {first_epoch}',flush=True)
    if args.evaluate_test:
        checkpoint=torch.load(output/'model.pt',map_location='cuda',weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        tx=torch.as_tensor(data['test']['X'],device='cuda');ty=torch.as_tensor(data['test']['y'],device='cuda')
        loss,accuracy,pred=evaluate(model,tx,ty,checkpoint['mean'],checkpoint['std'])
        metrics=Evaluate_Classification(10);metrics.data={'true_y':data['test']['y'],'pred_y':pred}
        result=metrics.evaluate();result['loss']=loss
        (output/'test_metrics.json').write_text(json.dumps(result,indent=2))
        np.savetxt(output/'test_predictions.csv',np.column_stack([data['test']['y'],pred]),delimiter=',',fmt='%d',
                   header='true_label,predicted_label',comments='')
        print(f'TEST ACCURACY {accuracy:.2%}',flush=True)
        return
    if not args.resume:
        (output/'experiment_plan.json').write_text(json.dumps(cfg,indent=2))
    print(f'WRN-{args.depth}-{args.width}: {sum(p.numel() for p in model.parameters()):,} parameters',flush=True)
    start=time.perf_counter()
    for epoch in range(first_epoch,args.epochs+1):
        if epoch<=5:lr=.1*epoch/5
        else:lr=1e-5+(.1-1e-5)*.5*(1+math.cos(math.pi*(epoch-5)/(args.epochs-5)))
        for group in optimizer.param_groups:group['lr']=lr
        model.train();order=torch.randperm(len(yt),device='cuda');total_loss=0.
        for indices in order.split(args.batch_size):
            xb=augment_cifar(xt[indices].float().div_(255));yb=yt[indices]
            xb=(xb-mean)/std
            centers=torch.randint(0,32,(len(indices),2),device='cuda')
            yy=torch.arange(32,device='cuda')[None,:,None]
            xx=torch.arange(32,device='cuda')[None,None,:]
            mask=((yy-centers[:,0,None,None]).abs()>=8)|((xx-centers[:,1,None,None]).abs()>=8)
            xb=xb*mask[:,None,:,:]
            lam=1.;mixed_y=yb
            if np.random.rand()<.5:
                lam=float(np.random.beta(.2,.2))
                permutation=torch.randperm(len(indices),device='cuda')
                xb=lam*xb+(1-lam)*xb[permutation];mixed_y=yb[permutation]
            xb=xb.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',dtype=torch.float16):
                logits=model(xb)
                loss=lam*F.cross_entropy(logits,yb)+(1-lam)*F.cross_entropy(logits,mixed_y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            total_loss+=loss.item()*len(indices)
        vl,va,pred=evaluate(model,xv,yv,mean,std)
        row={'epoch':epoch,'learning_rate':lr,'train_loss':total_loss/len(yt),
             'validation_loss':vl,'validation_accuracy':va}
        history.append(row)
        if (va,-vl)>(best_accuracy,-best_loss):
            best_accuracy,best_loss,best_epoch=va,vl,epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'state_dict':state,'config':cfg,'mean':mean.cpu(),'std':std.cpu(),
                        'best_epoch':best_epoch},output/'model.pt')
            np.savetxt(output/'validation_predictions.csv',np.column_stack([vi,data['train']['y'][vi],pred]),
                       delimiter=',',fmt='%d',header='train_row,true_label,predicted_label',comments='')
        with (output/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        s=np.random.get_state()
        torch.save({'state_dict':model.state_dict(),'config':cfg,'optimizer':optimizer.state_dict(),
                    'scaler':scaler.state_dict(),'history':history,'epoch':epoch,
                    'best_accuracy':best_accuracy,'best_loss':best_loss,'best_epoch':best_epoch,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                    'python_rng':random.getstate(),'numpy_rng':(s[0],s[1].tolist(),s[2],s[3],s[4])},
                   output/'latest.pt')
        (output/'progress.json').write_text(json.dumps({'epoch':epoch,'best_epoch':best_epoch,
            'best_validation_accuracy':best_accuracy,'validation_accuracy':va,
            'elapsed_seconds':time.perf_counter()-start,'config':cfg},indent=2))
        print(f'WRN epoch {epoch}/{args.epochs}: loss={row["train_loss"]:.4f}, val={va:.2%}, '
              f'best={best_accuracy:.2%}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
    report={'config':cfg,'best_epoch':best_epoch,'validation_accuracy':best_accuracy,
            'validation_loss':best_loss,'fit_samples':len(ti),'validation_samples':len(vi),
            'parameters':sum(p.numel() for p in model.parameters()),
            'normalization_mean':mean.flatten().cpu().tolist(),'normalization_std':std.flatten().cpu().tolist(),
            'test_evaluation':'Not performed; use --evaluate-test after validation selection'}
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    print(f'VALIDATION TRAINING COMPLETE {output}',flush=True)


if __name__=='__main__':main()
