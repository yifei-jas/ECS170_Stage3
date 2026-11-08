"""Train CNNs with an internal validation split and no test-driven selection."""
import copy
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from local_code.base_class.setting import setting
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Evaluate_Classification import Evaluate_Classification

MODELS={'MNIST':Method_CNN_MNIST,'ORL':Method_CNN_ORL,'CIFAR':Method_CNN_CIFAR}


def prepare_images(raw, name, device):
    x=torch.as_tensor(raw,device=device).float().div_(255)
    if name=='ORL':
        x=F.interpolate(x,size=(56,46),mode='bilinear',align_corners=False)
    return x


def augment_cifar(x):
    # Independently crop each image after four-pixel reflection padding.
    n,c,h,w=x.shape
    padded=F.pad(x,(4,4,4,4),mode='reflect')
    offsets=torch.randint(0,9,(n,2),device=x.device)
    yi=offsets[:,0,None,None]+torch.arange(h,device=x.device)[None,:,None]
    xi=offsets[:,1,None,None]+torch.arange(w,device=x.device)[None,None,:]
    cropped=padded.permute(0,2,3,1)[torch.arange(n,device=x.device)[:,None,None],yi,xi].permute(0,3,1,2)
    flipped=torch.rand(n,device=x.device)<.5
    return torch.where(flipped[:,None,None,None],cropped.flip(-1),cropped)


@torch.inference_mode()
def measure(model,x,y,mean,std,batch_size=128):
    model.eval()
    total_loss,correct=0.,0
    predictions=[]
    for start in range(0,len(y),batch_size):
        xb=(x[start:start+batch_size]-mean)/std
        yb=y[start:start+batch_size]
        logits=model(xb)
        total_loss+=F.cross_entropy(logits,yb,reduction='sum').item()
        prediction=logits.argmax(1)
        correct+=(prediction==yb).sum().item()
        predictions.append(prediction.cpu().numpy())
    return total_loss/len(y),correct/len(y),np.concatenate(predictions)


class Setting_Image_Classification(setting):
    def __init__(self,config,loaded,output):
        super().__init__('Image classification','Fixed instructor split with training holdout')
        self.config=config
        self.loaded=loaded
        self.output=Path(output)

    def load_run_save_evaluate(self):
        cfg=self.config
        self.output.mkdir(parents=True,exist_ok=False)
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        device=torch.device(cfg['device'])
        name=cfg['dataset']
        raw=self.loaded['train']
        ids=np.arange(len(raw['y']))
        ti,vi=train_test_split(ids,test_size=40 if name=='ORL' else .1,
                              stratify=raw['y'],random_state=cfg['seed'])
        np.savez(self.output/'split_indices.npz',train=ti,validation=vi)
        x=prepare_images(raw['X'],name,device)
        y=torch.as_tensor(raw['y'],device=device)
        xt,yt=x[ti],y[ti]
        xv,yv=x[vi],y[vi]
        del x,y
        mean=xt.mean(dim=(0,2,3),keepdim=True)
        std=xt.std(dim=(0,2,3),keepdim=True).clamp_min(1e-6)
        model=MODELS[name](cfg['variant']).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=1e-5)
        best=(-1.,-float('inf'))
        best_state=None
        best_epoch=0
        history=[]
        start=time.perf_counter()
        for epoch in range(1,cfg['epochs']+1):
            model.train()
            order=torch.randperm(len(yt),device=device)
            loss_sum,correct=0.,0
            epoch_lr=optimizer.param_groups[0]['lr']
            for indices in order.split(cfg['batch_size']):
                xb,yb=xt[indices],yt[indices]
                if name=='CIFAR': xb=augment_cifar(xb)
                xb=(xb-mean)/std
                optimizer.zero_grad(set_to_none=True)
                logits=model(xb)
                loss=F.cross_entropy(logits,yb)
                loss.backward()
                optimizer.step()
                loss_sum+=loss.item()*len(yb)
                correct+=(logits.detach().argmax(1)==yb).sum().item()
            vl,va,_=measure(model,xv,yv,mean,std)
            history.append({'epoch':epoch,'learning_rate':epoch_lr,'train_loss':loss_sum/len(yt),
                            'train_accuracy':correct/len(yt),'validation_loss':vl,'validation_accuracy':va})
            if (va,-vl)>best:
                best=(va,-vl); best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            scheduler.step()
            if epoch<=3 or epoch%5==0 or epoch==cfg['epochs']:
                print(f'{name}/{cfg["variant"]} epoch {epoch}/{cfg["epochs"]}: '
                      f'loss={loss_sum/len(yt):.4f}, val={va:.2%}, '
                      f'elapsed={time.perf_counter()-start:.1f}s',flush=True)
            with (self.output/'history.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
        elapsed=time.perf_counter()-start
        model.load_state_dict(best_state)
        vl,va,pred=measure(model,xv,yv,mean,std)
        evaluator=Evaluate_Classification(40 if name=='ORL' else 10)
        evaluator.data={'true_y':yv.cpu().numpy(),'pred_y':pred}
        report={'config':cfg,'best_epoch':best_epoch,'validation_loss':vl,'validation':evaluator.evaluate(),
                'fit_samples':len(ti),'validation_samples':len(vi),'test_samples':len(self.loaded['test']['y']),
                'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':elapsed,
                'normalization_mean':mean.flatten().cpu().tolist(),'normalization_std':std.flatten().cpu().tolist(),
                'input_shape':list(xt.shape[1:]),'label_offset':1 if name=='ORL' else 0,
                'initialization':'PyTorch Conv2d/Linear/BatchNorm defaults',
                'environment':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}}
        (self.output/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        torch.save({'state_dict':best_state,'config':cfg,'mean':mean.cpu(),'std':std.cpu(),
                    'best_epoch':best_epoch},self.output/'model.pt')
        np.savetxt(self.output/'validation_predictions.csv',np.column_stack([vi,raw['y'][vi],pred]),
                   delimiter=',',fmt='%d',header='train_row,true_label,predicted_label',comments='')
        print(f'SAVED {name}/{cfg["variant"]}: val={va:.2%}, epoch={best_epoch}, time={elapsed:.1f}s',flush=True)
        return report
