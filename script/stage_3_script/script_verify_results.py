"""Reproduce saved test predictions without retraining or model selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Setting_Image_Classification import MODELS,prepare_images,measure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    rows=json.loads((args.run_dir/'comparison.json').read_text())
    for name in dict.fromkeys(r['dataset'] for r in rows):
        selection=json.loads((args.run_dir/name/'selection.json').read_text())
        with (ROOT/'stage_3_data'/name).open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==selection['dataset_sha256']
        data=Dataset_Loader(ROOT/'stage_3_data',name).load()
        for row in [r for r in rows if r['dataset']==name]:
            path=args.run_dir/name/row['variant']
            config=json.loads((path/'metrics.json').read_text())['config']
            device=torch.device(config['device'])
            checkpoint=torch.load(path/'model.pt',map_location=device,weights_only=True)
            model=MODELS[name](config['variant']).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            tx=prepare_images(data['test']['X'],name,device)
            ty=torch.as_tensor(data['test']['y'],device=device)
            _,acc,pred=measure(model,tx,ty,checkpoint['mean'],checkpoint['std'])
            saved=np.loadtxt(path/'test_predictions.csv',delimiter=',',skiprows=1,dtype=int)
            np.testing.assert_array_equal(saved[:,0],data['test']['y'])
            np.testing.assert_array_equal(saved[:,1],pred)
            np.testing.assert_allclose(acc,row['test_accuracy'],rtol=0,atol=1e-12)
            with np.load(path/'split_indices.npz') as split:
                assert len(set(split['train']).intersection(split['validation']))==0
                assert len(split['train'])+len(split['validation'])==len(data['train']['y'])
                assert set(data['train']['y'][split['train']])==set(range(Dataset_Loader(ROOT/'stage_3_data',name).classes))
            print(f'VERIFIED {name}/{row["variant"]}: {len(pred)} predictions, accuracy={acc:.2%}',flush=True)
            del tx,ty,model
            if device.type=='cuda':torch.cuda.empty_cache()
    print('All saved models and predictions verified.',flush=True)


if __name__=='__main__':
    main()
