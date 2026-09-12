"""Denoise raw noisy [N,L]/[N,1,L] arrays; restore original scale."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from models import build_model, reconstruct
from run import load_weights, device_for


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='auto')
    a=p.parse_args()
    if Path(a.output).exists(): raise FileExistsError('Output exists')
    cfg=json.loads((Path(a.run_dir)/'config.json').read_text(encoding='utf-8'))
    device=device_for(a.device)
    m=build_model(cfg['model'],**cfg['model_kwargs']).to(device)
    m.load_state_dict(load_weights(Path(a.run_dir)/'best.pt',device)); m.eval()
    y=np.load(a.input,mmap_mode='r',allow_pickle=False)
    if y.ndim not in (2,3) or (y.ndim==3 and y.shape[1]!=1): raise ValueError('Expected [N,L] or [N,1,L]')
    if y.shape[-1]!=cfg['signal_length']: raise ValueError('Input length must match training length')
    result=[]
    with torch.no_grad():
        for raw in y:
            raw=np.array(raw,dtype=np.float32).reshape(1,1,-1)
            if not np.isfinite(raw).all(): raise ValueError('Non-finite input')
            mean,std=raw.mean(),raw.std()+1e-8
            norm=torch.from_numpy((raw-mean)/std).to(device)
            clean=reconstruct(norm,m(norm),cfg['target']).cpu().numpy()*std+mean
            result.append(clean.reshape(-1))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    np.save(a.output,np.stack(result),allow_pickle=False)


if __name__=='__main__': main()
