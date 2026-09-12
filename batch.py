"""Explicit experiment launcher; preview by default, use --execute to train."""
import argparse
import itertools
from pathlib import Path
import subprocess
import sys

from run import ROOT, MODEL_NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=['audio','seismic','ecg','vibration'],
                   default=['audio','seismic','ecg','vibration'])
    p.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument('--noises', nargs='+', choices=['white','pink'], default=['white','pink'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42,43,44,45,46])
    p.add_argument('--vit-configurations', nargs='+', type=int, choices=range(1,6), default=[3])
    p.add_argument('--output', default='outputs')
    p.add_argument('--device', default='auto')
    p.add_argument('--execute', action='store_true')
    args=p.parse_args()
    jobs=[]
    for ds,model,noise,target,seed in itertools.product(args.datasets,args.models,args.noises,
                                                       ['x-pred','n-pred'],args.seeds):
        for v in args.vit_configurations if model=='1D-ViT' else [3]:
            folder=Path(args.output)/ds/(model+('_c%d'%v if model=='1D-ViT' else ''))/noise/target/str(seed)
            cmd=[sys.executable,str(ROOT/'run.py'),'run','--config',str(ROOT/'configs'/(ds+'.json')),
                 '--model',model,'--noise',noise,'--target',target,'--seed',str(seed),
                 '--vit-configuration',str(v),'--run-dir',str(folder),'--device',args.device]
            jobs.append(cmd)
    print('Runs:',len(jobs),'(preview only)' if not args.execute else '(executing)',flush=True)
    for cmd in jobs:
        print(subprocess.list2cmdline(cmd),flush=True)
        if args.execute: subprocess.run(cmd,check=True)


if __name__=='__main__': main()
