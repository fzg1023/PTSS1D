"""Seed-level summary and paired C(s); sample SD and seed SD remain distinct."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from run import write_csv


def summarize(root, output):
    if Path(output).exists() and any(Path(output).iterdir()):
        raise FileExistsError('Use an empty summary directory to avoid stale result files')
    rows=[]
    for f in sorted(Path(root).rglob('metrics.csv')):
        with f.open(encoding='utf-8') as stream: rows.extend(csv.DictReader(stream))
    if not rows: raise ValueError('No metrics.csv found')
    keys=['dataset','model','configuration','input_snr_db','protocol_sha256']
    groups=defaultdict(list)
    paired={}
    for r in rows:
        ident=tuple(r[k] for k in keys)+ (r['noise'],r['target'],r['seed'])
        if ident in paired: raise ValueError('Duplicate experiment; remove duplicate inputs: '+str(ident))
        paired[ident]=r
        groups[ident[:-1]].append(r)
    summary=[]
    for key,rs in groups.items():
        row=dict(zip(keys+['noise','target'],key))
        row['n_seeds']=len(rs)
        row['seeds']=';'.join(sorted(r['seed'] for r in rs))
        for metric in ['snr_improvement_db','rmse','fpcc']:
            values=np.array([float(r[metric]) for r in rs])
            values=values[np.isfinite(values)]
            row[metric+'_mean']=float(values.mean()) if len(values) else float('nan')
            row[metric+'_seed_std']=float(values.std(ddof=1)) if len(values)>1 else float('nan')
            row[metric+'_valid_seeds']=len(values)
        summary.append(row)
    cs=[]
    bases={tuple(r[k] for k in keys)+(r['seed'],) for r in rows}
    missing=0
    for base in sorted(bases):
        prefix,seed=base[:-1],base[-1]
        needed=[prefix+(n,t,seed) for n,t in [('white','x-pred'),('white','n-pred'),
                                               ('pink','x-pred'),('pink','n-pred')]]
        if not all(k in paired for k in needed):
            missing+=1
            continue
        wx,wn,px,pn=[float(paired[k]['snr_improvement_db']) for k in needed]
        row=dict(zip(keys,prefix)); row.update(seed=seed,white_gap_db=wx-wn,
                                              pink_gap_db=px-pn,C_s_db=(wx-wn)-(px-pn))
        cs.append(row)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'all_metrics.csv',rows)
    write_csv(out/'seed_summary.csv',summary)
    if cs:
        write_csv(out/'C_s_per_seed.csv',cs)
        grouped=defaultdict(list)
        for r in cs: grouped[tuple(r[k] for k in keys)].append(r)
        result=[]
        for key,rs in grouped.items():
            v=np.array([r['C_s_db'] for r in rs])
            r=dict(zip(keys,key)); r.update(n_seeds=len(v),C_s_mean_db=float(v.mean()),
                     C_s_seed_std_db=float(v.std(ddof=1)) if len(v)>1 else float('nan'))
            result.append(r)
        write_csv(out/'C_s_summary.csv',result)
    print('Runs rows:',len(rows),'paired C(s):',len(cs),'incomplete quartets:',missing)
    print('Seed counts:',sorted({r['n_seeds'] for r in summary}),'; single-seed SD is NaN, not zero')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='outputs')
    p.add_argument('--output',default='summary')
    a=p.parse_args(); summarize(a.root,a.output)
