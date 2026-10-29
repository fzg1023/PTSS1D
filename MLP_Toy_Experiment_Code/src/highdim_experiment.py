"""Higher-dimensional phase signals; unchanged baseline model/training/evaluation."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

import run_experiment as base
from run_experiment import *


PROTOCOL = {**base.PROTOCOL,
    'highdim_extension': {'d':[16,32,64], 'D':[256], 'H':[32,64,128,256], 'sigma':[1.0]},
    'combined_grid': {'d':[8,16,32,64], 'D':[256], 'H':[32,64,128,256], 'sigma':[1.0]},
    'phase_generation_extension': 'Original eight phase columns unchanged; 56 additional iid phases from an independent fixed stream, shared across d>8.',
}


def make_cases():
    return [dict(stage='highdim_extension',d=d,D=256,H=H,sigma=1.0)
            for d in (16,32,64) for H in (32,64,128,256)]


def split_data(count,d,D,sigma,seed,split,device):
    if d<=8:
        return base.split_data(count,d,D,sigma,seed,split,device)
    if d not in (16,32,64) or D<=2*d:
        raise ValueError('High-dimensional extension requires d=16/32/64 and D>2d.')
    phase_rng=np.random.default_rng(base.seed_for('phases',seed,split))
    extra_rng=np.random.default_rng(base.seed_for('extended_phases_v1',seed,split))
    legacy=phase_rng.uniform(0.0,2.0*np.pi,size=(count,8))
    extra=extra_rng.uniform(0.0,2.0*np.pi,size=(count,56))
    phases=np.concatenate((legacy,extra),axis=1)[:,:d]
    grid=np.arange(D,dtype=np.float64)/D
    x=np.zeros((count,D),dtype=np.float64)
    for k in range(1,d+1):
        x+=np.cos(2.0*np.pi*k*grid[None,:]+phases[:,k-1:k])
    x*=math.sqrt(2.0/d)
    noise_rng=np.random.default_rng(base.seed_for('epsilon',seed,D,split))
    n=sigma*noise_rng.standard_normal(size=(count,D))
    assert np.max(np.abs(np.mean(x*x,axis=1)-1.0))<1e-11
    digest=hashlib.sha256(x.astype('<f4').tobytes()+n.astype('<f4').tobytes()).hexdigest()
    xt=torch.as_tensor(x.astype(np.float32),device=device)
    nt=torch.as_tensor(n.astype(np.float32),device=device)
    return dict(x=xt,n=nt,y=xt+nt,hash=digest)


def self_test(device):
    base.self_test(device)
    previous=None
    for d in (8,16,32,64):
        a=split_data(128,d,256,1.0,2026,'extension_selftest',device)
        b=split_data(128,d,256,4.0,2026,'extension_selftest',device)
        assert torch.equal(a['x'],b['x']) and torch.equal(4*a['n'],b['n'])
        if previous is not None:
            assert torch.equal(previous['n'],a['n'])
        assert torch.max(torch.abs(a['x'].square().mean(dim=1)-1.0))<1e-5
        spectrum=torch.fft.rfft(a['x'].double(),dim=1)
        expected_magnitude=256/math.sqrt(2*d)
        assert torch.allclose(spectrum[:,1:d+1].abs(),torch.full_like(spectrum[:,1:d+1].abs(),expected_magnitude),rtol=1e-5,atol=1e-5)
        outside=spectrum[:,d+1:].abs().square().sum()+spectrum[:,0].abs().square().sum()
        assert (outside/spectrum.abs().square().sum()).item()<1e-10
        if previous is not None:
            # Nested phase coordinates share the same lower-frequency coefficients.
            old_d=d//2
            old_spectrum=torch.fft.rfft(previous['x'].double(),dim=1)
            assert torch.allclose(spectrum[:,1:old_d+1]*math.sqrt(d),old_spectrum[:,1:old_d+1]*math.sqrt(old_d),rtol=1e-5,atol=1e-4)
        previous=a
    print('HIGHDIM_SELF_TEST_OK: unit power, expected Fourier basis, nested phases, shared Gaussian noise',flush=True)


def train_one(*args,**kwargs):
    # The controller redirects this module's write_json per worker. Bridge that
    # hook into the unchanged baseline trainer to keep progress writes isolated.
    original=base.write_json
    base.write_json=globals()['write_json']
    try:
        return base.train_one(*args,**kwargs)
    finally:
        base.write_json=original
