"""End-to-end check using ephemeral synthetic signals; no bundled checkpoints."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from data_pipeline import make_pair, Signals
from models import build_model, vit_config
from run import config_from_args, train, evaluate, ROOT, MODEL_NAMES
from summarize import summarize


def main():
    torch.set_num_threads(2)
    raw=np.sin(np.arange(1024)*0.04).astype(np.float32)
    # Verify AWGN is not normalized and follows the stated normalization exactly.
    seed=12; snr=-12
    noise=np.random.default_rng(seed).standard_normal(raw.size)
    noisy=(raw+noise*np.sqrt((np.mean(raw.astype(np.float64)**2)+1e-12)/10**(snr/10))).astype(np.float32)
    x,y=make_pair(raw,snr,'white',np.random.default_rng(seed))
    np.testing.assert_allclose(y,(noisy-noisy.mean())/(noisy.std()+1e-8))
    np.testing.assert_allclose(x,(raw-noisy.mean())/(noisy.std()+1e-8))
    with tempfile.TemporaryDirectory(prefix='denoise_release_smoke_') as tmp:
        tmp=Path(tmp)
        rng=np.random.default_rng(7)
        np.save(tmp/'train.npy',rng.normal(size=(5,1024)).astype(np.float32))
        np.save(tmp/'test.npy',rng.normal(size=(2,1024)).astype(np.float32))
        for name in MODEL_NAMES:
            for noise_type in ('white','pink'):
                for target in ('x-pred','n-pred'):
                    args=argparse.Namespace(config=str(ROOT/'configs/vibration.json'),model=name,
                         target=target,noise=noise_type,seed=42,epochs=1,batch_size=2,
                         train_path=str(tmp/'train.npy'),test_path=str(tmp/'test.npy'),
                         vit_configuration=3,model_options=None)
                    cfg=config_from_args(args)
                    out=tmp/'outputs'/name/noise_type/target
                    train(cfg,out,torch.device('cpu'))
                    rows=evaluate(out,torch.device('cpu'))
                    assert len(rows)==7
                    assert all(np.isfinite(r['rmse']) and np.isfinite(r['snr_improvement_db']) for r in rows)
        summarize(tmp/'outputs',tmp/'summary')
        import csv
        with (tmp/'summary/C_s_per_seed.csv').open() as f: cs=list(csv.DictReader(f))
        assert len(cs)==35
        ds1=Signals(tmp/'test.npy',1024,'pink',seed=42,snr=-12)
        ds2=Signals(tmp/'test.npy',1024,'pink',seed=42,snr=-12)
        assert torch.equal(ds1[0][1],ds2[0][1])
        del ds1, ds2  # Release Windows memory-mapped file handles before cleanup.
        import subprocess
        checkpoint=tmp/'outputs/TEM1Dformer/white/x-pred'
        subprocess.run([sys.executable,'-B',str(ROOT/'infer.py'),'--run-dir',str(checkpoint),
                        '--input',str(tmp/'test.npy'),'--output',str(tmp/'denoised.npy'),
                        '--device','cpu'],check=True)
        assert np.load(tmp/'denoised.npy').shape==(2,1024)
        # Capacity presets and input lengths.
        for dataset in ('Audio','Seismic','ECG','Vibration'):
            for c in range(1,6):
                config=vit_config(dataset,c)
                model=build_model('1D-ViT',**config).eval()
                with torch.no_grad(): out=model(torch.randn(1,1,config['signal_length']))
                assert out.shape[-1]==config['signal_length']
    print('PASS: 20 train/test jobs, 20 ViT presets, paired noise, AWGN formula, 35 C(s) rows.')


if __name__=='__main__': main()
