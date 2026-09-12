"""Train/test entry point. Run from this release directory."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline import Signals
from metrics import metrics
from models import DATASET_LENGTHS, build_model, get_config, vit_config, reconstruct, parameter_count


ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ("1D-ViT", "1D-U-Net", "1D-DnCNN", "L-WPT", "TEM1Dformer")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def device_for(value):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if value == 'auto' else torch.device(value)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_weights(path, device):
    # Only use checkpoints created by this release; never unpickle arbitrary objects.
    return torch.load(path, map_location=device, weights_only=True)


def protocol_hash(c):
    # Same protocol across architectures, target parameterizations and noise types.
    protocol = {k:c[k] for k in ('dataset','signal_length','epochs','batch_size',
                'learning_rate','weight_decay','grad_clip','validation_fraction','split_seed',
                'min_snr','max_snr','test_snrs','train_path','test_path')}
    protocol['implementation'] = 'uniform-awgn-release-v1'
    return hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()


def train(config, output, device):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Refusing to overwrite a nonempty run directory: ' + str(output))
    if Path(config['train_path']).resolve() == Path(config['test_path']).resolve():
        raise ValueError('Training and test arrays must be distinct, independently prepared splits')
    seed_all(config['seed'])
    count = len(np.load(config['train_path'], mmap_mode='r', allow_pickle=False))
    indices = np.random.default_rng(config['split_seed']).permutation(count)
    n_val = max(1, int(count * config['validation_fraction']))
    train_indices, val_indices = indices[n_val:], indices[:n_val]
    common = dict(path=config['train_path'], length=config['signal_length'],
                  noise_type=config['noise'], seed=config['seed'],
                  min_snr=config['min_snr'], max_snr=config['max_snr'])
    train_data = Signals(**common, indices=train_indices, training=True)
    val_data = Signals(**common, indices=val_indices)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(config, train_count=len(train_data), validation_count=len(val_data),
                  split_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                  torch_version=str(torch.__version__), numpy_version=np.__version__)
    (output/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    # Explicit split indices are small metadata, not signal data.
    (output/'split.json').write_text(json.dumps(dict(train=train_indices.tolist(),
                                                     validation=val_indices.tolist())), encoding='utf-8')
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                        num_workers=0, generator=generator)
    validation = DataLoader(val_data, batch_size=config['batch_size'], num_workers=0)
    model = build_model(config['model'], **config['model_kwargs']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    best, history = float('inf'), []
    for epoch in range(config['epochs']):
        train_data.epoch = epoch
        model.train()
        loss_sum = 0.
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            target = x if config['target'] == 'x-pred' else y-x
            loss = (model(y)-target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            loss_sum += loss.item()*x.shape[0]
        model.eval()
        val_sum = 0.
        with torch.no_grad():
            for x, y in validation:
                x, y = x.to(device), y.to(device)
                estimate = reconstruct(y, model(y), config['target'])
                val_sum += (estimate-x).square().mean().item()*x.shape[0]
        val_loss = val_sum/len(val_data)
        if not np.isfinite(val_loss):
            raise FloatingPointError('Nonfinite validation loss')
        scheduler.step()
        history.append(dict(epoch=epoch+1, train_mse=loss_sum/len(train_data), val_mse=val_loss))
        write_csv(output/'history.csv', history)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), output/'best.pt')
        print('%s %s %s epoch %d/%d train %.6g val %.6g' %
              (config['model'], config['noise'], config['target'], epoch+1, config['epochs'],
               history[-1]['train_mse'], val_loss), flush=True)
    return output/'best.pt'


def evaluate(run_dir, device, test_path=None, batch_size=None):
    run_dir = Path(run_dir)
    config = json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
    path = test_path or config['test_path']
    config['test_path'] = str(path)
    config['protocol_sha256'] = protocol_hash(config)
    if Path(path).resolve() == Path(config['train_path']).resolve():
        raise ValueError('Refusing to evaluate on the training file')
    model = build_model(config['model'], **config['model_kwargs']).to(device)
    model.load_state_dict(load_weights(run_dir/'best.pt', device), strict=True)
    model.eval()
    rows = []
    for snr in config['test_snrs']:
        ds = Signals(path, config['signal_length'], config['noise'], seed=config['seed']+100000,
                     snr=float(snr))
        loader = DataLoader(ds, batch_size=batch_size or config['batch_size'], num_workers=0)
        sample_rows = []
        with torch.no_grad():
            for x, y in loader:
                y_device = y.to(device)
                pred = reconstruct(y_device, model(y_device), config['target']).cpu().numpy()
                if not np.isfinite(pred).all():
                    raise FloatingPointError('Nonfinite test prediction')
                for a,b,c in zip(x.numpy(), y.numpy(), pred):
                    sample_rows.append(metrics(a,b,c))
        row = dict(dataset=config['dataset'], model=config['model'], target=config['target'],
                   noise=config['noise'], seed=config['seed'], input_snr_db=snr,
                   configuration=json.dumps(config['model_kwargs'], sort_keys=True),
                   parameters=parameter_count(model), samples=len(ds),
                   protocol_sha256=config['protocol_sha256'])
        for key in sample_rows[0]:
            values = np.array([r[key] for r in sample_rows])
            finite = values[np.isfinite(values)]
            row[key] = float(finite.mean()) if len(finite) else float('nan')
            row[key+'_sample_std'] = float(finite.std(ddof=1)) if len(finite)>1 else float('nan')
            row[key+'_valid_samples'] = len(finite)
        rows.append(row)
    write_csv(run_dir/'metrics.csv', rows)
    print('Saved', run_dir/'metrics.csv', flush=True)
    return rows


def config_from_args(args):
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    for key in ('dataset','model','target','noise','seed','epochs','batch_size','train_path','test_path'):
        value = getattr(args,key,None)
        if value is not None:
            c[key] = value
    if c['dataset'] not in DATASET_LENGTHS:
        raise ValueError('Unknown dataset')
    c['signal_length'] = DATASET_LENGTHS[c['dataset']]
    c['model_kwargs'] = (vit_config(c['dataset'], args.vit_configuration) if c['model']=='1D-ViT'
                         else get_config(c['model']))
    if args.model_options:
        c['model_kwargs'].update(json.loads(args.model_options))
    # Constructor validates model-specific options before a training run starts.
    build_model(c['model'], **c['model_kwargs'])
    if c['model']=='1D-ViT' and c['model_kwargs']['signal_length'] != c['signal_length']:
        raise ValueError('ViT signal_length must match dataset length')
    for key in ('epochs','batch_size'):
        if c[key] < 1: raise ValueError(key+' must be positive')
    if c['seed'] < 0 or not 0 < c['validation_fraction'] < 1:
        raise ValueError('Invalid seed or validation fraction')
    # Pair hash deliberately excludes architecture, target and random seed.
    c['protocol_sha256'] = protocol_hash(c)
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train','test','run'])
    p.add_argument('--config', default=str(ROOT/'configs/vibration.json'))
    p.add_argument('--model', choices=MODEL_NAMES)
    p.add_argument('--dataset', choices=list(DATASET_LENGTHS))
    p.add_argument('--target', choices=['x-pred','n-pred'])
    p.add_argument('--noise', choices=['white','pink'])
    p.add_argument('--seed', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--train-path')
    p.add_argument('--test-path')
    p.add_argument('--vit-configuration', type=int, choices=range(1,6), default=3)
    p.add_argument('--model-options', help='JSON object of native model constructor overrides')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    device = device_for(args.device)
    if args.action == 'test':
        evaluate(args.run_dir, device, args.test_path, args.batch_size)
    else:
        train(config_from_args(args), args.run_dir, device)
        if args.action == 'run': evaluate(args.run_dir, device)


if __name__ == '__main__':
    main()
