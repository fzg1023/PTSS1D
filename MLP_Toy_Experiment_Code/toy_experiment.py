"""Portable entry point for the paired phase-torus experiments.

The historical data generators and training implementation in src/ are unchanged.
This launcher removes dependencies on prior results and local machine paths.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import run_experiment as core
import highdim_experiment as extended
import numpy as np
import torch


def source_hashes():
    paths = ['run_toy.py', 'configs.json', 'src/run_experiment.py',
             'src/highdim_experiment.py']
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def make_cases(config, suite):
    # Deduplicate overlapping conditions so an identical pair is trained only once.
    names = config['paper_suites'] if suite == 'paper' else [suite]
    seen, cases = set(), []
    for name in names:
        spec = config['suites'][name]
        for d, D, H, sigma in itertools.product(
                spec['d'], spec['D'], spec['H'], spec['sigma']):
            key = (d, D, H, float(sigma))
            if key not in seen:
                seen.add(key)
                cases.append(dict(stage='toy', d=d, D=D, H=H, sigma=float(sigma)))
    return cases


def check_pair(case, seed, output):
    rows = [json.loads((output / 'records' /
            (core.run_key(case, seed, t) + '.json')).read_text(encoding='utf-8'))
            for t in ('x-pred', 'n-pred')]
    fields = ['initial_state_sha256', 'train_data_sha256',
              'validation_data_sha256', 'test_data_sha256', 'parameter_count']
    for field in fields:
        if rows[0][field] != rows[1][field]:
            raise RuntimeError('Pair integrity failure: ' + field)


def export_views(cases, seeds, output, config):
    rows = [json.loads(p.read_text(encoding='utf-8'))
            for p in sorted((output / 'records').glob('*.json'))]
    for name in config['paper_suites']:
        spec = config['suites'][name]
        chosen = [r for r in rows if all(r[k] in spec[k]
                  for k in ('d', 'D', 'H', 'sigma'))]
        core.write_csv(output / (name + '_runs.csv'), chosen)
    core.write_json(output / 'completion.json', dict(
        expected_runs=len(cases) * len(seeds) * 2, completed_runs=len(rows),
        complete=len(rows) == len(cases) * len(seeds) * 2,
        pairing='checked using initialization and all three data-split hashes'))


def main():
    config = json.loads((ROOT / 'configs.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['paper', 'selftest', 'smoke'] +
                        list(config['suites']), default='paper')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seeds', type=int, nargs='+', default=config['seeds'])
    for name in ('epochs', 'minimum_epochs', 'patience', 'n_train', 'n_val',
                 'n_test', 'batch_size'):
        parser.add_argument('--' + name.replace('_', '-'), type=int,
                            default=config['training'][name])
    parser.add_argument('--lr', type=float, default=config['training']['lr'])
    parser.add_argument('--dry-run', action='store_true', help='List workload; do not train.')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be unique.')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable. Use --device cpu explicitly.')
    if args.lr <= 0 or any(getattr(args, k) <= 0 for k in
            ('epochs', 'minimum_epochs', 'patience', 'n_train', 'n_val', 'n_test', 'batch_size')):
        parser.error('Training sizes, epoch budgets and learning rate must be positive.')
    if args.suite == 'smoke':
        cases = [dict(stage='smoke', d=1, D=32, H=16, sigma=1.0),
                 dict(stage='smoke', d=16, D=256, H=16, sigma=4.0)]
        args.seeds = [2026]
        args.epochs = args.minimum_epochs = args.patience = 2
        args.n_train, args.n_val, args.n_test, args.batch_size = 128, 64, 64, 64
    elif args.suite == 'selftest':
        core.configure()
        extended.self_test(args.device)
        return
    else:
        cases = make_cases(config, args.suite)
    if args.minimum_epochs > args.epochs:
        parser.error('--minimum-epochs must not exceed --epochs.')
    output = (args.output or ROOT / 'outputs' / args.suite).resolve()
    print(json.dumps(dict(suite=args.suite, conditions=len(cases), seeds=args.seeds,
          training_runs=len(cases)*len(args.seeds)*2, output=str(output)), indent=2))
    if args.dry_run:
        return
    core.configure()
    extended.self_test(args.device)
    settings = {k: v for k, v in vars(args).items() if k not in ('output', 'dry_run')}
    signature = dict(settings=settings, cases=cases, source_sha256=source_hashes())
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding='utf-8'))
        if previous['signature'] != signature:
            raise RuntimeError('Output belongs to different code/settings. Choose a new --output.')
    else:
        core.write_json(protocol_path, dict(signature=signature,
            scientific_protocol=core.PROTOCOL,
            note='Actual executed grid, seeds and settings are in signature, not the legacy grid.',
            created_at=core.stamp(), python=sys.version, torch=torch.__version__,
            numpy=np.__version__, platform=platform.platform()))
    total = len(cases) * len(args.seeds) * 2
    started = time.perf_counter()
    try:
        for case in cases:
            for seed in args.seeds:
                keys = [core.run_key(case, seed, t) for t in ('x-pred', 'n-pred')]
                if not all((output / 'records' / (k + '.json')).exists() for k in keys):
                    data = {split: extended.split_data(count, case['d'], case['D'],
                            case['sigma'], seed, split, args.device)
                            for split, count in [('train', args.n_train),
                            ('val', args.n_val), ('test', args.n_test)]}
                    model = core.make_model(case['D'], case['H'],
                            core.seed_for('init', seed, case['D'], case['H']), args.device)
                    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    del model
                    for target in ('x-pred', 'n-pred'):
                        core.train_one(case, seed, target, data, initial, args, output,
                                       dict(total_runs=total, suite=args.suite))
                    del data, initial
                check_pair(case, seed, output)
                core.summarize(output)
        export_views(cases, args.seeds, output, config)
        core.write_json(output / 'status.json', dict(state='completed',
            total_runs=total, completed_runs=core.summarize(output),
            seconds=time.perf_counter()-started, updated_at=core.stamp()))
    except BaseException as error:
        core.write_json(output / 'status.json', dict(state='failed', error=repr(error),
                                                   updated_at=core.stamp()))
        raise


if __name__ == '__main__':
    main()
