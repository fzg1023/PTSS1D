"""Paired MLP experiments on phase-torus signals. No pre-existing results are used."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from scipy.stats import t as student_t
from torch import nn

ROOT = Path(__file__).resolve().parent
PROTOCOL = {
    "schema_version": 1,
    "family": "phase_torus",
    "formula": "x[j]=sqrt(2/d)*sum_k cos(2*pi*k*j/D+phi[k]), k=1..d",
    "latent": "phi[k] iid Uniform(0,2*pi); fixed sampling grid j/D",
    "signal_power_per_record": 1.0,
    "linear_span_dimension": "2*d, distinct from intrinsic manifold dimension d",
    "noise": "n=sigma*epsilon, epsilon iid N(0,1), independent of x",
    "noise_rescaled_per_record": False,
    "normalization": "No sample-wise normalization. Unit signal power follows analytically from construction.",
    "main": {"d": [1, 2, 4, 8], "D": [32, 64, 128, 256, 512, 1024],
             "H": [32, 64, 128, 256], "sigma": [1.0]},
    "capacity_extension": {"d": [1, 2, 4, 8], "D": [256], "H": [8, 1024], "sigma": [1.0]},
    "noise_extension": {"d": [1, 2, 4, 8], "D": [256], "H": [32, 128], "sigma": [0.25, 4.0]},
    "seeds": [2026, 2027, 2028, 2029, 2030],
    "n_train": 8192, "n_val": 2048, "n_test": 4096,
    "model": "Linear(D,H)-ReLU-Linear(H,H)-ReLU-Linear(H,D); no global skip",
    "bias": "Hidden layers use PyTorch default bias initialization; final bias is zero. Identical within each pair.",
    "optimizer": "Adam", "initial_lr": 0.001, "weight_decay": 0.0,
    "batch_size": 1024, "max_epochs": 1000, "minimum_epochs": 200,
    "early_stop_patience": 100,
    "lr_scheduler": "ReduceLROnPlateau(validation reconstruction MSE, factor=0.5, patience=20, min_lr=1e-5)",
    "checkpoint_selection": "Lowest validation clean-reconstruction MSE; test never used for selection",
    "pairing": "Same data, noise, initial weights, and epoch-specific sample order; optimizer instances independent",
    "data_across_H": "Identical for a given seed,d,D,sigma",
    "latent_across_D": "Same phase parameters for a given seed and split; uniform grids change with D",
    "data_across_sigma": "Identical clean signals and underlying epsilon for a given seed,d,D",
    "loss": "Direct target MSE, equal algebraically to clean-reconstruction MSE",
    "primary_metric": "Clean-reconstruction MSE averaged over test records and coordinates",
    "secondary_metrics": ["RMSE", "mean record-wise output SNR", "mean record-wise SNR improvement", "mean Pearson correlation"],
    "uncertainty": "Sample SD across independent seeds; 95% Student-t CI for paired seed differences",
    "scope": "A controlled phase-torus family; not a universal manifold or optimization-free experiment",
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def configure():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def make_model(D, H, seed, device):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, D)).to(device)
    nn.init.zeros_(model[-1].bias)
    return model


def split_data(count, d, D, sigma, seed, split, device):
    # Data seeds deliberately do not include H or sigma. Clean and epsilon are shared across those controls.
    phase_rng = np.random.default_rng(seed_for("phases", seed, split))
    noise_rng = np.random.default_rng(seed_for("epsilon", seed, D, split))
    phases = phase_rng.uniform(0.0, 2.0 * np.pi, size=(count, 8))[:, :d]
    grid = np.arange(D, dtype=np.float64) / D
    x = np.zeros((count, D), dtype=np.float64)
    for k in range(1, d + 1):
        x += np.cos(2.0 * np.pi * k * grid[None, :] + phases[:, k - 1:k])
    x *= math.sqrt(2.0 / d)
    epsilon = noise_rng.standard_normal(size=(count, D))
    n = sigma * epsilon
    assert D > 2 * d, "Use D>2d to avoid the Nyquist-frequency degeneracy."
    assert np.max(np.abs(np.mean(x * x, axis=1) - 1.0)) < 1e-11
    digest = hashlib.sha256(x.astype("<f4").tobytes() + n.astype("<f4").tobytes()).hexdigest()
    xt = torch.as_tensor(x.astype(np.float32), device=device)
    nt = torch.as_tensor(n.astype(np.float32), device=device)
    return {"x": xt, "n": nt, "y": xt + nt, "hash": digest}


@torch.inference_mode()
def evaluate(model, data, target):
    prediction = model(data["y"])
    reconstructed = prediction if target == "x-pred" else data["y"] - prediction
    x, n = data["x"].double(), data["n"].double()
    error = (reconstructed.double() - x).square().mean(dim=1)
    signal_power = x.square().mean(dim=1)
    noise_power = n.square().mean(dim=1)
    in_snr = 10.0 * torch.log10(signal_power / noise_power)
    out_snr = 10.0 * torch.log10(signal_power / error.clamp_min(1e-30))
    xc = x - x.mean(dim=1, keepdim=True)
    rc = reconstructed.double() - reconstructed.double().mean(dim=1, keepdim=True)
    corr = (xc * rc).sum(dim=1) / (xc.square().sum(dim=1) * rc.square().sum(dim=1)).sqrt().clamp_min(1e-30)
    mse = error.mean().item()
    return {
        "mse": mse, "rmse": math.sqrt(mse), "nmse": (error / signal_power).mean().item(),
        "input_snr_db_mean": in_snr.mean().item(), "input_snr_db_sd_records": in_snr.std().item(),
        "input_snr_db_min": in_snr.min().item(), "input_snr_db_max": in_snr.max().item(),
        "output_snr_db_mean": out_snr.mean().item(),
        "snr_improvement_db_mean": (out_snr - in_snr).mean().item(),
        "pearson_mean": corr.mean().item(),
        "noise_power_mean": noise_power.mean().item(), "signal_power_mean": signal_power.mean().item(),
    }


def make_cases():
    cases = []
    for stage in ("main", "capacity_extension", "noise_extension"):
        spec = PROTOCOL[stage]
        for d, D, H, sigma in itertools.product(spec["d"], spec["D"], spec["H"], spec["sigma"]):
            cases.append({"stage": stage, "d": d, "D": D, "H": H, "sigma": sigma})
    return cases


def run_key(case, seed, target):
    return f'{case["stage"]}_d{case["d"]}_D{case["D"]}_H{case["H"]}_s{case["sigma"]:g}_seed{seed}_{target}'


def summarize(output):
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((output / "records").glob("*.json"))]
    write_csv(output / "all_runs.csv", rows)
    groups, pairs = {}, {}
    for row in rows:
        key = tuple(row[k] for k in ("stage", "d", "D", "H", "sigma"))
        groups.setdefault(key + (row["target"],), []).append(row)
        pairs.setdefault(key, {}).setdefault(row["seed"], {})[row["target"]] = row
    summaries = []
    for key, records in sorted(groups.items()):
        result = dict(zip(("stage", "d", "D", "H", "sigma", "target"), key))
        result["n_seeds_completed"] = len(records)
        for metric in ("mse", "rmse", "snr_improvement_db_mean", "output_snr_db_mean", "pearson_mean"):
            values = np.array([r[metric] for r in records])
            result[metric + "_mean"] = float(values.mean())
            result[metric + "_sd_seeds"] = float(values.std(ddof=1)) if len(values) > 1 else ""
        summaries.append(result)
    write_csv(output / "summary_mean_sd.csv", summaries)
    differences = []
    for key, seeds in sorted(pairs.items()):
        complete = {s: p for s, p in seeds.items() if len(p) == 2}
        if not complete:
            continue
        result = dict(zip(("stage", "d", "D", "H", "sigma"), key))
        result["n_pairs_completed"] = len(complete)
        for name, metric, polarity in (("delta_mse_n_minus_x", "mse", 1),
                                       ("delta_snr_improvement_x_minus_n", "snr_improvement_db_mean", -1)):
            values = np.array([polarity * (p["n-pred"][metric] - p["x-pred"][metric]) for p in complete.values()])
            result[name + "_mean"] = float(values.mean())
            result[name + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else ""
            half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
            result[name + "_ci95_low"] = float(values.mean() - half) if half is not None else ""
            result[name + "_ci95_high"] = float(values.mean() + half) if half is not None else ""
            result[name + "_positive_pairs"] = int((values > 0).sum())
        differences.append(result)
    write_csv(output / "paired_differences_ci95.csv", differences)
    return len(rows)


def train_one(case, seed, target, data, initial_state, args, output, progress):
    key = run_key(case, seed, target)
    record_path = output / "records" / (key + ".json")
    if record_path.exists():
        return json.loads(record_path.read_text(encoding="utf-8"))
    model = make_model(case["D"], case["H"], seed_for("init", seed, case["D"], case["H"]), args.device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, foreach=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=1e-5)
    best_loss, best_epoch, best_state = math.inf, 0, None
    history = []
    started = time.perf_counter()
    epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator = torch.Generator(device=args.device)
        generator.manual_seed(seed_for("order", seed, epoch))
        order = torch.randperm(args.n_train, generator=generator, device=args.device)
        total = torch.zeros((), device=args.device)
        for ids in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(data["train"]["y"][ids])
            truth = data["train"]["x" if target == "x-pred" else "n"][ids]
            loss = torch.nn.functional.mse_loss(prediction, truth)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {key}, epoch={epoch}")
            loss.backward()
            optimizer.step()
            total += loss.detach() * ids.numel()
        train_mse = (total / args.n_train).item()
        model.eval()
        with torch.inference_mode():
            predicted = model(data["val"]["y"])
            restored = predicted if target == "x-pred" else data["val"]["y"] - predicted
            val_mse = torch.nn.functional.mse_loss(restored, data["val"]["x"]).item()
        if not math.isfinite(val_mse):
            raise RuntimeError(f"Non-finite validation loss: {key}, epoch={epoch}")
        if val_mse < best_loss:
            best_loss, best_epoch = val_mse, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        scheduler.step(val_mse)
        history.append({"epoch": epoch, "train_target_mse": train_mse,
                        "validation_clean_mse": val_mse, "lr": optimizer.param_groups[0]["lr"]})
        epochs = epoch
        if epoch % 10 == 0 or epoch == 1:
            write_csv(output / "curves" / (key + ".csv"), history)
            write_json(output / "status.json", {**progress, "state": "running", "pid": os.getpid(),
                       "updated_at": stamp(), "current_run": key, "epoch": epoch, "best_epoch": best_epoch,
                       "best_validation_mse": best_loss, "elapsed_run_seconds": time.perf_counter() - started})
        if epoch >= args.minimum_epochs and epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    # Linear output confines every prediction to an affine subspace of dimension <=H.
    # Noise has covariance sigma^2 I. The clean phase family has 2d equal
    # nonzero eigenvalues D/(2d). Bounds are population quantities; a finite
    # test-set estimate may lie slightly below a population lower bound.
    subspace_bound = (case["sigma"] ** 2 * max(case["D"] - case["H"], 0) / case["D"]
                      if target == "n-pred" else max(2 * case["d"] - case["H"], 0) / (2 * case["d"]))
    result = {**case, "seed": seed, "target": target,
              "nominal_snr_db": -20 * math.log10(case["sigma"]),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "epochs_run": epochs, "best_epoch": best_epoch, "best_validation_mse": best_loss,
              "initial_lr": args.lr, "final_lr": optimizer.param_groups[0]["lr"],
              "population_output_subspace_mse_lower_bound": subspace_bound,
              "last_validation_mse": history[-1]["validation_clean_mse"],
              "tail_validation_relative_improvement": (history[-min(30, len(history))]["validation_clean_mse"] - history[-1]["validation_clean_mse"]) / max(history[-min(30, len(history))]["validation_clean_mse"], 1e-30),
              "training_seconds": time.perf_counter() - started,
              "initial_state_sha256": state_hash(initial_state),
              "train_data_sha256": data["train"]["hash"], "validation_data_sha256": data["val"]["hash"],
              "test_data_sha256": data["test"]["hash"],
              "n_train": args.n_train, "n_validation": args.n_val, "n_test": args.n_test,
              **evaluate(model, data["test"], target), "completed_at": stamp()}
    write_csv(output / "curves" / (key + ".csv"), history)
    write_json(record_path, result)
    print(json.dumps({"finished": key, "mse": result["mse"], "best_epoch": best_epoch,
                      "epochs": epochs, "seconds": round(result["training_seconds"], 2)}), flush=True)
    return result


def state_hash(state):
    digest = hashlib.sha256()
    for k, value in state.items():
        digest.update(k.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def self_test(device):
    for d in (1, 2, 4, 8):
        a = split_data(128, d, 32, 1.0, 2026, "selftest", device)
        b = split_data(128, d, 32, 4.0, 2026, "selftest", device)
        assert torch.equal(a["x"], b["x"])
        assert torch.equal(4.0 * a["n"], b["n"])
        assert torch.max(torch.abs(a["x"].square().mean(dim=1) - 1)) < 1e-5
        p = torch.zeros_like(a["x"])
        target_loss = torch.nn.functional.mse_loss(p, a["n"])
        reconstruction_loss = torch.nn.functional.mse_loss(a["y"] - p, a["x"])
        assert torch.allclose(target_loss, reconstruction_loss, rtol=1e-6, atol=1e-6)
        wider = split_data(128, d, 64, 1.0, 2026, "selftest", device)
        assert torch.equal(a["x"], wider["x"][:, ::2])
    print("SELF_TEST_OK: power, Gaussian scaling, paired samples, loss equivalence", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "pilot", "all", "main"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--minimum-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--n-val", type=int, default=2048)
    parser.add_argument("--n-test", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    configure()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this configured run; refusing a silent CPU fallback.")
    self_test(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "smoke":
        args.epochs, args.minimum_epochs, args.patience = 2, 2, 2
        args.n_train, args.n_val, args.n_test, args.batch_size = 128, 64, 64, 64
    cases = make_cases()
    seeds = PROTOCOL["seeds"]
    if args.mode in ("smoke", "pilot"):
        cases = [{"stage": args.mode, "d": 4, "D": 128, "H": 32, "sigma": 1.0}]
        seeds = [2026]
    elif args.mode == "main":
        cases = [c for c in cases if c["stage"] == "main"]
    protocol_path = output / "protocol.json"
    runtime = {**PROTOCOL, "actual_arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "created_at": stamp(), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
               "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
               "case_count": len(cases), "training_run_count": len(cases) * len(seeds) * 2}
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous["actual_arguments"] != runtime["actual_arguments"] or previous["code_sha256"] != runtime["code_sha256"]:
            raise RuntimeError("Output directory belongs to another protocol/code version. Use a new directory.")
    else:
        write_json(protocol_path, runtime)
    total = len(cases) * len(seeds) * 2
    print(json.dumps({"mode": args.mode, "cases": len(cases), "runs": total, "output": str(output)}), flush=True)
    completed = summarize(output)
    wall_start = time.perf_counter()
    for case in cases:
        for seed in seeds:
            keys = [run_key(case, seed, t) for t in ("x-pred", "n-pred")]
            if all((output / "records" / (k + ".json")).exists() for k in keys):
                continue
            data = {split: split_data(count, case["d"], case["D"], case["sigma"], seed, split, args.device)
                    for split, count in (("train", args.n_train), ("val", args.n_val), ("test", args.n_test))}
            base = make_model(case["D"], case["H"], seed_for("init", seed, case["D"], case["H"]), args.device)
            initial = {k: v.detach().clone() for k, v in base.state_dict().items()}
            del base
            for target in ("x-pred", "n-pred"):
                result = train_one(case, seed, target, data, initial, args, output,
                                   {"completed_runs": completed, "total_runs": total, "mode": args.mode})
                completed = len(list((output / "records").glob("*.json")))
            summarize(output)
            print(f"PROGRESS {completed}/{total}", flush=True)
            del data, initial
    completed = summarize(output)
    write_json(output / "status.json", {"state": "completed", "pid": os.getpid(), "updated_at": stamp(),
               "completed_runs": completed, "total_runs": total, "wall_seconds_this_invocation": time.perf_counter() - wall_start})
    print(f"COMPLETED {completed}/{total}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        # A failed invocation must never leave a stale 'running' status.
        try:
            parser_index = sys.argv.index("--output")
            failed_output = Path(sys.argv[parser_index + 1])
        except (ValueError, IndexError):
            failed_output = ROOT / "results"
        write_json(failed_output / "status.json", {"state": "failed", "updated_at": stamp(),
                   "pid": os.getpid(), "error": repr(error)})
        raise
