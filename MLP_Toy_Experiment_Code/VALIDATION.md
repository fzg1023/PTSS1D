# Release validation / 发布前检查

Date: 2026-10-29. These checks validate packaging and implementation continuity; the complete formal grid was not retrained during packaging.

## Checks performed

- Original `src/run_experiment.py` and `src/highdim_experiment.py` are byte-for-byte identical to the completed-experiment implementation.
- Historical sample identity: all 12 split hashes matched saved experimental records for seed 2026, D=256, sigma=1, d=1/8/16/64, separately checking train/validation/test sets at their full sizes.
- Existing physics/data self-tests passed: exact unit signal power, Gaussian amplitude scaling, shared samples, loss equivalence, extended-phase nesting and expected Fourier support.
- `--suite paper --dry-run` produced 50 distinct conditions, five seeds and 500 target-training runs.
- CPU smoke test completed all four training runs (two conditions, two targets), including training curves, validation-based checkpoint selection, held-out evaluation and aggregate CSV/JSON outputs.
- CUDA smoke test completed the same four runs successfully.
- CPU resume invocation reused all four completed records and verified paired data/initialization hashes.
- Python source syntax was checked.

Validation environment: Windows, Python 3.9 environment, PyTorch 2.7.1+cu118, NumPy 2.0.1, SciPy 1.13.1. This is distinct from the historical reference environment recorded in the README. CPU and GPU smoke metrics are not expected to match bit-for-bit.

本次没有重新完成500次正式训练，smoke数据不能作为论文结果。正式设置、五次重复与早停参数保存在configs.json中。原训练实现与数据生成实现保持不变；独立入口仅替换了历史脚本中的路径依赖、结果复用调度及命令行入口。

No weights, generated validation outputs, private datasets, Python caches or local experiment paths are included in the release archive.
