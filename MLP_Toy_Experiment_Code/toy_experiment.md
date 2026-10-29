# Paired MLP Toy Experiments / 配对 MLP 玩具实验

Code accompanying **Structure–Capacity Matching in Neural Network Learning: Prediction Target Selection for 1D Signal Denoising**.

This package contains the revised multi-phase synthetic experiment, including intrinsic dimensions up to 64, hidden widths up to 256, noise-strength controls, and five paired random-seed repetitions. All signals are generated locally. No external dataset, pretrained model, historical result directory, or model weights are required.

本代码对应修改稿中的多相位合成信号实验，包含本征维度、观测维度、网络宽度和噪声强度对照。数据由程序生成，不需要下载真实数据集，也不依赖作者电脑上的历史实验结果。默认按单组顺序运行；每个条件分别训练 x-pred 和 n-pred。代码不保存模型权重。

## 1. Files / 文件

```text
run_toy.py                 Portable training and evaluation entry point
configs.json               Experiment grids and training settings
src/run_experiment.py      Original model, optimizer, training and metrics
src/highdim_experiment.py  Original extension to d = 16, 32, 64
requirements.txt           Dependencies
SOURCE_MANIFEST.json       Source provenance and file hashes
VALIDATION.md              Release checks and their scope
```

The two files in `src/` preserve the implementation used in the completed experiments. The portable launcher reruns each selected condition directly, instead of requiring earlier runs to exist. Use `run_toy.py` as the entry point.

`src/`中保留原训练实现和扩展数据生成实现。新增入口只负责选取条件、记录配置、顺序调度及汇总，不需要先运行旧实验再拼接结果。科学设置中的 d 是相位流形的本征维度，其线性张成空间维度为 2d；不要将其与论文线性高斯命题中的子空间秩直接等同。

## 2. Installation / 环境

Python 3.9 or newer is recommended. The original recorded environment used Python 3.9, PyTorch 2.5.1+cu118 and NumPy 2.0.2. The release-validation environment is listed separately in `VALIDATION.md`.

```bash
python -m venv .venv
# Activate the environment using your operating system's command.
python -m pip install -r requirements.txt
```

For GPU execution, install a CUDA-compatible build of PyTorch for your machine. The default device is `cuda:0` when CUDA is available, otherwise CPU. You can explicitly choose `--device cpu` or `--device cuda:0`.

GPU 环境需安装与本机驱动兼容的 PyTorch。CPU 可运行，但正式网格耗时较长。不同硬件或 PyTorch 版本可能导致数值差异；不承诺跨版本逐位一致。

## 3. Quick check / 快速检查

Run from this folder:

```bash
python run_toy.py --suite selftest --device cpu
python run_toy.py --suite smoke --device cpu
python run_toy.py --suite paper --dry-run
```

`selftest` checks unit signal power, shared noise, phase nesting, Fourier support and MSE equivalence. `smoke` trains two small target pairs for two epochs, including the high-dimensional generator. These are implementation checks, not paper results.

自检验证信号功率、共享噪声、嵌套相位及损失关系。smoke 仅验证训练—验证—测试和汇总流程能否运行，不能作为论文实验数据。

## 4. Reproduce the revised experiments / 运行修改稿实验

```bash
python run_toy.py --suite paper --device cuda:0
```

The default `paper` suite combines the following grids and deduplicates overlapping conditions. It runs **50 distinct conditions × 5 seeds × 2 targets = 500 training runs**. Both targets are evaluated even where the manuscript plots only x-pred. The observation suite includes the full original D sweep; select the corresponding rows when displaying a subset in a figure.

| Suite | Purpose | d | D | H | Noise standard deviation |
|---|---|---|---|---|---|
| `observation` | Observation dimension and width, Fig. 2(a) family | 1 | 32, 64, 128, 256, 512, 1024 | 32, 64, 128, 256 | 1 |
| `width` | Width versus intrinsic dimension, Fig. 2(b) family | 8, 16, 32, 64 | 256 | 16, 32, 64, 128, 256 | 1 |
| `intrinsic` | Target gap as d increases, Table 1 | 2, 4, 8, 16, 32, 64 | 256 | 32 | 1 |
| `noise` | Noise-strength comparison, Fig. 2(c) | 1 | 256 | 32, 128 | 4, 1, 0.25 |

默认同时保留 H=256 的扩展结果。各组可以单独运行：

```bash
python run_toy.py --suite width --device cuda:0
python run_toy.py --suite intrinsic --device cuda:0
python run_toy.py --suite noise --device cuda:0
```

`legacy_main`, `legacy_capacity`, `legacy_noise` and `legacy_narrow` expose the wider historical grids. They are optional and are not included in the default paper suite.

To inspect the workload without training, use `--dry-run`. To change the output folder:

```bash
python run_toy.py --suite paper --output outputs/my_run --device cuda:0
```

Rerun the identical command to resume completed pairs. Settings and source hashes must match the stored protocol. A partial pair is completed using the same generated data and initial weights. Use a new output folder after changing code, grids, seeds or settings. Do not start multiple processes against the same output folder.

默认正式设置为五次随机种子重复。`--seeds 2026`可用于开发排查，但不能将单种子结果当作五次重复结果。对输出目录的断点续跑只适用于同一份代码和配置。

## 5. Signal and noise construction / 信号与噪声

For j = 0, ..., D−1:

```text
x[j] = sqrt(2/d) * sum(k=1..d) cos(2*pi*k*j/D + phi[k])
phi[k] independently follows Uniform(0, 2*pi)
n = sigma * epsilon, epsilon follows N(0, I)
y = x + n
```

All grids satisfy D > 2d. Clean signals have unit average power analytically. Noise realizations are not rescaled to force an exact record-wise SNR. No sample-wise or target-wise normalization is applied. Nominal input SNR is −20 log10(sigma): sigma = 4, 1, 0.25 corresponds to approximately −12.04, 0 and +12.04 dB. Actual record-wise SNR is measured and reported.

相位独立均匀采样。训练、验证和测试数据分别生成；两种目标在同一次重复中共享数据、噪声、初始参数和逐轮样本顺序。改变 H 时共享数据；改变 sigma 时共享纯净信号和基础高斯噪声。扩展生成器保持原前8个相位的随机流，额外56个相位来自独立固定随机流，以兼容已有 d≤8 的实验。

## 6. Model and training / 模型与训练

The MLP has two hidden fully connected layers, both of width H with ReLU activations, followed by a linear D-dimensional output. There is no global input-to-output identity skip. Hidden-layer biases use PyTorch's default initialization; the output bias is zero. Within each target pair, all initial parameters are identical.

| Setting | Value |
|---|---|
| Seeds | 2026, 2027, 2028, 2029, 2030 |
| Training / validation / test records | 8192 / 2048 / 4096 |
| Optimizer | Adam, initial LR 0.001, weight decay 0 |
| Batch size | 1024 |
| Epoch budget | At most 1000; early stopping enabled after at least 200 |
| Early-stop patience | 100 epochs without a new validation minimum |
| LR scheduler | ReduceLROnPlateau, factor 0.5, patience 20, minimum LR 1e-5 |
| Checkpoint selection | Lowest validation clean-reconstruction MSE |
| Arithmetic | FP32 training; TF32 disabled; deterministic PyTorch operations |

The best weights are retained in memory and evaluated on the held-out test set; weights are not written to disk. The test set is not used to select epochs or learning rates.

Both networks minimize direct target MSE. Clean reconstruction is `f(y)` for x-pred and `y - g(y)` for n-pred. Their target MSE and reconstruction MSE are algebraically equal for additive observations, but the learnable branches represent different quantities.

## 7. Outputs and interpretation / 输出

Results are written to `outputs/<suite>/` by default:

| File | Meaning |
|---|---|
| `protocol.json` | Actual grids, seeds, settings, environment and source hashes |
| `records/*.json` | One result per seed and prediction target, including pairing hashes |
| `curves/*.csv` | Training target MSE, validation reconstruction MSE and LR by epoch |
| `all_runs.csv` | All completed individual runs |
| `summary_mean_sd.csv` | Mean and sample SD across completed seeds |
| `paired_differences_ci95.csv` | Paired differences, sample SD and 95% Student-t CI |
| `observation_runs.csv`, etc. | Subsets for the corresponding experiment families |
| `completion.json` | Completed versus expected run count |
| `status.json` | Progress / completed / failed state |

Primary metric: clean-reconstruction MSE averaged over records and coordinates. Secondary metrics: RMSE, mean record-wise SNR improvement, output SNR and **time-domain** Pearson correlation (not FPCC). RMSE is computed within each run before aggregation. Paired differences use MSE(n-pred) minus MSE(x-pred), or SNR improvement(x-pred) minus SNR improvement(n-pred).

标准差为跨随机种子的样本标准差，只有一个完成种子时留空。置信区间使用配对差值计算；五次配对时自由度为4。区间没有多重比较校正。结果文件中的 population output-subspace lower bound 仅是总体分布下由输出子空间造成的重构下界，不是有限测试集必须严格满足的界，也不是通用可学习性定律。

## 8. Scope / 说明

This release packages the computational experiment, not a new full training run. Historical model weights, pilot runs, temporary logs and private paths are excluded. No license is imposed by this packaging step; the authors should choose a license before publishing the repository. No repository upload is performed by these scripts.

本文件夹可作为仓库内容或压缩包上传。本次整理不等于重新完成500次正式训练；验证范围见 `VALIDATION.md`。作者可按投稿与开源要求自行选择许可证。
