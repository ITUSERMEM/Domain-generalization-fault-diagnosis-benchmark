# AGENTS.md — Domain Generalization Fault Diagnosis Benchmark

> This file is intended for AI coding agents. It describes the project structure, conventions, and critical details you need to know before modifying code.

---

## Project Overview

This is a **PyTorch-based research codebase** for benchmarking domain-generalization (DG) and domain-adaptation (DA) algorithms on **machine fault diagnosis** tasks. The input data are 1-D vibration signals (bearing and gearbox datasets). The project implements and compares several methods:

| Script | Method | Key Idea |
|--------|--------|----------|
| `ERM.py` | Empirical Risk Minimization (baseline) | Standard cross-entropy on source domains |
| `DANN.py` | Domain-Adversarial Neural Network | Adversarial domain classifier with gradient reversal |
| `DDC.py` | Deep Domain Confusion | MMD loss between source domains |
| `DCORAL.py` | Deep CORAL | CORAL (covariance alignment) loss between source domains |
| `CCDG.py` | Contrastive Cross-Domain Generalization | Supervised contrastive loss across domains |
| `CNN-C.py` | CNN + Center Loss | Adds center-loss regularization on features |
| `DGNIS.py` | Domain-Generalization with Domain-Specific classifiers | Multiple domain-specific heads + domain-gating for target inference |
| `IEDGNet.py` | Improved Ensemble DG Network | Source training + synthetic noisy target training, adversarial domain loss + triplet loss |

All methods follow the same evaluation protocol: **three source domains** (different operating loads) are used for training; **one held-out target domain** is used for testing. The protocol is repeated across 6 datasets and 4 task splits.

---

## Technology Stack

- **Language**: Python 3
- **Deep Learning**: PyTorch
- **Scientific Computing**: NumPy, SciPy
- **Data Format**: MATLAB `.mat` files (loaded via `scipy.io.loadmat`)
- **Hardware**: NVIDIA GPU expected (CUDA hardcoded to device `"1"` in every script)

### Dependencies (inferred)

No `requirements.txt`, `pyproject.toml`, or `setup.py` exists. The following packages must be installed:

```bash
pip install torch numpy scipy
```

> **Note**: The code was written for older PyTorch versions (uses `Variable`, `.data`, and `iter.next()` style APIs). It may emit deprecation warnings on PyTorch ≥ 1.13.

---

## Project Structure

```
.
├── data_loader_1d.py      # Data I/O: loads .mat files, applies z-score & optional FFT
├── resnet18_1d.py         # Network definitions: CNN_1D, DeepALL_ADV, IEDG, DGNIS, etc.
├── utils.py               # Loss functions: MMD, CORAL, CenterLoss, TripletLoss, helpers
├── ERM.py                 # Baseline ERM training script
├── DANN.py                # DANN training script
├── DDC.py                 # DDC (MMD-based) training script
├── DCORAL.py              # DCORAL training script
├── CCDG.py                # Contrastive DG training script
├── CNN-C.py               # CNN with center-loss training script
├── DGNIS.py               # DGNIS training script
├── IEDGNet.py             # IEDGNet training script
├── README.md              # Minimal project description
└── LICENSE                # MIT License
```

There are **no test files**, **no CI/CD pipelines**, and **no packaging configuration**.

---

## How to Run

Each algorithm is a **standalone executable script**. Run directly with Python:

```bash
python ERM.py
python DANN.py
python DDC.py
# ... etc
```

### Important Runtime Details

1. **Hardcoded CUDA Device**: Every script sets:
   ```python
   os.environ["CUDA_VISIBLE_DEVICES"] = "1"
   ```
   If you only have one GPU or want to use CPU, change this to `"0"` or comment it out.

2. **Hardcoded Data Path**: Data paths point to an absolute directory on the original author's machine:
   ```python
   root_path = '/home/zhaochao/research/DTL/data/' + dataset + 'data' + str(class_num) + '.mat'
   ```
   **You must change this path** to wherever you place the `.mat` data files.

3. **Datasets Expected**:
   - `C-CWRU` (10 classes)
   - `C-LW` (4 classes)
   - `C-PU` (12 classes)
   - `C-PHM` (6 classes)
   - `C-SQbearing` (9 classes)
   - `C-SQgearbox` (3 classes)

   Each dataset file should be named like: `C-CWRUdata10.mat`, `C-LWdata4.mat`, etc.

4. **MAT File Variables**: The scripts expect variables named `load{X}_train` and `load{X}_test` inside the `.mat` file, where `{X}` is the load index (e.g., `load0_train`, `load1_test`).

5. **Hyperparameters**: Hyperparameters (learning rate, batch size, iterations) are hardcoded at the bottom of each script inside `if __name__ == '__main__':`. Typical values:
   - `iteration = 5000`
   - `batch_size = 256`
   - Learning rates vary per method (e.g., `0.001`, `0.005`, `0.0001`)

---

## Data Loading (`data_loader_1d.py`)

Key functions:

- `load_training(root_path, dir1, dir2, dir3, src_list, fft1, class_num, batch_size, kwargs)`  
  Loads three source domains, assigns domain labels `[0, 1, 2]`, returns a `DataLoader`.

- `load_testing(root_path, dir, fft1, class_num, batch_size, kwargs)`  
  Loads target test domain (no domain label).

- `load_source_training(...)` / `load_target_training(...)`  
  Used by `IEDGNet.py` to create clean source data and synthetic noisy target data (with amplitude perturbation `AP` and additive white Gaussian noise `SNR`).

**Preprocessing Pipeline**:
1. Optional FFT: `abs(fft(signal))[:, 0:1600]`
2. Optional log-min-max: `np.log(Z - Zmin + 1)`
3. Z-score normalization per sample: `(x - min) / (max - min)`

Labels are constructed as 2-D tensors: `[:, 0] = class_id`, `[:, 1] = domain_id`.

---

## Model Architecture (`resnet18_1d.py`)

Despite the filename, the file does **not** contain a 1-D ResNet-18. It defines:

- `CNN` — A custom 1-D CNN (4 conv blocks → adaptive max pool → 256-D feature vector).
- `CNN_1D` — Wrapper around `CNN` + classification head.
- `DeepALL_ADV` — `CNN` + cls head + multi-domain adversarial head (3 domains).
- `IEDG` — `CNN` + cls head + 2-domain adversarial head.
- `DGNIS` — Two `CNN` backbones + three domain-specific classifiers + domain-gating classifier for target inference.
- `AdversarialNetwork` / `AdversarialNetwork_multi` — Simple MLP domain discriminators.

All models expect input shape `(batch_size, signal_length)` (usually 1-D time-series or 1600-D FFT coefficients).

---

## Loss Functions & Utilities (`utils.py`)

- `mmd_rbf_noaccelerate(source, target)` — RBF-based Maximum Mean Discrepancy.
- `CORAL(source, target)` — CORAL (second-order covariance alignment) loss.
- `Center_loss(src_class)` — Intra-class feature variance loss (MSE to class centroid).
- `TripletLoss(margin)` — Hard or soft-margin triplet loss.
- `BatchHardTripletSelector` — Selects hard positive/negative pairs within a batch.
- `setup_seed(seed)` — Reproducibility helper (sets NumPy, random, torch seeds).
- Various entropy / cross-entropy / BCE helpers.

---

## Code Style & Conventions

- **No type hints** are used.
- **Global variables** are heavily used (`src_loader`, `tgt_test_loader`, `cuda`, `iteration`, etc.).
- **Mixed language comments**: Some Chinese comments appear (e.g., `计算模型训练参数个数`, `逐元素比较`).
- **Formatting is inconsistent**: Spacing around operators, blank lines, and indentation vary.
- **No logging framework**: All output is via `print()`.
- **No argument parsers**: Everything is hardcoded; there are no command-line interfaces.
- **Old PyTorch patterns**: Uses `Variable`, `.data`, and `iter.next()` (deprecated in PyTorch 1.x+).

When modifying code, try to **match the existing style** of the surrounding file. If you introduce new functionality, prefer adding it to `utils.py` or `resnet18_1d.py` rather than duplicating code across algorithm scripts.

---

## Testing

There are **no automated tests** in this repository. Validation is done by running the training scripts and inspecting printed accuracy values.

Each script prints:
- Training loss every `log_interval` iterations (default 10).
- Source and target accuracy every `log_interval * 10` iterations.

To verify correctness after changes, run a short training session (reduce `iteration` to e.g., 100) and confirm the loss decreases and accuracies are printed without runtime errors.

---

## Known Limitations & Security Considerations

1. **Hardcoded absolute paths** (`/home/zhaochao/research/DTL/data/`) will cause `FileNotFoundError` on any other machine.
2. **Hardcoded CUDA device** `"1"` will fail on single-GPU machines or CPU-only environments.
3. **No input validation** on `.mat` file contents; malformed files can cause cryptic NumPy/PyTorch errors.
4. **No model checkpointing**; trained weights are not saved. All results are ephemeral console output.
5. **Data leakage risk**: Some methods (e.g., `DGNIS`, `IEDGNet`) use target-domain information during training (domain labels or synthetic target data). Ensure you understand the protocol before using this code for production fault-diagnosis systems.
6. **No reproducibility logging**: Random seeds are set but not logged; exact hyperparameters per run are not saved.

---

## Development Tips for Agents

- If adding a new algorithm, **copy an existing script** (e.g., `ERM.py`) as a template and modify the `train()` loop and model instantiation.
- If adding a new dataset, add a new dictionary to the `Tasksetting` list in each script, or refactor the task settings into a shared config (recommended).
- To make the code runnable out-of-the-box, the **minimum required change** is updating the `root_path` variable in every script.
- If modernizing the codebase, consider:
  - Replacing `iter.next()` with `next(iter)`.
  - Removing `Variable` wrappers (unnecessary in PyTorch ≥ 1.0).
  - Using `argparse` for hyperparameters and paths.
  - Adding `torch.save()` / `torch.load()` for checkpointing.
