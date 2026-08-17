# Federated Data Science Framework for Privacy-Preserving Healthcare Analytics

Artefact for **7005SCN Individual Research Project**, Coventry University.

**Student:** Milan Rahul Jeyakumar (SID 10145294)
**Supervisor:** Roohi Barzamini
**Submission:** August 2026

This repository contains the code and data for the federated learning pilot reported in the
project report. It is public, and will remain unchanged for at least three months following
submission.

---

## Contents

| File | Description |
|---|---|
| `Dataset_Corrected.ipynb` | Complete pipeline: data preparation, EDA, centralised baselines, FedAvg training, evaluation |
| `heart.csv` | UCI Heart Disease dataset (Kaggle redistribution, 1,025 rows) |
| `requirements.txt` | Library versions needed to reproduce the environment |

---

## Reproducing the results

```bash
git clone <this-repository-url>
cd <repository-folder>
pip install -r requirements.txt
jupyter notebook Dataset_Corrected.ipynb
```

In Jupyter, choose **Kernel → Restart & Run All**. Runtime is roughly two to three minutes.

All random seeds are fixed at 42 across NumPy, PyTorch and scikit-learn, so the figures below
reproduce exactly.

---

## Results (held-out test set, n = 61)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Centralised logistic regression | 0.8525 | 0.8750 | 0.8485 | 0.8615 | 0.9058 |
| Centralised neural network | 0.8033 | 0.8387 | 0.7879 | 0.8125 | 0.8723 |
| Federated (FedAvg, 5 clients, 30 rounds) | 0.8689 | 0.8788 | 0.8788 | 0.8788 | 0.9134 |

Confusion matrix (federated): TN 24, FP 4, FN 4, TP 29.

The federated model records marginally higher values than the centralised baselines. This is
**not** evidence that federation improves prediction. With 61 test patients, one additional
correct classification shifts accuracy by 1.6 percentage points, so the difference falls within
the sampling variability of a single train/test split.

---

## Important note on the dataset

The source file contains 1,025 rows but only **302 distinct patients** — the remaining 723 are
exact duplicates introduced by the Kaggle redistribution.

An earlier version of this pipeline counted the duplicates but did not remove them before the
train/test split. This caused **202 of 205 test records (98.5%) to also appear in the training
set**, so the reported accuracy of 79.51% measured memorisation rather than generalisation.

The current notebook removes duplicates before any partitioning. This correction, and the defect
audit that produced it, are discussed in Sections 3.9 and 5.5 of the project report.

---

## Configuration

| Parameter | Value |
|---|---|
| Clients | 5 simulated hospitals (IID partition) |
| Communication rounds | 30 |
| Local epochs per round | 5 |
| Batch size | 32 |
| Optimiser | Adam, learning rate 0.001 |
| Loss | BCEWithLogitsLoss |
| Architecture | 20 – 64 – 32 – 16 – 1, ReLU, dropout 0.3 |
| Aggregation | FedAvg weighted by client sample size (nk/n) |
| Random seed | 42 |

---

## Scope

**Implemented and evaluated:** FedAvg aggregation with sample-size weighting; five simulated
clients; shared preprocessing fitted on training data only; centralised neural network and
logistic regression baselines; defect audit and reproducibility verification.

**Proposed but not implemented:** local differential privacy (DP-SGD); secure aggregation;
adaptive gradient compression; non-IID (Dirichlet) partitioning; FedProx comparison;
MIMIC-III and eICU datasets.

---

## Outputs written on run

`preprocessor.joblib`, `global_model.pt`, `federated_round_history.csv`, `model_comparison.csv`

---

## Data licence

The UCI Heart Disease dataset (Janosi, Steinbrunn, Pfisterer & Detrano, 1989) is publicly
available from the UCI Machine Learning Repository under a CC BY 4.0 licence. It is fully
anonymised and contains no direct or indirect patient identifiers.
