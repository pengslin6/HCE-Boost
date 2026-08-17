# HCE-Boost

Minimal reproducibility repository for **HCE-Boost: hierarchy-constrained expert boosting for imbalanced multiclass intrusion detection**.

HCE-Boost combines equally weighted XGBoost and LightGBM global experts with validation-triggered, probability-mass-conserving specialists. To keep the repository focused, it contains only the critical experiment and analysis code. The four analysis-ready CSV files representing the three evaluated datasets are provided as assets of the `v1.0.0` GitHub release. Generated result files, manuscript figures, and submission documents are intentionally not versioned here.

## Repository contents

- `hce_boost_detailed.py`: HCE-Boost training, validation-triggered specialist construction, and class-wise evaluation.
- `strong_tabular_audit.py`: validation-first global-expert fitting and development-set refitting.
- `revision_experiments.py`: dataset loaders, preprocessing, XGBoost reference, and scratch-MLP reference.
- `run_five_seed_neural_baselines.py`: resumable five-seed runner for the neural baselines.
- `run_edge_legacy_baselines.py`: DNN, RNN, LSTM-AE, VLSTM, and CNN-BiLSTM definitions and training loop.
- `finalize_five_seed_results.py`: completeness gate and source-data/statistics generator.
- `make_full_class_table.py`: complete 33-class Table 3 source-data generator.
- `make_hce_result_figures.py`: manuscript result-figure generator.
- `requirements-reported.txt`: reported software environment.
- Release `v1.0.0`: the four analysis-ready CSV files representing the three evaluated datasets.

The scripts create their result and figure directories when run. Those generated outputs are ignored by Git so the public repository remains limited to code and the four CSV files.

## Data inventory

Download the four CSV assets from [release `v1.0.0`](https://github.com/pengslin6/HCE-Boost/releases/tag/v1.0.0) and place them in a local `data/` directory before running the scripts.

| Dataset | File | Records | Size (bytes) | SHA-256 |
|---|---|---:|---:|---|
| NSL-KDD training | `data/KDDTrain.csv` | 125,973 | 13,065,689 | `61098521993990E771C5B656D28D2CDAF48EC1503CC75B1E31B05AF55D74221A` |
| NSL-KDD test | `data/KDDTest.csv` | 18,793 | 1,940,994 | `A38148840573EA47A615112862CD606F2C3A4FDE62EE0F4E5A5F3CF2D322F6E3` |
| CIC-IDS2017 supplied subset | `data/CICIDS2017.csv` | 543,734 | 191,714,289 | `98BBACBC2F0F1560F838B455504AC4F8D64126088E3E0F5B0C8317CC0DD2120E` |
| Edge-IIoTset | `data/ML-EdgeIIoT-dataset.csv` | 157,800 | 82,184,390 | `53101FAD091AF20BEE815860962A5016B802EE45D456A141042F6183094C3C1A` |

The fixed train/validation/test partitions use random state 42. Model seeds are 11, 22, 33, 44, and 55. Validation data determine early stopping and specialist activation; test labels are used only for final evaluation.

## Environment

The reported environment is recorded in `requirements-reported.txt`.

```bash
python -m pip install -r requirements-reported.txt
```

The scripts read the CSV files from `data/` by default. To use another location, set the `HCE_DATA_DIR` environment variable.

## Reproduction order

```bash
python hce_boost_detailed.py --datasets nsl cic edge
python run_five_seed_neural_baselines.py --dataset nsl --threads 8
python run_five_seed_neural_baselines.py --dataset cic --threads 8
python run_five_seed_neural_baselines.py --dataset edge --threads 8
python finalize_five_seed_results.py
python make_full_class_table.py
python make_hce_result_figures.py
```

The independent XGBoost and scratch-MLP result files are generated with `revision_experiments.py`. The finalization script refuses to complete unless every listed method contains seeds 11, 22, 33, 44, and 55.

## Reported HCE-Boost macro F1

- NSL-KDD: 64.99 ± 0.41%
- CIC-IDS2017 supplied subset: 93.73 ± 0.16%
- Edge-IIoTset: 92.387 ± 0.008%

These values apply to the supplied preprocessing and fixed splits. They do not establish embedded-device latency, memory, energy, or end-to-end deployment feasibility.

## Dataset provenance and terms

- NSL-KDD: [Canadian Institute for Cybersecurity dataset page](https://www.unb.ca/cic/datasets/nsl.html).
- CIC-IDS2017: [Canadian Institute for Cybersecurity dataset page](https://www.unb.ca/cic/datasets/ids-2017.html).
- Edge-IIoTset: Ferrag *et al.*, *IEEE Access* 10, 40281–40306 (2022), [doi:10.1109/ACCESS.2022.3165809](https://doi.org/10.1109/ACCESS.2022.3165809).

The files in `data/` are analysis-ready copies used by the study. The original providers' terms remain applicable; this repository does not assert a new licence over third-party datasets. No repository-wide software licence has been added.

## Repository scope

This repository is intentionally limited to critical code and metadata needed to run it; the associated release contains only the four CSV files listed above. It does not contain manuscript source files, reviewer-response files, rendered figures, or cached/generated experiment outputs.
