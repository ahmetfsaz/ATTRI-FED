# ATTRI-FED

Official implementation associated with **“Attri-Fed: A GIB Framework for Attribute-Based Privacy and Communication-Efficient Federated Learning,”** published at the 2023 IEEE 24th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC). Read the paper via [IEEE Xplore](https://doi.org/10.1109/SPAWC53906.2023.10304534).

If you use this repository in academic work, please cite the paper using the [citation](#citation) below.

## Overview

ATTRI-FED is an attribute-based privacy framework for federated learning. Rather than attempting to conceal every distinction in a client's input, it learns a compressed representation that:

- remains informative about a target attribute or task;
- obscures a designated private attribute from an inferential adversary; and
- reduces communication by transmitting a lower-dimensional representation.

The method is based on the Generalized Information Bottleneck (GIB). Its training objective combines target-task utility with two variational privacy components:

- **Encouragement:** encourages the representation to preserve target-relevant information that is not explained solely by the private attribute.
- **Enforcement:** adversarially discourages prediction of the private attribute from the learned representation.

The experiments use colored MNIST: digit identity is the target variable, while color is the private attribute. The original implementation also contains exploratory experiments using EMNIST as auxiliary public data.

## Unified implementation

The original repository contained 23 large experiment scripts that repeated the same model, data-processing, training, and evaluation code. They differed mainly in configuration choices such as the objective, number of clients, bottleneck dimension, initialization procedure, active training stages, and aggregation rule.

The consolidated implementation is:

```text
attri_fed.py
```

It preserves every historical filename as a named preset while exposing the meaningful differences as command-line options. This makes it possible to reproduce or inspect an old experiment without editing source code or maintaining a separate copy of the entire program.

List all available presets:

```bash
python attri_fed.py --list-presets
```

Inspect a preset without starting an experiment:

```bash
python attri_fed.py --preset Both_Yashas_N10.py --dry-run
```

## Methods represented by the presets

| Method | Objective | Representative preset |
|---|---|---|
| Baseline | Target-task utility without local attribute privatization | `Baseline_FL_N10.py` |
| ECO | Encouragement and rate terms, without explicit enforcement | `Only_Yashas_N10.py` |
| EEO | Enforcement, encouragement, and rate terms | `Both_Yashas_N10.py` |
| Enforcement-only | Target utility and the private-attribute adversary | `Just_Us_N10.py` |
| Legacy noisy aggregation | Historical clipped/Laplace aggregation experiment | `Baseline_DP_FL_N4.py` |

ECO and EEO correspond to the principal attribute-privacy variants. The enforcement-only configuration and several public-data configurations are retained as historical ablations.

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/ahmetfsaz/ATTRI-FED.git
cd ATTRI-FED

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the dependencies used by the implementation:

```bash
pip install numpy matplotlib scikit-learn \
    tensorflow tensorflow-compression tensorflow-datasets
```

The implementation uses TensorFlow's v1 compatibility mode and TensorFlow Compression's `ContinuousBatchedEntropyModel`. TensorFlow and TensorFlow Compression must therefore be installed as mutually compatible versions. The historical repository did not include a pinned environment, so creating and recording a tested lock file is recommended before reporting new numerical results.

Configuration-only commands such as `--list-presets` and `--dry-run` work even when the TensorFlow stack is not installed.

## Quick start

### Full EEO method

Train the full enforcement-and-encouragement privatizer and evaluate private-attribute leakage:

```bash
python attri_fed.py --preset Both_Yashas_N10.py
```

This produces `runs/Both_Yashas_MNIST_10.npz`. Use the saved representations for downstream federated target-task evaluation:

```bash
python attri_fed.py --preset Both10.py
```

### ECO method

Train the encouragement-only privatizer:

```bash
python attri_fed.py --preset Only_Yashas_N10.py
```

Then run downstream federated target-task evaluation:

```bash
python attri_fed.py --preset Only10.py
```

### Unprotected baseline

```bash
python attri_fed.py --preset Baseline_FL_N10.py
```

### Change the number of participating clients

The dataset is divided into 100 shards, and `--num-clients` selects how many shards participate:

```bash
python attri_fed.py \
    --preset Both_Yashas_N10.py \
    --num-clients 6 \
    --output-npz Both_Yashas_MNIST_6.npz
```

The paper evaluates multiple client counts. A single implementation can now run the full sweep:

```bash
for n in 2 4 6 8 10 12; do
    python attri_fed.py \
        --preset Both_Yashas_N10.py \
        --num-clients "$n" \
        --output-npz "Both_Yashas_MNIST_${n}.npz"
done
```

## Configuration overrides

Every preset is an `ExperimentConfig` instance defined near the beginning of `attri_fed.py`. Command-line arguments can override the most important fields without changing the file.

```bash
python attri_fed.py \
    --preset Both_Yashas_N10.py \
    --num-clients 10 \
    --latent-dim 50 \
    --beta-enforcement 1.5 \
    --beta-encouragement 1.5 \
    --beta-independence 1.5 \
    --adversarial-epochs 5000 \
    --posthoc-epochs 1000
```

Important options include:

| Option | Meaning |
|---|---|
| `--preset` | Historical experiment configuration to load |
| `--objective` | `baseline`, `eco`, `eeo`, or `enforcement` |
| `--num-clients` | Number of participating clients |
| `--num-data-shards` | Number of MNIST data shards; historically 100 |
| `--latent-dim` | Dimension of the transmitted representation |
| `--beta-enforcement` | Weight of the enforcement term |
| `--beta-encouragement` | Weight of the encouragement term |
| `--beta-independence` | Additional rate/independence weight |
| `--initialization` | Initialization recipe used before privatization |
| `--privatization` | Local privatization procedure |
| `--attack-input` | Evaluate an adversary on `compressed` or `raw` inputs |
| `--aggregation` | `average` or `legacy_laplace_dp` |
| `--input-npz` | Representation archive consumed by an evaluation preset |
| `--output-npz` | Filename for generated private-data representations |
| `--public-input-npz` | Optional precomputed public-data representations |
| `--public-output-npz` | Filename for generated public-data representations |
| `--output-dir` | Root directory for artifacts and run logs; default: `runs` |
| `--dry-run` | Print the resolved configuration without training |

Run `python attri_fed.py --help` for the complete interface.

## Workflows

The unified program represents the historical scripts through five workflows:

| Workflow | Purpose |
|---|---|
| `privatize` | Initialize and train local encoders/privatizers, save representations, and evaluate private-attribute leakage |
| `federated_current` | Encode the current dataset and train a federated target classifier |
| `federated_npz` | Train a federated target classifier from a saved representation archive |
| `precomputed_attack` | Evaluate private-attribute leakage from an existing archive |
| `public_transfer_attack` | Generate or load public representations and evaluate a transfer-trained adversary |

## Outputs

Generated representation archives are written directly under the selected output directory so producer and consumer presets can find one another. Logs, plots, and the resolved configuration are placed in a timestamped run directory.

```text
runs/
├── Both_Yashas_MNIST_10.npz
├── Both_Yashas_N10-YYYYMMDD-HHMMSS/
│   ├── config.json
│   └── ... plots and logs ...
└── Both10-YYYYMMDD-HHMMSS/
    ├── config.json
    └── ... plots and logs ...
```

Private-data representation archives use the following keys:

```text
tr_x, tr_priv, tr_tgt
val_x, val_priv, va_tgt
te_x, te_priv, te_tgt
```

Public EMNIST archives use:

```text
etr_x, etr_priv, etr_tgt
eval_x, eval_priv, eva_tgt
ete_x, ete_priv, ete_tgt
```

### Differential-privacy baseline

The paper reports a conventional differential-privacy comparison implemented using Opacus. The committed file, however, applies tensor-level clipping and Laplace noise during aggregation and does not contain the paper's stated Opacus implementation.

The consolidated program preserves this committed behavior under:

```text
aggregation = legacy_laplace_dp
```

It should be treated as a **historical noisy-aggregation experiment**, not presented as a faithful reconstruction of the paper's Opacus baseline. The original Opacus code or an independently validated replacement is required to reproduce that reported comparison.

## Citation

Ahmet Faruk Saz, Yashas Malur Saidutta, Faramarz Fekri, Mustafa Riza Akdeniz, Brandon Edwards, and Nageen Himayat, “Attri-Fed: A GIB Framework for Attribute-Based Privacy and Communication-Efficient Federated Learning,” in *2023 IEEE 24th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC)*, pp. 366–370, 2023. [doi:10.1109/SPAWC53906.2023.10304534](https://doi.org/10.1109/SPAWC53906.2023.10304534)

```bibtex
@inproceedings{saz2023attrifed,
  author    = {Saz, Ahmet Faruk and Saidutta, Yashas Malur and
               Fekri, Faramarz and Akdeniz, Mustafa Riza and
               Edwards, Brandon and Himayat, Nageen},
  title     = {{Attri-Fed}: A {GIB} Framework for Attribute-Based Privacy
               and Communication-Efficient Federated Learning},
  booktitle = {2023 IEEE 24th International Workshop on Signal Processing
               Advances in Wireless Communications (SPAWC)},
  pages     = {366--370},
  year      = {2023},
  doi       = {10.1109/SPAWC53906.2023.10304534}
}
```

## Acknowledgments

The work reported in the paper was supported by the National Science Foundation under award **MLWiNS-2003002** and by a gift from Intel Corporation.

