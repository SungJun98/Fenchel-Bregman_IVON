# Fenchel-Bregman IVON

## Code layout

```text
repository/
├── fb.py                      # shared finite-K FB weighting and failure EMA
├── optim/
│   ├── ivon.py                # recursive-reasoning IVON adapter
│   └── fbi_ivon.py            # autoregressive-LM IVON implementation
├── auto_lm/
│   ├── fbi.py                 # autoregressive FBI objective and failure state
│   ├── run.py                 # training and posterior-evaluation entry point
│   ├── build_base_eval_suite.py
│   ├── prepare_generation_parquet.py
│   ├── aggregate_pass_at_k.py
│   └── MMPO/                  # vendored MMPO/verl runtime
└── rrm/
    ├── fprm.py                # fixed-point reasoning model
    ├── ptrm.py                # TRM and PTRM model
    ├── gram.py                # GRAM model
    ├── fbi.py                 # recursive-reasoning FBI objective and tracker
    ├── fenchel_bregman.py     # compatibility import for the previous API
    ├── train.py               # training entry point
    ├── evaluation.py          # evaluation and aggregation entry point
    ├── arc.py                 # ARC-AGI data, voting, and evaluation entry point
    ├── utils.py               # data and checkpoint utilities
    └── sh/                    # launchers
```

The FPRM implementation follows [`nilskiKonjIzDunava/fprm`](https://github.com/nilskiKonjIzDunava/fprm).
The TRM and PTRM code follows [`SamsungSAILMontreal/TinyRecursiveModels`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
The autoregressive LM workflow is described [below](#autoregressive-lm-reproduction) and in the [detailed guide](auto_lm/README.md).

## Recursive reasoning installation

Create a CUDA-enabled PyTorch environment and install the dependencies:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Each launcher reads a preprocessed dataset directory from `DATASET`.
Evaluation checkpoints come from `CHECKPOINT`.
FPRM checkpoints require the released `all_config.yaml` in the same directory as the checkpoint.
Download public FPRM files from [`fixed-point-reasoners/fprm`](https://huggingface.co/fixed-point-reasoners/fprm).

## Table 1 launchers

Each task directory contains launchers for the six code-evaluated methods in this release.
The HRM values in Table 1 are paper-reported, so this repository does not provide an HRM launcher.

| Method | Maze-Hard | Sudoku-Extreme |
| --- | --- | --- |
| FPRM | `rrm/sh/maze_hard/fprm.sh` | `rrm/sh/sudoku_extreme/fprm.sh` |
| GRAM | `rrm/sh/maze_hard/gram.sh` | `rrm/sh/sudoku_extreme/gram.sh` |
| TRM | `rrm/sh/maze_hard/trm.sh` | `rrm/sh/sudoku_extreme/trm.sh` |
| PTRM | `rrm/sh/maze_hard/ptrm.sh` | `rrm/sh/sudoku_extreme/ptrm.sh` |
| W-PTRM | `rrm/sh/maze_hard/w_ptrm.sh` | `rrm/sh/sudoku_extreme/w_ptrm.sh` |
| FB | `rrm/sh/maze_hard/fb.sh` | `rrm/sh/sudoku_extreme/fb.sh` |

Use `rrm/sh/maze_hard/fb_fprm.sh` for the Maze-Hard FPRM-backbone FB experiment.

## ARC-AGI launchers

ARC-AGI-1 and ARC-AGI-2 use a dedicated evaluator in `rrm/arc.py` for augmentation inversion, Q-aware voting, task-normalized Pass@K, and submission generation.
`DATASET` must contain `test/dataset.json`, the five `test/all__*.npy` arrays, `identifiers.json`, and `test_puzzles.json` produced by the ARC preprocessing pipeline.

| Method | ARC-AGI-1 | ARC-AGI-2 |
| --- | --- | --- |
| PTRM | `rrm/sh/agi-1/ptrm.sh` | `rrm/sh/agi-2/ptrm.sh` |
| W-PTRM | `rrm/sh/agi-1/w_ptrm.sh` | `rrm/sh/agi-2/w_ptrm.sh` |

The launchers use eight `torchrun` processes, 25 candidates, and inference depth 16 by default.
PTRM defaults to global batch size 32 and latent-noise scale 0.2, while W-PTRM defaults to global batch size 768 and parameter-perturbation scale 0.3.
Set `NPROC_PER_NODE`, `CANDIDATE_COUNT`, `DEPTH`, `GLOBAL_BATCH_SIZE`, `SEED`, `LATENT_NOISE_SCALE`, or `PARAMETER_PERTURBATION_SCALE` to override these values.
Set `CONFIG` only when `all_config.yaml` is not next to the checkpoint, and set `GPU_IDS` to restrict the visible GPUs.

```bash
DATASET=/path/to/preprocessed-arc-agi-1 \
CHECKPOINT=/path/to/checkpoint.pt \
GPU_IDS=0,1,2,3,4,5,6,7 \
bash rrm/sh/agi-1/ptrm.sh
```

Complete runs write `metrics.json`, `submission.json`, `resolved_config.json`, per-rank runtime files, and rank prediction files under `OUTPUT_DIR`.

Run an evaluation with a trained checkpoint:

```bash
DATASET=/path/to/maze-hard \
CHECKPOINT=/path/to/checkpoint.pt \
OUTPUT_DIR=/path/to/output \
DEVICE=cuda:0 \
bash rrm/sh/maze_hard/fb.sh
```

`GRAM`, `FB`, and `FB-FPRM` train before evaluation when `CHECKPOINT` is unset.
Maze-Hard GRAM and both FB backbones read their initialization from `BASE_CHECKPOINT`.
Sudoku-Extreme GRAM trains from scratch.

```bash
DATASET=/path/to/sudoku-extreme \
BASE_CHECKPOINT=/path/to/base-checkpoint.pt \
OUTPUT_DIR=/path/to/output \
DEVICE=cuda:0 \
bash rrm/sh/sudoku_extreme/fb.sh
```

`fb_fprm.sh` uses eight training processes by default.
Set `NPROC_PER_NODE` to change that count.

The sharded launchers process shards in sequence on `DEVICE`.
They preserve the paper's shard boundaries and sampling resets without managing several GPU processes in shell.
Single runs write `evaluation/metrics.json`.
Sharded runs write `evaluation/aggregate/metrics.json`.
Each run records the measured selected accuracy and Pass@K in its output directory.

## Autoregressive LM reproduction

The Qwen3-4B-Base experiment uses the shared FB finite-K weighting and failure EMA in `fb.py`, the LM-specific FBI state in `auto_lm/fbi.py`, and posterior sampling from `optim/fbi_ivon.py`.
The `auto_lm/MMPO/` directory contains the MMPO/verl runtime adapted for this experiment.
The model, datasets, checkpoints, and generated outputs are external to this repository.

Use a separate CUDA environment with PyTorch, Ray, vLLM, and eight GPUs in total, either on one machine or across two four-GPU nodes.
Run the following commands from the repository root, and make the same repository and external input paths available on every Ray node.

```bash
python -m pip install -r auto_lm/MMPO/requirements.txt
python -m pip install -e 'auto_lm/MMPO[vllm]'
```

The training data and evaluation JSONL files come from MMPO commit `0ca3087f5eda8ebbd9dbe3a73bb5170eb99d2e2d`.
Set the paths below to external locations and supply a local Qwen3-4B-Base model directory.

```bash
DATA_ROOT=/absolute/path/to/data
RUN_ROOT=/absolute/path/to/runs
MODEL_PATH=/absolute/path/to/Qwen3-4B-Base
mkdir -p "$DATA_ROOT" "$RUN_ROOT"
git clone https://github.com/e3trange/MMPO.git "$DATA_ROOT/MMPO"
git -C "$DATA_ROOT/MMPO" checkout 0ca3087f5eda8ebbd9dbe3a73bb5170eb99d2e2d
TRAIN_FILE="$DATA_ROOT/MMPO/data/train.parquet"
EVAL_ROOT="$DATA_ROOT/MMPO/data/eval"
```

The training parquet must retain `data_source` and `extra_info.index` as stable problem identifiers for the FBI failure tracker.
The evaluation helpers combine MATH500, Olympiad-math, AMC23, AIME2024, and AIME2025 into a fixed Base suite and prepare the parquet used by the generator.

```bash
python auto_lm/build_base_eval_suite.py \
  --eval-root "$EVAL_ROOT" --output-dir "$RUN_ROOT/eval_suite" \
  --source-commit 0ca3087f5eda8ebbd9dbe3a73bb5170eb99d2e2d
python auto_lm/prepare_generation_parquet.py \
  --source "$RUN_ROOT/eval_suite/base_eval.jsonl" \
  --output "$RUN_ROOT/eval_suite/eval.parquet"
```

Stage 1 trains with lagged finite-K weights; stage 2 resumes its step-200 checkpoint and trains with normalized weights through step 300.
The example below assumes a running Ray cluster reachable through `RAY_ADDRESS`; use `auto` when launching on its head node.
On a single eight-GPU machine, replace each `--ray-address "$RAY_ADDRESS"` with `--nodes 1 --gpus-per-node 8`.

```bash
RAY_ADDRESS=auto
python auto_lm/run.py train --stage 1 \
  --model "$MODEL_PATH" --train-file "$TRAIN_FILE" \
  --output "$RUN_ROOT/stage1" --ray-address "$RAY_ADDRESS"
python auto_lm/run.py train --stage 2 \
  --model "$MODEL_PATH" --train-file "$TRAIN_FILE" \
  --checkpoint "$RUN_ROOT/stage1/global_step_200" \
  --output "$RUN_ROOT/stage2" --ray-address "$RAY_ADDRESS"
python auto_lm/run.py eval \
  --model "$MODEL_PATH" --train-file "$TRAIN_FILE" \
  --checkpoint "$RUN_ROOT/stage2/global_step_300" \
  --eval-file "$RUN_ROOT/eval_suite/eval.parquet" \
  --manifest "$RUN_ROOT/eval_suite/base_eval_manifest.json" \
  --output "$RUN_ROOT/evaluation" --ray-address "$RAY_ADDRESS"
```

Evaluation draws sixteen independent IVON posterior samples, scores answers with Math-Verify, and writes `evaluation/metrics.json` and `evaluation/candidates.jsonl` with Pass@1, Pass@4, Pass@8, and Pass@16 aggregation.
Add `--dry-run` to any `auto_lm/run.py` command to inspect the resolved configuration without starting Ray.
The [auto-LM guide](auto_lm/README.md) provides the training-file checksum, evaluation-file layout, and remaining protocol details.
