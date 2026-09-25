# Autoregressive FBI reproduction

This directory contains the Qwen3-4B-Base Fenchel–Bregman IVON (FBI) training and evaluation code.
`fbi.py` implements the autoregressive experiment's FBI logic using the shared FB weighting and EMA in `../fb.py`.
`MMPO/` is the experiment's MMPO/verl source based on upstream commit `0ca3087f5eda8ebbd9dbe3a73bb5170eb99d2e2d`, with FBI, IVON, and posterior evaluation support.
MMPO loads the LM-specific IVON implementation from `../optim/fbi_ivon.py`; its SHA-256 is `ec3d8943c826a71776b38abc00b57ca9f6857c834facc5152e38411617e2122c`.
The existing `optim/ivon.py` wrapper has a different sampling API and remains separate.
The MMPO source retains its Apache-2.0 license, and the LM-specific IVON source derives from [ivon-opt](https://github.com/team-approx-bayes/ivon) under GPLv3+ with its license in `../optim/IVON_LICENSE`.

## Prepare the environment and inputs

Use a CUDA environment with eight GPUs, PyTorch, Ray, vLLM, and the dependencies in `MMPO/requirements.txt`.
The original run used a PyTorch 25.06 container and two nodes with four GPUs each.
Install the local source in a separate environment from the recursive reasoning experiments on every Ray node, with the same paths visible on each node:

```bash
python -m pip install -r auto_lm/MMPO/requirements.txt
python -m pip install -e 'auto_lm/MMPO[vllm]'
```

Supply a local Qwen3-4B-Base model directory and obtain the MMPO data outside this repository:

```bash
DATA_ROOT=/absolute/path/to/external/data
RUN_ROOT=/absolute/path/to/external/runs
MODEL_PATH=/absolute/path/to/Qwen3-4B-Base
mkdir -p "$DATA_ROOT" "$RUN_ROOT"
git clone https://github.com/e3trange/MMPO.git "$DATA_ROOT/MMPO"
git -C "$DATA_ROOT/MMPO" checkout 0ca3087f5eda8ebbd9dbe3a73bb5170eb99d2e2d
TRAIN_FILE="$DATA_ROOT/MMPO/data/train.parquet"
EVAL_ROOT="$DATA_ROOT/MMPO/data/eval"
sha256sum "$TRAIN_FILE" # dd43f030222a4797a642c9ac050d1435830632154f4a4b39ad6650b829b24698
```

The 7,500-row training parquet must retain stable `data_source` and `extra_info.index` values, which identify problems for the FBI failure tracker.
The five evaluation JSONL files live under `EVAL_ROOT` with this layout:

```text
MATH500/math500.jsonl
Olympiad-math/olympiad.jsonl
AMC23/amc23.jsonl
AIME2024/aime2024.jsonl
AIME2025/aime2025.jsonl
```

The model, datasets, checkpoints, and generated outputs are external inputs or artifacts and are not included here.
The evaluation helpers construct a fixed 1,275-problem suite and its manifest from those JSONL files:

```bash
python auto_lm/build_base_eval_suite.py \
  --eval-root "$EVAL_ROOT" --output-dir "$RUN_ROOT/eval_suite" \
  --source-commit 0ca3087f5eda8ebbd9dbe3a73bb5170eb99d2e2d
python auto_lm/prepare_generation_parquet.py \
  --source "$RUN_ROOT/eval_suite/base_eval.jsonl" \
  --output "$RUN_ROOT/eval_suite/eval.parquet"
```

## Train and evaluate

Run the two training stages in order so the second stage resumes the model, optimizer, data order, and failure tracker from step 200.
Stage 1 uses lagged finite-K weights; stage 2 uses the normalized weights of the selected FBI run.
The commands below assume a running eight-GPU Ray cluster at `RAY_ADDRESS`.
For one machine with eight GPUs, replace `--ray-address "$RAY_ADDRESS"` with `--nodes 1 --gpus-per-node 8` in each command.

```bash
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

Evaluation uses sixteen independent IVON posterior draws, one generated answer per problem and draw, temperature 0.6, top-p 0.95, and Math-Verify binary scoring.
It writes `evaluation/metrics.json` and `evaluation/candidates.jsonl`; the aggregation script computes Pass@1, Pass@4, Pass@8, and Pass@16 from the generated candidates.
Use `--dry-run` with any `run.py` command to inspect its resolved command without starting Ray or loading a model.
