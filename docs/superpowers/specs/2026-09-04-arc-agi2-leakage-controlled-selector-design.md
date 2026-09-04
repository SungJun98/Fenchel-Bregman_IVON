# ARC-AGI-2 Leakage-Controlled Candidate Selector Design

## Objective

The experiment will test whether a lightweight candidate-level selector trained only on ARC-AGI-2 `training2` tasks can improve the frozen FB candidate ranking on ARC-AGI-2 evaluation without using evaluation outputs for training, model selection, policy selection, or prediction generation.
The primary target is to exceed the strongest recorded baselines at Pass@1, Pass@2, and Pass@100 with no external pretrained model.
The running ARC-AGI-1 evaluation is outside this experiment and must not be modified, interrupted, or slowed materially.

## Current Evidence and Constraints

The frozen FB posterior pool already contains more correct ARC-AGI-2 evaluation outputs than the current low-budget ranking exposes.
The saved evaluation portfolio reaches a 13.75% label-oracle ceiling, while its label-free posterior-vote ranking reaches 4.58%, 6.25%, and 12.78% at Pass@1, Pass@2, and Pass@100.
The strongest recorded baselines reach 5.00%, 7.50%, and 11.94% at the same budgets.
The existing full-run code computes model outputs from `inputs` and `puzzle_identifiers`, and the loss wrapper reads labels only after logits and Q values have been produced.
The existing verified run also records that target labels were not used for its frozen row-support policy selection.
However, the current posterior portfolio writer stores candidate hashes and aggregate statistics without populating its candidate-grid map, so the saved payload cannot support a content-aware verifier for every candidate.
The previously frozen demo-only symbolic solver produces no accepted prediction on the 120 evaluation tasks, so it remains a safe abstaining ablation but cannot be the primary improvement mechanism.

## Contamination Contract

The selector-facing training process may read only `training2` challenges, `training2` solutions, and candidate traces generated for `training2` tasks.
The prediction process may read evaluation challenges, frozen model artifacts, frozen selector artifacts, and posterior samples, but it may not read evaluation solutions or any result derived from them.
The scoring process is a separate executable that starts only after the prediction manifest and all output hashes have been atomically locked.
No module may import the task-specific post-hoc symbolic recovery code or contain evaluation task identifiers.
Previously observed evaluation diagnostics may motivate the generic feature families, but no feature, threshold, model, or fallback decision may be changed after the protocol is frozen.
The final report must disclose that evaluation labels were exposed in earlier project work even though the new executable boundary is label-free.

## Considered Approaches

### Statistics-only offline reranker

A statistics-only reranker could reuse the current saved hashes and run immediately without GPUs.
It is not the primary approach because vote, row-support, Q, and MBR-style aggregate signals have already failed to recover enough low-rank correct candidates on the hard evaluation distribution.

### Content-aware lightweight selector

The recommended approach combines posterior statistics with generic demonstration-to-candidate transformation features and trains a small pairwise linear ranker on complete `training2` tasks.
This approach directly targets the missing candidate-specific evidence while keeping the model small, auditable, and inexpensive to train.

### Neural Q-head retraining

Retraining the native Q-head would require saving hidden states or rerunning model inference and would still need a carefully designed candidate-verification target.
It is deferred until the lightweight selector establishes that demonstration-aware features transfer, because it adds substantially more GPU cost and confounds the diagnostic with representation learning.

## Architecture

### Candidate trace adapter

The adapter reconstructs one deduplicated candidate list for each task and test pair from all augmentations and posterior draws.
It records vote count, row support, Q mean, Q maximum, shard stability, max-Q presence, original ranks, the inverse-augmentation-normalized grid, and stable hashes.
The adapter never receives a solution in prediction mode.

### Demonstration-aware feature extractor

The extractor compares each candidate with transformations visible in the task demonstrations.
The fixed feature schema covers output shape rules, color additions and removals, color histograms, foreground bounding boxes, connected components, symmetry changes, changed-cell fractions, candidate consensus, and agreement with any frozen generic demo-only DSL program.
All features are invariant to task identifiers and contain no manually authored evaluation-specific rule.

### Pairwise selector

The primary selector is an L2-regularized pairwise logistic ranker trained on feature differences between an exact candidate and hard wrong candidates from the same pair.
Hard negatives are restricted to high-vote, high-Q, high-support, or demonstration-similar wrong candidates so that easy invalid outputs do not dominate training.
The searched regularization and blend grid is deliberately small and fixed before any new evaluation scoring.
Tree-based and statistics-only models are diagnostic ablations and cannot replace the primary selector based on evaluation results.

### Ranking policy

The frozen policy starts from the label-free posterior-vote ranking and reranks only its first 100 candidates using a preregistered blend of baseline rank and selector score.
Restricting changes to the same top-100 set makes Pass@100 invariant to low-budget reranking.
The frozen demo-only DSL is reported as a separate preregistered ablation, while the primary selector-only ranking never changes the top-100 candidate set.
The DSL ablation may insert a unanimous accepted prediction at rank one, while abstention preserves the complete candidate order.

### Prediction-only evaluator

The evaluator loads augmented inputs and identifiers without opening the corresponding label arrays, creates dummy ignore-label tensors required by the model wrapper, loads evaluation challenges without outputs for demonstration features, and writes full normalized candidate grids plus statistics.
It does not instantiate an ARC scorer or load an evaluation solution file.
The evaluator first verifies on a smoke batch that predictions and Q values are invariant when dummy labels are changed.

### Locked scorer

The scorer validates source, checkpoint, selector, protocol, data, and prediction hashes before loading evaluation solutions.
It independently computes task-weighted Pass@1, Pass@2, and Pass@100 and compares the frozen selector with max-Q, posterior vote, row support, TRM, PTRM, and W-PTRM references.

## Training and Selection Protocol

The existing 128-task `training2` posterior trace supplies the initial candidate bank and is split by complete task rather than candidate row.
Nested task-grouped cross-validation selects the regularization and rank-blend coefficient using a lexicographic objective that first rejects any Pass@100 regression, then maximizes the worst Pass@1 and Pass@2 delta across folds, and finally minimizes the number of harmed tasks.
The same task folds are reused across the available posterior-scale traces to test robustness to posterior sampling conditions.
The selector is fit once on all permitted training tasks only after the configuration passes the cross-validation gate.
The model, feature schema, rank policy, DSL code, and all hashes are then frozen in a read-only manifest.

## Training-Task Pre-Evaluation Gate

Before the full evaluation, a small candidate trace is generated for a deterministic selector-heldout subset of `training2` tasks that is disjoint from the 128-task fitting bank and uses a fixed reduced augmentation budget.
The heldout trace is predicted before its solutions are supplied to the separate scorer, and its prediction hash is locked before scoring.
The full evaluation is authorized only if Pass@1 and Pass@2 both improve over the preregistered baseline, Pass@100 proxy does not decrease, no fold or heldout result shows a catastrophic task-level regression, and the label-invariance test passes.
The selector and thresholds cannot be revised after this heldout score, regardless of whether the gate passes.

## Final Evaluation

After the ARC-AGI-1 process reaches a terminal state and releases the GPUs, the prediction-only evaluator uses GPUs 0 through 9 with the frozen FB checkpoint, posterior bank, scale, candidate count, depth, and augmentation budget.
The candidate payload and frozen ranking are finalized before the scorer reads evaluation solutions.
The primary paper gate requires strict improvement over 5.00% Pass@1, 7.50% Pass@2, and 11.94% Pass@100.
A result is considered compelling only if the Pass@1 and Pass@2 gains are at least one percentage point or their paired task-bootstrap intervals exclude zero.
All component ablations use the identical candidate payload, so they require no additional model inference.

## Compute Isolation and Scheduling

All preparation while ARC-AGI-1 is active runs with `CUDA_VISIBLE_DEVICES` empty, at most four CPU threads, `nice` level 19, idle I/O priority, and single-threaded BLAS backends.
Candidate artifacts are loaded once per stage to avoid repeated reads from `/data1`.
A detached state-based follower may start the GPU stages after ARC-AGI-1 terminates, but it may not poll GPUs at high frequency or alter the ARC-AGI-1 process tree.
The preparation stage pauses automatically if the observed ARC-AGI-1 batch duration rises by more than three percent over its established baseline.

## Failure Handling

Any provenance mismatch, evaluation-solution access during prediction, label-invariance failure, incomplete candidate payload, non-finite feature, or score replay mismatch fails closed and preserves all diagnostic artifacts.
Failure of the training or training-task holdout gate stops the expensive full evaluation and is reported as a negative result.
Failure of the primary paper gate after final scoring does not trigger evaluation-driven retuning under this protocol.

## Verification and Artifacts

Unit tests cover split isolation, label masking, label invariance, candidate deduplication, feature invariance, pairwise examples, prefix preservation, DSL abstention, task-weighted metrics, and hash enforcement.
An independent result-only verifier recomputes all reported metrics and added or lost pairs without importing the training or ranking implementation.
Source files live under a new ARC-AGI-2 experiment directory, while large runtime outputs live under `/data1/lsj9862/VR/experiments`.
Slack updates are posted only at meaningful checkpoints to thread `1788507328.161409` in channel `C0B967ZF85C`.

## Expected Runtime

CPU-only provenance, feature, training, and cross-validation work is expected to take four to six hours under the isolation limits.
The reduced training-task holdout trace should take less than one hour on ten GPUs.
If the holdout gate passes, the full 1,000-augmentation prediction run is expected to take approximately seven to eight hours on ten GPUs, followed by less than one hour for locked rescoring and independent verification.
