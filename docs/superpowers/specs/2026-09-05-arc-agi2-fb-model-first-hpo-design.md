# ARC-AGI-2 FB Model-First HPO Design

## Objective

This campaign will search the existing FB training and posterior hyperparameters without adding a new model, head, loss, learned selector, external model, or inference-time verifier.
The deployable target is one FB posterior whose official ARC aggregation strictly exceeds the strongest recorded baseline at Pass@1, Pass@2, and Pass@100 while retaining the existing method definition.
The strongest recorded baseline thresholds are 5.000%, 7.500%, and 11.944%, respectively.

## Interpretation of the ARC-AGI-2 Evaluation Split

Previous project work repeatedly inspected ARC-AGI-2 evaluation labels, and this HPO will use those labels for paired candidate comparison.
The 120-task public evaluation split is therefore a development set for this campaign rather than an untouched test set.
Any result reported from it must be described as evaluation-tuned or diagnostic, and a generalization claim requires a hidden test set or another untouched benchmark.
Released evaluation-task demonstrations may be used as model inputs and posterior-training examples, but their held-out target grids are never supplied as training labels.

## Immutable Inputs and Method Boundary

The base model is the locally reproduced ARC-AGI-2 TRM checkpoint at `/data1/lsj9862/VR/experiments/arc_agi2_trm_train_20260827/official_seed0/step_723914`.
Its SHA-256 is `0f4956d3102938d380c4d721da3f86013faa86abb315bcf6b060fd241602200a`.
The preprocessed source contains 1,280 groups, of which groups 1,000 through 1,119 are the 120 released evaluation-task demonstration groups.
The model architecture, frozen puzzle embeddings, depth 16, global training batch size 128, twelve dense posterior tensors, official count-first and mean-Q-second aggregation, and final `K_eval=25` remain fixed.
Every deployable candidate starts from the same base checkpoint and trains on all 1,280 source groups.
Continuation from 100 to 400 to 800 updates is permitted only when every other training hyperparameter is identical, so a promoted run is not redundantly restarted.

## Evaluation-Demonstration Replay Hyperparameter

The current implementation already includes evaluation-task demonstrations in posterior training because it excludes them only from validation eligibility, not from the training stream.
The campaign will therefore tune their relative sampling exposure rather than introduce a new demonstration mechanism.
Replay multipliers `1`, `2`, `4`, and `8` correspond to approximate evaluation-demonstration sampling shares of 9.38%, 17.14%, 29.27%, and 45.28%.
Multiplier `0` is retained only as a diagnostic control and cannot be selected as the final method because it changes the intended all-data training contract.
Replay weighting changes only the source-group sampler and continues to use the existing FB task loss and Q auxiliary target.
Shape, color, area, and symmetry relationships derived from demonstrations may be used for post-run diagnostics, but they are not model inputs, loss terms, or selector features.

## Search Space

The coarse training space covers learning rate log-uniformly from `1e-5` to `3e-4`, `K_train` in `{2, 5, 10, 25}`, failure temperature from `0.25` to `4`, tracker beta in `{0.90, 0.95, 0.99, 0.995}`, and initial dual weight in `{0.02, 0.1, 0.5, 1.0}`.
The initial failure state is derived from the initial dual weight and `K_train` so that candidate-count comparisons begin at the same intended weighting state.
The remaining coarse dimensions are task loss in `{mean, smooth_weakest}`, Q mode in `{native, three_part}`, Q coefficient log-uniformly from `0.1` to `2`, weight decay in `{0, 0.05, 0.1, 0.225, 0.5}`, and evaluation-demonstration replay multiplier in `{1, 2, 4, 8}`.
One multiplier-zero control is evaluated outside the selectable 64 configurations.

Posterior refinement covers IVON effective sample size in `{10000, 30000, 100000}`, Hessian initialization in `{1, 3, 10}`, beta2 in `{0.999, 0.9999, 0.99999}`, curvature interval in `{10, 20, 40}`, and curvature-proxy scale in `{0.75, 1.084, 1.5}`.
Evaluation posterior scale is explored from `0.5` through `2.5`, followed by local refinement at increments of `0.125` around the best stable region.
The initial 64 configurations use a deterministic Sobol sequence, and subsequent promotions use BOHB-style multi-fidelity allocation rather than an exhaustive Cartesian grid.

## Search Stages

### Stage 0: Preflight and calibration

The campaign verifies GPU identity and availability, checkpoint and data hashes, source revision, group counts, evaluation task and pair counts, and existing baseline metrics.
A two-update training smoke and one-batch compiled evaluation smoke must pass before the HPO starts.
The smoke also measures actual examples per second and updates the ETA without changing any search decision.

### Stage 1: Coarse search

Sixty-four selectable configurations run for 100 updates with 16 common augmentation prefixes and 10 common posterior draws.
Ten single-GPU trials may run concurrently on physical GPUs 0 through 9.
The stage promotes 24 candidates using the union of the empirical Pareto frontier and candidates whose paired-bootstrap uncertainty still permits improvement on all three target metrics.

### Stage 2: Fine search

The 24 promoted candidates continue or train to at most 400 updates and are evaluated with 64 common augmentation prefixes and 25 common posterior draws.
Candidates with a confidently dominated metric vector are stopped early, while ambiguous candidates retain their full budget.
The stage promotes at most 18 candidates to posterior refinement.

### Stage 3: Posterior refinement

At most 18 candidates continue or train to at most 800 updates while the IVON and posterior-scale dimensions are refined with 128 common augmentations and 25 common draws.
Only the small final Pareto set is extended from the cached 128-prefix result to 256 augmentations.
No candidate is selected by sacrificing one target metric for a gain in another.

### Stage 4: Stability check and final evaluation

HPO training uses seed 0 only, with identical augmentation and posterior-noise seeds across candidates to maximize paired comparison power.
There is no k-fold cross-validation and no multi-seed sweep during HPO.
The single finalist is trained afresh from the immutable base checkpoint under training seed 1 only for one 128-augmentation sanity check, which is reported as stability evidence and is not used to tune hyperparameters.
If the seed-1 direction reverses on any target metric, the campaign marks the candidate unstable and does not silently replace it with an evaluation-selected runner-up.
If the finalist passes the development gate, the seed-0 artifact is evaluated once with 1,000 augmentations, 25 posterior draws, depth 16, and all ten GPUs under the official aggregation protocol.

## Candidate Comparison and Gates

All candidate comparisons use the same task order, augmentation prefixes, and standard-normal posterior draws.
Task-level paired bootstrap intervals are computed offline from cached per-task outcomes and do not require model reruns.
Early stages retain uncertain Pareto candidates rather than applying a noisy hard threshold.
The 256-augmentation gate requires positive point-estimate deltas over the strongest baseline at Pass@1, Pass@2, and Pass@100 and no bootstrap evidence of a material regression.
The final adoption gate requires Pass@1 above 5.000%, Pass@2 above 7.500%, and Pass@100 above 11.944% under the 1,000-augmentation protocol.
Failure of any one condition is reported directly and blocks automatic paper-table replacement.

## Compute Scheduling and Runtime

Independent HPO candidates occupy one GPU each so that up to ten configurations run concurrently.
The final 1,000-augmentation evaluation uses distributed ranks on physical GPUs 0 through 9.
The expected wall time is 45 to 50 hours, with a conservative upper bound of 55 hours, based on the prior 6.45-hour ten-GPU full evaluation and measured candidate-training runtimes.
Training is not the main bottleneck; reduced and full evaluations dominate the schedule.
A detached state-driven supervisor advances phases, records process identities and exit codes, and avoids high-frequency Codex polling.

## Failure Handling and Reproducibility

The pipeline fails closed on a checkpoint or data hash mismatch, non-finite training state, missing rank payload, incomplete task coverage, GPU out-of-memory error, inconsistent rescore, or unauthorized method change.
Every configuration receives a canonical identifier derived from its complete configuration and immutable input hashes.
Stage outputs are atomic and resumable, and a completed candidate is never rerun unless its artifact verification fails.
The supervisor preserves logs and terminal failure records and does not automatically alter the frozen search space after observing evaluation results.

## Verification, Paper, and Slack Reporting

An independent rescorer must exactly reproduce Pass@1, Pass@2, and Pass@100 from the final rank payloads and confirm 120 tasks, 172 test pairs, and zero missing pairs.
The campaign stores the full configuration ledger, Sobol points, promotion decisions, paired-bootstrap tables, checkpoint and posterior hashes, runtime records, official metrics, and independent metrics under a new experiment root in `/data1/lsj9862/VR/experiments`.
The current paper statement that all evaluation-task groups were excluded from posterior training is inconsistent with the executed code and must be corrected before any verified result is incorporated.
No LaTeX or PDF value is changed until final scoring, independent rescoring, and the disclosure wording are all verified.
Slack reporting uses the new thread rooted at timestamp `1788606339.092529` in channel `C0B967ZF85C`.
Updates are limited to meaningful phase boundaries, failures, ETA changes, and the final conclusion in accordance with `Rules/slack.md`.
