# ARC-AGI-2 FB Model-First HPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a resumable ten-GPU ARC-AGI-2 FB hyperparameter campaign that preserves the existing method boundary and adopts a result only when Pass@1, Pass@2, and Pass@100 all exceed the recorded baselines.

**Architecture:** Create a clean worktree from the hardened ARC-AGI-2 FB branch, add deterministic replay weighting and search-space primitives to the existing trainer, and place prefix-data construction, task-paired scoring, trial execution, and campaign supervision in separate focused modules. The detached supervisor owns state transitions and atomic artifacts, while Slack updates are emitted only at stage boundaries through the already-created thread.

**Tech Stack:** Python 3.12, PyTorch 2.8, NumPy, pytest, torch.compile, IVON, torch.distributed, NVIDIA RTX A6000 GPUs, JSON and pickle artifacts, Bash process launch.

**Spec:** `docs/superpowers/specs/2026-09-05-arc-agi2-fb-model-first-hpo-design.md`

## Global Constraints

The immutable base checkpoint is `/data1/lsj9862/VR/experiments/arc_agi2_trm_train_20260827/official_seed0/step_723914` with SHA-256 `0f4956d3102938d380c4d721da3f86013faa86abb315bcf6b060fd241602200a`.
The immutable source data are `/data1/lsj9862/VR/experiments/arc_agi2_trm_eval_20260825/data/arc2concept-aug-1000` with 1,280 training groups, 120 evaluation tasks, 172 test pairs, and 1,000 augmentations.
The architecture, frozen puzzle embeddings, twelve dense posterior tensors, global training batch size 128, inference depth 16, final `K_eval=25`, and count-first then mean-Q aggregation cannot change.
No new model, learned head, loss, selector, external model, or inference-time verifier may be introduced.
Every selectable HPO trial trains on all 1,280 source groups, and evaluation-demonstration replay multipliers are restricted to `1`, `2`, `4`, and `8`.
Training seed 0 is used throughout HPO, and only the single finalist receives a fresh seed-1 128-augmentation stability run.
The public evaluation labels are development labels for this campaign and cannot support an untouched-test generalization claim.
All GPU jobs use physical GPUs 0 through 9, and the final full evaluation uses all ten devices.
The runtime root is `/data1/lsj9862/VR/experiments/arc_agi2_fb_model_first_hpo_20260905`.
Slack updates use channel `C0B967ZF85C` and thread `1788606339.092529`.

---

### Task 1: Create an isolated worktree and campaign contract

**Files:**
- Create: `fb_model_first_contract.py`
- Create: `tests/test_fb_model_first_contract.py`
- Verify: `run_fb_arc.py`
- Verify: `evaluate_arc_checkpoint.py`

**Interfaces:**
- Consumes: the hardened source branch at commit `be09231` and the immutable paths in Global Constraints.
- Produces: `CampaignContract`, `sha256_file(path: Path) -> str`, and `validate_preflight(contract: CampaignContract) -> dict[str, object]`.

- [ ] **Step 1: Create the isolated worktree**

Run:

```bash
git worktree add -b codex/arc-agi2-fb-model-first-hpo-20260905 /home/lsj9862/new_topic/bayes_reasoning/experiments/arc_agi2_fb_model_first_hpo_20260905/TinyRecursiveModels be09231
```

Expected: the new worktree is on `codex/arc-agi2-fb-model-first-hpo-20260905`, and the dirty source checkout remains unchanged.

- [ ] **Step 2: Write the failing contract tests**

```python
def test_contract_fixes_method_and_dataset_counts(tmp_path: Path) -> None:
    contract = CampaignContract.defaults(runtime_root=tmp_path)
    assert contract.training_groups == 1280
    assert contract.evaluation_tasks == 120
    assert contract.evaluation_pairs == 172
    assert contract.depth == 16
    assert contract.k_eval == 25
    assert contract.gpu_ids == tuple(range(10))


def test_preflight_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"wrong")
    contract = replace(CampaignContract.defaults(tmp_path), checkpoint=checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint SHA-256"):
        validate_preflight(contract)
```

- [ ] **Step 3: Run the tests and verify failure**

Run: `pytest -q tests/test_fb_model_first_contract.py`

Expected: FAIL because `fb_model_first_contract` does not exist.

- [ ] **Step 4: Implement the immutable contract and preflight**

```python
@dataclass(frozen=True)
class CampaignContract:
    runtime_root: Path
    data_root: Path
    checkpoint: Path
    checkpoint_sha256: str
    training_groups: int = 1280
    evaluation_tasks: int = 120
    evaluation_pairs: int = 172
    depth: int = 16
    k_eval: int = 25
    gpu_ids: tuple[int, ...] = tuple(range(10))

    @classmethod
    def defaults(cls, runtime_root: Path) -> "CampaignContract":
        return cls(
            runtime_root=runtime_root,
            data_root=Path("/data1/lsj9862/VR/experiments/arc_agi2_trm_eval_20260825/data/arc2concept-aug-1000"),
            checkpoint=Path("/data1/lsj9862/VR/experiments/arc_agi2_trm_train_20260827/official_seed0/step_723914"),
            checkpoint_sha256="0f4956d3102938d380c4d721da3f86013faa86abb315bcf6b060fd241602200a",
        )
```

`validate_preflight` must verify the checkpoint hash, train group count, test task count, test-pair count, dataset metadata, available GPU indices, and absence of active compute processes before it returns an atomic JSON-serializable record.

- [ ] **Step 5: Run the focused and existing CPU tests**

Run: `pytest -q tests/test_fb_model_first_contract.py tests/test_fb_arc.py tests/test_run_fb_arc.py`

Expected: PASS.

- [ ] **Step 6: Commit the contract**

```bash
git add fb_model_first_contract.py tests/test_fb_model_first_contract.py
git commit -m "feat: add ARC-AGI-2 model-first campaign contract"
```

---

### Task 2: Add deterministic demonstration replay and expose all training hyperparameters

**Files:**
- Modify: `fb_arc.py`
- Modify: `run_fb_arc.py`
- Modify: `tests/test_fb_arc.py`
- Modify: `tests/test_run_fb_arc.py`

**Interfaces:**
- Consumes: source-group IDs from the 1,280-group training arrays.
- Produces: `demo_group_multipliers(group_ids: np.ndarray, replay: int) -> np.ndarray`, `initial_failure_from_dual_weight(candidate_count: int, initial_dual_weight: float) -> float`, and complete candidate CLI/config serialization.

- [ ] **Step 1: Write failing replay and initialization tests**

```python
def test_demo_replay_builds_exact_deterministic_epoch_share() -> None:
    groups = np.arange(1280, dtype=np.int32)
    multipliers = demo_group_multipliers(groups, replay=4)
    assert multipliers[:1000].tolist() == [1] * 1000
    assert multipliers[1000:1120].tolist() == [4] * 120
    assert multipliers[1120:].tolist() == [1] * 160
    assert multipliers.sum() == 1640


@pytest.mark.parametrize("k,weight", [(2, 0.02), (5, 0.1), (10, 0.5), (25, 1.0)])
def test_initial_failure_reconstructs_requested_dual_weight(k: int, weight: float) -> None:
    failure = initial_failure_from_dual_weight(k, weight)
    assert k * failure ** (k - 1) == pytest.approx(weight)
```

Add a stream-state round-trip test that uses nonuniform multipliers and proves the next batch is bitwise identical after resume.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `pytest -q tests/test_fb_arc.py tests/test_run_fb_arc.py -k 'demo_replay or initial_failure or stream_state'`

Expected: FAIL because the replay and initialization interfaces do not exist.

- [ ] **Step 3: Implement integer-weighted epoch sampling**

```python
def demo_group_multipliers(group_ids: np.ndarray, replay: int) -> np.ndarray:
    if replay not in {0, 1, 2, 4, 8}:
        raise ValueError("evaluation-demo replay must be one of 0, 1, 2, 4, or 8")
    values = np.ones(group_ids.shape, dtype=np.int16)
    mask = (group_ids >= 1000) & (group_ids < 1120)
    values[mask] = replay
    if not values.sum():
        raise ValueError("weighted training pool is empty")
    return values


def initial_failure_from_dual_weight(candidate_count: int, initial_dual_weight: float) -> float:
    if candidate_count < 2 or not 0 < initial_dual_weight <= candidate_count:
        raise ValueError("invalid initial dual weight")
    return (initial_dual_weight / candidate_count) ** (1.0 / (candidate_count - 1))
```

Change `ARCTrainingStream` to accept an aligned integer `group_multipliers` vector, permute `np.repeat(group_ids, group_multipliers)` at epoch boundaries, and include the vector in its exact-resume state contract.

- [ ] **Step 4: Extend `CandidateConfig` and the candidate CLI**

Expose and validate `candidate_count`, `tracker_beta`, `initial_dual_weight`, `ivon_ess`, `ivon_hess_init`, `ivon_beta2`, `weight_decay`, `curvature_interval`, `curvature_proxy_scale`, and `evaluation_demo_replay`.
Require `train_split=full` for selectable campaign trials, derive `initial_failure` from the requested dual weight, and write every value to `config.json`.

- [ ] **Step 5: Run focused tests**

Run: `pytest -q tests/test_fb_arc.py tests/test_run_fb_arc.py`

Expected: PASS.

- [ ] **Step 6: Commit replay-aware training**

```bash
git add fb_arc.py run_fb_arc.py tests/test_fb_arc.py tests/test_run_fb_arc.py
git commit -m "feat: add replay-aware FB candidate training"
```

---

### Task 3: Generate the deterministic coarse and refinement spaces

**Files:**
- Create: `fb_model_first_search.py`
- Create: `tests/test_fb_model_first_search.py`

**Interfaces:**
- Consumes: complete candidate dictionaries and stage metric records.
- Produces: `generate_coarse_configs(seed: int = 0, count: int = 64) -> list[dict[str, object]]`, `propose_fine_configs(promoted: Sequence[Mapping[str, object]], count: int = 24, seed: int = 1) -> list[dict[str, object]]`, `canonical_config_id(config: Mapping[str, object]) -> str`, `training_fingerprint(config: Mapping[str, object]) -> str`, `propose_posterior_refinements(promoted: Sequence[Mapping[str, object]], count: int = 18) -> list[dict[str, object]]`, and `select_uncertain_pareto(records: Sequence[Mapping[str, object]], keep: int) -> list[str]`.

- [ ] **Step 1: Write failing deterministic-space tests**

```python
def test_coarse_space_is_deterministic_unique_and_in_bounds() -> None:
    first = generate_coarse_configs()
    second = generate_coarse_configs()
    assert first == second
    assert len(first) == len({canonical_config_id(item) for item in first}) == 64
    assert {item["candidate_count"] for item in first} == {2, 5, 10, 25}
    assert {item["evaluation_demo_replay"] for item in first} == {1, 2, 4, 8}
    assert all(1e-5 <= item["learning_rate"] <= 3e-4 for item in first)


def test_training_fingerprint_ignores_only_evaluation_scale() -> None:
    config = generate_coarse_configs()[0]
    assert training_fingerprint(config) == training_fingerprint({**config, "posterior_scale": 2.0})
    assert training_fingerprint(config) != training_fingerprint({**config, "tracker_beta": 0.95})
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest -q tests/test_fb_model_first_search.py`

Expected: FAIL because the search module does not exist.

- [ ] **Step 3: Implement Sobol mapping and stable identifiers**

Use `torch.quasirandom.SobolEngine(dimension=10, scramble=True, seed=0)` and map continuous coordinates by fixed formulas.

```python
def log_map(value: float, lower: float, upper: float) -> float:
    return math.exp(math.log(lower) + value * (math.log(upper) - math.log(lower)))


def categorical(value: float, choices: Sequence[T]) -> T:
    return choices[min(int(value * len(choices)), len(choices) - 1)]
```

Canonical IDs must hash sorted compact JSON and include every training and posterior field except stage resource budgets.

- [ ] **Step 4: Implement refinement proposals and uncertainty-preserving promotion**

`propose_fine_configs` must retain the strongest exact anchors and use a second deterministic Sobol stream to perturb continuous dimensions locally within the original bounds and move categorical dimensions only to adjacent ordered choices.
An exact anchor may resume from 100 updates, while any changed training fingerprint starts afresh from the immutable checkpoint.
Posterior refinements must draw only from ESS `{10000, 30000, 100000}`, Hessian initialization `{1, 3, 10}`, beta2 `{0.999, 0.9999, 0.99999}`, curvature interval `{10, 20, 40}`, curvature-proxy scale `{0.75, 1.084, 1.5}`, and posterior scale `[0.5, 2.5]`.
`select_uncertain_pareto` must retain the empirical Pareto frontier first, then records whose paired confidence intervals still overlap a non-dominated region, and finally fill remaining slots by the largest minimum standardized delta across Pass@1, Pass@2, and Pass@100.

- [ ] **Step 5: Run the search tests**

Run: `pytest -q tests/test_fb_model_first_search.py`

Expected: PASS with exactly 64 unique coarse configurations and at most 18 valid posterior refinements.

- [ ] **Step 6: Commit the search space**

```bash
git add fb_model_first_search.py tests/test_fb_model_first_search.py
git commit -m "feat: define deterministic FB model-first search"
```

---

### Task 4: Build nested evaluation-prefix datasets without regenerating augmentations

**Files:**
- Create: `build_arc_evaluation_prefix.py`
- Create: `tests/test_build_arc_evaluation_prefix.py`

**Interfaces:**
- Consumes: the immutable `arc2concept-aug-1000` test arrays.
- Produces: `select_prefix_puzzles(group_indices: np.ndarray, augmentations: int) -> np.ndarray`, `write_prefix_dataset(source: Path, destination: Path, augmentations: int) -> dict[str, object]`, and prefix roots for `16`, `64`, `128`, and `256` augmentations.

- [ ] **Step 1: Write failing synthetic-prefix tests**

```python
def test_prefix_selection_is_group_balanced_and_nested() -> None:
    group_indices = np.array([0, 3, 5, 9], dtype=np.int64)
    first = select_prefix_puzzles(group_indices, augmentations=1)
    second = select_prefix_puzzles(group_indices, augmentations=2)
    assert first.tolist() == [0, 3, 5]
    assert second.tolist() == [0, 1, 3, 4, 5, 6]
    assert set(first).issubset(second)


def test_written_prefix_rebuilds_all_indices_and_preserves_identifier_space(tmp_path: Path) -> None:
    source = make_synthetic_arc_dataset(tmp_path / "source")
    result = write_prefix_dataset(source, tmp_path / "prefix", augmentations=2)
    assert result["groups"] == 3
    assert result["max_augmentations_per_group"] == 2
    assert json.loads((tmp_path / "prefix/test/dataset.json").read_text())["num_puzzle_identifiers"] == 99
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest -q tests/test_build_arc_evaluation_prefix.py`

Expected: FAIL because the prefix builder does not exist.

- [ ] **Step 3: Implement deterministic prefix extraction**

For every original test group, select the first `min(augmentations, group_size)` puzzle rows, copy the corresponding contiguous example slices, and rebuild `puzzle_indices` and `group_indices` from zero.
Preserve original `puzzle_identifiers` values and the original `num_puzzle_identifiers` metadata so the immutable checkpoint embedding shape remains valid.
Symlink `identifiers.json` and `test_puzzles.json` to the immutable source, write arrays through temporary files followed by `os.replace`, and record source hashes plus selected counts in `prefix_manifest.json`.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/test_build_arc_evaluation_prefix.py`

Expected: PASS.

- [ ] **Step 5: Build and audit real prefix roots**

Run:

```bash
python build_arc_evaluation_prefix.py --source /data1/lsj9862/VR/experiments/arc_agi2_trm_eval_20260825/data/arc2concept-aug-1000 --output-root /data1/lsj9862/VR/experiments/arc_agi2_fb_model_first_hpo_20260905/data-prefixes --augmentations 16 64 128 256
```

Expected: four manifests, 120 tasks, 172 test pairs, unchanged identifier-space size, and strictly nested selected puzzle identifiers.

- [ ] **Step 6: Commit the prefix builder**

```bash
git add build_arc_evaluation_prefix.py tests/test_build_arc_evaluation_prefix.py
git commit -m "feat: build nested ARC evaluation prefixes"
```

---

### Task 5: Add task-paired metrics, bootstrap intervals, and prefix baselines

**Files:**
- Create: `fb_model_first_metrics.py`
- Create: `tests/test_fb_model_first_metrics.py`
- Reuse: `rescore_arc_payloads.py`

**Interfaces:**
- Consumes: rank prediction payloads, labeled `test_puzzles.json`, and fixed baseline records.
- Produces: `score_task_vectors(test_puzzles: Mapping[str, object], payloads: Sequence[tuple[dict, dict]], pass_ks: Sequence[int]) -> dict[str, list[float]]`, `paired_bootstrap(candidate: Mapping[str, Sequence[float]], baseline: Mapping[str, Sequence[float]], seed: int = 0, draws: int = 10000) -> dict[str, object]`, and `strongest_metricwise_baseline(records: Sequence[Mapping[str, object]]) -> dict[str, object]`.

- [ ] **Step 1: Write failing paired-metric tests**

```python
def test_task_vectors_average_multi_pair_tasks_before_tasks() -> None:
    vectors = score_task_vectors(TEST_PUZZLES, PAYLOADS, pass_ks=(1, 2, 100))
    assert vectors["ARC/pass@1"] == [0.5, 1.0]


def test_paired_bootstrap_uses_identical_task_resamples() -> None:
    candidate = {"ARC/pass@1": [1.0, 0.0, 1.0]}
    baseline = {"ARC/pass@1": [0.0, 0.0, 1.0]}
    result = paired_bootstrap(candidate, baseline, seed=0, draws=1000)
    assert result["ARC/pass@1"]["point_delta"] == pytest.approx(1 / 3)
    assert result == paired_bootstrap(candidate, baseline, seed=0, draws=1000)
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest -q tests/test_fb_model_first_metrics.py`

Expected: FAIL because the metrics module does not exist.

- [ ] **Step 3: Implement exact task-normalized scoring and paired bootstrap**

Use the same count-first then mean-Q candidate ordering as `rescore_arc_payloads.score_payloads`, but retain one fractional correctness value per task and Pass@K.
Bootstrap a single task-index matrix from NumPy Philox seed 0 and reuse it for every candidate and metric comparison.
Return the point delta, 2.5th and 97.5th percentiles, probability of positive delta, gained tasks, and harmed tasks.

- [ ] **Step 4: Add prefix-baseline execution to the campaign contract**

At prefix budgets `16`, `64`, `128`, and `256`, evaluate the frozen TRM, PTRM, and W-PTRM artifacts with their existing method settings and identical prefix roots.
For each metric, identify the method with the highest aggregate score and use that method's complete task vector for paired comparison.
Persist the metric-to-method mapping and every method-specific record so no per-task oracle or synthetic single-model claim is created.

- [ ] **Step 5: Run focused scoring tests**

Run: `pytest -q tests/test_fb_model_first_metrics.py tests/test_rescore_arc_payloads.py`

Expected: PASS and exact agreement with the existing aggregate scorer.

- [ ] **Step 6: Commit paired scoring**

```bash
git add fb_model_first_metrics.py tests/test_fb_model_first_metrics.py
git commit -m "feat: add paired ARC HPO metrics"
```

---

### Task 6: Implement one atomic, resumable training-and-evaluation trial

**Files:**
- Create: `run_fb_model_first_trial.py`
- Create: `tests/test_run_fb_model_first_trial.py`
- Reuse: `run_fb_arc.py`
- Reuse: `evaluate_arc_checkpoint.py`

**Interfaces:**
- Consumes: `TrialSpec`, one physical GPU ID, one prefix data root, and immutable contract paths.
- Produces: `run_trial(spec: TrialSpec) -> dict[str, object]`, `trial_complete(run_dir: Path, expected_fingerprint: str) -> bool`, `trial.json`, `metrics.json`, per-task vectors, runtime records, posterior manifest, and immutable command records.

- [ ] **Step 1: Write failing trial-state tests**

```python
def test_complete_trial_requires_matching_hashes_and_terminal_status(tmp_path: Path) -> None:
    write_json(tmp_path / "trial.json", {"status": "complete", "fingerprint": "abc", "artifacts": []})
    assert not trial_complete(tmp_path, expected_fingerprint="abc")


def test_training_resume_is_used_only_for_identical_fingerprint() -> None:
    assert resume_checkpoint(EARLIER, LATER_SAME_HP).name == "checkpoint.pt"
    assert resume_checkpoint(EARLIER, LATER_CHANGED_ESS) is None
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest -q tests/test_run_fb_model_first_trial.py`

Expected: FAIL because the trial runner does not exist.

- [ ] **Step 3: Implement `TrialSpec` and fail-closed artifact transitions**

```python
@dataclass(frozen=True)
class TrialSpec:
    candidate_id: str
    stage: str
    training: dict[str, object]
    updates: int
    augmentations: int
    posterior_draws: int
    posterior_scale: float
    seed: int
    gpu_id: int
```

Write status through `trial.json.tmp.<pid>` followed by `os.replace`, record every command as a JSON array before launch, and mark complete only after checkpoint, posterior, payload, aggregate metric, task-vector, and runtime hashes verify.

- [ ] **Step 4: Implement the subprocess sequence**

Launch `run_fb_arc.py candidate` with `train_split=full` and every serialized HP, then launch `evaluate_arc_checkpoint.py` on the requested prefix root with local batch size 96, compile enabled, fixed depth, fixed posterior seed, and the requested posterior draws and scale.
Run the independent aggregate rescorer and `score_task_vectors`, require exact aggregate agreement, and preserve stdout, stderr, exit code, GPU UUID, and peak memory.

- [ ] **Step 5: Run trial tests with mocked subprocesses**

Run: `pytest -q tests/test_run_fb_model_first_trial.py`

Expected: PASS for fresh runs, exact resumes, already-complete skips, and injected nonzero exit codes.

- [ ] **Step 6: Commit the trial runner**

```bash
git add run_fb_model_first_trial.py tests/test_run_fb_model_first_trial.py
git commit -m "feat: run atomic FB model-first trials"
```

---

### Task 7: Implement the multi-fidelity ten-GPU campaign orchestrator

**Files:**
- Create: `run_fb_model_first_hpo.py`
- Create: `tests/test_run_fb_model_first_hpo.py`

**Interfaces:**
- Consumes: the campaign contract, generated configuration ledger, prefix datasets, prefix baselines, and completed trial records.
- Produces: `run_stage(stage: StageSpec) -> StageResult`, `schedule_gpu_waves(trials: Sequence[TrialSpec], gpu_ids: Sequence[int]) -> list[list[TrialSpec]]`, `promote_stage(records: Sequence[Mapping[str, object]], keep: int) -> list[str]`, `campaign_state.json`, and stage selection ledgers.

- [ ] **Step 1: Write failing scheduling and resume tests**

```python
def test_scheduler_uses_each_gpu_at_most_once_per_wave() -> None:
    waves = schedule_gpu_waves(make_trials(24), tuple(range(10)))
    assert [len(wave) for wave in waves] == [10, 10, 4]
    assert all(len({trial.gpu_id for trial in wave}) == len(wave) for wave in waves)


def test_campaign_resume_skips_verified_trials_and_requeues_corrupt_trials(tmp_path: Path) -> None:
    state = make_campaign_state(tmp_path, complete=("a",), corrupt=("b",))
    pending = pending_trials(state)
    assert [trial.candidate_id for trial in pending] == ["b", "c"]
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest -q tests/test_run_fb_model_first_hpo.py`

Expected: FAIL because the campaign orchestrator does not exist.

- [ ] **Step 3: Implement stage definitions and deterministic ledgers**

Define coarse as 64 trials at 100 updates, 16 augmentations, and 10 draws; fine as at most 24 trials at 400 updates, 64 augmentations, and 25 draws; posterior refinement as at most 18 trials at 800 updates, 128 augmentations, and 25 draws; and finalist extension as the final Pareto set at 256 augmentations and 25 draws.
Write the full proposed ledger before launching a stage, and write promotion decisions with candidate metric vectors, bootstrap intervals, ranking reasons, and source artifact hashes.

- [ ] **Step 4: Implement dynamic one-GPU scheduling**

Maintain at most one subprocess per physical GPU, immediately reuse a GPU when its trial exits, and never wait for a full wave when shorter jobs finish early.
Use `CUDA_VISIBLE_DEVICES=<physical-id>` for each child and separate `TORCHINDUCTOR_CACHE_DIR` roots per GPU and model signature.
On one trial failure, preserve the failure record, retry once only for recognized transient CUDA or process-launch errors, and fail the stage for deterministic data, hash, or metric errors.

- [ ] **Step 5: Implement finalist and seed-stability gates**

Require positive 256-prefix point deltas for all three target metrics and absence of bootstrap evidence for material regression before the seed-1 run.
Train the selected HP afresh from the immutable checkpoint with seed 1, evaluate it at 128 augmentations, and mark the campaign unstable if any metric direction reverses.
Do not replace the finalist automatically after observing the seed-1 result.

- [ ] **Step 6: Run orchestrator tests**

Run: `pytest -q tests/test_run_fb_model_first_hpo.py tests/test_fb_model_first_search.py tests/test_run_fb_model_first_trial.py`

Expected: PASS for stage counts, dynamic scheduling, exact resume, promotion limits, retry policy, and final gating.

- [ ] **Step 7: Commit the orchestrator**

```bash
git add run_fb_model_first_hpo.py tests/test_run_fb_model_first_hpo.py
git commit -m "feat: orchestrate ten-GPU FB model-first HPO"
```

---

### Task 8: Add detached supervision, phase notifications, and final verification

**Files:**
- Create: `supervise_fb_model_first_hpo.py`
- Create: `tests/test_supervise_fb_model_first_hpo.py`
- Create: `verify_fb_model_first_result.py`
- Create: `tests/test_verify_fb_model_first_result.py`

**Interfaces:**
- Consumes: campaign state transitions and final rank payloads.
- Produces: `supervisor.status`, `supervisor.pid`, `supervisor.log`, append-only `slack_outbox.jsonl`, `final_result.json`, and `verification.json`.

- [ ] **Step 1: Write failing supervisor-state tests**

```python
def test_notification_outbox_emits_each_phase_once(tmp_path: Path) -> None:
    notify_phase(tmp_path, phase="coarse_complete", payload={"completed": 64})
    notify_phase(tmp_path, phase="coarse_complete", payload={"completed": 64})
    assert len((tmp_path / "slack_outbox.jsonl").read_text().splitlines()) == 1


def test_final_verifier_rejects_any_metric_disagreement(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="independent metric mismatch"):
        verify_final_result(make_mismatched_result(tmp_path))
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest -q tests/test_supervise_fb_model_first_hpo.py tests/test_verify_fb_model_first_result.py`

Expected: FAIL because the supervisor and verifier do not exist.

- [ ] **Step 3: Implement the detached supervisor**

Acquire an exclusive campaign lock, record the supervisor PID and process start time, run preflight, prefix construction, baseline prefixes, HPO stages, stability gate, and final evaluation in order, and atomically update `supervisor.status` at each transition.
Append one Slack-ready record only at preflight completion, coarse completion, fine completion, posterior-refinement completion, final-evaluation start, terminal failure, and terminal completion.
The supervisor advances from child process exit events and file state and contains no fixed high-frequency GPU polling loop.

- [ ] **Step 4: Implement ten-GPU final evaluation and independent verification**

Launch one `torchrun --standalone --nproc-per-node=10` evaluation on the immutable 1,000-augmentation root with global batch size 960, posterior candidates 25, depth 16, and the frozen seed-0 finalist.
Require ten runtime records, 120 tasks, 172 test pairs, zero missing pairs, unchanged model and posterior hashes, and exact official-versus-independent Pass@1, Pass@2, and Pass@100 equality.
Write the adoption decision against thresholds 5.000%, 7.500%, and 11.944% without modifying LaTeX.

- [ ] **Step 5: Run supervisor and verifier tests**

Run: `pytest -q tests/test_supervise_fb_model_first_hpo.py tests/test_verify_fb_model_first_result.py`

Expected: PASS for notification idempotence, stale-lock recovery, failure preservation, metric verification, and adoption gating.

- [ ] **Step 6: Commit supervision and verification**

```bash
git add supervise_fb_model_first_hpo.py verify_fb_model_first_result.py tests/test_supervise_fb_model_first_hpo.py tests/test_verify_fb_model_first_result.py
git commit -m "feat: supervise and verify FB model-first campaign"
```

---

### Task 9: Verify the complete implementation, run smoke tests, and launch

**Files:**
- Verify: all source and test files from Tasks 1 through 8.
- Create at runtime: `/data1/lsj9862/VR/experiments/arc_agi2_fb_model_first_hpo_20260905/preflight.json`
- Create at runtime: `/data1/lsj9862/VR/experiments/arc_agi2_fb_model_first_hpo_20260905/smoke/`
- Create at runtime: `/data1/lsj9862/VR/experiments/arc_agi2_fb_model_first_hpo_20260905/supervisor.status`

**Interfaces:**
- Consumes: the complete tested campaign implementation.
- Produces: a stable detached campaign with a verified PID, live status artifact, and first-stage ETA.

- [ ] **Step 1: Run the complete CPU test suite**

Run: `pytest -q`

Expected: all tests pass with no collection errors.

- [ ] **Step 2: Run source and artifact hygiene checks**

Run:

```bash
git diff --check
python -m py_compile fb_model_first_contract.py fb_model_first_search.py build_arc_evaluation_prefix.py fb_model_first_metrics.py run_fb_model_first_trial.py run_fb_model_first_hpo.py supervise_fb_model_first_hpo.py verify_fb_model_first_result.py
```

Expected: zero output and exit code 0.

- [ ] **Step 3: Run immutable preflight**

Run:

```bash
python supervise_fb_model_first_hpo.py --runtime-root /data1/lsj9862/VR/experiments/arc_agi2_fb_model_first_hpo_20260905 --preflight-only
```

Expected: checkpoint and data hashes match, physical GPUs 0 through 9 are available, and `preflight.json` records 1,280 groups, 120 tasks, and 172 pairs.

- [ ] **Step 4: Run the two-update and one-batch GPU smoke**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python supervise_fb_model_first_hpo.py --runtime-root /data1/lsj9862/VR/experiments/arc_agi2_fb_model_first_hpo_20260905 --smoke-only
```

Expected: finite FB loss, frozen puzzle-embedding guard, valid posterior manifest, one complete compiled evaluation batch, and peak memory below 45 GiB.

- [ ] **Step 5: Verify the launch revision is clean**

Run: `git status --short`

Expected: no source, cache, or runtime artifact remains untracked or modified because every intentional source and test change was committed in the preceding tasks.

- [ ] **Step 6: Launch the detached campaign**

Run:

```bash
setsid /data1/lsj9862/VR/venvs/trm-cu128-train-20260827/bin/python supervise_fb_model_first_hpo.py --runtime-root /data1/lsj9862/VR/experiments/arc_agi2_fb_model_first_hpo_20260905 </dev/null >/data1/lsj9862/VR/experiments/arc_agi2_fb_model_first_hpo_20260905/supervisor.launch.log 2>&1 &
```

Expected: `supervisor.pid` identifies the live process, `supervisor.status` advances to `baseline_prefixes` or `coarse`, and the launcher exits without retaining a controlling terminal.

- [ ] **Step 7: Post the smoke-complete Slack update**

Reply once to thread `1788606339.092529` with the verified source commit, preflight counts, smoke runtime and peak memory, detached PID, current phase, and recalibrated ETA.

---

### Task 10: Handle terminal results without evaluation-driven method changes

**Files:**
- Verify at runtime: `/data1/lsj9862/VR/experiments/arc_agi2_fb_model_first_hpo_20260905/final_result.json`
- Modify only after adoption: `workshops/neurips2026_raaai/latex/body/experiments.tex`
- Modify only after adoption: `workshops/neurips2026_raaai/latex/body/app_experimental_details.tex`

**Interfaces:**
- Consumes: a terminal verified campaign result.
- Produces: a final Slack conclusion and, only for an adopted result, corrected paper prose and regenerated `main.pdf`.

- [ ] **Step 1: Verify the terminal artifact independently**

Run:

```bash
python verify_fb_model_first_result.py --runtime-root /data1/lsj9862/VR/experiments/arc_agi2_fb_model_first_hpo_20260905
```

Expected: `verification.json` either records exact metric agreement and an explicit adoption decision or fails closed with a concrete reason.

- [ ] **Step 2: Post the final Slack thread update**

Report the conclusion, Pass@1/2/100, baseline deltas, bootstrap stability, seed-1 sanity result, source and checkpoint hashes, artifact paths, and whether paper adoption is authorized.

- [ ] **Step 3: Update the paper only after adoption**

Correct the inaccurate claim that evaluation-task demonstration groups were excluded from posterior training, explicitly disclose evaluation-tuned model selection, update only verified metric cells, rebuild `main.pdf`, and inspect the rendered table and experimental-details page.

- [ ] **Step 4: Preserve a negative result without retuning**

If any target metric misses its gate, leave the paper values unchanged, mark the campaign `not_adopted`, preserve every artifact, and report the failed metric rather than launching an evaluation-driven follow-up automatically.
