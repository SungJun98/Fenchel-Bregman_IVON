# ARC-AGI-2 Leakage-Controlled Candidate Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, freeze, and evaluate a training2-only candidate selector whose ARC-AGI-2 evaluation predictions are generated before any evaluation solution is opened.

**Architecture:** A CPU package reconstructs posterior candidates, extracts generic demonstration-aware features, and fits a pairwise logistic ranker with task-grouped validation. A separate GPU executable reads only augmented inputs and challenge JSON, writes complete candidate payloads, and exits before an independently hashed scorer reads solutions.

**Tech Stack:** Python 3.12, NumPy 1.26, scikit-learn 1.4, PyTorch 2.8, pytest, JSON, pickle, and the frozen TinyRecursiveModels evaluation utilities.

**Spec:** `docs/superpowers/specs/2026-09-04-arc-agi2-leakage-controlled-selector-design.md`

## Global Constraints

The selector must use only ARC-AGI-2 `training2` labels for fitting and model selection.
The prediction-only process must not open an evaluation label array, evaluation solution JSON, or a previously scored evaluation result.
No source file may import the post-hoc `symbolic_consistency.py` or `symbolic_recovery.py` modules or contain an evaluation task identifier.
The primary ranking may reorder only the fixed posterior-vote top-100 set so Pass@100 membership is invariant.
No external pretrained model is permitted.
The existing ARC-AGI-1 artifacts and process definitions must not be modified.
Runtime artifacts must be written atomically under `/data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905`.
All evaluation features, selector parameters, blend parameters, and hashes must be frozen before target prediction scoring.
All source prose must keep one complete sentence per source line.

---

### Task 1: Protocol, provenance, and task boundaries

**Files:**
- Create: `experiments/arc_agi2_leakage_selector_20260905/protocol.py`
- Create: `experiments/arc_agi2_leakage_selector_20260905/make_protocol.py`
- Test: `experiments/arc_agi2_leakage_selector_20260905/tests/test_protocol.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str` and `atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None`.
- Produces: `hash_split(task_ids: Sequence[str], excluded: Collection[str], count: int, namespace: str) -> tuple[str, ...]`.
- Produces: `freeze_protocol(output: Path, artifacts: Mapping[str, Path], policy: Mapping[str, Any]) -> dict[str, Any]`.
- Produces: an immutable `protocol.json` containing all allowed input paths, source hashes, fit task IDs, heldout task IDs, feature schema, search grid, and success gates.

- [ ] **Step 1: Write failing protocol tests.**

```python
def test_hash_split_is_disjoint_deterministic_and_exact():
    ids = [f"task-{index}" for index in range(20)]
    first = hash_split(ids, {"task-0"}, 5, "selector-heldout-v1")
    second = hash_split(list(reversed(ids)), {"task-0"}, 5, "selector-heldout-v1")
    assert first == second
    assert len(first) == len(set(first)) == 5
    assert "task-0" not in first

def test_freeze_protocol_records_content_hashes(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    frozen = freeze_protocol(tmp_path / "protocol.json", {"source": source}, {"prefix": 100})
    assert frozen["artifacts"]["source"]["sha256"] == sha256_file(source)
```

- [ ] **Step 2: Run the focused test and confirm missing-module failure.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_protocol.py`
Expected: collection fails because `protocol.py` does not exist.

- [ ] **Step 3: Implement hashing, atomic writes, deterministic splitting, and protocol freezing.**

```python
def hash_split(task_ids, excluded, count, namespace):
    eligible = sorted(
        set(task_ids).difference(excluded),
        key=lambda task: hashlib.sha256(f"{namespace}:{task}".encode()).hexdigest(),
    )
    if len(eligible) < count:
        raise ValueError("not enough eligible tasks")
    return tuple(eligible[:count])
```

- [ ] **Step 4: Generate a protocol draft without reading evaluation solutions.**

Run: `env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 ionice -c3 python experiments/arc_agi2_leakage_selector_20260905/make_protocol.py --draft`
Expected: the output lists 128 fit tasks and 100 disjoint heldout tasks and omits every evaluation solution path.

- [ ] **Step 5: Run the focused tests and commit.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_protocol.py`
Expected: all tests pass.

Commit: `git add -f experiments/arc_agi2_leakage_selector_20260905 && git commit -m "test: lock ARC-AGI-2 selector protocol"`.

### Task 2: Candidate reconstruction and generic features

**Files:**
- Create: `experiments/arc_agi2_leakage_selector_20260905/candidate_data.py`
- Create: `experiments/arc_agi2_leakage_selector_20260905/features.py`
- Test: `experiments/arc_agi2_leakage_selector_20260905/tests/test_candidate_data.py`
- Test: `experiments/arc_agi2_leakage_selector_20260905/tests/test_features.py`

**Interfaces:**
- Produces: `CandidateRecord`, `PairRecord`, and `reconstruct_trace(trace: Mapping[str, np.ndarray], identifiers: Sequence[str], puzzles: Mapping[str, Any]) -> list[PairRecord]`.
- Produces: `feature_dict(pair: PairRecord, candidate: CandidateRecord) -> dict[str, float]`.
- Produces: `FEATURE_NAMES: tuple[str, ...]` with a stable order shared by training and prediction.
- Consumes: TinyRecursiveModels `inverse_aug`, `_crop`, and `grid_hash` functions through an explicit adapter path.

- [ ] **Step 1: Write failing reconstruction tests.**

```python
def test_reconstruction_deduplicates_inverse_augmented_outputs():
    pairs = reconstruct_trace(tiny_trace_with_two_equivalent_augmentations(), ["a", "a__identity"], tiny_puzzles())
    assert len(pairs) == 1
    assert pairs[0].row_count == 2
    assert pairs[0].candidates[0].vote_count == 2
    assert pairs[0].candidates[0].row_support == 2

def test_prediction_mode_pair_has_no_label_field():
    pair = PairRecord.for_prediction(task="a", pair_index=0, test_input=np.array([[1]]), demonstrations=[])
    assert not hasattr(pair, "label_hash")
```

- [ ] **Step 2: Run the reconstruction test and confirm missing-interface failures.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_candidate_data.py`
Expected: collection fails because the candidate interfaces do not exist.

- [ ] **Step 3: Implement deterministic candidate aggregation with grid-hash collision checks.**

```python
@dataclass(frozen=True)
class CandidateRecord:
    output_hash: str
    grid: np.ndarray
    vote_count: int
    q_sum: float
    q_max: float
    row_support: int
    first_seen: int
```

- [ ] **Step 4: Write failing feature tests for shape, color, component, symmetry, consensus, and task-ID invariance.**

```python
def test_feature_vector_is_independent_of_task_identifier():
    left = feature_dict(make_pair(task="task-a"), make_candidate())
    right = feature_dict(make_pair(task="task-b"), make_candidate())
    assert left == right
    assert tuple(sorted(left)) == FEATURE_NAMES
```

- [ ] **Step 5: Implement the frozen generic feature schema.**

The implementation must include normalized vote, support, Q, rank, output-shape, input-output color-set, histogram, foreground-box, component-density, symmetry, and changed-cell features.
The feature extractor must return finite values for empty foregrounds and reject invalid grids.

- [ ] **Step 6: Run focused tests and a 16-pair real-trace smoke.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_candidate_data.py experiments/arc_agi2_leakage_selector_20260905/tests/test_features.py`
Run: `env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 ionice -c3 python experiments/arc_agi2_leakage_selector_20260905/candidate_data.py --smoke-pairs 16`
Expected: both runs pass, every feature is finite, and no evaluation solution is opened.

- [ ] **Step 7: Commit the candidate and feature layer.**

Commit: `git add -f experiments/arc_agi2_leakage_selector_20260905 && git commit -m "feat: reconstruct ARC posterior candidates"`.

### Task 3: Pairwise selector, nested validation, and freezing

**Files:**
- Create: `experiments/arc_agi2_leakage_selector_20260905/selector.py`
- Create: `experiments/arc_agi2_leakage_selector_20260905/train_selector.py`
- Test: `experiments/arc_agi2_leakage_selector_20260905/tests/test_selector.py`

**Interfaces:**
- Produces: `make_pairwise_examples(pairs: Sequence[LabeledPairRecord], hard_negative_limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]`.
- Produces: `rank_top_prefix(pair: PairRecord, scores: Mapping[str, float], prefix_size: int, alpha: float) -> list[str]`.
- Produces: `nested_task_cv(pairs: Sequence[LabeledPairRecord], grid: Sequence[SelectorSpec], folds: int) -> SelectionResult`.
- Produces: `frozen_selector.joblib` and `selector_manifest.json` after the training gate passes.

- [ ] **Step 1: Write failing tests for hard negatives and prefix preservation.**

```python
def test_pairwise_examples_use_only_within_pair_hard_negatives():
    x, y, groups = make_pairwise_examples(two_labeled_pairs(), hard_negative_limit=2)
    assert x.shape[0] == 8
    assert set(y.tolist()) == {0, 1}
    assert set(groups.tolist()) == {"task-a", "task-b"}

def test_rank_top_prefix_preserves_top100_membership():
    baseline = [str(index) for index in range(150)]
    ranked = rank_top_prefix(make_pair(baseline), reverse_scores(baseline), 100, 1.0)
    assert set(ranked[:100]) == set(baseline[:100])
    assert ranked[100:] == baseline[100:]
```

- [ ] **Step 2: Run the focused test and confirm missing-interface failures.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_selector.py`
Expected: collection fails because `selector.py` does not exist.

- [ ] **Step 3: Implement balanced pairwise logistic training and deterministic task folds.**

```python
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=spec.c, penalty="l2", class_weight="balanced", max_iter=3000, random_state=260905),
)
```

The fixed search grid is `C in {0.01, 0.1, 1.0}` and `alpha in {0.125, 0.25, 0.5, 1.0, 2.0}`.
Each exact-versus-wrong feature difference must be added in both directions to prevent intercept shortcuts.

- [ ] **Step 4: Implement nested task-grouped selection and the preregistered gate.**

The selector must reject configurations that change top-100 membership, regress Pass@100, or have a negative worst-fold Pass@1 or Pass@2 delta.
Among survivors, it maximizes the minimum Pass@1/Pass@2 delta, then their sum, then minimizes harmed tasks and model complexity.

- [ ] **Step 5: Run unit tests and training-only cross-validation under CPU limits.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_selector.py`
Run: `env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 ionice -c3 python experiments/arc_agi2_leakage_selector_20260905/train_selector.py --protocol-draft /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/protocol_draft.json`
Expected: the command writes cross-validation predictions, fold metrics, the selected configuration, and either a fail-closed gate or frozen selector hashes.

- [ ] **Step 6: Independently replay every fold metric and commit.**

Run: `python experiments/arc_agi2_leakage_selector_20260905/train_selector.py --verify-only /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/cv_result.json`
Expected: `TRAINING_CV_VERIFIED`.

Commit: `git add -f experiments/arc_agi2_leakage_selector_20260905 && git commit -m "feat: train ARC candidate selector"`.

### Task 4: Label-free prediction-only GPU evaluator

**Files:**
- Create: `experiments/arc_agi2_leakage_selector_20260905/label_free_dataset.py`
- Create: `experiments/arc_agi2_leakage_selector_20260905/prediction_only.py`
- Test: `experiments/arc_agi2_leakage_selector_20260905/tests/test_prediction_only.py`

**Interfaces:**
- Produces: `LabelFreePuzzleDataset`, which opens `inputs`, `puzzle_identifiers`, and `puzzle_indices` arrays but never opens a labels array.
- Produces: `PredictionOnlyCollector.update_batch(batch, predictions) -> None` and `PredictionOnlyCollector.write_rank(path: Path, rank: int) -> Path`.
- Produces: one atomic `rank_N_candidates.pkl` and `rank_N_runtime.json` per rank without a final distributed scoring collective.
- Consumes: frozen TinyRecursiveModels model loading and `learned_posterior_batch` functions.

- [ ] **Step 1: Write failing tests that trap forbidden file access.**

```python
def test_label_free_dataset_never_opens_label_array(monkeypatch, tiny_dataset):
    real_load = np.load
    def guarded_load(path, *args, **kwargs):
        assert "labels" not in str(path)
        return real_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded_load)
    assert next(iter(LabelFreePuzzleDataset(tiny_dataset)))[1]["labels"].eq(-100).all()

def test_prediction_module_has_no_solution_path_constant():
    source = Path(prediction_only.__file__).read_text()
    assert "evaluation2_solutions" not in source
```

- [ ] **Step 2: Run the focused test and confirm missing-interface failures.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_prediction_only.py`
Expected: collection fails because the prediction-only modules do not exist.

- [ ] **Step 3: Implement the label-free loader with dummy ignore labels.**

The loader must reproduce the original world-size slicing exactly and must record the opened paths in each runtime manifest.

- [ ] **Step 4: Implement the complete-grid collector and remove final collectives.**

```python
for raw_prediction, q_value in zip(row_predictions, row_q, strict=True):
    grid = inverse(_crop(raw_prediction))
    output_hash = grid_hash(grid)
    self.grids.setdefault(output_hash, grid)
    self.statistics[task][input_hash][output_hash].observe(float(q_value))
```

Each rank must write its payload atomically and exit independently after CUDA synchronization.
The supervisor, not rank zero, checks that all rank files exist after `torchrun` exits.

- [ ] **Step 5: Add a one-batch label-invariance smoke.**

The smoke runs identical inputs with two different dummy label tensors and requires bitwise-equal predictions and Q logits before any long run is authorized.

- [ ] **Step 6: Run CPU tests, Python compilation, and a one-GPU one-batch smoke.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_prediction_only.py`
Run: `python -m py_compile experiments/arc_agi2_leakage_selector_20260905/*.py`
Run: `CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 experiments/arc_agi2_leakage_selector_20260905/prediction_only.py --mode evaluation --protocol /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/protocol.json --output-root /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/smoke_one_gpu --max-batches 1 --label-invariance-check`
Expected: the smoke writes one complete rank payload, reports label invariance, and opens no label or solution file.

- [ ] **Step 7: Commit the prediction-only evaluator.**

Commit: `git add -f experiments/arc_agi2_leakage_selector_20260905 && git commit -m "feat: add label-free ARC prediction runner"`.

### Task 5: Training-task holdout gate

**Files:**
- Create: `experiments/arc_agi2_leakage_selector_20260905/prepare_holdout.py`
- Create: `experiments/arc_agi2_leakage_selector_20260905/run_holdout_gate.py`
- Test: `experiments/arc_agi2_leakage_selector_20260905/tests/test_holdout_gate.py`

**Interfaces:**
- Produces: a reduced-augmentation dataset for the 100 protocol-heldout training2 tasks while preserving the frozen checkpoint identifier mapping.
- Produces: `heldout_predictions.lock.json`, `heldout_metrics.json`, and `holdout_gate.json`.
- Consumes: only the frozen selector manifest during prediction and training2 solutions during the separate scoring phase.

- [ ] **Step 1: Write failing tests for identifier preservation and score-after-lock ordering.**

```python
def test_holdout_identifiers_are_rows_from_frozen_mapping():
    built = build_holdout_fixture(challenges(), frozen_identifiers())
    assert set(built.identifier_ids).issubset(set(range(len(frozen_identifiers()))))

def test_scorer_rejects_unlocked_predictions(tmp_path):
    with pytest.raises(RuntimeError, match="prediction lock"):
        score_holdout(tmp_path / "predictions.json", solutions_path())
```

- [ ] **Step 2: Run the focused test and confirm missing-interface failures.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_holdout_gate.py`
Expected: collection fails because holdout modules do not exist.

- [ ] **Step 3: Implement deterministic reduced-augmentation holdout construction and provenance hashes.**

The augmentation budget is fixed in `protocol.json` before generation and cannot be changed after scoring.

- [ ] **Step 4: Implement prediction locking and the separate training2 scorer.**

The gate requires positive Pass@1 and Pass@2 deltas, an unchanged Pass@100 candidate set, and no catastrophic fold or task-family regression.

- [ ] **Step 5: Run the heldout GPU trace only after Tasks 1 through 4 pass.**

Run: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 torchrun --standalone --nproc-per-node=10 experiments/arc_agi2_leakage_selector_20260905/prediction_only.py --mode training-holdout --protocol /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/protocol.json --output-root /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/holdout_prediction`
Expected: all ten atomic rank payloads and a prediction lock are present without a metric file.

- [ ] **Step 6: Score once, enforce the gate, post a Slack checkpoint, and commit.**

Run: `env CUDA_VISIBLE_DEVICES='' python experiments/arc_agi2_leakage_selector_20260905/run_holdout_gate.py --protocol /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/protocol.json --prediction-lock /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/holdout_prediction/predictions.lock.json --solutions /home/lsj9862/new_topic/bayes_reasoning/references/TinyRecursiveModels/kaggle/combined/arc-agi_training2_solutions.json --output /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/holdout_gate.json`
Expected: `holdout_gate.json` records either `passed` or a fail-closed reason.

Commit: `git add -f experiments/arc_agi2_leakage_selector_20260905 && git commit -m "test: gate selector on heldout ARC training tasks"`.

### Task 6: Locked evaluation scorer and independent verifier

**Files:**
- Create: `experiments/arc_agi2_leakage_selector_20260905/score_locked.py`
- Create: `experiments/arc_agi2_leakage_selector_20260905/verify_result.py`
- Test: `experiments/arc_agi2_leakage_selector_20260905/tests/test_score_locked.py`

**Interfaces:**
- Produces: `rank_with_frozen_selector(payloads: Sequence[Payload], challenges: Mapping[str, Any], selector: FrozenSelector) -> dict[str, list[str]]`.
- Produces: `score_rankings(rankings: Mapping[str, Sequence[str]], solutions: Mapping[str, Any], budgets: Sequence[int]) -> dict[str, Any]`.
- Produces: `final_result.json` and `independent_verification.json`.

- [ ] **Step 1: Write failing tests for task weighting, top-100 preservation, and hash enforcement.**

```python
def test_task_weighting_averages_pairs_before_tasks():
    scores = score_rankings(two_task_fixture(), two_task_solutions(), [1])
    assert scores["1"]["score_percent"] == 75.0

def test_scorer_rejects_changed_selector(tmp_path):
    manifest = frozen_manifest(tmp_path)
    Path(manifest["selector"]["path"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_frozen_artifacts(manifest)
```

- [ ] **Step 2: Run the focused test and confirm missing-interface failures.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_score_locked.py`
Expected: collection fails because scorer modules do not exist.

- [ ] **Step 3: Implement target ranking without solutions and serialize its lock.**

The ranking stage validates every rank payload, applies the selector only inside the top-100 set, writes the top-100 grid hashes and grids, and atomically records its SHA-256 digest.

- [ ] **Step 4: Implement a separate solution-owning scorer and independent verifier.**

The verifier must recompute max-Q, posterior vote, row support, selector metrics, pair-hit counts, added and lost pairs, and source hashes without importing `selector.py`.

- [ ] **Step 5: Run focused and regression tests and commit.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_score_locked.py experiments/arc_agi2_leakage_selector_20260905/tests/test_selector.py`
Expected: all tests pass.

Commit: `git add -f experiments/arc_agi2_leakage_selector_20260905 && git commit -m "feat: add locked ARC selector scoring"`.

### Task 7: Campaign orchestration, full evaluation, and reporting

**Files:**
- Create: `experiments/arc_agi2_leakage_selector_20260905/run_campaign.py`
- Create: `experiments/arc_agi2_leakage_selector_20260905/REPORT.md`
- Test: `experiments/arc_agi2_leakage_selector_20260905/tests/test_campaign.py`

**Interfaces:**
- Produces: a fail-closed state machine with states `preflight`, `training_gate`, `holdout_gate`, `target_prediction`, `prediction_locked`, `scored`, `verified`, and `failed`.
- Produces: `state.json`, `campaign.status`, command manifests, immutable logs, and Slack-ready checkpoint text.
- Consumes: the passed holdout gate, ten idle GPUs, and all frozen artifacts.

- [ ] **Step 1: Write failing transition and restart-safety tests.**

```python
def test_target_prediction_requires_passed_holdout_gate():
    with pytest.raises(RuntimeError, match="holdout gate"):
        Campaign(failed_holdout()).transition("target_prediction")

def test_scoring_requires_prediction_lock():
    with pytest.raises(RuntimeError, match="prediction lock"):
        Campaign(valid_holdout()).transition("scored")
```

- [ ] **Step 2: Run the focused test and confirm missing-interface failures.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests/test_campaign.py`
Expected: collection fails because the campaign state machine does not exist.

- [ ] **Step 3: Implement idempotent orchestration and one-time target scoring.**

Every external command must be recorded before launch, and a restart must resume from the first incomplete verified state without deleting valid payloads.

- [ ] **Step 4: Run all CPU tests and a ten-GPU one-batch distributed smoke.**

Run: `pytest -q experiments/arc_agi2_leakage_selector_20260905/tests`
Run: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 torchrun --standalone --nproc-per-node=10 experiments/arc_agi2_leakage_selector_20260905/prediction_only.py --mode evaluation --protocol /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/protocol.json --output-root /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/smoke_ten_gpu --max-batches 1 --label-invariance-check`
Expected: ten rank files are complete and no final NCCL aggregation barrier is used.

- [ ] **Step 5: Launch the full target prediction only if every preflight and holdout gate passes.**

Run: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python experiments/arc_agi2_leakage_selector_20260905/run_campaign.py --protocol /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/protocol.json`
Expected: target prediction runs on GPUs 0 through 9, locks predictions, scores once, and reaches `verified` or preserves a precise `failed` state.

- [ ] **Step 6: Write the final report, verify it, and post the final Slack thread update.**

The report must separate clean protocol results from earlier post-hoc diagnostics and include Pass@1, Pass@2, Pass@100, paired task bootstrap intervals, component ablations, provenance hashes, elapsed time, and limitations.

- [ ] **Step 7: Run final verification and commit.**

Run: `python experiments/arc_agi2_leakage_selector_20260905/verify_result.py --result /data1/lsj9862/VR/experiments/arc_agi2_leakage_selector_20260905/final_result.json`
Expected: `INDEPENDENT_VERIFICATION_OK`.

Commit: `git add -f experiments/arc_agi2_leakage_selector_20260905 docs/superpowers/plans/2026-09-05-arc-agi2-leakage-controlled-selector.md && git commit -m "exp: verify leakage-controlled ARC-AGI-2 selector"`.
