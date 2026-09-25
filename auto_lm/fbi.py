"""Autoregressive-LM FBI objective and lagged failure state."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, NamedTuple

import numpy as np
import torch

from fb import FB


STATE_VERSION = 1


def training_decoding_seeds(problem_ids, step, rollout_n, base_seed):
    return np.asarray([
        int.from_bytes(hashlib.blake2s(f"train:{base_seed}:{step}:{identity}:{candidate}".encode(), digest_size=4).digest(), "little") & 0x7FFFFFFF
        for identity in problem_ids for candidate in range(rollout_n)
    ], dtype=np.int64)


def validate_fb_training_contract(config) -> None:
    """Reject configurations that break one-draw finite-K training semantics."""

    objective = config.algorithm.adv_estimator
    if objective not in {"fb_rloo", "fb_grpo"}:
        return
    fb = config.algorithm.fb
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    if not fb.enabled:
        raise ValueError("an FB objective requires algorithm.fb.enabled=True")
    temperature = float(fb.get("weight_temperature", 1.0) if hasattr(fb, "get") else getattr(fb, "weight_temperature", 1.0))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("FB weight_temperature must be finite and positive")
    if fb.k_train != 8 or rollout.n != fb.k_train:
        raise ValueError("FB requires rollout.n=algorithm.fb.k_train=8")
    if config.data.train_batch_size != actor.ppo_mini_batch_size or actor.ppo_epochs != 1:
        raise ValueError("FB requires one optimizer transaction per rollout with ppo_epochs=1")
    if actor.optim.optimizer.lower() != "ivon":
        raise ValueError("FB requires the IVON optimizer")
    if config.algorithm.use_kl_in_reward or actor.use_kl_loss:
        raise ValueError("FB training does not permit KL reward shaping or actor KL loss")
    if getattr(actor, "entropy_coeff", 0) != 0:
        raise ValueError("FB training requires entropy_coeff=0")
    rollout_correction = (
        config.algorithm.get("rollout_correction")
        if hasattr(config.algorithm, "get")
        else getattr(config.algorithm, "rollout_correction", None)
    )
    if rollout_correction and any(
        rollout_correction.get(key) not in (None, False) for key in ("rollout_is", "rollout_rs", "bypass_mode")
    ):
        raise ValueError("FB training does not permit rollout correction")
    trainer = getattr(config, "trainer", None)
    if getattr(trainer, "critic_warmup", 0) != 0:
        raise ValueError("FB training requires critic_warmup=0")
    world_size = getattr(trainer, "nnodes", 1) * getattr(trainer, "n_gpus_per_node", 1)
    fsdp_size = getattr(getattr(actor, "fsdp_config", None), "fsdp_size", -1)
    if fsdp_size not in (-1, world_size):
        raise ValueError("FB training requires one fully sharded FSDP mesh without replication")
    loss_mode = actor.policy_loss.loss_mode
    if objective == "fb_rloo":
        if loss_mode != "fb_rloo":
            raise ValueError("FB-RLOO requires the direct fb_rloo policy loss")
        if actor.loss_scale_factor != config.data.max_response_length:
            raise ValueError("FB-RLOO requires a fixed max-response-length normalizer")
    elif loss_mode != "vanilla" or any(
        value != 0.2 for value in (actor.clip_ratio, actor.clip_ratio_low, actor.clip_ratio_high)
    ):
        raise ValueError("FB-GRPO requires vanilla PPO clipping at 0.2")


def validate_binary_rewards(rewards: torch.Tensor) -> torch.Tensor:
    rewards = torch.as_tensor(rewards)
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("binary rewards must be a nonempty vector")
    if not bool((torch.isfinite(rewards) & ((rewards == 0) | (rewards == 1))).all()):
        raise ValueError("binary rewards must contain only finite values in {0, 1}")
    return rewards.detach().to(torch.float32)


def _groups(group_ids: Sequence[object]) -> dict[object, list[int]]:
    result: dict[object, list[int]] = defaultdict(list)
    for position, group_id in enumerate(group_ids):
        if isinstance(group_id, np.generic):
            group_id = group_id.item()
        if group_id is None:
            raise ValueError("group IDs must be non-null")
        result[group_id].append(position)
    return dict(result)


def _problem_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("problem IDs must be nonempty strings")
    return value


def stable_problem_ids(data_sources: Sequence[object], source_indices: Sequence[object]) -> np.ndarray:
    if len(data_sources) != len(source_indices) or len(data_sources) == 0:
        raise ValueError("stable problem ID metadata must be nonempty and aligned")
    result = []
    for source, index in zip(data_sources, source_indices, strict=True):
        source = str(source).strip()
        index = str(index).strip()
        if not source or not index:
            raise ValueError("stable problem ID metadata must be nonempty")
        result.append(f"{source}:{index}")
    return np.asarray(result, dtype=object)


def fb_sequence_policy_loss(log_prob, weighted_advantage, response_mask, fixed_normalizer):
    if not isinstance(fixed_normalizer, (int, float)) or not math.isfinite(fixed_normalizer) or fixed_normalizer <= 0:
        raise ValueError("fixed_normalizer must be finite and positive")
    if log_prob.ndim != 2 or weighted_advantage.shape != log_prob.shape or response_mask.shape != log_prob.shape:
        raise ValueError("FB policy tensors must have identical two-dimensional shapes")
    per_sequence = -(log_prob * weighted_advantage.detach() * response_mask).sum(dim=-1) / fixed_normalizer
    return per_sequence.mean()


def rloo_advantages(rewards: torch.Tensor, group_ids: Sequence[object]) -> torch.Tensor:
    rewards = validate_binary_rewards(rewards)
    if len(group_ids) != rewards.numel():
        raise ValueError("group IDs and rewards must be aligned")
    result = torch.empty_like(rewards)
    for group_id, positions in _groups(group_ids).items():
        if len(positions) < 2:
            raise ValueError(f"group {group_id!r} must contain at least two responses")
        selector = torch.tensor(positions, device=rewards.device)
        values = rewards.index_select(0, selector)
        result.index_copy_(0, selector, values - (values.sum() - values) / (len(positions) - 1))
    return result.detach()


def grpo_advantages(rewards: torch.Tensor, group_ids: Sequence[object], epsilon: float = 1e-6) -> torch.Tensor:
    rewards = validate_binary_rewards(rewards)
    if len(group_ids) != rewards.numel():
        raise ValueError("group IDs and rewards must be aligned")
    result = torch.empty_like(rewards)
    for group_id, positions in _groups(group_ids).items():
        if len(positions) < 2:
            raise ValueError(f"group {group_id!r} must contain at least two responses")
        selector = torch.tensor(positions, device=rewards.device)
        values = rewards.index_select(0, selector)
        result.index_copy_(0, selector, (values - values.mean()) / (values.std() + epsilon))
    return result.detach()


class PendingFailureUpdate(NamedTuple):
    observed_failures: dict[str, float]
    raw_reward_mean: float


class FailureStateStore(FB):
    def __init__(self, k_train: int, ema_beta: float) -> None:
        self.k_train = self.validate_k(k_train)
        if not math.isfinite(ema_beta) or not 0 <= ema_beta < 1:
            raise ValueError("ema_beta must be finite and in [0, 1)")
        self.ema_beta = float(ema_beta)
        self._initial = self.initial_failure_probability(k_train)
        self._failures: dict[str, torch.Tensor] = {}

    @property
    def num_keys(self) -> int:
        return len(self._failures)

    def failure(self, problem_id: str) -> float:
        key = _problem_id(problem_id)
        return float(self._failures[key].item()) if key in self._failures else self._initial

    def weights(self, problem_ids: Sequence[str]) -> torch.Tensor:
        failures = torch.tensor([self.failure(key) for key in problem_ids], dtype=torch.float32)
        return self.finite_k_weight(failures, self.k_train)

    def stage(self, problem_ids: Sequence[str], group_ids: Sequence[object], rewards: torch.Tensor) -> PendingFailureUpdate:
        rewards = validate_binary_rewards(rewards)
        if len(problem_ids) != rewards.numel() or len(group_ids) != rewards.numel():
            raise ValueError("problem IDs, group IDs, and rewards must be aligned")
        keys = [_problem_id(value) for value in problem_ids]
        visits: dict[str, list[float]] = defaultdict(list)
        for group_id, positions in _groups(group_ids).items():
            group_keys = {keys[position] for position in positions}
            if len(positions) < 2 or len(group_keys) != 1:
                raise ValueError(f"group {group_id!r} must map at least two responses to one problem")
            selector = torch.tensor(positions, device=rewards.device)
            visits[next(iter(group_keys))].append(float(1.0 - rewards.index_select(0, selector).mean().item()))
        observed = {key: math.fsum(values) / len(values) for key, values in sorted(visits.items())}
        return PendingFailureUpdate(observed, float(rewards.mean().item()))

    def commit(self, pending: PendingFailureUpdate, optimizer_step_succeeded: bool) -> None:
        if not optimizer_step_succeeded or self.k_train == 1:
            return
        for key, observed in pending.observed_failures.items():
            if not math.isfinite(observed) or not 0 <= observed <= 1:
                raise ValueError("observed failure must be finite and in [0, 1]")
            value = self.ema_failure(self.failure(key), observed, self.ema_beta)
            self._failures[key] = torch.tensor(value, dtype=torch.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"version": STATE_VERSION, "k_train": self.k_train, "ema_beta": self.ema_beta, "failures": {key: value.clone() for key, value in sorted(self._failures.items())}}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != STATE_VERSION or state.get("k_train") != self.k_train or state.get("ema_beta") != self.ema_beta:
            raise ValueError("failure state checkpoint is incompatible")
        values = state.get("failures")
        if not isinstance(values, Mapping):
            raise ValueError("failure state values must be a mapping")
        restored = {}
        for key, value in values.items():
            key = _problem_id(key)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape
                or value.dtype != torch.float32
                or not bool(torch.isfinite(value))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError("failure state values must be finite scalar FP32 tensors in [0, 1]")
            restored[key] = value.detach().cpu().clone()
        self._failures = restored


def _weighted(base: torch.Tensor, problem_ids: Sequence[str], store: FailureStateStore) -> torch.Tensor:
    weights = store.weights(problem_ids).to(device=base.device, dtype=base.dtype)
    return (base * weights).detach()


def weighted_rloo_advantages(rewards, group_ids, problem_ids, store):
    rewards = validate_binary_rewards(rewards)
    pending = store.stage(problem_ids, group_ids, rewards)
    return _weighted(rloo_advantages(rewards, group_ids), problem_ids, store), pending


def weighted_grpo_advantages(rewards, group_ids, problem_ids, store):
    rewards = validate_binary_rewards(rewards)
    pending = store.stage(problem_ids, group_ids, rewards)
    return _weighted(grpo_advantages(rewards, group_ids), problem_ids, store), pending


class FBI(FB):
    """Driver-owned transactional FB state."""

    def __init__(self, objective: str, k_train: int, ema_beta: float, weight_mode: str = "lagged", weight_temperature: float = 1.0, freeze_failure_updates: bool = False) -> None:
        if objective not in {"fb_rloo", "fb_grpo"}:
            raise ValueError("FB objective must be fb_rloo or fb_grpo")
        self.objective = objective
        if weight_mode not in {"lagged", "uniform", "normalized"}:
            raise ValueError("FB weight mode must be lagged, uniform, or normalized")
        if not math.isfinite(weight_temperature) or weight_temperature <= 0:
            raise ValueError("FB weight temperature must be finite and positive")
        self.weight_mode = weight_mode
        self.weight_temperature = float(weight_temperature)
        self.freeze_failure_updates = bool(freeze_failure_updates)
        self.last_metrics = {}
        self.failure_state = FailureStateStore(k_train, ema_beta)
        self.pending: PendingFailureUpdate | None = None
        self.logical_perturbation_counter = 0

    def next_perturbation_id(self, global_step: int) -> str:
        self.logical_perturbation_counter += 1
        return f"fb-{global_step}-{self.logical_perturbation_counter}"

    def compute_advantages(self, rewards, response_mask, group_ids, problem_ids):
        if self.pending is not None:
            raise RuntimeError("previous FB failure update is still pending")
        self.pending = self.failure_state.stage(problem_ids, group_ids, rewards)
        function = rloo_advantages if self.objective == "fb_rloo" else grpo_advantages
        base = function(rewards, group_ids)
        intended = self.failure_state.weights(problem_ids).to(base)
        if self.weight_mode == "uniform":
            applied = torch.ones_like(intended)
        else:
            applied = intended.pow(self.weight_temperature)
            if self.weight_mode == "normalized":
                active = base != 0
                if bool(active.any()):
                    applied = applied / applied[active].mean()
        scalar = (base * applied).detach()
        groups = list(_groups(group_ids).values())
        group_weights = applied[[positions[0] for positions in groups]]
        active_groups = [positions for positions in groups if bool((base[positions] != 0).any())]
        active_weights = applied[[positions[0] for positions in active_groups]]
        if active_weights.numel():
            active_ess = active_weights.sum().square() / (active_weights.numel() * active_weights.square().sum())
            top_count = max(1, math.ceil(0.1 * active_weights.numel()))
            active_top_mass = active_weights.topk(top_count).values.sum() / active_weights.sum()
        else:
            active_ess = active_top_mass = torch.zeros((), device=base.device)
        pre_energy = base.square().mean()
        post_energy = scalar.square().mean()
        quantiles = torch.quantile(group_weights, torch.tensor([0., .25, .5, .75, 1.], device=base.device))
        self.last_metrics = {
            "weight_mode": self.weight_mode,
            "weight_temperature": self.weight_temperature,
            "group_count": len(groups),
            "problem_batch_sha256": hashlib.sha256("\n".join(sorted(problem_ids[p[0]] for p in groups)).encode()).hexdigest(),
            "revisit_fraction": sum(problem_ids[p[0]] in self.failure_state._failures for p in groups) / len(groups),
            "nonunit_active_groups": sum(bool((base[p] != 0).any()) and abs(float(applied[p[0]]) - 1.) > 1e-5 for p in groups),
            "active_group_fraction": len(active_groups) / len(groups),
            "active_weight_ess_fraction": float(active_ess),
            "active_weight_top10_mass": float(active_top_mass),
            "weighted_advantage_energy_ratio": float(post_energy / pre_energy) if bool(pre_energy > 0) else 0.0,
            "applied_weight_quantiles": quantiles.tolist(),
            "pre_weight_abs_mean": float(base.abs().mean()),
            "post_weight_abs_mean": float(scalar.abs().mean()),
        }
        return scalar.unsqueeze(-1) * response_mask

    def commit(self, optimizer_step_succeeded: bool) -> None:
        if self.pending is None:
            raise RuntimeError("FB failure update is not pending")
        self.failure_state.commit(self.pending, optimizer_step_succeeded and not self.freeze_failure_updates)
        self.pending = None

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError("cannot checkpoint with a pending FB update")
        return {
            "version": STATE_VERSION,
            "objective": self.objective,
            "failure_state": self.failure_state.state_dict(),
            "logical_perturbation_counter": self.logical_perturbation_counter,
        }

    def load_state_dict(self, state):
        if state.get("version") != STATE_VERSION or state.get("objective") != self.objective:
            raise ValueError("FB trainer checkpoint is incompatible")
        counter = state.get("logical_perturbation_counter")
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise ValueError("FB logical perturbation counter is invalid")
        self.failure_state.load_state_dict(state.get("failure_state"))
        self.logical_perturbation_counter = counter

    def save(self, checkpoint_root) -> None:
        checkpoint_root = Path(checkpoint_root)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".fb_failure_state.", dir=str(checkpoint_root))
        try:
            with os.fdopen(descriptor, "wb") as handle:
                torch.save(self.state_dict(), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, checkpoint_root / "fb_failure_state.pt")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, checkpoint_root) -> None:
        path = Path(checkpoint_root) / "fb_failure_state.pt"
        if not path.is_file():
            raise ValueError("FB checkpoint is missing fb_failure_state.pt")
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
