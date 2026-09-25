"""Chen et al. analytical Pass@K response-relative advantages."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _comb_ratio(numerator_n: int, denominator_n: int, k: int) -> float:
    if k < 0 or denominator_n < k:
        return 0.0
    if numerator_n < k:
        return 0.0
    return math.comb(numerator_n, k) / math.comb(denominator_n, k)


def calculate_group_advantages(
    rewards: torch.Tensor,
    *,
    k: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return the analytical response-relative advantages for one rollout group."""

    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a nonempty one-dimensional tensor")
    group_size = int(rewards.numel())
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= group_size:
        raise ValueError(f"Pass@K requires 1 <= k <= group size, got k={k}, group_size={group_size}")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("Pass@K rewards must be finite and binary")
    if not bool(torch.logical_or(rewards == 0, rewards == 1).all()):
        raise ValueError("Pass@K rewards must be binary values in {0, 1}")

    success_count = int(rewards.sum().item())
    failure_count = group_size - success_count
    pass_at_k = 1.0 - _comb_ratio(failure_count, group_size, k)
    sigma = math.sqrt(pass_at_k * (1.0 - pass_at_k))

    if sigma == 0.0:
        return torch.zeros_like(rewards, dtype=torch.float32)

    positive_advantage = (1.0 - pass_at_k) / (sigma + epsilon)
    negative_success_probability = 1.0 - _comb_ratio(failure_count - 1, group_size - 1, k - 1)
    negative_advantage = (negative_success_probability - pass_at_k) / (sigma + epsilon)
    positive = torch.full_like(rewards, positive_advantage, dtype=torch.float32)
    negative = torch.full_like(rewards, negative_advantage, dtype=torch.float32)
    return torch.where(rewards == 1, positive, negative)


def _config_k(config: Any) -> int:
    if config is None:
        raise ValueError("Pass@K estimator requires algorithm.passk.K")
    passk_config = config.get("passk")
    if passk_config is None:
        raise ValueError("Pass@K estimator requires algorithm.passk.K")
    value = passk_config.get("K")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("algorithm.passk.K must be an integer")
    return value


def compute_passk_chen_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Any = None,
    *,
    K: int | None = None,
    epsilon: float = 1e-6,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the analytical Pass@K estimator used by Chen et al."""

    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("token_level_rewards and response_mask must have the same shape")
    scores = token_level_rewards.sum(dim=-1).to(torch.float32)
    if scores.numel() != len(index):
        raise ValueError("prompt index length must match the response batch")
    k = _config_k(config) if K is None else K
    scalar_advantages = torch.zeros_like(scores, dtype=torch.float32)

    with torch.no_grad():
        for group_id in np.unique(index):
            group_mask = torch.as_tensor(index == group_id, device=scores.device, dtype=torch.bool)
            scalar_advantages[group_mask] = calculate_group_advantages(
                scores[group_mask],
                k=k,
                epsilon=epsilon,
            )

    advantages = scalar_advantages.unsqueeze(-1) * response_mask
    return advantages, advantages
