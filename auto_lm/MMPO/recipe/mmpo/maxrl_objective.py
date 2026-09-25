"""MaxRL outcome advantages from Tajwar et al. (2026), Equation 10."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def calculate_group_advantages(rewards: torch.Tensor, *, epsilon: float = 1e-6) -> torch.Tensor:
    """Return the control-variate MaxRL advantages for one binary rollout group.

    For N rollouts with K successes, this is ``r_i / (K / N) - 1``.
    The epsilon only defines the all-failure group as a zero update.
    """

    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("MaxRL rewards must be a nonempty one-dimensional tensor")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("MaxRL epsilon must be finite and positive")
    if not torch.isfinite(rewards).all().item() or not torch.logical_or(rewards == 0, rewards == 1).all().item():
        raise ValueError("MaxRL rewards must be finite binary values in {0, 1}")

    success_rate = rewards.to(torch.float32).mean()
    return (rewards.to(torch.float32) - success_rate) / (success_rate + float(epsilon))


def compute_maxrl_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    *,
    epsilon: float = 1e-6,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute MaxRL advantages, targeting the order-N truncated ML objective."""

    if token_level_rewards.ndim != 2 or token_level_rewards.shape != response_mask.shape:
        raise ValueError("MaxRL requires matching rank-two token rewards and response mask")
    if not torch.isfinite(token_level_rewards).all().item() or not torch.isfinite(response_mask).all().item():
        raise ValueError("MaxRL rewards and response mask must be finite")
    if not torch.any(response_mask != 0, dim=-1).all().item():
        raise ValueError("MaxRL does not accept empty responses")
    group_ids = np.asarray(index)
    if group_ids.ndim != 1 or len(group_ids) != token_level_rewards.shape[0]:
        raise ValueError("MaxRL prompt index must be one-dimensional and match the response batch")

    scores = token_level_rewards.sum(dim=-1).to(torch.float32)
    scalar_advantages = torch.zeros_like(scores)
    with torch.no_grad():
        for group_id in np.unique(group_ids):
            group_mask = torch.as_tensor(group_ids == group_id, device=scores.device, dtype=torch.bool)
            scalar_advantages[group_mask] = calculate_group_advantages(scores[group_mask], epsilon=epsilon)

    advantages = scalar_advantages.unsqueeze(-1) * response_mask
    return advantages, advantages
