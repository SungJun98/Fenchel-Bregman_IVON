"""Pure DMPO group objective and microbatch-safe gradient coefficients."""

from __future__ import annotations

import math

import numpy as np
import torch


def calculate_group_statistics(rewards, sequence_log_probs, *, temperature):
    if rewards.ndim != 1 or sequence_log_probs.ndim != 1 or rewards.shape != sequence_log_probs.shape:
        raise ValueError("rewards and sequence_log_probs must be matching nonempty vectors")
    if rewards.numel() == 0:
        raise ValueError("DMPO groups must be nonempty")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not torch.isfinite(rewards).all() or not torch.isfinite(sequence_log_probs).all():
        raise ValueError("DMPO rewards and log probabilities must be finite")
    target = torch.softmax(rewards / temperature, dim=0)
    policy = torch.softmax(sequence_log_probs, dim=0)
    delta = policy - target
    mse = delta.square().mean()
    coefficient = 2.0 / rewards.numel() * policy * (delta - torch.sum(policy * delta))
    return target, policy, mse, coefficient


def compute_dmpo_batch(
    token_level_rewards,
    old_log_probs,
    response_mask,
    index,
    *,
    matching_weight,
    temperature,
    group_size,
):
    if token_level_rewards.ndim != 2 or token_level_rewards.shape != old_log_probs.shape or old_log_probs.shape != response_mask.shape:
        raise ValueError("DMPO expects matching rank-two reward, log-probability, and mask tensors")
    if not math.isfinite(matching_weight) or matching_weight <= 0:
        raise ValueError("matching_weight must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size < 2:
        raise ValueError("group_size must be an integer of at least two")
    if index is None:
        raise ValueError("DMPO batch dimensions and prompt UIDs must match")
    index = np.asarray(index, dtype=object)
    if index.ndim != 1:
        raise ValueError("DMPO prompt UIDs must be one-dimensional")
    if len(index) != old_log_probs.shape[0] or old_log_probs.shape[0] == 0:
        raise ValueError("DMPO batch dimensions and prompt UIDs must match")
    if not torch.isfinite(token_level_rewards).all() or not torch.isfinite(old_log_probs).all() or not torch.isfinite(response_mask).all():
        raise ValueError("DMPO rewards, log probabilities, and masks must be finite")
    lengths = response_mask.sum(-1)
    if torch.any(lengths <= 0):
        raise ValueError("DMPO responses must not be empty")
    group_ids = np.unique(index)
    for group_id in group_ids:
        if int(np.count_nonzero(index == group_id)) != group_size:
            raise ValueError(f"DMPO requires exactly {group_size} responses per prompt")
    rewards = token_level_rewards.sum(-1).to(torch.float64)
    phi = (old_log_probs * response_mask).sum(-1).to(torch.float64) / lengths.to(torch.float64)
    coefficient = torch.zeros_like(phi)
    repeated_mse = torch.zeros_like(phi)
    response_scale = old_log_probs.shape[0] / len(group_ids)
    with torch.no_grad():
        for group_id in group_ids:
            group_mask = torch.as_tensor(index == group_id, device=phi.device, dtype=torch.bool)
            _, _, mse, derivative = calculate_group_statistics(
                rewards[group_mask], phi[group_mask], temperature=temperature
            )
            coefficient[group_mask] = matching_weight * response_scale * derivative
            repeated_mse[group_mask] = mse
    return coefficient.to(old_log_probs.dtype), repeated_mse.to(old_log_probs.dtype)
