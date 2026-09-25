"""Shared finite-K Fenchel--Bregman failure weighting."""

from __future__ import annotations

import torch
from torch import Tensor


class FB:
    """Math shared by the recursive and autoregressive FBI experiments."""

    @staticmethod
    def validate_k(candidate_count: int) -> int:
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 1:
            raise ValueError("candidate_count must be an integer of at least one")
        return candidate_count

    @classmethod
    def initial_failure_probability(cls, candidate_count: int) -> float:
        candidate_count = cls.validate_k(candidate_count)
        return 1.0 if candidate_count == 1 else float(candidate_count ** (-1.0 / (candidate_count - 1)))

    @classmethod
    def finite_k_weight(cls, failure: Tensor, candidate_count: int) -> Tensor:
        candidate_count = cls.validate_k(candidate_count)
        if not isinstance(failure, Tensor):
            raise TypeError("failure must be a tensor")
        if not bool((torch.isfinite(failure) & (failure >= 0) & (failure <= 1)).all()):
            raise ValueError("failure must contain finite values in [0, 1]")
        return (
            torch.ones_like(failure)
            if candidate_count == 1
            else candidate_count * failure.pow(candidate_count - 1)
        ).detach()

    @staticmethod
    def ema_failure(previous: Tensor | float, observed: Tensor | float, beta: float) -> Tensor | float:
        """Advance a failure EMA, preserving tensor in-place update order."""
        if isinstance(previous, Tensor):
            return previous.mul_(beta).add_(observed, alpha=1 - beta)
        return beta * previous + (1 - beta) * observed


if __name__ == "__main__":
    assert torch.equal(FB.finite_k_weight(torch.tensor([0.5]), 2), torch.tensor([1.0]))
    assert FB.ema_failure(0.5, 1.0, 0.5) == 0.75
