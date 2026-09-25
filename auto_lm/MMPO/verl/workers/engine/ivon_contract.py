"""Fail-closed lifecycle helpers for one-draw IVON updates."""

from __future__ import annotations


def inject_ivon_covariance(optimizer, donor_state_dict, *, require_noop: bool = False) -> None:
    """Copy only IVON's diagonal Hessian from a compatible optimizer checkpoint."""
    import torch

    donor_groups = donor_state_dict.get("param_groups")
    if not isinstance(donor_groups, list) or len(donor_groups) != len(optimizer.param_groups):
        raise ValueError("IVON covariance donor has incompatible parameter groups")
    for index, (target, donor) in enumerate(zip(optimizer.param_groups, donor_groups, strict=True)):
        for key in ("ess", "weight_decay", "hess_init", "local_numel"):
            if target.get(key) != donor.get(key):
                raise ValueError(f"IVON covariance donor group {index} differs on {key}")
        source_hessian = donor.get("hess")
        target_hessian = target.get("hess")
        if not isinstance(source_hessian, torch.Tensor) or not isinstance(target_hessian, torch.Tensor):
            raise ValueError(f"IVON covariance donor group {index} is missing Hessian state")
        if source_hessian.shape != target_hessian.shape or source_hessian.dtype != target_hessian.dtype:
            raise ValueError(f"IVON covariance donor group {index} has incompatible Hessian state")
        source_hessian = source_hessian.to(target_hessian.device, non_blocking=False)
        if require_noop and not torch.equal(target_hessian, source_hessian):
            raise ValueError(f"IVON covariance self-injection is not a no-op for group {index}")
        target_hessian.copy_(source_hessian)
        if not torch.equal(target_hessian, source_hessian):
            raise RuntimeError(f"IVON covariance injection verification failed for group {index}")


def _require_perturbation_id(perturbation_id: str | None) -> str:
    if not isinstance(perturbation_id, str) or not perturbation_id:
        raise RuntimeError("FB backward requires a nonempty logical perturbation ID")
    return perturbation_id


def assert_ivon_perturbation(optimizer, expected_id: str | None) -> None:
    expected_id = _require_perturbation_id(expected_id)
    if not bool(getattr(optimizer, "_is_noised", False)):
        raise RuntimeError("FB backward requires an active IVON perturbation")
    actual_id = getattr(optimizer, "_logical_perturbation_id", None)
    if actual_id != expected_id:
        raise RuntimeError(f"IVON perturbation mismatch: expected {expected_id}, observed {actual_id}")


def sample_ivon_parameters(optimizer, perturbation_id: str | None, noise_seed: int | None = None) -> None:
    perturbation_id = _require_perturbation_id(perturbation_id)
    sample = getattr(optimizer, "_sample_params", None)
    if sample is None:
        raise RuntimeError("configured IVON optimizer does not expose _sample_params")
    saved_rng = None
    if noise_seed is not None:
        import torch.distributed as dist

        if isinstance(noise_seed, bool) or not isinstance(noise_seed, int) or noise_seed < 0:
            raise ValueError("evaluation noise seed must be a nonnegative integer")
        if optimizer._is_noised:
            raise RuntimeError("cannot reseed an active IVON perturbation")
        saved_rng = optimizer._generator.get_state()
        rank = dist.get_rank() if dist.is_initialized() else 0
        optimizer._generator.manual_seed(noise_seed + rank)
    try:
        sample(logical_perturbation_id=perturbation_id)
    except TypeError as exc:
        raise RuntimeError("installed IVON lacks logical perturbation ID support") from exc
    finally:
        if saved_rng is not None:
            optimizer._generator.set_state(saved_rng)
            optimizer._sync_checkpoint_metadata()
    assert_ivon_perturbation(optimizer, perturbation_id)


def restore_ivon_parameter_mean(optimizer, perturbation_id: str | None, *, train: bool) -> None:
    assert_ivon_perturbation(optimizer, perturbation_id)
    restore = getattr(optimizer, "_restore_param_average", None)
    if restore is None:
        raise RuntimeError("configured IVON optimizer does not expose _restore_param_average")
    restore(train=train)
    if bool(getattr(optimizer, "_is_noised", False)):
        raise RuntimeError("IVON failed to restore the posterior mean")
