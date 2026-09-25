import numpy as np
import torch

from verl.trainer.ppo.core_algos import agg_loss, register_adv_est, register_policy_loss
from recipe.mmpo.maxrl_objective import compute_maxrl_outcome_advantage
from recipe.mmpo.passk_chen import compute_passk_chen_outcome_advantage


@register_adv_est("passk_chen")
def compute_passk_chen_advantage(*args, **kwargs):
    return compute_passk_chen_outcome_advantage(*args, **kwargs)


@register_adv_est("maxrl")
def compute_maxrl_advantage(*args, **kwargs):
    return compute_maxrl_outcome_advantage(*args, **kwargs)


@register_policy_loss("fb_rloo")
def compute_fb_rloo_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode=None,
    config=None,
    rollout_is_weights=None,
):
    del old_log_prob, loss_agg_mode
    if config is None:
        raise ValueError("FB-RLOO policy loss requires actor config")
    if rollout_is_weights is not None:
        raise ValueError("FB-RLOO policy loss does not accept rollout correction weights")
    loss = agg_loss(
        loss_mat=-log_prob * advantages.detach(),
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-sum-norm",
        **config.global_batch_info,
    )
    return loss, {"actor/fb_rloo_unclipped": 1.0}


@register_policy_loss("grpo_dmpo")
def compute_grpo_dmpo_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
    rollout_is_weights=None,
    dmpo_coefficients=None,
    dmpo_group_mse=None,
):
    if config is None or dmpo_coefficients is None or dmpo_group_mse is None:
        raise ValueError("GRPO+DMPO requires actor config, coefficients, and group MSE")
    from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

    pg_loss, metrics = compute_policy_loss_vanilla(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )
    surrogate = agg_loss(
        loss_mat=log_prob * dmpo_coefficients.unsqueeze(-1),
        loss_mask=response_mask,
        loss_agg_mode="seq-mean-token-mean",
        **config.global_batch_info,
    )
    metrics.update(
        {
            "actor/dmpo_loss": dmpo_group_mse.float().mean().detach().item(),
            "actor/dmpo_surrogate": surrogate.detach().item(),
        }
    )
    return pg_loss + surrogate, metrics

@register_adv_est("mmpo")
def compute_mmpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    unbiased: bool,
    T: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        token_level_rewards:
            Tensor of shape [batch_size, response_length].
            Each response has a binary outcome reward in {0, 1}.
        response_mask:
            Tensor of shape [batch_size, response_length].
        index:
            Array of shape [batch_size]. Responses with the same index
            correspond to rollouts from the same problem.
        unbiased:
            Whether to use the unbiased estimator.
        T:
            Truncation order. The unbiased estimator requires T <= G.

    Returns:
        advantages:
            Token-level advantages of shape
            [batch_size, response_length].
        returns:
            Same as advantages.
    """
    if unbiased:
        return _compute_unbiased_outcome_advantage(
            token_level_rewards, response_mask, index, T
        )
    return _compute_biased_outcome_advantage(
        token_level_rewards, response_mask, index, T
    )


def _compute_unbiased_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unbiased MMPO outcome advantage for T <= G.
    For rollout j of a problem with G rollouts, define
        M_{-j} = sum_{l != j} (1 - r_l),
    where M_{-j} is the number of failures among the other G - 1
    rollouts. The unbiased leave-one-out advantage is
        A_j = [sum_{k=1}^{T} k * C(M_{-j}, k-1) / C(G-1, k-1)] * (r_j - 1 + M_{-j} / (G-1)).
    """

    device = token_level_rewards.device

    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        scalar_advantages = torch.zeros_like(scores, dtype=torch.float32)

        for idx in np.unique(index):
            group_mask = torch.as_tensor(index == idx, device=device, dtype=torch.bool)
            group_scores = scores[group_mask].to(torch.float32)
            G = int(group_scores.numel())

            if G < 2:
                raise ValueError("The unbiased estimator requires G >= 2.")
            if T > G:
                raise ValueError("The unbiased estimator requires T <= G.")

            failures = 1.0 - group_scores
            total_failures = failures.sum()

            # M_{-j}: number of failures excluding rollout j.
            M_minus_j = total_failures - failures
            binomial_ratio = torch.ones_like(M_minus_j)
            unbiased_weight = torch.ones_like(M_minus_j)
            for k in range(2, T + 1):
                h = k - 1
                binomial_ratio = binomial_ratio * (M_minus_j - h + 1.0) / float(G - h)
                unbiased_weight = unbiased_weight + float(k) * binomial_ratio

            centered_reward = group_scores - 1.0 + M_minus_j / float(G - 1)

            group_advantages = unbiased_weight * centered_reward

            scalar_advantages[group_mask] = group_advantages.to(torch.float32)

        advantages = scalar_advantages.unsqueeze(-1) * response_mask

    return advantages, advantages


def _compute_biased_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    T: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Biased MMPO outcome advantage.
    For each question x, we have G rollouts sharing the same index.
    Let r_i in {0, 1} be the outcome reward of rollout i.
        q_hat_old(x) = mean_i r_i
        w_T(q_hat) = sum_{j=1}^{T} j * (1 - q_hat)^{j - 1}
        A_i = w_T(q_hat_old(x)) * (r_i - q_hat_old(x))
    """

    device = token_level_rewards.device

    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        scalar_advantages = torch.zeros_like(scores, dtype=torch.float32)

        for idx in np.unique(index):
            group_mask = torch.tensor(index == idx, device=device, dtype=torch.bool)
            group_scores = scores[group_mask]

            # q_hat_old(x)
            q_hat_old = group_scores.mean()

            # w_T(q_hat) = sum_{j=1}^{T} j * (1 - q_hat)^{j - 1}
            w_T = torch.zeros((), device=device, dtype=torch.float32)
            for j in range(1, T+1):
                w_T = w_T + j * torch.pow(1.0 - q_hat_old, j - 1)

            # A_i = w_T(q_hat_old(x)) * (r_i - q_hat_old(x))
            scalar_advantages[group_mask] = w_T * (group_scores - q_hat_old)

        advantages = scalar_advantages.unsqueeze(-1) * response_mask

    return advantages, advantages
