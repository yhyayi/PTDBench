

"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""
__all__ = ['register_adv_est', 'get_adv_estimator_fn', 'AdvantageEstimator']
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional
import numpy as np
import torch
from omegaconf import DictConfig
import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig
PolicyLossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, Optional[DictConfig | ActorConfig], torch.Tensor | None], tuple[torch.Tensor, dict[str, Any]]]
POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}

def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func
    return decorator

def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(f'Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}')
    return POLICY_LOSS_REGISTRY[loss_name]

class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    REMAX = 'remax'
    RLOO = 'rloo'
    OPO = 'opo'
    GRPO_PASSK = 'grpo_passk'
    GPG = 'gpg'
    RLOO_VECTORIZED = 'rloo_vectorized'
    GRPO_VECTORIZED = 'grpo_vectorized'
    OPTIMAL_TOKEN_BASELINE = 'optimal_token_baseline'
    TIR_OPTIMAL_TOKEN_BASELINE = 'tir_optimal_token_baseline'
    GDPO = 'gdpo'
ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}

def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(f'Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}')
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn
    return decorator

def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f'Unknown advantage estimator simply: {name}')
    return ADV_ESTIMATOR_REGISTRY[name]

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass

def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == 'fixed':
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == 'adaptive':
        assert kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {kl_ctrl.horizon}'
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

@register_adv_est(AdvantageEstimator.GAE)
def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return (advantages, returns)

@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float=1e-06, norm_adv_by_std_in_grpo: bool=True, config: Optional[AlgoConfig]=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f'no score in prompt index: {idx}')
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
    return (scores, scores)

@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float=1e-06, norm_adv_by_std_in_grpo: bool=True, config: Optional[AlgoConfig]=None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return (advantages, advantages)

@register_adv_est(AdvantageEstimator.GDPO)
def compute_gdpo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float=1e-06, norm_adv_by_std_in_grpo: bool=True, config: Optional[AlgoConfig]=None, non_tensor_batch=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GDPO advantage: outcome reward normalized by group mean and std,
    multiplied by response mask.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) group indices
        epsilon: small constant for numerical stability
        norm_adv_by_std_in_grpo: whether to divide by std
        config: algorithm config
        non_tensor_batch: optional dict with extra fields

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0, device=scores.device)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return (advantages, advantages)

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)
def compute_reinforce_plus_plus_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float=1e-06, norm_adv_by_std_in_grpo: bool=True, config: Optional[AlgoConfig]=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute REINFORCE++ advantage: discounted cumulative returns with whitening.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) group indices (unused, for interface compatibility)
        epsilon: small constant for numerical stability
        norm_adv_by_std_in_grpo: whether to whiten advantages
        config: algorithm config

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    with torch.no_grad():
        gamma = config.gamma if config is not None else 1.0
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        gen_len = token_level_rewards.shape[-1]
        for t in reversed(range(gen_len)):
            running_return = token_level_rewards[:, t] + gamma * running_return * response_mask[:, t]
            returns[:, t] = running_return
        if norm_adv_by_std_in_grpo:
            advantages = verl_F.masked_whiten(returns, response_mask)
        else:
            advantages = returns
        advantages = advantages * response_mask
        return (advantages, returns)

@register_adv_est(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float=1e-06, norm_adv_by_std_in_grpo: bool=True, config: Optional[AlgoConfig]=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RLOO advantage: leave-one-out baseline.

    For each group, baseline = mean of all OTHER samples in the group.
    Advantage = score - leave-one-out baseline.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) group indices
        epsilon: small constant for numerical stability
        norm_adv_by_std_in_grpo: whether to normalize by std
        config: algorithm config

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, counts_g = group_mean_std(scores, g, eps=0.0, device=scores.device)

        # Leave-one-out: baseline = (sum_group - score_i) / (count - 1)
        group_sum = mean_g[g] * counts_g[g]
        baseline = (group_sum - scores) / (counts_g[g] - 1).clamp(min=1)

        if norm_adv_by_std_in_grpo:
            scalars = (scores - baseline) / (std_g[g] + epsilon)
        else:
            scalars = scores - baseline

        advantages = scalars.unsqueeze(-1) * response_mask
        return (advantages, advantages)

@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE)
def compute_reinforce_plus_plus_baseline_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float=1e-06, norm_adv_by_std_in_grpo: bool=True, config: Optional[AlgoConfig]=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute REINFORCE++ with group baseline advantage.

    Combines REINFORCE++ discounted returns with group-based baseline subtraction.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) group indices
        epsilon: small constant
        norm_adv_by_std_in_grpo: whether to whiten
        config: algorithm config

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    with torch.no_grad():
        gamma = config.gamma if config is not None else 1.0
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        gen_len = token_level_rewards.shape[-1]
        for t in reversed(range(gen_len)):
            running_return = token_level_rewards[:, t] + gamma * running_return * response_mask[:, t]
            returns[:, t] = running_return

        # Group baseline
        scores = returns.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=0.0, device=scores.device)

        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]

        advantages = scalars.unsqueeze(-1) * response_mask
        return (advantages, returns)

def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    This function combines the token-level scores (outcome rewards) with a KL
    divergence penalty between the current policy (old_log_prob) and the reference
    policy (ref_log_prob). The KL penalty uses the k1 estimator (log_prob - ref_log_prob).

    Args:
        token_level_scores (torch.Tensor): Token-level scores, shape (bs, response_length).
        old_log_prob (torch.Tensor): Log-probabilities under the current policy,
            shape (bs, response_length).
        ref_log_prob (torch.Tensor): Log-probabilities under the reference policy,
            shape (bs, response_length).
        kl_ratio (float): Coefficient for the KL penalty.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied,
            shape (bs, response_length).
    """
    kl_div = old_log_prob - ref_log_prob
    token_level_rewards = token_level_scores - kl_ratio * kl_div
    return token_level_rewards

@register_policy_loss('vanilla')
def compute_policy_loss_vanilla(old_log_prob: torch.Tensor, log_prob: torch.Tensor, advantages: torch.Tensor, response_mask: torch.Tensor, loss_agg_mode: str='token-mean', config: Optional[DictConfig | ActorConfig]=None, rollout_is_weights: torch.Tensor | None=None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the clipped policy objective and related metrics for PPO.

    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L844

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode. Defaults to "token-mean".
        config: `(DictConfig | ActorConfig | None)`:
            Configuration for the loss.
    """
    assert config is not None
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.max(pg_losses1, pg_losses2)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean(torch.lt(pg_losses2, pg_losses1).float() * (advantages < 0).float(), response_mask)
    pg_metrics = {'actor/pg_clipfrac': pg_clipfrac.detach().item(), 'actor/ppo_kl': ppo_kl.detach().item(), 'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item()}
    return (pg_loss, pg_metrics)

@register_policy_loss('gspo')
def compute_policy_loss_gspo(old_log_prob: torch.Tensor, log_prob: torch.Tensor, advantages: torch.Tensor, response_mask: torch.Tensor, loss_agg_mode: str='token-mean', config: Optional[DictConfig | ActorConfig]=None, rollout_is_weights: torch.Tensor | None=None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute GSPO policy loss.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode. Defaults to "token-mean".
        config: `(DictConfig | ActorConfig | None)`:
            Configuration for the loss.
    """
    assert config is not None
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    # GSPO uses per-sequence importance ratio
    seq_ratio = torch.exp((negative_approx_kl * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-08))
    seq_advantage = (advantages * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + 1e-08)
    pg_losses = -seq_advantage * torch.clamp(seq_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    pg_loss = torch.mean(pg_losses)
    pg_clipfrac = verl_F.masked_mean((torch.abs(seq_ratio.unsqueeze(-1) - 1) > clip_ratio).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    pg_metrics = {'actor/pg_clipfrac': pg_clipfrac.detach().item(), 'actor/ppo_kl': ppo_kl.detach().item(), 'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item()}
    return (pg_loss, pg_metrics)

@register_policy_loss('gpg')
def compute_policy_loss_gpg(old_log_prob: torch.Tensor, log_prob: torch.Tensor, advantages: torch.Tensor, response_mask: torch.Tensor, loss_agg_mode: str='token-mean', config: Optional[DictConfig | ActorConfig]=None, rollout_is_weights: torch.Tensor | None=None) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute GPG policy loss.

    Adapted from https://arxiv.org/abs/2502.02657

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode. Defaults to "token-mean".
        config: `(DictConfig | ActorConfig | None)`:
            Configuration for the loss.
    """
    assert config is not None
    clip_ratio = config.clip_ratio
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    ratio = torch.exp(negative_approx_kl)
    pg_losses = -advantages * ratio
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info)
    pg_clipfrac = torch.tensor(0.0, device=pg_loss.device)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    pg_metrics = {'actor/pg_clipfrac': pg_clipfrac.detach().item(), 'actor/ppo_kl': ppo_kl.detach().item(), 'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item()}
    return (pg_loss, pg_metrics)

@register_policy_loss('gmpo')
def compute_policy_loss_gmpo(old_log_prob: torch.Tensor, log_prob: torch.Tensor, advantages: torch.Tensor, response_mask: torch.Tensor, loss_agg_mode: str='token-mean', config: Optional[DictConfig | ActorConfig]=None, rollout_is_weights: torch.Tensor | None=None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-08))
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-08)
    pg_losses = -advantage * ratio
    if rollout_is_weights is not None:
        seq_is_weights = torch.exp((torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-08))
        pg_losses = pg_losses * seq_is_weights
    pg_loss = torch.mean(pg_losses)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {'actor/pg_clipfrac': pg_clipfrac.detach().item(), 'actor/ppo_kl': ppo_kl.detach().item(), 'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item()}
    return (pg_loss, pg_metrics)

@register_policy_loss('cispo')
def compute_policy_loss_cispo(old_log_prob: torch.Tensor, log_prob: torch.Tensor, advantages: torch.Tensor, response_mask: torch.Tensor, loss_agg_mode: str='token-mean', config: Optional[DictConfig | ActorConfig]=None, rollout_is_weights: torch.Tensor | None=None) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for CISPO.

    See https://arxiv.org/pdf/2506.13585 for more details.
    """
    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clipped_ratio_sg = clipped_ratio.detach()
    pg_losses = -clipped_ratio_sg * advantages * log_prob
    pg_clipfrac = verl_F.masked_mean((ratio != clipped_ratio).float(), response_mask)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)
    pg_metrics = {'actor/pg_clipfrac': pg_clipfrac.detach().item(), 'actor/ppo_kl': ppo_kl.detach().item(), 'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item()}
    return (pg_loss, pg_metrics)

def compute_entropy_loss(logits, response_mask, loss_agg_mode: str='token-mean'):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    token_entropy = verl_F.entropy_from_logits(logits)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss

def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str='token-mean', dp_size: int=1, batch_num_tokens: Optional[int]=None, global_batch_size: Optional[int]=None, loss_scale_factor: Optional[int]=None):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        dp_size (int, optional):
            Data parallel size, forwarded to `agg_loss` for global-batch normalization. Defaults to 1.
        batch_num_tokens (Optional[int], optional):
            Number of valid tokens in the global batch, forwarded to `agg_loss`. Defaults to None
            (normalize by the local micro-batch token count).
        global_batch_size (Optional[int], optional):
            Global batch size, forwarded to `agg_loss` for the seq-mean modes. Defaults to None.
        loss_scale_factor (Optional[int], optional):
            Scale factor for the "seq-mean-token-sum-norm" mode, forwarded to `agg_loss`. Defaults to None.

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, dp_size=dp_size, batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size, loss_scale_factor=loss_scale_factor)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return (vf_loss, vf_clipfrac)

def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Uses straight-through estimator on ref_logprob for unbiased KL gradient estimation
    (k2 / mse uses ref_logprob.detach() so gradient only flows through logprob).
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob (torch.FloatTensor): Log-probabilities under the current policy.
        ref_logprob (torch.FloatTensor): Log-probabilities under the reference policy.
        kl_penalty (str): Type of KL penalty to apply:
            - 'kl'/'k1': logprob - ref_logprob (reverse KL estimator)
            - 'abs': |logprob - ref_logprob|
            - 'mse'/'k2': 0.5 * (logprob - ref_logprob)^2 (with straight-through)
            - 'low_var_kl'/'k3': low-variance KL estimator
            - 'full': full KL divergence (not implemented)

    Returns:
        torch.FloatTensor: KL divergence estimate, same shape as inputs.
    """
    if kl_penalty in ('kl', 'k1'):
        return logprob - ref_logprob
    if kl_penalty == 'abs':
        return (logprob - ref_logprob).abs()
    if kl_penalty in ('mse', 'k2'):
        # Straight-through estimator: detach ref_logprob for unbiased gradient
        return 0.5 * (logprob - ref_logprob.detach()).square()
    if kl_penalty in ('low_var_kl', 'k3'):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    if kl_penalty == 'full':
        raise NotImplementedError
    raise NotImplementedError

def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ('kl', 'k1'):
        return logprob - ref_logprob
    if kl_penalty == 'abs':
        return (logprob - ref_logprob).abs()
    if kl_penalty in ('mse', 'k2'):
        return 0.5 * (logprob - ref_logprob).square()
    if kl_penalty in ('low_var_kl', 'k3'):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)
    if kl_penalty == 'full':
        raise NotImplementedError
    raise NotImplementedError

def compute_pf_ppo_reweight_data(data, reweight_method: str='pow', weight_pow: float=2.0):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == 'pow':
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == 'max_min':
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == 'max_random':
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f'Unsupported reweight_method: {reweight_method}')
        return weights
    scores = data.batch['token_level_scores'].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-08, min=1e-08)
    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)
    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}
    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]
    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list):
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from verl import DataProto
    resampled_data = DataProto(
        batch=resampled_batch,
        non_tensor_batch=resampled_non_tensor_batch,
        meta_info=resampled_meta_info
    )
    return resampled_data

def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str='token-mean', dp_size: int=1, batch_num_tokens: Optional[int]=None, global_batch_size: Optional[int]=None, loss_scale_factor: Optional[int]=None):
    """Aggregate loss matrix according to the specified mode.

    Args:
        loss_mat (torch.Tensor): Loss matrix, shape (bs, response_length).
        loss_mask (torch.Tensor): Mask for valid tokens, shape (bs, response_length).
        loss_agg_mode (str): Aggregation mode. One of:
            - 'token-mean': mean over all valid tokens.
            - 'token-sum': sum over all valid tokens.
            - 'seq-mean-token-sum': mean over sequences of per-sequence token sums.
            - 'seq-mean-token-mean': mean over sequences of per-sequence token means.
        dp_size (int): Data parallel size for global batch normalization.
        batch_num_tokens (Optional[int]): Total valid tokens in the global batch.
        global_batch_size (Optional[int]): Total batch size for seq-mean modes.
        loss_scale_factor (Optional[int]): Scale factor for seq-mean-token-sum-norm mode.

    Returns:
        torch.Tensor: Aggregated loss scalar.
    """
    if loss_agg_mode == 'token-mean':
        if batch_num_tokens is not None:
            loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
        else:
            loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == 'token-sum':
        loss = verl_F.masked_sum(loss_mat, loss_mask)
    elif loss_agg_mode == 'seq-mean-token-sum':
        seq_losses = (loss_mat * loss_mask).sum(dim=-1)
        if global_batch_size is not None:
            loss = seq_losses.sum() / global_batch_size * dp_size
        else:
            loss = seq_losses.mean()
    elif loss_agg_mode == 'seq-mean-token-mean':
        seq_mask_sum = loss_mask.sum(dim=-1).clamp(min=1)
        seq_losses = (loss_mat * loss_mask).sum(dim=-1) / seq_mask_sum
        if global_batch_size is not None:
            loss = seq_losses.sum() / global_batch_size * dp_size
        else:
            loss = seq_losses.mean()
    elif loss_agg_mode == 'seq-mean-token-sum-norm':
        seq_losses = (loss_mat * loss_mask).sum(dim=-1)
        if loss_scale_factor is not None:
            seq_losses = seq_losses / loss_scale_factor
        if global_batch_size is not None:
            loss = seq_losses.sum() / global_batch_size * dp_size
        else:
            loss = seq_losses.mean()
    else:
        raise ValueError(f'Unsupported loss_agg_mode: {loss_agg_mode}')
    return loss
