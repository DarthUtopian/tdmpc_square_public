# TODO: modify TDMPC2 using offline RL policy update.
import numpy as np
import torch
import torch.nn.functional as F

from tdmpc_square.common import math
from tdmpc_square.common.scale import RunningScale
from tdmpc_square.common.world_model import WorldModel

from copy import deepcopy


class TDMPC2:
	"""
	Modified TD-MPC2 agent. Implements training + inference.
	Current implementation supports both state and pixel observations.
	Only support Single-task setting is supported.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		if torch.cuda.is_available():
			self.device = torch.device("cuda")
		else:
			self.device = torch.device("cpu")
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam(
			[
				{
					"params": self.model._encoder.parameters(),
					"lr": self.cfg.lr * self.cfg.enc_lr_scale,
				},
				{"params": self.model._dynamics.parameters()},
				{"params": self.model._reward.parameters()},
				{"params": self.model._Qs.parameters()},
				{
					"params": self.model._task_emb.parameters()
					if self.cfg.multitask
					else []
				},
			],
			lr=self.cfg.lr,
		)
		self.pi_optim = torch.optim.Adam(
			self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5
		)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.log_pi_scale = RunningScale(cfg)	
		self.cfg.iterations += 2 * int(
			cfg.action_dim >= 20
		)  # Heuristic for large action spaces
		self.discount = (
			torch.tensor(
				[self._get_discount(ep_len) for ep_len in cfg.episode_lengths],
				device="cuda",
			)
			if self.cfg.multitask
			else self._get_discount(cfg.episode_length)
		)

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
				episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
				float: Discount factor for the task.
		"""
		frac = episode_length / self.cfg.discount_denom
		return min(
			max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max
		)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
				fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
				fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None, use_pi=False):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
				obs (torch.Tensor): Observation from the environment.
				t0 (bool): Whether this is the first observation in the episode.
				eval_mode (bool): Whether to use the mean of the action distribution.
				task (int): Task index (only used for multi-task experiments).

		Returns:
				torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		z = self.model.encode(obs, task)
		if self.cfg.mpc and not use_pi:
			a, mu, std = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
		else:
			mu, pi, log_pi, log_std = self.model.pi(z, task)
			if eval_mode:
				a = mu[0]
			else:
				a = pi[0]
			mu, std = mu[0], log_std.exp()[0]
		return a.cpu(), mu.cpu(), std.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task, horizon, eval_mode=False):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G += discount * reward
			discount *= (
				self.discount[torch.tensor(task)]
				if self.cfg.multitask
				else self.discount
			)
		return G + discount * self.model.Q(
			z, self.model.pi(z, task)[1], task, return_type="avg"
		)

	@torch.no_grad()
	def _estimate_value_parallel(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[:, t], task), self.cfg)
			z = self.model.next(z, actions[:, t], task)
			G = G + discount * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
		return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

	@torch.no_grad()
	def plan(self, z, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
				z (torch.Tensor): Latent state from which to plan.
				t0 (bool): Whether this is the first observation in the episode.
				eval_mode (bool): Whether to use the mean of the action distribution.
				task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
				torch.Tensor: Action to take in the environment.
		"""
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(
				self.cfg.horizon,
				self.cfg.num_pi_trajs,
				self.cfg.action_dim,
				device=self.device,
			)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon - 1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std * torch.ones(
			self.cfg.horizon, self.cfg.action_dim, device=self.device
		)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(
			self.cfg.horizon,
			self.cfg.num_samples,
			self.cfg.action_dim,
			device=self.device,
		)
		if self.cfg.num_pi_trajs > 0:
			actions[:, : self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):
			# Sample actions
			actions[:, self.cfg.num_pi_trajs :] = (
				mean.unsqueeze(1)
				+ std.unsqueeze(1)
				* torch.randn(
					self.cfg.horizon,
					self.cfg.num_samples - self.cfg.num_pi_trajs,
					self.cfg.action_dim,
					device=std.device,
				)
			).clamp(-1, 1)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task, self.cfg.horizon).nan_to_num_(0)
			elite_idxs = torch.topk(
				value.squeeze(1), self.cfg.num_elites, dim=0
			).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0)[0]
			score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			score /= score.sum(0)
			mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (
				score.sum(0) + 1e-9
			)
			std = torch.sqrt(
				torch.sum(
					score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1
				)
				/ (score.sum(0) + 1e-9)
			).clamp_(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		score = score.squeeze(1).cpu().numpy()
		actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
		self._prev_mean = mean
		mu, std = actions[0], std[0]
		if not eval_mode:
			a = mu + std * torch.randn(self.cfg.action_dim, device=std.device)
		else:
			a = mu
		return a.clamp_(-1, 1), mu, std
	
	@torch.no_grad()
	def _plan(self, z, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.
		for parallelized version of plan.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		num_envs = z.size(0)
		# Sample policy trajectories
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(num_envs, self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[:,t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[:,t], task)
			pi_actions[:,-1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
		mean = torch.zeros(num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std*torch.ones(num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		if not t0:
			mean[:, :-1] = self._prev_mean[:, 1:]
		actions = torch.empty(num_envs, self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :, :self.cfg.num_pi_trajs] = pi_actions
	
		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(num_envs, self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, :, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value_parallel(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(2), self.cfg.num_elites, dim=1).indices
			elite_value = torch.gather(value, 1, elite_idxs.unsqueeze(2))
			elite_actions = torch.gather(actions, 2, elite_idxs.unsqueeze(1).unsqueeze(3).expand(-1, self.cfg.horizon, -1, self.cfg.action_dim))

			# Update parameters
			max_value = elite_value.max(1).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value.unsqueeze(1)))
			score = (score / score.sum(1, keepdim=True))
			mean = (score.unsqueeze(1) * elite_actions).sum(2) / (score.sum(1, keepdim=True) + 1e-9)
			std = ((score.unsqueeze(1) * (elite_actions - mean.unsqueeze(2)) ** 2).sum(2) / (score.sum(1, keepdim=True) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(2), dim=1)  # gumbel_softmax_sample is compatible with cuda graphs
		actions = elite_actions[torch.arange(num_envs), :, rand_idx]
		action, std = actions[:, 0], std[:, 0]
		if not eval_mode:
			action = action + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return action.clamp(-1, 1)

	def update_pi(self, zs, action, mu, std, task):
		"""
		Update policy using a sequence of latent states.

		Args:
				zs (torch.Tensor): Sequence of latent states.
				action (torch.Tensor): Sequence of actions.
				task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
				float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		_, pis, log_pis, _ = self.model.pi(zs, task)
		qs = self.model.Q(zs, pis, task, return_type="avg")
		self.scale.update(qs[0])
		qs = self.scale(qs)
			
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		if self.cfg.actor_mode=="sac":
			# TD-MPC2 baseline setting.
			pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
			prior_loss = torch.zeros_like(pi_loss) # Not used
			q_loss = pi_loss.detach().clone()

		elif self.cfg.actor_mode=="awac":
			# Loss for AWAC-MPC
			with torch.no_grad():
				vs = self.model.Q(zs, action, task, return_type="avg")
				vs = self.scale(vs)
			adv = (qs - vs).detach()
			weights = torch.clamp(torch.exp(adv / self.cfg.awac_lambda), self.cfg.exp_adv_min, self.cfg.exp_adv_max)
			log_pis_action = self.model.log_pi_action(zs, action, task)
			pi_loss = (( - weights * log_pis_action).mean(dim=(1, 2)) * rho).mean()
			q_loss = torch.zeros_like(pi_loss)
			prior_loss = torch.zeros_like(pi_loss)

		elif self.cfg.actor_mode=="residual":
			# Loss for TD-M(PC)^2
			action_dims = None if not self.cfg.multitask else self.model._action_masks.size(-1)
			std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
			eps = (pis - mu) / std
			log_pis_prior = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)
			log_pis_prior = torch.clamp(log_pis_prior, -50000, 0.0)
			self.log_pi_scale.update(log_pis_prior[0]) # Update scale

			if self.scale.value <= self.cfg.scale_threshold:
				log_pis_prior = torch.zeros_like(log_pis_prior)
			else:
				log_pis_prior = self.scale(log_pis_prior)

			q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
			prior_loss = - (log_pis_prior.mean(dim=-1) * rho).mean()
			pi_loss = q_loss + (self.cfg.prior_coef * self.cfg.action_dim / 61) * prior_loss

		elif self.cfg.actor_mode=="bc_sac": 
			# Vanilla BC-SAC loss for policy learning
			q_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1, 2)) * rho).mean()
			prior_loss = (((pis - action) ** 2).sum(dim=-1).mean(dim=1) * rho).mean()
			pi_loss = q_loss + self.cfg.prior_coef * prior_loss

		elif self.cfg.actor_mode=="bc":
			# Loss for BC-MPC baseline
			action_dims = None if not self.cfg.multitask else self.model._action_masks.size(-1)
			std = torch.max(std, self.cfg.min_std * torch.ones_like(std))
			eps = (pis - mu) / std
			log_pis_prior = math.gaussian_logprob(eps, std.log(), size=action_dims).mean(dim=-1)
			log_pis_prior = torch.clamp(log_pis_prior, -50000, 0.0)
			self.log_pi_scale.update(log_pis_prior[0])
			
			log_pis_prior = self.scale(log_pis_prior)
			pi_loss = - (log_pis_prior.mean(dim=-1) * rho).mean()
			prior_loss = pi_loss.detach().clone()
			q_loss = torch.zeros_like(pi_loss) # Not used

		else:
			raise NotImplementedError

		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(
			self.model._pi.parameters(), self.cfg.grad_clip_norm
		)
		
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item(), q_loss.item(), prior_loss.item()

	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
				next_z (torch.Tensor): Latent state at the following time step.
				reward (torch.Tensor): Reward at the current time step.
				task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
				torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z, task)[1]
		discount = (
			self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		)
		return reward + discount * self.model.Q(
			next_z, pi, task, return_type="min", target=True
		)

	def update(self, buffer):	
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
				buffer (common.buffer.Buffer): Replay buffer.

		Returns:
				dict: Dictionary of training statistics.
		"""
		if self.cfg.multitask and self.cfg.task in {"mt30","mt80"}:
			# offline training
			obs, action, reward, task = buffer.sample()
			mu = action.detach().clone()
			std = torch.full_like(action, self.cfg.max_std)
		else:
			# online training
			obs, action, mu, std, reward, task = buffer.sample() # mu and std are from Gaussian policy used for data collection	

		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)
			
		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(
			self.cfg.horizon + 1,
			self.cfg.batch_size,
			self.cfg.latent_dim,
			device=self.device,
		)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			z = self.model.next(z, action[t], task)
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
			zs[t + 1] = z


		#####################
		if self.cfg.record_q_scale:
			# in-distribution q value scale
			with torch.no_grad():
				qs = self.model.Q(zs[:-1], action, task, return_type="all")
				q_val_sacle = torch.cat([q.view(-1) for q in qs]).mean().item()
		#####################

		#####################
		if self.cfg.record_pi_q_scale:
			# out-of-distribution q value scale
			with torch.no_grad():
				pi = self.model.pi(zs[:-1], task)[1]
				qs = self.model.Q(zs[:-1], pi, task, return_type="all")
				q_val_sacle_ood = torch.cat([q.view(-1) for q in qs]).mean().item()
		#####################

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type="all")
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += (
				math.soft_ce(reward_preds[t], reward[t], self.cfg).mean()
				* self.cfg.rho**t
			)
			for q in range(self.cfg.num_q):
				value_loss += (
					math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean()
					* self.cfg.rho**t
				)
		consistency_loss *= 1 / self.cfg.horizon
		reward_loss *= 1 / self.cfg.horizon
		value_loss *= 1 / (self.cfg.horizon * self.cfg.num_q)

		total_loss = (
			self.cfg.consistency_coef * consistency_loss
			+ self.cfg.reward_coef * reward_loss
			+ self.cfg.value_coef * value_loss
		)
			
		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(
			self.model.parameters(), self.cfg.grad_clip_norm
		)
		self.optim.step()

		# Update policy
		pi_loss, pi_loss_q, pi_loss_prior  = self.update_pi(_zs.detach(), action.detach(), mu.detach(), std.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"pi_loss_q": pi_loss_q,
			"pi_loss_prior": pi_loss_prior,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
			"log_pi_scale": float(self.log_pi_scale.value),	
			"q_val_scale": q_val_sacle if self.cfg.record_q_scale else 0,
			"q_val_scale_ood": q_val_sacle_ood if self.cfg.record_pi_q_scale else 0,
		}
