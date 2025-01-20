################################### 导入依赖包 ###################################
try:
    from easypip import easyimport
except ModuleNotFoundError:
    from subprocess import run

    assert (
        run(["pip", "install", "easypip"]).returncode == 0
    ), "Could not install easypip"
    from easypip import easyimport

easyimport("swig")
easyimport("bbrl_utils>=0.5").setup()

import copy
import os

import gymnasium as gym
import torch
import torch.nn as nn
from bbrl.workspace import Workspace
from bbrl.agents import Agent, Agents, TemporalAgent, KWAgentWrapper
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import build_mlp, setup_optimizer, soft_update_params
from bbrl_utils.notebook import setup_tensorboard
from omegaconf import OmegaConf
from torch.distributions import (
    Normal,
    Independent,
    TransformedDistribution,
    TanhTransform,
)
import bbrl_gymnasium

########################################################################################

class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action


class Actor(Agent):
    """Computes probabilities over action"""

    def __init__(self, state_dim, hidden_layers, action_dim, min_std=1e-4):
        """Creates a new Squashed Gaussian actor

        :param state_dim: The dimension of the state space
        :param hidden_layers: Hidden layer sizes
        :param action_dim: The dimension of the action space
        :param min_std: The minimum standard deviation, defaults to 1e-4
        """
        super().__init__()
        self.min_std = min_std
        backbone_dim = [state_dim] + list(hidden_layers)
        self.layers = build_mlp(backbone_dim, activation=nn.ReLU())
        self.backbone = nn.Sequential(*self.layers)
        self.last_mean_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.last_std_layer = nn.Linear(hidden_layers[-1], action_dim)
        self.softplus = nn.Softplus()

        # cache_size avoids numerical infinites or NaNs when
        # computing log probabilities
        self.tanh_transform = TanhTransform(cache_size=1)

    def normal_dist(self, obs:torch.Tensor):
        """compute normal distribution given observation(s)"""

        backbone_output = self.backbone(obs)
        mean = self.last_mean_layer(backbone_output)
        std_out = self.last_std_layer(backbone_output)
        std = self.softplus(std_out) + self.min_std

        # Independent ensures that we have a multivariate
        # Gaussian with a diagonal covariance matrix (given as a vector `std`)
        return Independent(Normal(mean, std), 1)

    def forward(self, t: int, stochastic=True):
        """Computes the action a_t and its log-probability p(a_t| s_t)

        :param stochastic: True when sampling
        """

        # Computes probabilities over actions
        normal_dist = self.normal_dist(self.get(("env/env_obs", t)))
        action_dist = TransformedDistribution(normal_dist, [self.tanh_transform])
        if stochastic:
            # Uses the re-parametrization trick
            action = action_dist.rsample()
        else:
            # Directly uses the mode of the distribution
            action = self.tanh_transform(normal_dist.mode)

        log_prob = action_dist.log_prob(action)

        # This line allows to deepcopy the actor...
        self.tanh_transform._cached_x_y = [None, None]
        self.set(("action", t), action)
        self.set(("action_logprob", t), log_prob)


# 选择 action , 在上一个 class 中已实现，不需要了(?)
class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        # Selects the best actions according to the policy
        pass


# 没有 state 时，随机生成的 action，不需要更改
class SamplingActor(Agent):
    """Samples random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))
