from operator import is_
from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np
from gym.spaces import Box
from torch.distributions.normal import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from dynamic_input_models.dynamic_input_handler import DynamicInputModel

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []

    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]

    return nn.Sequential(*layers)

class ActorCritic(nn.Module):

    def __init__(self, action_space, proprioceptive_state_dim=6, exteroceptive_state_dim=8, 
    rnn_hidden_dim=64, hidden_sizes_ac=(256, 256), hidden_sizes_v=(16, 16), 
    activation=nn.ReLU, output_activation=nn.Tanh, output_activation_v= nn.Identity, device="mps", mode='BiGRU', drop_p=0):
        super().__init__()

        self.device = device
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
        
        obs_dim = (rnn_hidden_dim + proprioceptive_state_dim)

        self.dynamic_input_model = DynamicInputModel(input_dim=exteroceptive_state_dim, hidden_dim=rnn_hidden_dim, device=device, mode=mode)

        self.pi = GaussianActor(obs_dim, action_space.shape[0], hidden_sizes_ac, activation, output_activation, device=device)

        # build value function
        self.v = Critic(obs_dim, hidden_sizes_v, activation, output_activation_v, device=device)

    def get_intermediate_representation(self, proprioceptive_observation, exteroceptive_observation, is_batch=True):
        if is_batch:
                proprioceptive_observation = np.array(proprioceptive_observation)
                proprioceptive_observation = torch.tensor(proprioceptive_observation).to(self.device, dtype=torch.float32)
        else:
            # convert a list of numpy arrays to a tensor
            proprioceptive_observation = torch.stack([torch.from_numpy(proprioceptive_observation)]).to(self.device, dtype=torch.float32)

        obs = self.dynamic_input_model(exteroceptive_observation)
        if is_batch:
            obs = torch.cat((obs, proprioceptive_observation), 1)
        else:
            obs = torch.cat((obs, proprioceptive_observation), 1)
        
        return obs

    def step(self, proprioceptive_observation, exteroceptive_observation, std_factor=1, is_batch=True):
        with torch.no_grad():
            obs = self.get_intermediate_representation(proprioceptive_observation, exteroceptive_observation, is_batch)
            pi_dis = self.pi._distribution(obs, std_factor)
            a = pi_dis.sample()
            logp_a = self.pi._log_prob_from_distribution(pi_dis, a)
            v = self.v(obs)
        a = a.squeeze()
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, proprioceptive_observation, exteroceptive_observation, std_factor=1, is_batch=True):
        return self.step(proprioceptive_observation, exteroceptive_observation, std_factor, is_batch)[0]


class GaussianActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation, device="mps"):
        super().__init__()

        self.device = device
        self.net_out=mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation)

        log_std = -1 * np.ones(act_dim, dtype=np.float32)

        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std, device=self.device))
        self.net_out=self.net_out.to(self.device)

    def _distribution(self, obs, std_factor=1):

        obs = obs.to(self.device)
        net_out = self.net_out(obs)
        
        mu = net_out 
        std = torch.exp(self.log_std)
        std = std_factor * std
        
        return Normal(mu, std)
        
    def _log_prob_from_distribution(self, pi, act):

        act = act.to(self.device)

        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None, std_factor=1):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, std_factor)
        logp_a = None

        if act is not None:   
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, logp_a

class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, output_activation, device="mps"):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation, output_activation)
        self.v_net = self.v_net.to(device)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)