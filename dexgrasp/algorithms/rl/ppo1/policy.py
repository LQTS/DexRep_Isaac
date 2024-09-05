import os

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from dexgrasp.algorithms.rl.backbones.pre_model import Encoder_DexRep, Encoder_GeoDex, Encoder_GeoDex_cold, PointNetfeatTwoStream


class ActorCriticDexRep(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticDexRep, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # create BN
        self.bn_type = encoder_cfg["bn_type"]
        if encoder_cfg["bn_type"] == "part":
            self.bn_pnl = nn.BatchNorm1d(env_cfg['obs_dim']['dexrep_pnl'])
        elif encoder_cfg["bn_type"] == "full":
            self.bn_pnl = nn.BatchNorm1d(sum(self.obs_dim[1:]))
        elif encoder_cfg["bn_type"] == "null":
            self.bn_pnl = None
        else:
            raise NotImplementedError(f"bn_type not impleted")
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]
        self.dexrep_sensor_enc = nn.Linear(env_cfg['obs_dim']['dexrep_sensor'], emb_dim)
        self.dexrep_pointL_enc = nn.Linear(env_cfg['obs_dim']['dexrep_pnl'], emb_dim)
        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(len(self.obs_dim)*emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        # self.extractors_vf = nn.ModuleList(LinearEncoder(self.obs_dim[i], hidden_size, obs_emb) for i in range(self.obs_dim.__len__()))
        critic_layers = []

        critic_layers.append(nn.Linear(len(self.obs_dim)*emb_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.obs_enc)
        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

        # init link FC layers
        torch.nn.init.orthogonal_(self.state_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.dexrep_sensor_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.dexrep_pointL_enc.weight, gain=np.sqrt(2))

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        if self.bn_type == "part":
            self.bn_pnl.eval()
            state, dexrep_sensor, dexrep_pnl_raw = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_norm = self.bn_pnl(dexrep_pnl_raw)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl_norm)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "full":
            self.bn_pnl.eval()
            observations[:, self.obs_dim[0]:] = self.bn_pnl(observations[:, self.obs_dim[0]:])
            state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "null":
            state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        else:
            raise NotImplementedError(f"bn_type not impleted")

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(),\
               observations[:, state.shape[1]:].detach()

    @torch.no_grad()
    def act_inference(self, observations):
        if self.bn_type == "part":
            self.bn_pnl.eval()
            state, dexrep_sensor, dexrep_pnl_raw = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_norm = self.bn_pnl(dexrep_pnl_raw)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl_norm)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "full":
            self.bn_pnl.eval()
            observations[:, self.obs_dim[0]:] = self.bn_pnl(observations[:, self.obs_dim[0]:])
            state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "null":
            state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        else:
            raise NotImplementedError(f"bn_type not impleted")

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):
        if self.bn_type == "part":
            self.bn_pnl.train()
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features[:, :-self.obs_dim[-1]])
            dexrep_pnl = self.bn_pnl(obs_features[:, -self.obs_dim[-1]:])
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "full":
            self.bn_pnl.train()
            obs_features_norm = self.bn_pnl(obs_features)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features_norm[:, :-self.obs_dim[-1]])
            dexrep_pnl_emb = self.dexrep_pointL_enc(obs_features_norm[:, -self.obs_dim[-1]:])
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "null":
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features[:, :-self.obs_dim[-1]])
            dexrep_pnl_emb = self.dexrep_pointL_enc(obs_features[:, -self.obs_dim[-1]:])
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        else:
            raise NotImplementedError(f"bn_type not impleted")

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None