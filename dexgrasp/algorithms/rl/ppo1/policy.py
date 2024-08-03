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

class ActorCriticDexRep2g(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticDexRep2g, self).__init__()

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
        self.obj2goal_enc = nn.Linear(env_cfg['obs_dim']['dexrep_sensor2'], emb_dim)
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
        torch.nn.init.orthogonal_(self.obj2goal_enc.weight, gain=np.sqrt(2))

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
            state, dexrep_sensor, dexrep_pnl_raw, dexrep_sensor2 = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_sensor2_emb = self.obj2goal_enc(dexrep_sensor2)
            dexrep_pnl_norm = self.bn_pnl(dexrep_pnl_raw)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl_norm)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
            dexrep_sensor2_emb = F.normalize(dexrep_sensor2_emb, dim=-1)
        elif self.bn_type == "full":
            self.bn_pnl.eval()
            observations[:, self.obs_dim[0]:] = self.bn_pnl(observations[:, self.obs_dim[0]:])
            state, dexrep_sensor, dexrep_pnl, dexrep_sensor2 = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_sensor2_emb = self.obj2goal_enc(dexrep_sensor2)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
            dexrep_sensor2_emb = F.normalize(dexrep_sensor2_emb, dim=-1)
        elif self.bn_type == "null":
            state, dexrep_sensor, dexrep_pnl, dexrep_sensor2 = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_sensor2_emb = self.obj2goal_enc(dexrep_sensor2)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
            dexrep_sensor2_emb = F.normalize(dexrep_sensor2_emb, dim=-1)
        else:
            raise NotImplementedError(f"bn_type not impleted")

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb, dexrep_sensor2_emb], dim=1)

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
            state, dexrep_sensor, dexrep_pnl_raw, dexrep_sensor2 = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_sensor2_emb = self.obj2goal_enc(dexrep_sensor2)
            dexrep_pnl_norm = self.bn_pnl(dexrep_pnl_raw)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl_norm)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
            dexrep_sensor2_emb = F.normalize(dexrep_sensor2_emb, dim=-1)
        elif self.bn_type == "full":
            self.bn_pnl.eval()
            observations[:, self.obs_dim[0]:] = self.bn_pnl(observations[:, self.obs_dim[0]:])
            state, dexrep_sensor, dexrep_pnl, dexrep_sensor2 = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_sensor2_emb = self.obj2goal_enc(dexrep_sensor2)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
            dexrep_sensor2_emb = F.normalize(dexrep_sensor2_emb, dim=-1)
        elif self.bn_type == "null":
            state, dexrep_sensor, dexrep_pnl, dexrep_sensor2 = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_sensor2_emb = self.obj2goal_enc(dexrep_sensor2)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
            dexrep_sensor2_emb = F.normalize(dexrep_sensor2_emb, dim=-1)
        else:
            raise NotImplementedError(f"bn_type not impleted")

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb, dexrep_sensor2_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):
        if self.bn_type == "part":
            self.bn_pnl.train()
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features[:, :-self.obs_dim[-1]-self.obs_dim[-2]])
            dexrep_pnl = self.bn_pnl(obs_features[:, -self.obs_dim[-1]-self.obs_dim[-2]:-self.obs_dim[-1]])
            dexrep_sensor2_emb = self.obj2goal_enc(obs_features[:, -self.obs_dim[-1]:])
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
            dexrep_sensor2_emb = F.normalize(dexrep_sensor2_emb, dim=-1)
        elif self.bn_type == "full":
            self.bn_pnl.train()
            obs_features_norm = self.bn_pnl(obs_features)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features_norm[:, :-self.obs_dim[-1] - self.obs_dim[-2]])
            dexrep_pnl_emb = self.dexrep_pointL_enc(obs_features_norm[:, -self.obs_dim[-1]-self.obs_dim[-2]:-self.obs_dim[-1]])
            dexrep_sensor2_emb = self.obj2goal_enc(obs_features_norm[:, -self.obs_dim[-1]:])
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
            dexrep_sensor2_emb = F.normalize(dexrep_sensor2_emb, dim=-1)
        elif self.bn_type == "null":
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features[:, :-self.obs_dim[-1] - self.obs_dim[-2]])
            dexrep_pnl_emb = self.dexrep_pointL_enc(obs_features[:, -self.obs_dim[-1] - self.obs_dim[-2]:-self.obs_dim[-1]])
            dexrep_sensor2_emb = self.obj2goal_enc(obs_features[:, -self.obs_dim[-1]:])
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
            dexrep_sensor2_emb = F.normalize(dexrep_sensor2_emb, dim=-1)
        else:
            raise NotImplementedError(f"bn_type not impleted")

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb, dexrep_sensor2_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

class ActorCriticSurf(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticSurf, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]
        self.sensor_enc = nn.Linear(env_cfg['obs_dim']['sensor'], emb_dim)
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
        torch.nn.init.orthogonal_(self.sensor_enc.weight, gain=np.sqrt(2))

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
        state, sensor = self.obs_division(observations)
        state_emb = self.state_enc(state)
        sensor_emb = self.sensor_enc(sensor)
        sensor_emb = F.normalize(sensor_emb, dim=-1)

        joint_emb = torch.cat([state_emb, sensor_emb], dim=1)

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
        state, sensor = self.obs_division(observations)
        state_emb = self.state_enc(state)
        sensor_emb = self.dexrep_sensor_enc(sensor)
        sensor_emb = F.normalize(sensor_emb, dim=-1)

        joint_emb = torch.cat([state_emb, sensor_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):
        state_emb = self.state_enc(state)
        sensor_emb = self.sensor_enc(obs_features)
        sensor_emb = F.normalize(sensor_emb, dim=-1)

        joint_emb = torch.cat([state_emb, sensor_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)



class ActorCriticDexRep_Normall(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticDexRep_Normall, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]
        self.dexrep_sensor_enc = nn.Linear(env_cfg['obs_dim']['dexrep_sensor'], emb_dim)
        self.bn_pnl = nn.BatchNorm1d(env_cfg['obs_dim']['dexrep_pnl'])
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

    # def Nan_Detect_SetZero(self, detect_tensor):
    #     nan_exist = torch.isnan(detect_tensor)
    #     if nan_exist.any():
    #         nan_loc = torch.where(nan_exist == True)

    @torch.no_grad()
    def act(self, observations):
        # self.bn_pnl.eval()
        state, dexrep_sensor, dexrep_pnl_raw = self.obs_division(observations)
        state_emb = self.state_enc(state)
        dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
        dexrep_pnl_norm = self.bn_pnl(dexrep_pnl_raw)
        dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl_norm)
        # dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
        # dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        dexrep_emb = torch.cat([dexrep_sensor_emb, dexrep_pnl_emb], dim=1)
        dexrep_emb = F.normalize(dexrep_emb, dim=-1)

        joint_emb = torch.cat([state_emb, dexrep_emb], dim=1)

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
        # self.obs_enc.eval()
        # self.bn_pnl.eval()
        state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
        state_emb = self.state_enc(state)
        dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
        dexrep_pnl = self.bn_pnl(dexrep_pnl)
        dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
        # dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
        # dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        dexrep_emb = torch.cat([dexrep_sensor_emb, dexrep_pnl_emb], dim=1)
        dexrep_emb = F.normalize(dexrep_emb, dim=-1)

        joint_emb = torch.cat([state_emb, dexrep_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):

        state_emb = self.state_enc(state)
        dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features[:, :-self.obs_dim[-1]])
        dexrep_pnl = self.bn_pnl(obs_features[:, -self.obs_dim[-1]:])
        dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
        # dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
        # dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        dexrep_emb = torch.cat([dexrep_sensor_emb, dexrep_pnl_emb], dim=1)
        dexrep_emb = F.normalize(dexrep_emb, dim=-1)

        joint_emb = torch.cat([state_emb, dexrep_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

class ActorCriticDoubleDexRep(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticDoubleDexRep, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]
        self.dexrep_left_sensor_enc = nn.Linear(env_cfg['obs_dim']['dexrep_left_sensor'], emb_dim)
        self.dexrep_right_sensor_enc = nn.Linear(env_cfg['obs_dim']['dexrep_right_sensor'], emb_dim)
        self.bn_left_pnl = nn.BatchNorm1d(env_cfg['obs_dim']['dexrep_left_pnl'])
        self.bn_right_pnl = nn.BatchNorm1d(env_cfg['obs_dim']['dexrep_right_pnl'])
        self.dexrep_left_pointL_enc = nn.Linear(env_cfg['obs_dim']['dexrep_left_pnl'], emb_dim)
        self.dexrep_right_pointL_enc = nn.Linear(env_cfg['obs_dim']['dexrep_right_pnl'], emb_dim)
        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)

        # split num
        self.split_idx1 = None
        self.split_idx2 = None
        self.split_idx3 = None

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

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

        torch.nn.init.orthogonal_(self.state_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.dexrep_left_sensor_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.dexrep_right_sensor_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.dexrep_left_pointL_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.dexrep_right_pointL_enc.weight, gain=np.sqrt(2))

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
        self.bn_left_pnl.eval()
        self.bn_right_pnl.eval()
        # observation split
        state, dexrep_left_sensor, dexrep_left_pnl, dexrep_right_sensor, dexrep_right_pnl = self.obs_division(observations)
        # state
        state_emb = self.state_enc(state)
        # dexrep sensor
        dexrep_left_sensor_emb = self.dexrep_left_sensor_enc(dexrep_left_sensor)
        dexrep_right_sensor_emb = self.dexrep_right_sensor_enc(dexrep_right_sensor)
        # dexrep pnl batch norm
        dexrep_left_pnl = self.bn_left_pnl(dexrep_left_pnl)
        dexrep_right_pnl = self.bn_right_pnl(dexrep_right_pnl)
        # dexrep pnl Linear
        dexrep_left_pnl_emb = self.dexrep_left_pointL_enc(dexrep_left_pnl)
        dexrep_right_pnl_emb = self.dexrep_right_pointL_enc(dexrep_right_pnl)
        # dexrep norm
        dexrep_left_sensor_emb = F.normalize(dexrep_left_sensor_emb, dim=-1)
        dexrep_left_pnl_emb = F.normalize(dexrep_left_pnl_emb)
        dexrep_right_sensor_emb = F.normalize(dexrep_right_sensor_emb)
        dexrep_right_pnl_emb = F.normalize(dexrep_right_pnl_emb)

        joint_emb = torch.cat([state_emb, dexrep_left_sensor_emb, dexrep_left_pnl_emb, dexrep_right_sensor_emb, dexrep_right_pnl_emb], dim=1)

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
        # self.obs_enc.eval()
        self.bn_left_pnl.eval()
        self.bn_right_pnl.eval()
        # observation split
        state, dexrep_left_sensor, dexrep_left_pnl, dexrep_right_sensor, dexrep_right_pnl = self.obs_division(observations)
        # state
        state_emb = self.state_enc(state)
        # dexrep sensor
        dexrep_left_sensor_emb = self.dexrep_left_sensor_enc(dexrep_left_sensor)
        dexrep_right_sensor_emb = self.dexrep_right_sensor_enc(dexrep_right_sensor)
        # dexrep pnl batch norm
        dexrep_left_pnl = self.bn_left_pnl(dexrep_left_pnl)
        dexrep_right_pnl = self.bn_right_pnl(dexrep_right_pnl)
        # dexrep pnl Linear
        dexrep_left_pnl_emb = self.dexrep_left_pointL_enc(dexrep_left_pnl)
        dexrep_right_pnl_emb = self.dexrep_right_pointL_enc(dexrep_right_pnl)
        # dexrep norm
        dexrep_left_sensor_emb = F.normalize(dexrep_left_sensor_emb, dim=-1)
        dexrep_left_pnl_emb = F.normalize(dexrep_left_pnl_emb)
        dexrep_right_sensor_emb = F.normalize(dexrep_right_sensor_emb)
        dexrep_right_pnl_emb = F.normalize(dexrep_right_pnl_emb)

        joint_emb = torch.cat(
            [state_emb, dexrep_left_sensor_emb, dexrep_left_pnl_emb, dexrep_right_sensor_emb, dexrep_right_pnl_emb],
            dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):
        # self.obs_enc.eval()
        self.bn_left_pnl.train()
        self.bn_right_pnl.train()

        state_emb = self.state_enc(state)
        # observation split
        if self.split_idx1 == None:
            self.split_idx1 = self.obs_dim[1]
            self.split_idx2 = self.obs_dim[1] + self.obs_dim[2]
            self.split_idx3 = self.split_idx2 + self.obs_dim[3]
        dexrep_left_sensor = obs_features[:, :self.split_idx1]
        dexrep_left_pnl = obs_features[:, self.split_idx1:self.split_idx2]
        dexrep_right_sensor = obs_features[:, self.split_idx2:self.split_idx3]
        dexrep_right_pnl = obs_features[:, self.split_idx3:]
        # dexrep sensor
        dexrep_left_sensor_emb = self.dexrep_left_sensor_enc(dexrep_left_sensor)
        dexrep_right_sensor_emb = self.dexrep_right_sensor_enc(dexrep_right_sensor)
        # dexrep pnl batch norm
        dexrep_left_pnl = self.bn_left_pnl(dexrep_left_pnl)
        dexrep_right_pnl = self.bn_right_pnl(dexrep_right_pnl)
        # dexrep pnl Linear
        dexrep_left_pnl_emb = self.dexrep_left_pointL_enc(dexrep_left_pnl)
        dexrep_right_pnl_emb = self.dexrep_right_pointL_enc(dexrep_right_pnl)
        # dexrep norm
        dexrep_left_sensor_emb = F.normalize(dexrep_left_sensor_emb, dim=-1)
        dexrep_left_pnl_emb = F.normalize(dexrep_left_pnl_emb)
        dexrep_right_sensor_emb = F.normalize(dexrep_right_sensor_emb)
        dexrep_right_pnl_emb = F.normalize(dexrep_right_pnl_emb)

        joint_emb = torch.cat(
            [state_emb, dexrep_left_sensor_emb, dexrep_left_pnl_emb, dexrep_right_sensor_emb, dexrep_right_pnl_emb],
            dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

class ActorCriticDexRepNoNorm(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticDexRepNoNorm, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]
        self.dexrep_sensor_enc = nn.Linear(env_cfg['obs_dim']['dexrep_sensor'], emb_dim)
        # self.bn_pnl = nn.BatchNorm1d(env_cfg['obs_dim']['dexrep_pnl'])
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

    @staticmethod
    def detect_nan_setZero(input_tensor, value=0):
        nan_exists = torch.isnan(input_tensor)
        if nan_exists.any():
            nan_loc = torch.where(nan_exists)
            nan_env = torch.unique(nan_loc[0])
            # set this env to zero
            for env_idx in nan_env:
                if value == 0:
                    input_tensor[env_idx] = torch.zeros((input_tensor.shape[1]))
                else:
                    input_tensor[env_idx] = torch.ones((input_tensor.shape[1])) * value
        return input_tensor

    @torch.no_grad()
    def act(self, observations):
        state, dexrep_sensor, dexrep_pnl_raw = self.obs_division(observations)
        state = self.detect_nan_setZero(state)
        state_emb = self.state_enc(state)
        dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
        # dexrep_pnl = self.bn_pnl(dexrep_pnl)
        dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl_raw)
        dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
        dexrep_pnl_emb_norm = F.normalize(dexrep_pnl_emb, dim=-1)

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb_norm], dim=1)

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
        self.obs_enc.eval()
        state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
        state = self.detect_nan_setZero(state)
        state_emb = self.state_enc(state)
        dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
        # dexrep_pnl = self.bn_pnl(dexrep_pnl)
        dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
        dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
        dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):
        state = self.detect_nan_setZero(state)
        state_emb = self.state_enc(state)
        dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features[:, :-self.obs_dim[-1]])
        # dexrep_pnl = self.bn_pnl(obs_features[:, -self.obs_dim[-1]:])
        dexrep_pnl_emb = self.dexrep_pointL_enc(obs_features[:, -self.obs_dim[-1]:])
        dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
        dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

class ActorCriticPNG(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticPNG, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]
        # self.bn_pnl = nn.BatchNorm1d(env_cfg['obs_dim']['pnG'])
        self.pointG_enc = nn.Linear(env_cfg['obs_dim']['pnG'], emb_dim)
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

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

        # Initial fn
        torch.nn.init.orthogonal_(self.state_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.pointG_enc.weight, gain=np.sqrt(2))

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
        # self.bn_pnl.eval()
        state, pnG = self.obs_division(observations)
        state_emb = self.state_enc(state)
        # pnG = self.bn_pnl(pnG)
        pnG_emb = self.pointG_enc(pnG)
        pnG_emb = F.normalize(pnG_emb, dim=-1)

        joint_emb = torch.cat([state_emb, pnG_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)
        # self.bn_pnl.train()

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(),\
               observations[:, state.shape[1]:].detach()

    @torch.no_grad()
    def act_inference(self, observations):
        # self.bn_pnl.eval()
        state, pnG = self.obs_division(observations)
        state_emb = self.state_enc(state)
        # pnG = self.bn_pnl(pnG)
        pnG_emb = self.pointG_enc(pnG)
        pnG_emb = F.normalize(pnG_emb, dim=-1)

        joint_emb = torch.cat([state_emb, pnG_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):

        state_emb = self.state_enc(state)
        # pnG = self.bn_pnl(obs_features)
        pnG_emb = self.pointG_enc(obs_features)
        pnG_emb = F.normalize(pnG_emb, dim=-1)
        joint_emb = torch.cat([state_emb, pnG_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


class ActorCriticGeoDex_cold(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticGeoDex_cold, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        obs_emb_dim = encoder_cfg["obs_emb_dim"]
        self.obs_enc = Encoder_GeoDex_cold(env_cfg, encoder_cfg)
        # load obs_enc
        pointnet_load_path = encoder_cfg["pretrain_dir"]
        fname = 'feat_model.pth'
        pretrain_weights = torch.load(os.path.join(pointnet_load_path, fname))
        self.obs_enc.pointnet.load_state_dict(pretrain_weights)
        # if not self.args.finetune_pointnet:
        for name, p in self.obs_enc.pointnet.named_parameters(): # frozen
            p.requires_grad = False
        print('*** successfully loaded pointnet feature model ***')

        # state_enc
        state_emb_dim = encoder_cfg["state_emb_dim"]
        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], state_emb_dim)

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
        # actor_layers.append(nn.Linear(obs_emb_dim + state_emb_dim, actor_hidden_dim[0]))
        actor_layers.append(nn.Linear(state_emb_dim, actor_hidden_dim[0]))
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

        # critic_layers.append(nn.Linear(obs_emb_dim + state_emb_dim, critic_hidden_dim[0]))
        critic_layers.append(nn.Linear(state_emb_dim, critic_hidden_dim[0]))
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

    def forward(self):
        raise NotImplementedError

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

    @torch.no_grad()
    def act(self, observations):
        self.obs_enc.eval()
        state, obj_pnts, obj_norms, goal_pnts, goal_norms = self.obs_division(observations)
        # state project
        state_emb = self.state_enc(state)
        # obs project
        # obs_emb, obs_feat = self.obs_enc.forward_obs(obj_pnts, obj_norms, goal_pnts, goal_norms)
        # joint_emb = torch.cat([state_emb, obs_emb], dim=1)
        joint_emb = state_emb

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)
        self.obs_enc.train()

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(), \
               observations[:, state.shape[1]:].detach()  # obs_feat.detach()

    @torch.no_grad()
    def act_inference(self, observations):
        self.obs_enc.eval()
        state, obj_pnts, obj_norms, goal_pnts, goal_norms = self.obs_division(observations)
        state_emb = self.state_enc(state)
        # obs_emb, _ = self.obs_enc.forward_obs(obj_pnts, obj_norms, goal_pnts, goal_norms)
        # joint_emb = torch.cat([state_emb, obs_emb], dim=1)
        joint_emb = state_emb

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):
        # project state
        state_emb = self.state_enc(state)
        # project obs feat
        # obs_emb = self.obs_enc.forward(obs_features)
        # joint emb
        # joint_emb = torch.cat([state_emb, obs_emb], dim=1)
        joint_emb = state_emb
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

class ActorCriticGeoDex_cold2(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticGeoDex_cold2, self).__init__()
        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]

        # state encoder None
        self.state_emb_dim = encoder_cfg["state_emb_dim"]
        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], self.state_emb_dim)

        # obs_Encoder
        self.obs_emb_dim = encoder_cfg["obs_emb_dim"]
        self.num_points = env_cfg["geodex"]["sample_num_points"]
        self.obs_enc = PointNetfeatTwoStream(
            output_dim=encoder_cfg["pointnet_output_dim"])
        # self.bn_pnl = nn.BatchNorm1d(encoder_cfg["pointnet_output_dim"])
        self.obs_fn = nn.Linear(
            encoder_cfg["pointnet_output_dim"], self.obs_emb_dim)
        # load obs_enc
        pointnet_load_path = encoder_cfg["pretrain_dir"]
        fname = 'feat_model.pth'
        pretrain_weights = torch.load(os.path.join(pointnet_load_path, fname))
        self.obs_enc.load_state_dict(pretrain_weights)
        # if not self.args.finetune_pointnet:
        for name, p in self.obs_enc.named_parameters():  # frozen
            p.requires_grad = False
        print('*** successfully loaded pointnet feature model ***')

        # Encoder
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
        actor_layers.append(nn.Linear(self.state_emb_dim + self.obs_emb_dim, actor_hidden_dim[0])) # add
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []

        # critic_layers.append(nn.Linear(*obs_shape, critic_hidden_dim[0]))
        critic_layers.append(nn.Linear(self.state_emb_dim + self.obs_emb_dim, critic_hidden_dim[0]))  # add
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

        # Initial fn
        torch.nn.init.orthogonal_(self.state_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.obs_fn.weight, gain=np.sqrt(2))

    # add
    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 0
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        self.obs_enc.eval()
        state, obj_pnts, obj_norms, goal_pnts, goal_norms = self.obs_division(observations)  # add
        # project state
        state_emb = self.state_enc(state)

        # project obs
        batch_num = obj_pnts.shape[0]
        assert len(obj_pnts.shape) == 2 and len(goal_pnts.shape) == 2
        obj_points = obj_pnts.reshape(
            [batch_num, self.num_points, 3])
        target_points = goal_pnts.reshape(
            [batch_num, self.num_points, 3])
        # reshape points
        assert len(obj_norms.shape) == 2 and len(goal_norms.shape) == 2
        obj_normals = obj_norms.reshape([batch_num, self.num_points, 3])
        target_normals = goal_norms.reshape([batch_num, self.num_points, 3])
        # get pointnet features ========================================================
        obj_obs = torch.cat([obj_points, obj_normals], dim=-1)
        target_obs = torch.cat([target_points, target_normals], dim=-1)
        # need to do transpose in order to use the conv implementation of pointnet
        obj_obs = obj_obs.transpose(2, 1)
        target_obs = target_obs.transpose(2, 1)
        obs_feat = self.obs_enc(obj_obs, target_obs)
        # obs fn
        # obs_emb = self.bn_pnl(obs_feat)
        obs_emb = self.obs_fn(obs_feat)
        # obs_emb = F.relu(obs_emb)
        obs_emb = F.normalize(obs_emb, dim=-1)

        # joint emb
        joint_emb = torch.cat(
            [state_emb, obs_emb], dim=1
        )

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(joint_emb)

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state, \
               obs_feat.detach()

    @torch.no_grad()
    def act_inference(self, observations):
        self.obs_enc.eval()

        state, obj_pnts, obj_norms, goal_pnts, goal_norms = self.obs_division(observations)  # add
        # project state
        state_emb = self.state_enc(state)

        # project obs
        batch_num = obj_pnts.shape[0]
        assert len(obj_pnts.shape) == 2 and len(goal_pnts.shape) == 2
        obj_points = obj_pnts.reshape(
            [batch_num, self.num_points, 3])
        target_points = goal_pnts.reshape(
            [batch_num, self.num_points, 3])
        # reshape points
        assert len(obj_norms.shape) == 2 and len(goal_norms.shape) == 2
        obj_normals = obj_norms.reshape([batch_num, self.num_points, 3])
        target_normals = goal_norms.reshape([batch_num, self.num_points, 3])
        # get pointnet features ========================================================
        obj_obs = torch.cat([obj_points, obj_normals], dim=-1)
        target_obs = torch.cat([target_points, target_normals], dim=-1)
        # need to do transpose in order to use the conv implementation of pointnet
        obj_obs = obj_obs.transpose(2, 1)
        target_obs = target_obs.transpose(2, 1)
        obs_feat = self.obs_enc(obj_obs, target_obs)
        # obs fn
        # obs_emb = self.bn_pnl(obs_feat)
        obs_emb = self.obs_fn(obs_feat)
        # obs_emb = F.relu(obs_emb)
        obs_emb = F.normalize(obs_emb, dim=-1)

        # joint emb
        joint_emb = torch.cat(
            [state_emb, obs_emb], dim=1
        )

        actions_mean = self.actor(joint_emb)
        return actions_mean


    def evaluate(self, observations, state, actions):
        # project state
        state_emb = self.state_enc(state)
        # project obs
        # obs_emb = self.bn_pnl(observations)
        obs_emb = self.obs_fn(observations)
        # obs_emb = F.relu(obs_emb)
        obs_emb = F.normalize(obs_emb, dim=-1)
        # cat
        joint_emb = torch.cat(
            [state_emb, obs_emb], dim=1
        )
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

class ActorCriticGeoDex2(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticGeoDex2, self).__init__()
        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]

        # state encoder None
        self.state_emb_dim = encoder_cfg["state_emb_dim"]
        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], self.state_emb_dim)

        # obs_Encoder
        self.obs_emb_dim = encoder_cfg["obs_emb_dim"]
        self.num_points = env_cfg["geodex"]["sample_num_points"]
        self.obs_enc = PointNetfeatTwoStream(
            output_dim=encoder_cfg["pointnet_output_dim"])
        # self.bn_pnl = nn.BatchNorm1d(encoder_cfg["pointnet_output_dim"])
        self.obs_fn = nn.Linear(
            encoder_cfg["pointnet_output_dim"], self.obs_emb_dim)
        # load obs_enc
        pointnet_load_path = encoder_cfg["pretrain_dir"]
        fname = 'feat_model.pth'
        pretrain_weights = torch.load(os.path.join(pointnet_load_path, fname))
        self.obs_enc.load_state_dict(pretrain_weights)
        print('*** successfully loaded pointnet feature model ***')

        self.split_idx1 = None
        self.split_idx2 = None
        self.split_idx3 = None

        # split batch to enc obs
        self.split_batch_num = 2000

        # Encoder
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
        actor_layers.append(nn.Linear(self.state_emb_dim + self.obs_emb_dim, actor_hidden_dim[0])) # add
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []

        # critic_layers.append(nn.Linear(*obs_shape, critic_hidden_dim[0]))
        critic_layers.append(nn.Linear(self.state_emb_dim + self.obs_emb_dim, critic_hidden_dim[0]))  # add
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

        # Initial fn
        torch.nn.init.orthogonal_(self.state_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.obs_fn.weight, gain=np.sqrt(2))

    # add
    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 0
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        self.obs_enc.eval()

        state, obj_pnts, obj_norms, goal_pnts, goal_norms = self.obs_division(observations)  # add
        # project state
        state_emb = self.state_enc(state)

        # project obs
        batch_num = obj_pnts.shape[0]
        assert len(obj_pnts.shape) == 2 and len(goal_pnts.shape) == 2
        obj_points = obj_pnts.reshape(
            [batch_num, self.num_points, 3])
        target_points = goal_pnts.reshape(
            [batch_num, self.num_points, 3])
        # reshape points
        assert len(obj_norms.shape) == 2 and len(goal_norms.shape) == 2
        obj_normals = obj_norms.reshape([batch_num, self.num_points, 3])
        target_normals = goal_norms.reshape([batch_num, self.num_points, 3])
        # get pointnet features ========================================================
        obj_obs = torch.cat([obj_points, obj_normals], dim=-1)
        target_obs = torch.cat([target_points, target_normals], dim=-1)
        # need to do transpose in order to use the conv implementation of pointnet
        obj_obs = obj_obs.transpose(2, 1)
        target_obs = target_obs.transpose(2, 1)
        obs_feat = self.obs_enc(obj_obs, target_obs)
        # obs fn
        # obs_emb = self.bn_pnl(obs_feat)
        obs_emb = F.normalize(obs_feat, dim=-1)
        obs_emb = self.obs_fn(obs_emb)

        # joint emb
        joint_emb = torch.cat(
            [state_emb, obs_emb], dim=1
        )

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(joint_emb)

        self.obs_enc.train()

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state, \
               observations[:, state.shape[1]:].detach()

    def evaluate(self, observations, state, actions):
        # project state
        state_emb = self.state_enc(state)
        # divide observation
        if self.split_idx1 == None:
            self.split_idx1 = self.obs_dim[1]
            self.split_idx2 = self.obs_dim[1] + self.obs_dim[2]
            self.split_idx3 = self.split_idx2 + self.obs_dim[3]
        obj_pnts = observations[:, :self.split_idx1]
        obj_norms = observations[:, self.split_idx1:self.split_idx2]
        goal_pnts = observations[:, self.split_idx2:self.split_idx3]
        goal_norms = observations[:, self.split_idx3:]
        # compute obs
        batch_num = obj_pnts.shape[0]
        assert len(obj_pnts.shape) == 2 and len(goal_pnts.shape) == 2
        obj_points = obj_pnts.reshape([batch_num, self.num_points, 3])
        target_points = goal_pnts.reshape([batch_num, self.num_points, 3])
        # reshape points
        assert len(obj_norms.shape) == 2 and len(goal_norms.shape) == 2
        obj_normals = obj_norms.reshape([batch_num, self.num_points, 3])
        target_normals = goal_norms.reshape([batch_num, self.num_points, 3])
        # get pointnet features ========================================================
        obj_obs = torch.cat([obj_points, obj_normals], dim=-1)
        target_obs = torch.cat([target_points, target_normals], dim=-1)
        # transpose
        obj_obs = obj_obs.transpose(2, 1)
        target_obs = target_obs.transpose(2, 1)
        # project obs splitly
        obj_obs_slices = [obj_obs[i * self.split_batch_num: (i + 1) * self.split_batch_num, ...] \
                             for i in range(int((obj_obs.shape[0] + self.split_batch_num - 1) / self.split_batch_num))]
        target_obs_slices = [target_obs[i * self.split_batch_num: (i + 1) * self.split_batch_num, ...] \
                             for i in range(int((target_obs.shape[0] + self.split_batch_num - 1) / self.split_batch_num))]
        obs_feat_slices = []
        assert len(obj_obs_slices) == len(target_obs_slices)
        for i in range(len(obj_obs_slices)):
            obs_feat_slice = self.obs_enc(obj_obs_slices[i], target_obs_slices[i])
            obs_feat_slices.append(obs_feat_slice)
        obs_feat = torch.cat(obs_feat_slices, dim=0)
        # obs_feat = self.obs_enc(obj_obs, target_obs)
        # obs_emb = self.bn_pnl(obs_feat)
        obs_emb = F.normalize(obs_feat, dim=-1)
        obs_emb = self.obs_fn(obs_emb)
        # cat
        joint_emb = torch.cat(
            [state_emb, obs_emb], dim=1
        )
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


class ActorCriticGeoDex(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticGeoDex, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        obs_emb_dim = encoder_cfg["obs_emb_dim"]
        self.obs_enc = Encoder_GeoDex(env_cfg, encoder_cfg)
        # load obs_enc
        pointnet_load_path = encoder_cfg["pretrain_dir"]
        fname = 'feat_model.pth'
        pretrain_weights = torch.load(os.path.join(pointnet_load_path, fname))
        self.obs_enc.pointnet.load_state_dict(pretrain_weights)
        # if not self.args.finetune_pointnet:
        # for name, p in self.obs_enc.pointnet.named_parameters(): # frozen
        #     p.requires_grad = False
        print('*** successfully loaded pointnet feature model ***')

        # state_enc
        state_emb_dim = encoder_cfg["state_emb_dim"]
        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], state_emb_dim)

        self.split_idx1 = None
        self.split_idx2 = None
        self.split_idx3 = None

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
        actor_layers.append(nn.Linear(obs_emb_dim + state_emb_dim, actor_hidden_dim[0]))
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

        critic_layers.append(nn.Linear(obs_emb_dim + state_emb_dim, critic_hidden_dim[0]))
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

    def forward(self):
        raise NotImplementedError

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

    @torch.no_grad()
    def act(self, observations):
        self.obs_enc.eval()
        state, obj_pnts, obj_norms, goal_pnts, goal_norms = self.obs_division(observations)
        # state project
        state_emb = self.state_enc(state)
        # obs project
        obs_emb = self.obs_enc.forward_obs(obj_pnts, obj_norms, goal_pnts, goal_norms)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)
        self.obs_enc.train()

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(), \
               observations[:, state.shape[1]:].detach()

    @torch.no_grad()
    def act_inference(self, observations):
        self.obs_enc.eval()
        state, obj_pnts, obj_norms, goal_pnts, goal_norms = self.obs_division(observations)
        state_emb = self.state_enc(state)
        obs_emb = self.obs_enc.forward_obs(obj_pnts, obj_norms, goal_pnts, goal_norms)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):
        # project state
        state_emb = self.state_enc(state)

        if self.split_idx1 == None:
            self.split_idx1 = self.obs_dim[1]
            self.split_idx2 = self.obs_dim[1] + self.obs_dim[2]
            self.split_idx3 = self.split_idx2 + self.obs_dim[3]
        obj_pnts = obs_features[:, :self.split_idx1]
        obj_norms = obs_features[:, self.split_idx1:self.split_idx2]
        goal_pnts = obs_features[:, self.split_idx2:self.split_idx3]
        goal_norms = obs_features[:, self.split_idx3:]
        # project obs feat
        obs_emb = self.obs_enc.forward(obj_pnts, obj_norms, goal_pnts, goal_norms)
        # joint emb
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)
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