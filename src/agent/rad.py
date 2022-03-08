import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .sac import SAC


class RAD(SAC):
    def __init__(self, model, device, action_shape, args):
        super().__init__(model, device, action_shape, args)


    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_rad()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.model.soft_update_params(self.critic_tau, self.encoder_tau)

