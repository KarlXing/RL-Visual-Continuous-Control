import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .sac import SAC


class CURL(SAC):
    def __init__(self, model, device, action_shape, args):
        super().__init__(model, device, action_shape, args)

        self.curl_update_freq = args.curl_update_freq
        self.encoder_tau = args.curl_encoder_tau

        self.curl_optimizer = torch.optim.Adam(
                self.model.curl.parameters(), lr=args.curl_lr)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        self.train()


    def train(self, training=True):
        self.training = training
        self.model.actor.train(training)
        self.model.critic.train(training)
        self.model.curl.train(training)

    
    def update_curl(self, x_a, x_pos, L, step):
        z_a = self.model.curl.encode(x_a)
        with torch.no_grad():
            z_pos = self.model.critic_target.encoder(x_pos)
        
        logits = self.model.curl.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        
        self.curl_optimizer.zero_grad()
        loss.backward()
        self.curl_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/curl_loss', loss, step)


    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_curl()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.model.soft_update_params(self.critic_tau, self.encoder_tau)
        
        if step % self.curl_update_freq == 0:
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            self.update_curl(obs_anchor, obs_pos, L, step)

        