import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .sac import SAC

def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class SACAE(SAC):
    def __init__(self, model, device, action_shape, args):
        super().__init__(model, device, action_shape, args)
        
        self.autoencoder_update_freq = args.sacae_update_freq
        self.encoder_tau = args.sacae_encoder_tau

        self.autoencoder_optimizer = torch.optim.Adam(
            self.model.autoencoder.parameters(), lr=args.sacae_autoencoder_lr, betas=(args.sacae_autoencoder_beta, 0.999))

        self.train()

    def train(self, training=True):
        self.training = training
        self.model.actor.train(training)
        self.model.critic.train(training)
        self.model.autoencoder.train(training)


    def update_autoencoder(self, x, L, step):
        recon_x = self.model.autoencoder.recon(x)
        target = preprocess_obs(x)
        recon_loss = F.mse_loss(recon_x, target)

        self.autoencoder_optimizer.zero_grad()
        recon_loss.backward()
        self.autoencoder_optimizer.step()
        
        if step % self.log_interval == 0:
            L.log('train/autoencoder_loss', recon_loss, step)


    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.model.soft_update_params(self.critic_tau, self.encoder_tau)
        
        if step % self.autoencoder_update_freq == 0:
            self.update_autoencoder(obs, L, step)
