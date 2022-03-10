import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .sac import SAC


class ATC(SAC):
    def __init__(self, model, device, action_shape, args):
        super().__init__(model, device, action_shape, args)
        
        # atc does not share encoder target with critic (different from sac_ae and curl)
        self.atc_encoder_tau = args.atc_encoder_tau
        self.atc_target_update_freq = args.atc_target_update_freq
        self.atc_update_freq = args.atc_update_freq
        self.atc_rl_clip_grad_norm = args.atc_rl_clip_grad_norm
        self.atc_cpc_clip_grad_norm = args.atc_cpc_clip_grad_norm

        # optimizers
        self.atc_optimizer = torch.optim.Adam(
            self.model.atc.parameters(), lr=args.atc_lr, betas=(args.atc_beta, 0.999))

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.model.critic_target.train()


    def train(self, training=True):
        self.training = training
        self.model.actor.train(training)
        self.model.critic.train(training)
        self.model.atc.train(training)


    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.model.actor(next_obs)
            target_Q1, target_Q2 = self.model.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.model.critic(
            obs, action, detach=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.atc_rl_clip_grad_norm)
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.model.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.model.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.atc_rl_clip_grad_norm)
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


    def update_atc(self, x_a, x_pos, L, step):
        z_a = self.model.atc.encode(x_a)
        with torch.no_grad():
            z_pos = self.model.atc_encoder_target(x_pos)
        
        logits = self.model.atc.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)
        
        self.atc_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.atc.parameters(), self.atc_cpc_clip_grad_norm)
        self.atc_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/atc_loss', loss, step)        
    

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_atc()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)
        
        if step % self.atc_update_freq == 0:
            self.update_atc(obs, next_obs, L, step)
        
        if step % self.critic_target_update_freq == 0:
            self.model.soft_update_params(self.critic_tau, self.encoder_tau)
            
        if step % self.atc_target_update_freq == 0:
            self.model.soft_update_params_atc(self.atc_encoder_tau)
