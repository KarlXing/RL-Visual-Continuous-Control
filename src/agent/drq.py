import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .sac import SAC


class DRQ(SAC):
    def __init__(self, model, device, action_shape, args):
        super().__init__(model, device, action_shape, args)

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step, obs_aug, next_obs_aug):
                    
        with torch.no_grad():
            # target Q from next_obs
            _, policy_action, log_pi, _ = self.model.actor(next_obs)
            target_Q1, target_Q2 = self.model.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

            # target Q from next_obs_aug
            _, policy_action_aug, log_pi_aug, _ = self.model.actor(next_obs_aug)
            target_Q1_aug, target_Q2_aug = self.model.critic_target(next_obs_aug, policy_action_aug)
            target_V_aug = torch.min(target_Q1_aug, target_Q2_aug) - self.alpha.detach() * log_pi_aug
            target_Q_aug =  reward + (not_done * self.discount * target_V_aug)
            
            target_Q = (target_Q + target_Q_aug) / 2
        
        # get current Q estimates
        current_Q1, current_Q2 = self.model.critic(obs, action, detach=self.detach_encoder)
        current_Q1_aug, current_Q2_aug = self.model.critic(obs_aug, action, detach=self.detach_encoder)
        
        critic_loss = F.mse_loss(current_Q1,target_Q) + \
                      F.mse_loss(current_Q2, target_Q) + \
                      F.mse_loss(current_Q1_aug,target_Q) +  \
                      F.mse_loss(current_Q2_aug, target_Q)
    
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample_drq()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step, obs_aug, next_obs_aug)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.model.soft_update_params(self.critic_tau, self.encoder_tau)

