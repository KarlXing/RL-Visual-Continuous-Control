import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy


def parse_args():
    parser = argparse.ArgumentParser()
    ##### Common #####
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) 
    parser.add_argument('--critic_encoder_tau', default=0.05, type=float) 
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)

    ##### Algorithm-Specific Parameters
    parser.add_argument('--agent', default='curl', type=str, help='curl, sacae, sac, rad, drq, atc')
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)

    # curl
    parser.add_argument('--curl_update_freq', default=1, type=int)
    parser.add_argument('--curl_lr', default=1e-3, type=float)
    parser.add_argument('--curl_encoder_tau', default=0.05, type=float)

    # sac_ae
    parser.add_argument('--sacae_update_freq', default=1, type=int)
    parser.add_argument('--sacae_autoencoder_lr', default=1e-3, type=float)
    parser.add_argument('--sacae_autoencoder_beta', default=0.9, type=float)
    parser.add_argument('--sacae_encoder_tau', default=0.05, type=float)

    # drq & atc
    parser.add_argument('--image_pad', default=4, type=int)

    # atc
    parser.add_argument('--atc_update_freq', default=1, type=int)
    parser.add_argument('--atc_lr', default=1e-3, type=float)
    parser.add_argument('--atc_beta', default=0.9, type=float)
    parser.add_argument('--atc_encoder_tau', default=0.01, type=float)
    parser.add_argument('--atc_target_update_freq', default=1, type=int)
    parser.add_argument('--atc_encoder_feature_dim', default=128, type=int)
    parser.add_argument('--atc_hidden_feature_dim', default=512, type=int)
    parser.add_argument('--atc_rl_clip_grad_norm', default=1000000, type=float)
    parser.add_argument('--atc_cpc_clip_grad_norm', default=10, type=float)

    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='./log', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--log_interval', default=25, type=int)
    parser.add_argument('--tag', default='', type=str)
    args = parser.parse_args()
    
    # verification
    assert (args.agent in ['curl', 'sacae', 'sac', 'rad', 'drq', 'atc'])

    if args.agent in ['curl', 'rad']:
        args.env_image_size = 100
        args.agent_image_size = 84
    elif args.agent in ['sacae', 'sac', 'drq', 'atc']:
        args.env_image_size = 84
        args.agent_image_size = 84
    
    if args.agent not in ['drq', 'atc']:
        args.image_pad = None

    return args

