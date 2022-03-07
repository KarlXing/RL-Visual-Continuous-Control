from operator import imod
from pyexpat import model
from parso import parse
import torch
import numpy as np
from utils.argument import parse_args
from utils.misc import set_seed_everywhere, make_dir, VideoRecorder, eval_mode
from utils.logger import Logger
from memory import ReplayBuffer
from model import CURL_Model
from env import make_envs
from agent import make_agent
import dmc2gym
import time
import os
import json


def main():
    # prepare workspace
    args = parse_args()
    set_seed_everywhere(args.seed)
    
    ts = time.strftime("%m-%d", time.gmtime())    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.env_image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.agent
    args.work_dir = args.work_dir + '/'  + exp_name
    make_dir(args.work_dir)
    video_dir = make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = make_dir(os.path.join(args.work_dir, 'buffer'))
    video = VideoRecorder(video_dir if args.save_video else None)
    
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    
    # prepare env
    env = make_envs(args)
    
    # prepare memory
    action_shape = env.action_space.shape
    agent_obs_shape = (3*args.frame_stack, args.agent_image_size, args.agent_image_size)
    env_obs_shape = (3*args.frame_stack, args.env_image_size, args.env_image_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    replay_buffer = ReplayBuffer(
        obs_shape=env_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.agent_image_size,
    )
    
    # prepare model
    model = CURL_Model(agent_obs_shape, 
                       action_shape, 
                       args.hidden_dim,
                       args.encoder_feature_dim,
                       args.actor_log_std_min,
                       args.actor_log_std_max,
                       args.num_layers,
                       args.num_filters,
                       device)


    # prepare agent
    agent = make_agent(
        model=model,
        device=device,
        action_shape=action_shape,
        args=args
    )
    
    # run
    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        # if step % args.eval_freq == 0:
        #     L.log('eval/episode', episode, step)
        #     evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
            # if args.save_model:
            #     agent.save_curl(model_dir, step)
            # if args.save_buffer:
            #     replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


    
    
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()