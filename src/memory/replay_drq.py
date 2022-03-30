# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import kornia
from skimage.util.shape import view_as_windows


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return episode['s'].shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, replay_dir):
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, state, action, reward, done):
        if state is not None:
            self._current_episode['s'].append(state)
        if action is not None:
            self._current_episode['a'].append(action)
        if reward is not None:
            self._current_episode['r'].append(reward)
        
        if done:
            episode = dict()
            episode['s'] = np.array(self._current_episode['s'], np.uint8)
            episode['a'] = np.array(self._current_episode['a'], np.float32)
            episode['r'] = np.array(self._current_episode['r'], np.float32)

            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBufferDataset(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self.cnt = 0

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        # print('try fetch start, ', len(self._episodes.keys()), self._samples_since_last_fetch, self._fetch_every)
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break
        self.cnt += 1
        # print('try fetch, ', len(self._episodes.keys()), self._samples_since_last_fetch, self._fetch_every)
        # print('fetch cnt', self.cnt)

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1)
        obs = episode['s'][idx]
        action = episode['a'][idx]
        next_obs = episode['s'][idx + self._nstep]
        reward = np.zeros_like(episode['r'][idx])
        discount = 1
        for i in range(self._nstep):
            step_reward = episode['r'][idx]
            reward += discount * step_reward
            discount *= self._discount
        return (obs, action, reward, next_obs)

    def __iter__(self):
        self.cnt += 1
        while True:
            yield self._sample()


class ReplayBuffer(object):
    def __init__(self, iter, obs_shape, device, image_size=84, image_pad=None):
        self.iter = iter
        self.device = device
        self.image_size = image_size
        self.image_pad = image_pad

        if image_pad is not None:
            self.aug_trans = nn.Sequential(
                nn.ReplicationPad2d(image_pad),
                kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))
    
    def sample(self):
        (obs, action, reward, next_obs) = next(self.iter)
        reward = torch.unsqueeze(reward, dim=-1)
        not_done = torch.ones_like(reward)  # episode ends because of maximal step limits 
        
        obs = obs.float().to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.float().to(self.device)
        not_done = not_done.to(self.device)
        
        return obs, action, reward, next_obs, not_done
    
    def sample_drq(self):
        (obs, action, reward, next_obs) = next(self.iter)
        reward = torch.unsqueeze(reward, dim=-1)
        not_done = torch.ones_like(reward)  # episode ends because of maximal step limits 
        
        obs_aug = obs.clone()
        next_obs_aug = next_obs.clone()
        
        obs = obs.float().to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.float().to(self.device)
        not_done = not_done.to(self.device)

        obs_aug = obs_aug.float().to(self.device)
        next_obs_aug = next_obs_aug.float().to(self.device)

        obs = self.aug_trans(obs)
        next_obs = self.aug_trans(next_obs)

        obs_aug = self.aug_trans(obs_aug)
        next_obs_aug = self.aug_trans(next_obs_aug)
        
        return obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug    
    

    def sample_atc(self):
        (obs, action, reward, next_obs) = next(self.iter)
        reward = torch.unsqueeze(reward, dim=-1)
        not_done = torch.ones_like(reward)  # episode ends because of maximal step limits 
        
        obs = obs.float().to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.float().to(self.device)
        not_done = not_done.to(self.device)

        obs = self.aug_trans(obs)
        next_obs = self.aug_trans(next_obs)
        
        return obs, action, reward, next_obs, not_done    


    def sample_rad(self):
        (obs, action, reward, next_obs) = next(self.iter)
        reward = torch.unsqueeze(reward, dim=-1)
        not_done = torch.ones_like(reward)  # episode ends because of maximal step limits 

        obs = obs.float().to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.float().to(self.device)
        not_done = not_done.to(self.device)

        obs = random_crop(obs, size=self.image_size)
        next_obs = random_crop(next_obs, size=self.image_size)

        return obs, action, reward, next_obs, not_done    


    def sample_curl(self):
        (obs, action, reward, next_obs) = next(self.iter)
        reward = torch.unsqueeze(reward, dim=-1)
        not_done = torch.ones_like(reward)  # episode ends because of maximal step limits 
        
        pos = obs.clone()
        
        obs = obs.float().to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.float().to(self.device)
        not_done = not_done.to(self.device)
        pos = pos.float().to(self.device)             
        
        obs = random_crop(obs, self.image_size)
        next_obs = random_crop(next_obs, self.image_size)
        pos = random_crop(pos, self.image_size)
        
        cpc_kwargs = dict(obs_anchor=obs, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obs, action, reward, next_obs, not_done, cpc_kwargs


# referred from https://github.com/nicklashansen/dmcontrol-generalization-benchmark
def random_crop(x, size=84):
	"""Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
	assert isinstance(x, torch.Tensor) and x.is_cuda, \
		'input must be CUDA tensor'
	
	n = x.shape[0]
	img_size = x.shape[-1]
	crop_max = img_size - size

	if crop_max <= 0:
		return x

	x = x.permute(0, 2, 3, 1)

	w1 = torch.LongTensor(n).random_(0, crop_max)
	h1 = torch.LongTensor(n).random_(0, crop_max)

	windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
	cropped = windows[torch.arange(n), w1, h1]

	return cropped


def view_as_windows_cuda(x, window_shape):
	"""PyTorch CUDA-enabled implementation of view_as_windows"""
	assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
		'window_shape must be a tuple with same number of dimensions as x'
	
	slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
	win_indices_shape = [
		x.size(0),
		x.size(1)-int(window_shape[1]),
		x.size(2)-int(window_shape[2]),
		x.size(3)    
	]

	new_shape = tuple(list(win_indices_shape) + list(window_shape))
	strides = tuple(list(x[slices].stride()) + list(x.stride()))

	return x.as_strided(new_shape, strides)


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_buffer(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount, obs_shape, device, image_size, image_pad):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBufferDataset(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    buffer = ReplayBuffer(iter(loader), obs_shape, device, image_size, image_pad)
    
    return buffer
