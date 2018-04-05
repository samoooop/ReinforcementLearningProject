#!/usr/bin/env python3
import sys
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import multiprocessing
import tensorflow as tf

import gym
import gym_rle
from baselines.common.atari_wrappers import wrap_deepmind, make_atari, MaxAndSkipEnv
from baselines import bench

from baselines.common import set_global_seeds
from wrapper import *

def make_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = StateSaver2(env, load_chance = 0.5)
            env = EpisodicWrapper(env)
            env = wrap_deepmind(env, episode_life = False, clip_rewards = False, frame_stack = True)
            env = MaxAndSkipEnv(env, skip=4)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

env = make_env('GradiusIiiDeterministic-v0', 8, 0)
env.reset()
for i in range(10000):
    obs, rews, dones, infos = env.step(np.zeros(8, dtype = np.int))
    print(obs.shape, i, dones)