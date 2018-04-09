import gym
import gym_rle
import os
from wrapper import *

from baselines import logger
from baselines.common.atari_wrappers import wrap_deepmind, make_atari, MaxAndSkipEnv
from baselines import bench
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from realtime_env import RealtimeEnv
import numpy as np

def make_env(dying_penalty = 0):
    env = gym.make('GradiusIiiDeterministic-v0')
    env = StateSaver2(env, load_chance = 0.5)
    env = EpisodicWrapper(env, dying_penalty = dying_penalty)
    env = wrap_deepmind(env, episode_life = False, clip_rewards = False, frame_stack = True)
    env = MaxAndSkipEnv(env, skip=2)
    env = AutoShootWrapper(env)
    env = bench.Monitor(env, logger.get_dir())
    return env

def make_subproc_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            # env = StateSaver2(env, load_chance = 0.5)
            env = EpisodicWrapper(env)
            env = wrap_deepmind(env, episode_life = False, clip_rewards = False, frame_stack = True)
            env = MaxAndSkipEnv(env, skip=4)
            # env = AutoShootWrapper(env)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_realtime_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            # print('xxxxxx', env.observation_space)
            env = StateSaver2(env, load_chance = 0.5)
            env = EpisodicWrapper(env)
            env = WrapFrame(env)
            env = MaxAndSkipEnv(env, skip=4)
            # print('xxxxxx', env.observation_space)
            # env = wrap_deepmind(env, episode_life = False, clip_rewards = False, frame_stack = True)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    set_global_seeds(seed)
    return RealtimeEnv([make_env(i + start_index) for i in range(num_env)])
