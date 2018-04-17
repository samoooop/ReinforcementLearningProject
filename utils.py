import gym
import gym_rle
import os
from wrapper import *

from baselines import logger
from baselines.common.atari_wrappers import wrap_deepmind, make_atari, MaxAndSkipEnv, FrameStack
from baselines import bench
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from realtime_env import RealtimeEnv
import numpy as np

def make_env(dying_penalty = 0):
    env = gym.make('GradiusIiiDeterministic-v0')
    env = EpisodicWrapper(env)
    env = StateLoader(env, path = 'states/')
    env = WrapFrame(env)
    env = MaxAndSkipEnv(env, skip=2)
    env = FrameStack(env, 4)
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
            env = MaxAndSkipEnv(env, skip=2)
            # env = AutoShootWrapper(env)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def make_realtime_env_with_eval(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_eval_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            # env = EpisodicWrapper(env)
            env = WrapFrame(env)
            env = MaxAndSkipEnv(env, skip=2)
            env = FrameStack(env, 4)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env, True
        return _thunk
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = EpisodicWrapper(env)
            env = StateLoader(env, path = 'states/')
            env = WrapFrame(env)
            env = MaxAndSkipEnv(env, skip=2)
            env = FrameStack(env, 4)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env, False
        return _thunk
    set_global_seeds(seed)
    return RealtimeEnv([make_eval_env(0)] + [make_env(i + start_index) for i in range(1, num_env)])

def make_realtime_env_with_one_state(env_id, seed, path = 'states/', wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored env with each have one state to load
    """
    states = []
    # state should not contain extension
    for fn in listdir(path): 
        if '.' not in fn:
            states.append(path + fn)
            print(path + fn)
    print('Loadable states ' + str(listdir(path)))
    # appending none which represent just reset the env
    # states.append(None)
    num_env = len(states)
    
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank, state_path): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = OneStateLoader(env, state_path)
            env = EpisodicWrapper(env)
            env = WrapFrame(env)
            # env = wrap_deepmind(env, episode_life = False, clip_rewards = False, frame_stack = True)
            # env = MaxAndSkipEnv(env, skip=4)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    set_global_seeds(seed)
    return RealtimeEnv([make_env(i + start_index, state) for (i, state) in zip(range(num_env), states)])

