import gym
import gym_rle
from baselines.common.atari_wrappers import wrap_deepmind, FrameStack
from baselines.common.atari_wrappers import make_atari, MaxAndSkipEnv

from baselines import deepq
from wrapper import *

import time
import matplotlib.pyplot as plt

def main():
    env = gym.make('GradiusIiiDeterministic-v0')
    print(env.action_space)
    
    env = WrapFrame(env)
    env = MaxAndSkipEnv(env, skip=2)
    env = EndingSaver(env)
    # env = ObservationSaver(env)
    print(env.observation_space)
    # env = wrap_deepmind(env, episode_life = False, clip_rewards = False, frame_stack = True)
    act = deepq.load("saved_model/current_max_2m.pkl")
    sum_rew = 0
    neps = 5000
    fps = None
    for i in range(neps):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            if fps is not None:
                time.sleep(1./fps)
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
        sum_rew += episode_rew
    print(sum_rew / neps)	

if __name__ == '__main__':
    main()
