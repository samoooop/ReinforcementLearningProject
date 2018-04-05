import gym
import gym_rle
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common.atari_wrappers import make_atari, MaxAndSkipEnv

from baselines import deepq
from wrapper import *


def main():
    env = gym.make('GradiusIiiDeterministic-v0')
    print(env.action_space)
    # env = StateSaver2(env, load_chance = 0.5)
    # env = EpisodicWrapper(env)
    env = wrap_deepmind(env, episode_life = False, clip_rewards = False, frame_stack = True)
    env = MaxAndSkipEnv(env, skip=4)
    # env = bench.Monitor(env, logger.get_dir())
    
    # env = StateSaver2(env, load_chance = 0.0)
    # env = wrap_deepmind(env, episode_life = False, clip_rewards = False, frame_stack = True)
    # env = MaxAndSkipEnv(env, skip=4)
    act = deepq.load("10mstep4stack7261random.pkl")
    sum_rew = 0
    for i in range(100):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)
        sum_rew += episode_rew
    print(sum_rew)

if __name__ == '__main__':
    main()
