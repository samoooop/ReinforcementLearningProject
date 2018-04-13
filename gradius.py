import gym
import gym_rle
from wrapper import *
from utils import make_env, make_subproc_env

from baselines.common.atari_wrappers import wrap_deepmind
from baselines import deepq
from simple import learn
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari, MaxAndSkipEnv

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(2e6))
    args = parser.parse_args()
    save_dir = './logs/2skipthen4stack_2m_paramnoise_2xmodel/'
    logger.configure(dir = save_dir)
    set_global_seeds(args.seed)
    # env = make_env(dying_penalty = 0)
    env = make_env()
    print(logger.get_dir())
    
    # env = MaxAndSkipEnv(env, skip = 3)
    model = deepq.models.cnn_to_mlp(
        convs=[(128, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512],
        dueling=bool(args.dueling),
    )
    act = learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=100000,
        exploration_fraction=0.6,
        exploration_final_eps=0.01,
        train_freq=1,
        param_noise=True,
        learning_starts=10000,
        target_network_update_freq=5000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized)
    )
    act.save(save_dir + "gradius_model.pkl") 
    env.close()


if __name__ == '__main__':
    main()
