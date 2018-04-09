import gym
import gym_rle
from wrapper import *
from utils import make_env, make_subproc_env, make_realtime_env

from baselines.common.atari_wrappers import wrap_deepmind
from baselines import deepq
from deepq_mp import learn
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari, MaxAndSkipEnv
from realtime_env import RealtimeEnv


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(5e6))
    args = parser.parse_args()
    save_dir = './logs/4x_4skip_4stack-128batchsize-100kmem_stateloader/'
    logger.configure(dir = save_dir)
    set_global_seeds(args.seed)
    # env = make_env(dying_penalty = 0)
    env = make_realtime_env('GradiusIiiDeterministic-v0', 8, 0)
    print(logger.get_dir())
    
    # env = MaxAndSkipEnv(env, skip = 3)
    model = deepq.models.cnn_to_mlp(
        convs=[(64, 8, 4), (32, 4, 2), (32, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    act = learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=100000,
        exploration_fraction=0.4,
        exploration_final_eps=0.01,
        train_freq=1,
        batch_size=128,
        param_noise = True,
        learning_starts=10000,
        target_network_update_freq=5000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized)
    )
    act.save(save_dir + "gradius_model.pkl") 
    env.close()


if __name__ == '__main__':
    main()
