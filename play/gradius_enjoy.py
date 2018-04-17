import gym
import gym_rle
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common.atari_wrappers import make_atari, MaxAndSkipEnv

from baselines import deepq
# from wrapper import *
from multiprocessing import Process,Manager

from pynput import keyboard
from pynput.keyboard import Key
import time

from wrapper import *

import sys
from os.path import exists
import numpy as np
from skimage.io import imsave

manager = Manager()
keys = manager.dict()
keys['a'] = False

# keys = {}
keys['a'] = False
keys['b'] = False
keys['l'] = False
keys['r'] = False
keys['u'] = False
keys['d'] = False
keys['enter'] = False

kk = 'c'


def on_press(key):
    try:
        k = key.char
        if(k == 'a'): keys['a'] = True
        if(k == 'b'): keys['b'] = True
    except AttributeError:
        if(key == Key.left): keys['l'] = True
        if(key == Key.right): keys['r'] = True
        if(key == Key.up): keys['u'] = True
        if(key == Key.down): keys['d'] = True
        if(key == key.enter): keys['enter'] = True

def on_release(key):
    try:
        k = key.char
        if(k == 'a'): keys['a'] = False
        if(k == 'b'): keys['b'] = False
    except AttributeError:
        if(key == Key.left): keys['l'] = False
        if(key == Key.right): keys['r'] = False
        if(key == Key.up): keys['u'] = False
        if(key == Key.down): keys['d'] = False
        if(key == key.enter): keys['enter'] = False

        


def get_current_action():
    action = 19
    # if keys['b']: actions = 0
    # if keys['a']: actions = 1
    # if keys['a'] and keys['b']: action = 2
    # if keys['r']: action = 3
    # if keys['r'] and keys['a']: action = 4
    # if keys['l']: action = 5
    # if keys['l'] and keys['a']: action = 6
    # if keys['d']: action = 7
    # if keys['d'] and keys['a']: action = 8
    # if keys['d'] and keys['r']: action = 9
    # if keys['d'] and keys['r'] and key['a']: action = 10
    # if keys['d'] and keys['l']: action = 11
    # if keys['d'] and keys['l'] and key['a']: action = 12
    if keys['a']: return 1
    if keys['b']: return 0
    if keys['l']: return 6
    if keys['r']: return 4
    if keys['u']: return 14
    if keys['d']: return 8
    return 2

def save_state(env, step, obs):
    t = 0
    file_name = "./states/%d" % step
    while exists(file_name):
        t+=1
    file_name = file_name + '-' + str(t)
    env.unwrapped.rle.saveStateToFile(file_name)
    im = obs
    imsave(file_name + '.png', im)
    print('saved to', file_name)
    
def main():
    env = gym.make('GradiusIii-v0')
    action_set = env.unwrapped.get_action_meanings()
    for i, a in zip(range(len(action_set)), action_set):
        print(i, a)

    # env = WrapFrame(env)
    # env = StateLoader(env)
    # env = StateSaver2(env, load_chance = 1.0)
    # env = EpisodicWrapper(env)
    #env = wrap_deepmind(env, episode_life = False, clip_rewards = False, frame_stack = True)
    #env = MaxAndSkipEnv(env, skip=4)
    # env = bench.Monitor(env, logger.get_dir())
    
    # env = StateSaver2(env, load_chance = 0.0)
    # env = wrap_deepmind(env, episode_life = False, clip_rewards = False, frame_stack = True)
    # env = MaxAndSkipEnv(env, skip=4)
    #act = deepq.load("10mstep4stack7261random.pkl")
    fps = 30
    sum_rew = 0
    for i in range(100):
        obs, done = env.reset(), False
        print(sys.getsizeof(obs), obs.shape, type(obs), obs.nbytes)
        episode_rew = 0
        step = 0
        while not done:
            env.render(mode = 'human')
            a = get_current_action()
            obs, rew, done, _ = env.step(a)
            if keys['enter']:
                save_state(env, step, obs)
            episode_rew += rew
            time.sleep(1./fps)
            step += 1
        print("Episode reward", episode_rew)
        sum_rew += episode_rew
    print(sum_rew)
    
    
if __name__ == '__main__':
    d = manager.dict()
    d['a'] = False
    #t = Process(target = main, args = (d,))
    # t.start()
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    #t.join()
    main()
