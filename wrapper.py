import gym
import numpy as np
from skimage.io import imsave
import cv2
from gym import spaces

class SurviveEnv(gym.RewardWrapper):
    def _reward(self, reward):
        return 1

class EpisodicWrapper(gym.Wrapper):
    
    def __init__(self, env, dying_penalty = 0):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.dying_penalty = dying_penalty

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        # self.env.get_ram()
        lives = info['gradius_lives']
        if lives < self.lives:
            done = True
            # if die reduce the reward
            reward -= self.dying_penalty
        self.lives = lives
        return obs, reward, done, info
    
    # TODO Recheck logic!
    def _reset(self, **kwargs):
        if self.lives <= 0:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        return obs

class StateSaver(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.step_count = 0
        self.step_count_max = 0
        self.saved = False
        self.saved_step = 0
        self.load_chance = 0.8
        self.save_count = 0

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.step_count = self.step_count + 1
        if not done and self.step_count % 300 == 0:
            # print('saving')
            self.step_count_max = self.step_count
            self.env.unwrapped.rle.saveState()
            self.save_count =  self.save_count + 1
            self.saved = True
            self.saved_step = self.step_count
        return obs, reward, done, info

    def _reset(self, **kwargs):
        load = np.random.random() < self.load_chance
        if self.save_count > 0 and load:
            # print('loading')
            self.step_count = self.saved_step
            self.env.unwrapped.rle.loadState()
            self.save_count = self.save_count - 1
            self.saved = False
        else:
            self.step_count = 0
            obs = self.env.reset(**kwargs)
        obs, _, _, _ = self.env.step(0)
        return obs

class StateSaver2(gym.Wrapper):
    def __init__(self, env, load_chance = 0.5):
        gym.Wrapper.__init__(self, env)
        self.load_chance = load_chance
        self.ever_reset = False

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        # self.step_count = self.step_count + 1
        # print(self.step_count)
        # if not done and self.step_count == 730:
        #     self.env.unwrapped.rle.saveStateToFile("./states/%d.txt" % self.step_count)
        #     self.step_count_max = self.step_count
        #     self.env.unwrapped.rle.saveState()
        #     self.save_count =  self.save_count + 1
        #     self.saved = True
        #     self.saved_step = self.step_count
        return obs, reward, done, info

    def _reset(self, **kwargs):
        load = np.random.random() < self.load_chance
        if load:
            #print('loading')
            if not self.ever_reset:
                obs = self.env.reset(**kwargs)    
            self.env.unwrapped.rle.loadStateFromFile("./states/730.txt")
        else:
            obs = self.env.reset(**kwargs)
        obs, _, _, _ = self.env.step(0)
        return obs

class WrapFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1))

    def _observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        a = frame.reshape((self.width, self.height))
        return frame[:, :, None]

class ObservationSaver(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lastObservation = None
        self.step_count = 0

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.lastObservation is not None:
            im = self.lastObservation.reshape((obs.shape[0], obs.shape[1]))
            imsave("observations/{}.png".format(self.step_count), im)
        self.lastObservation = obs
        self.step_count = self.step_count + 1
        return obs, reward, done, info

    def _reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, _, _ = self.env.step(0)
        self.lastObservation = obs
        return obs    

class EndingSaver(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        print(env.observation_space.shape)
        self.height, self.width, _ = env.observation_space.shape
        self.stack_size = 50
        self.lastObservation = np.zeros((self.height, self.width * self.stack_size), dtype = np.uint8)
        self.episode_count = 0

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.lastObservation[:,0:self.width*(self.stack_size-1)] = self.lastObservation[:,self.width:self.width*self.stack_size]
        self.lastObservation[:,self.width*(self.stack_size-1):self.width*self.stack_size] = obs.reshape((self.height, self.width)) 
        return obs, reward, done, info

    def _reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, _, _ = self.env.step(0)
        imsave("observations/Ending_%06d.png" % self.episode_count, self.lastObservation[:,self.stack_size*10:self.stack_size*20])
        self.lastObservation = np.zeros((self.height, self.width * self.stack_size), dtype = np.uint8)
        self.episode_count = self.episode_count + 1
        return obs   



class WrapAndSaveFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.step_count = 0
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1))

    def _observation(self, frame):
        imsave("observations/{}-rgb.png".format(self.step_count), frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        print(frame.shape)
        imsave("observations/{}-grey.png".format(self.step_count), frame)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        imsave("observations/{}-resize.png".format(self.step_count), frame)
        a = frame.reshape((self.width, self.height))
        self.step_count += 1
        return frame[:, :, None]
    
class AutoShootWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.action_space = spaces.Discrete(5)
        self.action_map = [2, 4, 6, 8, 14]

    def _step(self, action):
        action = self.action_map[action]
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info