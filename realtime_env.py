import numpy as np
from multiprocessing import Process, Pipe, Array
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from time import sleep

def worker(remote, parent_remote, env_fn_wrapper, ready, rank):
    np.random.seed(rank * 1000)
    parent_remote.close()
    env, is_eval = env_fn_wrapper.x()
    # maybe it shouldn't be None
    ob = None
    total_rew = 0
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            new_ob, reward, done, info = env.step(data)
            if is_eval:
                env.render()
            if done:
                new_ob = env.reset()
            ready[rank] = 1
            total_rew += reward
            remote.send((ob, data, reward, new_ob, done, total_rew, is_eval))
            total_rew = 0 if done else total_rew
            ob = new_ob
            
        elif cmd == 'reset':
            total_rew = 0
            ob = env.reset()
            remote.send(ob)
            
        elif cmd == 'reset_task':
            total_rew = 0
            ob = env.reset_task()
            remote.send(ob)
            
        elif cmd == 'close':
            remote.close()
            break
            
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
            
        else:
            raise NotImplementedError

class RealtimeEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.tick_rate = 200
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.expecting_actions = np.ones((self.nenvs,), dtype = np.bool)
        self.ready = Array('b', np.ones(nenvs, dtype = np.int))
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn), self.ready, rank))
            for (rank, work_remote, remote, env_fn) in zip(range(nenvs), self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        # only send actions to envs that expecting an action
        c = 0
        for i, remote, expecting_action in zip(range(self.nenvs)
                                                       , self.remotes
                                                       , self.expecting_actions):
            if expecting_action:
                self.ready[i] = False
                self.expecting_actions[i] = False
                remote.send(('step', actions[c]))
                c += 1
        self.waiting = True

    def step_wait(self):
        results = []
        while len(results) == 0: # no point to return nothing
            sleep(1./self.tick_rate)
            for i, remote in zip(range(self.nenvs), self.remotes):
                if self.ready[i]:
                    self.expecting_actions[i] = True
                    results.append(remote.recv())
        self.waiting = False
        return results
        # obs, rews, dones, infos = zip(*results)
        #return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True