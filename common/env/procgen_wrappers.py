from collections.abc import Callable, Iterable, Mapping
import contextlib
import os
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import gym
from gym import spaces
import time
from collections import deque
import torch
import cv2
from queue import Queue
import numba as nb
from numba import jit
from threading import Thread

"""
Copy-pasted from OpenAI to obviate dependency on Baselines. Required for vectorized environments.
"""

class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
    step_async() is called again.
    """

    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)


class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
    step_wait() is called.
    """

    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = "ARGHH" #tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

    
class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(num_envs=venv.num_envs,
                        observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.venv, name)

    
class VecEnvObservationWrapper(VecEnvWrapper):
    @abstractmethod
    def process(self, obs):
        pass

    def reset(self):
        obs = self.venv.reset()
        return self.process(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self.process(obs), rews, dones, infos

    
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

        
@contextlib.contextmanager
def clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing
    Processes.
    """
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)

        
class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs
    
class VecExtractDictObs(VecEnvObservationWrapper):
    def __init__(self, venv, key):
        self.key = key
        super().__init__(venv=venv,
            observation_space=venv.observation_space.spaces[self.key])

    def process(self, obs):
        return obs[self.key]
    
    
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

        
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for i in range(len(infos)):
            infos[i]['env_reward'] = rews[i]
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)


class TransposeFrame(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(venv=env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.float32)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs.transpose(0,3,1,2), reward, done, info

    def reset(self):
        obs = self.venv.reset()
        return obs.transpose(0,3,1,2)


class ScaledFloatFrame(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(venv=env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs/255.0, reward, done, info

    def reset(self):
        obs = self.venv.reset()
        return obs/255.0
    
@jit(nopython=True)
def get_feature_(img=np.array([[]], dtype=np.int8), solution=64, basic=False, num_max_color=20, color_map=np.array([], dtype=int), color_cnt=0):
    """
    Vectorized version to support numba
    """
    blob_list = nb.typed.List()
    idx_set = np.array([[row,col] for row in range(img.shape[0]) for col in range(img.shape[1])])   # [64*64, 2], storing position of each pixel, row first counting
    tile_idx_set = idx_set // solution                                                              # [64*64, 2] // solution, storing tile position for each pixel
    tile_min_x = solution * tile_idx_set[:,0]                                                       # [64*64, 1], storing tile_min_x for each pixel
    tile_max_x = tile_min_x + solution - 1                                                          # [64*64, 1], storing tile_max_x for each pixel
    tile_min_y = solution * tile_idx_set[:,1]                                                       # [64*64, 1], storing tile_min_y for each pixel
    tile_max_y = tile_min_y + solution - 1                                                          # [64*64, 1], storing tile_max_y for each pixel
    offsets = np.array([[0,1],[0,-1],[1,0],[-1,0]])                                                 # 4 direction offsets for BFS
    queue = np.zeros((solution*solution, 2))                                                        # vector-based queue with max length of `solution*solution`
    relative_feature = np.zeros((num_max_color, num_max_color, 64//solution*2-1, 64//solution*2-1), dtype=np.int8)
    basic_feature = np.zeros((num_max_color, 64//solution, 64//solution), dtype=np.int8) if basic else None
    head = 0                                                                                        # queue head
    tail = 0                                                                                        # queue tail
    for pixel_idx in range(64*64):                                                                  # For every pixel
        if img[idx_set[pixel_idx][0]][idx_set[pixel_idx][1]] != 0:                        #   if not background:
            color = img[idx_set[pixel_idx][0]][idx_set[pixel_idx][1]]
            if not (color_map == color).any():
                color_map[color_cnt] = color
                color_cnt += 1
            color = np.argwhere(color_map == color)[0][0]
            img[idx_set[pixel_idx][0]][idx_set[pixel_idx][1]] = 0                 
            for dir in range(4):
                x = int(idx_set[pixel_idx][0] + offsets[dir][0])
                y = int(idx_set[pixel_idx][1] + offsets[dir][1])
                if x>=tile_min_x[pixel_idx] and y>=tile_min_y[pixel_idx] and x<=tile_max_x[pixel_idx] and y<=tile_max_y[pixel_idx] and img[x][y]==color:
                    queue[head] = np.array([x,y])
                    tail = (tail + 1) % (solution*solution)
                    img[x][y] = 0
            while tail > head:
                position = queue[head]
                head = (head + 1) % (solution*solution)
                for dir in range(4):
                    x = int(position[0] + offsets[dir][0])
                    y = int(position[1] + offsets[dir][1])
                    if x>=tile_min_x[pixel_idx] and y>=tile_min_y[pixel_idx] and x<=tile_max_x[pixel_idx] and y<=tile_max_y[pixel_idx] and img[x][y]==color:
                        queue[tail] = np.array([x,y])
                        tail = (tail + 1) % (solution*solution)
                        img[x][y] = 0  
            blob_list.append((color, tile_idx_set[pixel_idx][0], tile_idx_set[pixel_idx][1]))
    for idx1 in range(len(blob_list)):
        for idx2 in range(idx1,len(blob_list)):
            color1 = blob_list[idx1][0]
            color2 = blob_list[idx2][0]
            row1 = blob_list[idx1][1]
            row2 = blob_list[idx2][1]
            col1 = blob_list[idx1][2]
            col2 = blob_list[idx2][2]
            relative_feature[color1][color2][row1 - row2 + 64//solution - 1][col1 - col2 + 64//solution - 1] = 1
            #TODO Debug
            # print((color1, color2, row1 - row2 + 64//solution - 1, col1 - col2 + 64//solution - 1))
    if basic:
        for blob in blob_list:
            basic_feature[blob[0]][blob[1]][blob[2]] = 1
    feature = np.concatenate((relative_feature.reshape(-1),basic_feature.reshape(-1))) if basic else relative_feature.reshape(-1)
    return feature, color_map, color_cnt

@jit(nopython=True)
def get_feature_withoutBFS_(img=np.array([[]], dtype=np.int8), solution=32, basic=False, num_max_color=20, color_map=np.array([], dtype=int), color_cnt=0):
    """
    Vectorized version to support numba
    """
    tile_min_x = 0
    tile_min_y = 0
    tile_max_x = solution
    tile_max_y = solution
    tile_col = 0
    tile_row = 0
    num_block = 64//solution
    relative_feature = np.zeros((num_max_color, num_max_color, num_block*2-1, num_block*2-1), dtype=np.int32)
    basic_feature = np.zeros((num_max_color, num_block, num_block), dtype=np.int32) if basic else None
    storage = np.zeros(((num_block)**2, num_max_color), dtype=np.int32) 
    storage_num = np.zeros((num_block)**2,dtype=np.int32)
    for tile in range((num_block)**2):
        patch = img[tile_min_x:tile_max_x, tile_min_y:tile_max_y].copy()
        # print(patch)
        patch_color = np.unique(patch.reshape(-1))[1:]  # 0 is recognized as background
        # print(patch_color)
        storage_num[tile] = patch_color.size
        # print(storage_num)
        for p_idx, p_color in enumerate(patch_color):
            if not (p_color == color_map).any():
                color_map[color_cnt] = p_color
                color_cnt += 1
            patch_color[p_idx] = np.argwhere(color_map == p_color)[0][0]
            storage[tile][p_idx] = patch_color[p_idx]
        if basic:
            for p_color in patch_color:
                basic_feature[p_color][tile_row][tile_col] = 1
        if tile_max_y < 64:
            tile_min_y += solution
            tile_max_y += solution
        else:
            tile_min_y = 0
            tile_max_y = solution
            tile_min_x += solution
            tile_max_x += solution
    for idx1 in range((num_block)**2):
        for idx2 in range(idx1,(num_block)**2):
            row1 = idx1 // (num_block)
            row2 = idx1 // (num_block)
            col1 = idx1 - row1 * (num_block)
            col2 = idx2 - row2 * (num_block)
            for color1 in storage[idx1][:storage_num[idx1]]:
                for color2 in storage[idx2][:storage_num[idx2]]:
                    relative_feature[color1][color2][row1 - row2 + num_block -1][col1 - col2 + num_block -1] = 1
    feature = np.concatenate((relative_feature.reshape(-1),basic_feature.reshape(-1))) if basic else relative_feature.reshape(-1)
    return feature, color_map, color_cnt
    
class TBlobPROS(VecEnvWrapper):
    def __init__(self, env, solution=8, basic=False, max_color_num=20):
        super().__init__(venv=env)
        self.solution = solution
        self.basic = basic
        self.max_color_num = max_color_num
        self.block_num = 64 // self.solution
        self.color_map = np.zeros(max_color_num, dtype=int)
        self.color_cnt = 0
        if basic:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=((self.block_num*2-1)*(self.block_num*2-1)*max_color_num*max_color_num+max_color_num*self.block_num*self.block_num,))
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=((self.block_num*2-1)*(self.block_num*2-1)*max_color_num*max_color_num,))
    
    def reset(self, **kwargs):
        vec_obs = self.venv.reset(**kwargs)
        vec_feature = self.get_vec_feature(vec_obs)
        return vec_feature
    
    def step_wait(self):
        vec_obs, reward, done, info = self.venv.step_wait()
        vec_feature = self.get_vec_feature(vec_obs)
        return vec_feature, reward, done, info
        
    def gray_observation(self, observation):
        image_gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return image_gray
    
    def get_feature(self, obs):
        feature, self.color_map, self.color_cnt = get_feature_withoutBFS_(self.gray_observation(obs),
                                                                solution=self.solution,
                                                                basic=self.basic,
                                                                num_max_color=self.max_color_num,
                                                                color_map=self.color_map,
                                                                color_cnt=self.color_cnt)
        return feature
    
    def get_vec_feature(self, vec_obs):
        features = []
        num_env = vec_obs.shape[0]
        thread_pool = [MyThread(self.get_feature, (vec_obs[i],)) for i in range(num_env)]
        for thread in thread_pool:
            thread.start()
        for thread in thread_pool:
            thread.join()
            features.append(thread.get_result().reshape(1,-1))
        return np.concatenate(features, axis=0)

class MyThread(Thread):
    def __init__(self, func, args) -> None:
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)
    
    def get_result(self):
        return self.result