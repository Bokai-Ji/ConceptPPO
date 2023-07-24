from queue import Queue
from ..procgen_wrappers import VecEnvWrapper
import numpy as np
import gym
import cv2
cimport numpy as cnp

DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t

class Blob_PROS(VecEnvWrapper):
    def __init__(self, env, int solution, bint mute_basic, int max_color_num):
        super().__init__(venv=env)
        self.solution = solution
        self.mute_basic = mute_basic
        self.max_color_num = max_color_num
        self.block_num = <int>(64 // self.solution)
        self.color_map = {}
        self.color_cnt = 0
        if not mute_basic:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=((self.block_num*2-1)*(self.block_num*2-1)*max_color_num*max_color_num+max_color_num*self.block_num*self.block_num,))
        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=((self.block_num*2-1)*(self.block_num*2-1)*max_color_num*max_color_num,))
    
    def reset(self, **kwargs):
        cdef cnp.ndarray[DTYPE_t, ndim=3] vec_obs = self.venv.reset(**kwargs)
        cdef cnp.ndarray[DTYPE_t, ndim=2] vec_feature = self.get_vec_feature(vec_obs)
        return vec_feature
    
    def step_wait(self):
        cdef cnp.ndarray[DTYPE_t, ndim=3] vec_obs
        cdef float reward
        cdef bint done
        cdef dict info
        vec_obs, reward, done, info = self.venv.step_wait()
        cdef cnp.ndarray[DTYPE_t, ndim=2] vec_feature = self.get_vec_feature(vec_obs)
        return vec_feature, reward, done, info
        
    def gray_observation(self, observation):
        cdef cnp.ndarray[DTYPE_t, ndim=2] image_gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return image_gray

    def BFS_tile_boarder(self, cnp.ndarray[DTYPE_t, ndim=2] img, int color, tuple start_point, dict tile_boarder):
        cdef cnp.ndarray[DTYPE_t, ndim=2] offsets = np.array([[0,1],[0,-1],[1,0],[-1,0]])
        img[start_point[0]][start_point[1]] = 0
        vis_queue = Queue()
        cdef int x
        cdef int y
        for dir in range(4):
            x = start_point[0] + offsets[dir][0]
            y = start_point[1] + offsets[dir][1]
            if x>=tile_boarder['min_x'] and y>=tile_boarder['min_y'] and x<=tile_boarder['max_x'] and y<=tile_boarder['max_y']:
                if img[x, y] == color:
                    vis_queue.append((x,y))
                    img[x, y] = 0
                    max_x = max(x,max_x)
                    max_y = max(y,max_y)
        while not vis_queue.empty():
            pos = vis_queue.get()
            for dir in range(4):
                x = pos[0] + offsets[dir][0]
                y = pos[1] + offsets[dir][1]
                if x>=tile_boarder['min_x'] and y>=tile_boarder['min_y'] and x<=tile_boarder['max_x'] and y<=tile_boarder['max_y']:
                    if img[x, y] == color:
                        vis_queue.put((x,y))
                        img[x, y] = 0
        return img

    def get_blob_list_seperate(self, img, tile_width=8, tile_height=8):
        blob_list = []
        cdef int tile_row
        cdef int tile_col
        cdef int tile_min_x
        cdef int tile_max_x
        cdef int tile_min_y
        cdef int tile_max_y
        cdef int color
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if img[row][col] != 0:
                    tile_row = row // tile_height
                    tile_col = col // tile_width
                    tile_min_x = tile_height * tile_row
                    tile_max_x = tile_min_x + tile_height - 1
                    tile_min_y = tile_width * tile_col
                    tile_max_y = tile_min_y + tile_width - 1
                    tile_boarder = {'min_x': tile_min_x,
                                    'max_x': tile_max_x,
                                    'min_y': tile_min_y,
                                    'max_y': tile_max_y}
                    color = img[row][col]
                    img= self.BFS_tile_boarder(img,color,(row,col),tile_boarder)
                    blob_list.append({'color': color,
                                        'tile_row': tile_row,
                                        'tile_col': tile_col})
        return blob_list
    
    def get_relative_feature(self, blob_list):
        cdef cnp.ndarray[DTYPE_t, ndim=4] relative_feature = np.zeros((self.max_color_num, self.max_color_num, self.block_num*2-1, self.block_num*2-1))
        cdef int offset1
        cdef int offset2
        for idx1 in range(len(blob_list)):
            for idx2 in range(idx1+1, len(blob_list)):
                color1 = blob_list[idx1]['color']
                color2 = blob_list[idx2]['color']
                offset1 = blob_list[idx1]['tile_row'] - blob_list[idx2]['tile_row'] + self.block_num - 1
                offset2 = blob_list[idx1]['tile_col'] - blob_list[idx2]['tile_col'] + self.block_num - 1
                if color1 in self.color_map.keys():
                    color1 = <int>self.color_map[color1]
                else:
                    self.color_map[color1] = self.color_cnt
                    color1 = <int>self.color_map[color1]
                    self.color_cnt += 1
                if color2 in self.color_map.keys():
                    color2 = <int>self.color_map[color2]
                else:
                    self.color_map[color2] = self.color_cnt
                    color2 = <int>self.color_map[color2]
                    self.color_cnt += 1
                relative_feature[color1, color2, offset1, offset2] = 1
        return relative_feature.reshape(-1)
    
    def get_basic_feature(self, blob_list):
        basic_feature = np.zeros((self.max_color_num, self.block_num, self.block_num))
        for blob in blob_list:
            color = blob['color']
            x = blob['tile_row']
            y = blob['tile_col']
            basic_feature[color][x][y] = 1
        return basic_feature.reshape(-1)
    
    def get_feature(self, blob_list):
        relative_feature = self.get_relative_feature(blob_list)
        if not self.mute_basic:
            basic_feature = self.get_basic_feature(blob_list)
            return np.concatenate([relative_feature,basic_feature])
        return relative_feature
    
    def get_vec_feature(self, vec_obs):
        env_num = vec_obs.shape[0]
        features = []
        for i in range(env_num):
            obs = vec_obs[i]
            blob_list = self.get_blob_list_seperate(self.gray_observation(obs))
            features.append(self.get_feature(blob_list).reshape(1,-1))
        return np.concatenate(features, axis=0)