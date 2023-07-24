import os
import cv2
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from procgen import ProcgenEnv
from queue import Queue

ENV_NAME = 'starpilot'
PATH = f"./dataset/{ENV_NAME}/data_set/"
ZOOM_SCALE = 8      #! Scale of zooming the render image, adjust to meet your requirment
FONT_SIZE = 0.5     #! Number font size, can be [0,1], ajust to meet your requirment
FONT_THICKNESS = 1  #! Thickness of numbers, can be [1,2,...], adjust to meet your requirment

def gray_observation(observation):
    image = observation
    image = image.reshape(64,64,3)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image_gray

def BFS_tile_boarder(img,need_to_visit,color,start_point,tile_boarder):
    pixel_count = 1
    max_x = start_point[0]
    max_y = start_point[1]
    offsets = [[0,1],[0,-1],[1,0],[-1,0]]
    need_to_visit[start_point[0]][start_point[1]] = False
    vis_queue = Queue()
    for dir in range(4):
        x = start_point[0] + offsets[dir][0]
        y = start_point[1] + offsets[dir][1]
        if x>=tile_boarder['min_x'] and y>=tile_boarder['min_y'] and x<=tile_boarder['max_x'] and y<=tile_boarder['max_y']:
            if img[x][y] == color and need_to_visit[x][y]:
                vis_queue.put((x,y))
                need_to_visit[x][y] = False
                max_x = max(x,max_x)
                max_y = max(y,max_y)
    while not vis_queue.empty():
        pos = vis_queue.get()
        pixel_count += 1
        for dir in range(4):
            x = pos[0] + offsets[dir][0]
            y = pos[1] + offsets[dir][1]
            if x>=tile_boarder['min_x'] and y>=tile_boarder['min_y'] and x<=tile_boarder['max_x'] and y<=tile_boarder['max_y']:
                if img[x][y] == color and need_to_visit[x][y]:
                    vis_queue.put((x,y))
                    need_to_visit[x][y] = False
                    max_x = max(x,max_x)
                    max_y = max(y,max_y)
    min_x = start_point[0]
    min_y = start_point[1]
    return min_x,max_x,min_y,max_y,pixel_count,need_to_visit

def get_blob_list_seperate(img,tile_width=8,tile_height=8):
    blob_list = []
    need_to_visit = np.array(list(map(lambda x:x>0,img)))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if need_to_visit[row][col]:
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
                min_x,max_x,min_y,max_y,pix_cnt,need_to_visit= BFS_tile_boarder(img,need_to_visit,color,(row,col),tile_boarder)
                if pix_cnt > 2:
                    center_x = (min_x + max_x) // 2
                    center_y = (min_y + max_y) // 2
                    blob_list.append({'color': color,
                                      'tile_row': tile_row,
                                      'tile_col': tile_col,
                                      'center': (center_x,center_y),
                                      'pixel_count': pix_cnt})
    return blob_list

actions = {
            0: ("LEFT", "DOWN"),
            1: ("LEFT",),
            2: ("LEFT", "UP"),
            3: ("DOWN",),
            4: (),
            5: ("UP",),
            6: ("RIGHT", "DOWN"),
            7: ("RIGHT",),
            8: ("RIGHT", "UP"),
            9: ("D",),
            10: ("A",),
            11: ("W",),
            12: ("S",),
            13: ("Q",),
            14: ("E",),
}

data_list = os.listdir(PATH)
print(f"[INFO] {data_list}")
for path in data_list:
    episode = None
    with open(PATH+path, "rb") as f:
        episode = pickle.load(f)
        f.close()
    print(f"[INFO] Done reading {path}.")
    for idx,frame in enumerate(episode):
        img = frame['feature'][0]
        blob_list = get_blob_list_seperate(gray_observation(img))    
        img = cv2.resize(img, (64*ZOOM_SCALE,64*ZOOM_SCALE))
        for idx_,blob in enumerate(blob_list):
            center = blob['center']
            img[center[0]*ZOOM_SCALE][center[1]*ZOOM_SCALE] = np.array([255,255,255])
            cv2.putText(img, str(idx_), (center[1]*ZOOM_SCALE, center[0]*ZOOM_SCALE), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (255, 0, 0), FONT_THICKNESS)
        # plt.imshow(img)
        # plt.title(f"Action: {actions[frame['action']]}")
        # plt.show(block=False)
        print(f"[INFO] Entity selected: {frame['entities']}")
        # plt.pause(0.01)
        # plt.clf()
        entity_idx = 0
        while entity_idx < len(frame['entities']):
            entity_id = frame['entities'][entity_idx]
            try:
                blob_list[entity_id] = 0
                entity_idx += 1
            except Exception as e:
                print(f"[ERROR] {e}")
                print(f"[INFO ] Target entity: {entity_id}")
                plt.imshow(img)
                plt.title(f"Action: {actions[frame['action']]}")
                plt.show(block=False)
                plt.pause(0.01)
                plt.clf()
                new_entity = input("Please select another entity id or delete by entering `x`: ")
                new_entity = eval(f"[{new_entity}]")
                frame['entities'].remove(entity_id)
                if new_entity[0] != 'x':
                    frame['entities'].extend(new_entity)
        # plt.imshow(img)
        # plt.show(block=False)
        # plt.pause(1)
        # plt.clf()
    with open(PATH+path, "wb") as f:
        pickle.dump(episode, f)
        f.close()
    print("Episode done.")