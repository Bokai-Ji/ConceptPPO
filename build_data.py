import os
import cv2
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from queue import Queue

ENV_NAME = 'starpilot'
PATH = f"./dataset/{ENV_NAME}/data_set/"
entities_map = dict(pickle.load(open(f"./dataset/{ENV_NAME}/entities.pkl", "rb")))
concept_idx = dict(zip(set(list(entities_map.values())), list(range(len(entities_map)))))
NUM_CONCEPT = len(concept_idx)

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

imgs = []
concept_labels = []
action_labels = []
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
        print(f"[INFO] Entity selected: {frame['entities']}")
        entity_idx = 0
        concept_encoding = np.zeros(NUM_CONCEPT)
        concepts = []
        while entity_idx < len(frame['entities']):
            entity_id = frame['entities'][entity_idx]
            try:
                color = blob_list[entity_id]['color']
                concept = entities_map[color]
                concepts.append(concept)
                c_idx = concept_idx[concept]
                concept_encoding[c_idx] = 1
                entity_idx += 1
            except Exception as e:
                print(f"[ERROR] {e}")
        imgs.append(img)
        concept_labels.append(concept_encoding)
        action_labels.append(frame['action'])
        print(f'[INFO] Concepts selected: {concepts}')

data_set = list(zip(imgs, list(zip(concept_labels, action_labels))))
with open("./dataset/starpilot/data.pkl","wb") as f:
    pickle.dump(data_set, f)
    f.close()
# data_set = pd.DataFrame(list(zip(imgs,concept_labels,action_labels)), columns=['Observation', 'ConceptLabel', 'ActionLabel'])
# data_set.to_csv(PATH+f'{ENV_NAME}_dataset.csv',index=False)
print(f"Dataset built in {PATH+f'{ENV_NAME}_dataset.csv'}")