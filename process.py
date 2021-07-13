import os
from collections import defaultdict
import json
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Process data')
parser.add_argument('--buffer', type=int, default=0)
args = parser.parse_args()

if args.buffer == 0:
    print('left turn train')
    path = '/nfs/kun1/users/asap7772/Dataset/Left Turn/train/'
    out_path = '/nfs/kun1/users/asap7772/Dataset/left_turn_train.npy'
elif args.buffer == 1:
    print('overtake train')
    path = '/nfs/kun1/users/asap7772/Dataset/Overtake/train'
    out_path = '/nfs/kun1/users/asap7772/Dataset/overtake_train.npy'
elif args.buffer == 2:
    print('right turn train')
    path = '/nfs/kun1/users/asap7772/Dataset/Right Turn/train'
    out_path = '/nfs/kun1/users/asap7772/Dataset/right_turn_train.npy'
else:
    assert False

map_keys = {
    'S_past_world_frame':'ego_state',
    'S_future_world_frame':'ego_state_next', 
    'yaws': 'ego_yaw',
    'agent_presence':'agent_presence',
    'overhead_features':'lidar',
    'light_strings': 'traffic_light',
    'A_future_world_frame':'actor_state',
    'A_past_world_frame':'actor_state_next',
    'A_yaws': 'actor_yaw',
}

out_lst = []
sorted_paths = sorted(os.listdir(path), key=lambda p: (int(p.split('_')[2]), int(p.split('_')[-1].split('.')[0])))
last_ep = int(sorted_paths[0].split('_')[2])
current_path = defaultdict(lambda:[])
i = 0
for p in sorted_paths:
    ep = int(p.split('_')[2])
    frame = int(p.split('_')[-1].split('.')[0])
    
    if ep != last_ep:
        print(last_ep, 'done')
        out_lst.append(dict(current_path))
        current_path = defaultdict(lambda:[])
    
    f = open(os.path.join(path, p),)
    data = json.load(f)
    for key in data:
        current_path[map_keys[key]].append(np.array(data[key]))
    current_path['action'] = current_path['ego_state_next'][-1][0]-current_path['ego_state'][-1][-1]
    last_ep = ep
    f.close()
    if i % 50 == 0:
        np.save(out_path, out_lst)
    i += 1
np.save(out_path, out_lst)

