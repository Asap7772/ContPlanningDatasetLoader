import numpy as np
from collections import defaultdict

# def add_next_yaw(path, outpath):
#     data = np.load(path,allow_pickle=True)

#     for i in range(len(data)):
#         import ipdb; ipdb.set_trace()

def process(path, outpath):
    data = np.load(path,allow_pickle=True)

    SCALE_VAL = 3
    d = 10

    out = []
    for i in range(len(data)):
        if i % 10 == 0: print(i)
        curr_traj = defaultdict(list)
        prev_vel = 0
        for j in range(len(data[i]['ego_state'])-1):
            curr_pos = data[i]['ego_state'][j][-1]
            next_pos = data[i]['ego_state_next'][j][0]
            curr_yaw = data[i]['ego_yaw'][j]
            next_yaw = data[i]['ego_yaw'][j+1]

            actor_pos = data[i]['actor_state'][j][-1]
            actor_next_pos = data[i]['actor_state_next'][j][0]
            actor_yaw = data[i]['actor_yaw'][j]
            actor_next_yaw = data[i]['actor_yaw'][j+1]

            curr_yaw_rad = np.deg2rad(curr_yaw.item())

            diff = next_pos - curr_pos

            v = np.array([np.cos(curr_yaw_rad), np.sin(curr_yaw_rad)])
            w = diff/np.linalg.norm(diff)

            scaled_v = np.linalg.norm(diff) * v


            dot_prod = np.dot(v,w)
            angle = np.arccos(dot_prod)

            if scaled_v[0] < diff[0]:
                angle = -angle

            angle = np.clip(angle,-1,1)
            vel = np.linalg.norm(diff) * SCALE_VAL

            curr_traj['states'].append(np.array([curr_pos[0], curr_pos[1], curr_yaw, prev_vel, actor_pos[0],actor_pos[1],actor_yaw]))
            curr_traj['actions'].append(np.array((angle, vel)))
            curr_traj['next_state'].append(np.array([next_pos[0],next_pos[1], next_yaw, vel, actor_next_pos[0], actor_next_pos[1],actor_next_yaw]))
            curr_traj['rewards'].append(np.linalg.norm(curr_pos-actor_pos))
            curr_traj['terminals'].append(0)
            prev_vel = vel
        out.append(dict(curr_traj))
    print('saved', outpath)
    np.save(outpath, out)

# process('/nfs/kun1/users/asap7772/Dataset/left_turn_train.npy', '/nfs/kun1/users/asap7772/Dataset/multicar_left_turn_train.npy')
# process('/nfs/kun1/users/asap7772/Dataset/right_turn_train.npy', '/nfs/kun1/users/asap7772/Dataset/multicar_right_turn_train.npy')
process('/nfs/kun1/users/asap7772/Dataset/overtake_train.npy', '/nfs/kun1/users/asap7772/Dataset/multicar_overtake_train.npy')

