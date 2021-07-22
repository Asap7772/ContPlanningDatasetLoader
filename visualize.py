import numpy as np
path = '/home/asap7772/asap7772/Dataset/left_turn_train.npy'
# path = '/home/asap7772/asap7772/Dataset/overtake_train.npy'
data = np.load(path,allow_pickle=True)

traj = 0
curr_pos = np.array([data[traj]['ego_state'][j][-1] for j in range(len(data[traj]['ego_state']))])
import matplotlib.pyplot as plt

plt.plot(curr_pos[:,0],curr_pos[:,1])

plt.savefig('/home/asap7772/asap7772/Dataset/a.png')