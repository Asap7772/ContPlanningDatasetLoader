import numpy as np
from dummy_env_carla import DummyEnvCarla

def load_buffer(path, buffer_size=1E5, expl_env=DummyEnvCarla(), observation_key='state', rew_type='zero', all_zeros=True):
    data = np.load(path, allow_pickle=True)

    from railrl.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
    replay_buffer = ObsDictReplayBuffer(
        buffer_size,
        expl_env,
        observation_key=observation_key,
    )
    
    discount = 1-1/len(data[0]['ego_state'])  
    for dct in data:
        ego_state, actor_state = np.array(dct['ego_state'])[:,-1,:], np.array(dct['actor_state'])[:,-1,:]
        ego_state_next, actor_state_next = np.array(dct['ego_state_next'])[:,0,:], np.array(dct['actor_state_next'])[:,0,:]
        ego_state, actor_state, ego_state_next, actor_state_next = ego_state[:-1], actor_state[:-1], ego_state_next[:-1], actor_state_next[:-1] #Truncate Last

        ego_yaw, actor_yaw = np.array(dct['ego_yaw'])[None].T, np.array(dct['actor_yaw'])[None].T
        ego_yaw, actor_yaw, ego_yaw_next, actor_yaw_next = ego_yaw[:-1], actor_yaw[:-1], ego_yaw[1:], actor_yaw[1:]

        state = np.hstack((ego_state, ego_yaw, actor_state, actor_yaw))
        next_state = np.hstack((ego_state_next,ego_yaw_next, actor_state_next, actor_yaw_next))

        actions = ego_state_next - ego_state

        path = dict(
            observations=dict(state=state),
            actions=actions,
            rewards=getRewFunc(dct, rew_type=rew_type)[:-1],
            next_observations=dict(state=next_state),
            terminals = getTerminals(dct, all_zeros=all_zeros)[:-1],
        )

        replay_buffer.add_path(path)
    return replay_buffer, discount #discount factor

def getRewFunc(dct, rew_type=None):
    ego_state, actor_state = np.array(dct['ego_state'])[:,-1,:], np.array(dct['actor_state'])[:,-1,:]
    ego_state_next, actor_state_next = np.array(dct['ego_state_next'])[:,0,:], np.array(dct['actor_state_next'])[:,0,:]
    actions = ego_state_next - ego_state

    if rew_type == 'distance':
        return np.linalg.norm(ego_state_next - actor_state_next, 1, axis=1, keepdims=True) #l1 norm of the positions
    elif rew_type == 'graphnav':
        assert False
    elif rew_type == 'zero':
        return np.zeros((ego_state.shape[0],1))
    else:
        assert False

def getTerminals(dct, all_zeros=False):
    terminals = np.zeros((len(dct['ego_state']),1))
    if not all_zeros:
        terminals[-1] = 1
    return terminals


if __name__ == '__main__':
    buffer, discount = load_buffer('/nfs/kun1/users/asap7772/Dataset/left_turn_train.npy', rew_type='distance')
    import ipdb; ipdb.set_trace()
