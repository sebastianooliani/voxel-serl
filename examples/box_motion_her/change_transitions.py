# use this script to quickly evaluate how do the weights affect the reward function
# eventually, we can use this script to change the transitions of the demos
# without having to re-record them

from ur_env.utils.her import HER
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pandas as pd

def convert_pose_2_7dim(pose):
    return np.concatenate([pose[:3], 
                            R.from_mrp(pose[3:6]).as_quat(), 
                            pose[6:9], 
                            R.from_mrp(pose[9:]).as_quat()], 
                            axis=0)

if __name__ == '__main__':
    weights = {
            "step_weight": 0.1,
            "action_weight": 0.1,
            "orientation_weight": 10.,
            "position_weight": 15.,
            "distance_weight": 0.1,
            "grasping_weight": 0.5,
            "suction_weight": 0.5,
            "goal_weight": 20.,
            "success_weight": 200.,
            "penalty": 10,
            "safety_threshold": 0.13,
        }
    her = HER(scale=True,
              trans=True,
              weights_dict=weights)

    # transitions path
    path = '/home/sebastiano/voxel-serl/examples/box_motion_her/dual_20_her_transitions_2025-02-18_16-24-56.pkl'

    with open(path, 'rb') as f:
        transitions = pickle.load(f)

    breakpoint()

    new_transition = {}
    new_transitions = []

    for i, transition in enumerate(transitions):
        if i == 0:
            reset_pose = transition['observations'][39:51]
            reset_pose = convert_pose_2_7dim(reset_pose)

        # if i > 100: break

        new_transition["observations"] = transition["observations"].copy()
        new_transition["actions"] = transition["actions"].copy()
        new_transition["next_observations"] = transition["next_observations"].copy()
        new_transition["rewards"] = her.compute_reward_her(obs=transition['observations'].copy(), 
                            action=transition['actions'].copy(), 
                            goal_position=transition['observations'][-3:].copy(),
                            reset_pose=reset_pose,
                            )
        new_transition["masks"] = transition["masks"]
        new_transition["dones"] = transition["dones"]

        new_transitions.append(new_transition)
        new_transition = {}

    df = pd.DataFrame.from_dict(new_transitions)

    if len(new_transitions) == len(transitions):
        with open(f"mod_{path.split('/')[-1]}", 'wb') as f:
            pickle.dump(new_transitions, f)
            print(f"saved new transitions to mod_{path.split('/')[-1]}")

    plt.figure()
    plt.plot(df['rewards'])
    plt.savefig("test.png")