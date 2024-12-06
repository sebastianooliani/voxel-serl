import os
import datetime
import threading
import numpy as np
import copy
import pickle as pkl
from tqdm import tqdm
import gymnasium as gym
from pprint import pprint
from pynput import keyboard
import sys
import math
from scipy.spatial.transform import Rotation as R

sys.path.append("../../serl_robot_infra")
from ur_env.envs.wrappers import SpacemouseIntervention, Quat2MrpWrapper, DualQuat2MrpWrapper, TwoSpacemiceIntervention
from serl_launcher.wrappers.serl_obs_wrappers import SerlObsWrapperNoImages
from serl_launcher.wrappers.chunking import ChunkingWrapper
from ur_env.utils.sample_3d_points import sample_points_in_intersecting_boxes

from gymnasium.wrappers import TransformReward
from ur_env.envs.relative_env import RelativeFrame, DualRelativeFrame

from franka_env.utils.transformations import (
    pose_2_homogeneous_matrix,
    construct_homogeneous_matrix
)
from fast_kinematics import FastKinematics

exit_program = threading.Event()


def on_space(key, info_dict):
    if key == keyboard.Key.space:
        for key, item in info_dict.items():
            print(f'{key}:  {item}', end='   ')
        print()


def on_esc(key):
    if key == keyboard.Key.esc:
        exit_program.set()

############################################################################################################
#                         global variables to speed up HER computation                                     #
############################################################################################################
file_name="/home/sebastiano/voxel-serl/serl_robot_infra/robot_controllers/ur5.urdf"
link="ee_link"
N=1
robot_model = FastKinematics(file_name, N, link)
joint_positions = np.array([[- math.pi / 6. , -math.pi/2 + math.pi/24, math.pi/2 + math.pi/6, -math.pi/2 - math.pi/6 - math.pi/24, -math.pi/2, 0.,
                        math.pi + math.pi / 4, -math.pi/2 + math.pi/24, math.pi/2 + math.pi/6, -math.pi/2 - math.pi/6 - math.pi/24, -math.pi/2, 0.]], dtype=np.float32)
# output of forward kinematics is position and quaternion
curr_reset_pose = np.concatenate([robot_model.forward_kinematics(joint_positions[0, :6].transpose()), robot_model.forward_kinematics(joint_positions[0, 6:].transpose())], axis=0)
# curr_reset_pose = np.concatenate([curr_reset_pose[:3], (R.from_quat(curr_reset_pose[3:7])).as_mrp(), curr_reset_pose[7:10], (R.from_quat(curr_reset_pose[10:])).as_mrp()], axis=0)

############################################################################################################
#                                        HER: her reward computation                                       #
############################################################################################################
def compute_reward_her(obs, 
                   action, 
                   goal_position,
                   last_action=np.zeros((14,)),
                   T_O1_O2=np.array([[0., 1., 0., -0.945], 
                        [-1., 0., 0., -0.], 
                        [0., 0., 1., 0.01], 
                        [0., 0., 0., 1.]]),
                    T_EE_SC=np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., 0.130],
                        [0., 0., 0., 1.]]),
                    ) -> float:
        
        def convert_pose_2_7dim(pose):
            return np.concatenate([pose[:3], (R.from_mrp(pose[3:6])).as_quat(), pose[6:9], (R.from_mrp(pose[9:])).as_quat()], axis=0)
        
        def reached_goal_state_her(obs, goal_position) -> bool:
            return np.linalg.norm(goal_position - obs[69:72]) < 0.05 and 0.1 < obs[14:18][0] < 1. and 0.1 < obs[14:18][2] < 1.

        tcp_pose = obs[39:51]
        tcp_pose = convert_pose_2_7dim(tcp_pose)

        action_cost = 0.1 * np.sum(np.power(action, 2))
        action_diff_cost = 0.1 * np.sum(np.power(obs[:14] - last_action, 2))
        last_action[:] = action
        
        # STEP: penalize each step
        step_cost = 0.1

        # SUCTION: reward for successful grip and cost for unnecessary suctioning
        suction_reward = 5 * 0.3 * (float(obs[14:18][1] > 0.5) + float(obs[14:18][3] > 0.5))
        suction_cost = 0.5 * 3. * (float(obs[14:18][1] < -0.5) + float(obs[14:18][3] < -0.5))

        # ORIENTATION: penalize deviating too much from the starting pose
        orientation_cost = 0
        orientation_cost = 0.5 - sum(tcp_pose[3:7] * curr_reset_pose[3:7]) ** 2
        orientation_cost += 0.5 - sum(tcp_pose[10:] * curr_reset_pose[10:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * 25.

        # POSITION: penalize deviating too much from the starting pose
        max_pose_diff = 0.05  # set to 5cm
        pos_diff = np.concatenate([tcp_pose[:2] - curr_reset_pose[:2], tcp_pose[7:9] - curr_reset_pose[7:9]])
        position_cost = 10. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0)
        )

        # 3D DISTANCE: penalize the distance between the two robots' end-effectors
        # TODO: adjust reference frames and relative base positions
        T_O1_E1 = construct_homogeneous_matrix(tcp_pose[:7])
        T_O2_E2 = construct_homogeneous_matrix(tcp_pose[7:])
        T_O1_SC1 = T_O1_E1 @ T_EE_SC
        T_O1_SC2 = T_O1_O2 @ T_O2_E2 @ T_EE_SC
        distance_cost = 1. / np.linalg.norm(T_O1_SC1[:3, 3] - T_O1_SC2[:3, 3])
                
        if reached_goal_state_her(obs, goal_position):
            last_action[:] = 0.
            R_goal = 100.
            return R_goal - action_cost - orientation_cost - position_cost - action_diff_cost - distance_cost
        else:
            return 0. + suction_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost - action_diff_cost - distance_cost
        
############################################################################################################

DUAL = True

if __name__ == "__main__":
    env = gym.make("box_picking_camera_env_dual_robot_motion_planning",
                   camera_mode="none") if DUAL else gym.make("box_picking_camera_env", camera_mode="rgb")
    
    DUAL_SPACEMOUSE = env.env.env.env.config.DUAL
    HER = env.env.env.env.config.HER
    T = env.env.env.env.config.T_O1_O2
        
    # Example boxes
    box1_min = np.concatenate([env.env.env.env.config.ABS_POSE_LIMIT_LOW_ROBOT_1[:3], [1]])
    box1_max = np.concatenate([env.env.env.env.config.ABS_POSE_LIMIT_HIGH_ROBOT_1[:3], [1]])
    box2_min = np.concatenate([env.env.env.env.config.ABS_POSE_LIMIT_LOW_ROBOT_2[:3], [1]])
    box2_max = np.concatenate([env.env.env.env.config.ABS_POSE_LIMIT_HIGH_ROBOT_2[:3], [1]])

    env = TwoSpacemiceIntervention(env) if DUAL_SPACEMOUSE else SpacemouseIntervention(env)
    env = DualRelativeFrame(env) if DUAL_SPACEMOUSE else RelativeFrame(env)
    env = DualQuat2MrpWrapper(env) if DUAL_SPACEMOUSE else Quat2MrpWrapper(env)
    env = SerlObsWrapperNoImages(env)
    # env = TransformReward(env, lambda r: 10. * r)
    # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    transitions = []

    her_transitions = []
    augmented_transitions = []

    success_count = 0
    success_needed = 20
    total_count = 0
    pbar = tqdm(total=success_needed)

    info_dict = {'state': env.unwrapped.curr_pos, 'gripper_state': env.unwrapped.gripper_state,
                 'force': env.unwrapped.curr_force}
    listener_1 = keyboard.Listener(daemon=True, on_press=lambda event: on_space(event, info_dict=info_dict))
    listener_1.start()

    listener_2 = keyboard.Listener(on_press=on_esc, daemon=True)
    listener_2.start()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"ur5_test_{success_needed}_demos_{uuid}_her.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    try:
        iter = 0

        # Evaluate box limits in the correct reference frame
        box2_min = T @ box2_min
        box2_max = T @ box2_max

        # Sample points in the intersection
        intersection_points = sample_points_in_intersecting_boxes(
            box1_min[:3], box1_max[:3], box2_min[:3], box2_max[:3], 20
        )

        num_points = intersection_points.shape[0]

        while iter < num_points:
            # define goal position
            env.env.env.env.env.goal_position = intersection_points[iter]
            # print(f"Goal position: {intersection_points[iter]}")
            if exit_program.is_set():
                raise KeyboardInterrupt  # stop program, but clean up before
            
            action = np.zeros((14,)) if DUAL_SPACEMOUSE else np.zeros((7,))
            next_obs, rew, done, truncated, info = env.step(action=action)
            actions = info["intervene_action"]

            # Original transitions
            # Std experience replay
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                )
            )
            transitions.append(transition)
            # pprint(transition)

            obs = next_obs

            if done:
                last_obs = next_obs

                # HER transitions
                for trans in transitions:
                    her_transitions.append(
                        dict(
                            observations=np.concatenate([trans['observations'], last_obs], axis=0),
                            actions=trans['actions'],
                            next_observations=np.concatenate([trans['next_observations'], last_obs], axis=0),
                            # compute reward based on the new goal state
                            rewards=compute_reward_her(obs=trans['observations'],action=trans['actions'], goal_position=last_obs[-3:]), # TODO: implement this function
                            masks=trans['masks'],
                            dones=trans['dones'],
                        )
                    )
                    augmented_transitions.append(
                        dict(
                            observations=np.concatenate([trans['observations'], intersection_points[iter]], axis=0), # TODO: should I recompute the goal_box_position observation?
                            actions=trans['actions'],
                            next_observations=np.concatenate([trans['next_observations'], intersection_points[iter]], axis=0),
                            rewards=trans['rewards'],
                            masks=trans['masks'],
                            dones=trans['dones'],
                        )
                    )
                
                # Reset transitions
                transitions = []
                iter += 1
                
                total_count += 1
                print(
                    f"{rew}\tRecorded {iter}, {success_needed} needed."
                )
                pbar.update(1)
                obs, _ = env.reset()

        with open(file_path, "wb") as f:
            augmented_transitions.extend(her_transitions)
            pkl.dump(augmented_transitions, f)
            print(f"saved {success_needed} demos to {file_path}")

    except KeyboardInterrupt as e:
        print(f'\nProgram was interrupted from keyboard, cleaning up...  ', e.__str__())

    finally:
        if 'pbar' in locals() and not pbar.disable:
            pbar.close()
        env.close()
        print("Environment closed.")
        listener_1.stop()
        listener_2.stop()
        print("Program ended.")