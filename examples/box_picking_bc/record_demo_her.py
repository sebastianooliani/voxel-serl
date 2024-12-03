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

sys.path.append("../../serl_robot_infra")
from ur_env.envs.wrappers import SpacemouseIntervention, Quat2MrpWrapper, DualQuat2MrpWrapper, TwoSpacemiceIntervention
from serl_launcher.wrappers.serl_obs_wrappers import SerlObsWrapperNoImages
from serl_launcher.wrappers.chunking import ChunkingWrapper
from ur_env.utils.sample_3d_points import sample_points_in_intersecting_boxes

from gymnasium.wrappers import TransformReward
from ur_env.envs.relative_env import RelativeFrame, DualRelativeFrame

exit_program = threading.Event()


def on_space(key, info_dict):
    if key == keyboard.Key.space:
        for key, item in info_dict.items():
            print(f'{key}:  {item}', end='   ')
        print()


def on_esc(key):
    if key == keyboard.Key.esc:
        exit_program.set()

DUAL = True

if __name__ == "__main__":
    env = gym.make("box_picking_camera_env_dual_robot",
                   camera_mode="none", her=True) if DUAL else gym.make("box_picking_camera_env", camera_mode="rgb")
    
    DUAL_SPACEMOUSE = env.env.env.env.config.DUAL
    HER = env.env.env.env.config.HER

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
    success_needed = 5 if not DUAL_SPACEMOUSE else 10
    total_count = 0
    pbar = tqdm(total=success_needed)

    info_dict = {'state': env.unwrapped.curr_pos, 'gripper_state': env.unwrapped.gripper_state,
                 'force': env.unwrapped.curr_force}
    listener_1 = keyboard.Listener(daemon=True, on_press=lambda event: on_space(event, info_dict=info_dict))
    listener_1.start()

    listener_2 = keyboard.Listener(on_press=on_esc, daemon=True)
    listener_2.start()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"ur5_test_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    try:
        iter = 0
        T = env.env.env.env.config.T_O1_O2
        
        # Example boxes
        box1_min = env.env.env.env.config.ABS_POSE_LIMIT_LOW_ROBOT_1[:3]
        box1_max = env.env.env.env.config.ABS_POSE_LIMIT_HIGH_ROBOT_1[:3]
        box2_min = env.env.env.env.config.ABS_POSE_LIMIT_LOW_ROBOT_2[:3]
        box2_max = env.env.env.env.config.ABS_POSE_LIMIT_HIGH_ROBOT_2[:3]

        # Evaluate box limits in the correct reference frame
        box2_min = T @ box2_min
        box2_max = T @ box2_max

        # Sample points in the intersection
        intersection_points = sample_points_in_intersecting_boxes(
            box1_min, box1_max, box2_min, box2_max, 20
        )

        num_points = intersection_points.shape[0]

        while iter < num_points:
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

                def reached_goal_state_her(obs, box_position) -> bool:
                    state = obs["state"]
                    box_pos = state["box_position"]
                    goal_pos = box_position
                    return np.linalg.norm(goal_pos - box_pos) < 0.05 and 0.1 < state['gripper_state'][0] < 1. and 0.1 < state['gripper_state'][2] < 1.

                # HER transitions
                for i, trans in enumerate(transitions):
                    her_transitions.append(
                        dict(
                            observations=np.concatenate([trans['observations'], last_obs], axis=0),
                            actions=trans['actions'],
                            next_observations=np.concatenate([trans['next_observations'], last_obs], axis=0),
                            # compute reward based on the new goal state
                            rewards=reached_goal_state_her(box_position=last_obs['state']['box_position'], obs=trans['observations']), # TODO: implement this function
                            masks=trans['masks'],
                            dones=trans['dones'],
                        )
                    )
                    augmented_transitions.append(
                        dict(
                            observations=np.concatenate([trans['observations'], intersection_points[iter]], axis=0),
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
                
                success_count += int(rew > 0.99)
                total_count += 1
                print(
                    f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
                )
                pbar.update(int(rew > 0.99))
                obs, _ = env.reset()

        with open(file_path, "wb") as f:
            augmented_transitions.extend(her_transitions)
            pkl.dump(augmented_transitions, f)
            print(f"saved {success_needed} demos to {file_path}")

    except KeyboardInterrupt as e:
        print(f'\nProgram was interrupted from keyboard, cleaning up...  ', e.__str__())

    except ValueError as e:
        print(f'\nValue Error! Program was interrupted, cleaning up...  ', e.__str__())

    finally:
        if 'pbar' in locals() and not pbar.disable:
            pbar.close()
        env.close()
        env.env.env.env.env.close()
        print("Environment closed.")
        listener_1.stop()
        listener_2.stop()
        print("Program ended.")