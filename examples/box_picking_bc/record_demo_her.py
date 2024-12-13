import os
import datetime
import threading
import numpy as np
import copy
import pickle as pkl
from tqdm import tqdm
import gymnasium as gym
from pynput import keyboard
import math
from scipy.spatial.transform import Rotation as R
from pprint import pprint

from ur_env.envs.wrappers import SpacemouseIntervention, Quat2MrpWrapper, DualQuat2MrpWrapper, TwoSpacemiceIntervention, SampleGoalPositionsWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SerlObsWrapperNoImages

from ur_env.envs.relative_env import RelativeFrame, DualRelativeFrame

from ur_env.utils.her import HER

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
    env = gym.make("box_picking_camera_env_dual_robot_motion_planning",
                   camera_mode="none") if DUAL else gym.make("box_picking_camera_env", camera_mode="rgb")
    
    DUAL_SPACEMOUSE = env.env.env.env.config.DUAL
    HER_EPISODE = env.env.env.env.config.HER
        
    env = SampleGoalPositionsWrapper(env) if HER_EPISODE else env
    env = TwoSpacemiceIntervention(env) if DUAL_SPACEMOUSE else SpacemouseIntervention(env)
    env = DualRelativeFrame(env) if DUAL_SPACEMOUSE else RelativeFrame(env)
    env = DualQuat2MrpWrapper(env) if DUAL_SPACEMOUSE else Quat2MrpWrapper(env)
    env = SerlObsWrapperNoImages(env)

    obs, _ = env.reset()

    her = HER()
    transitions = []
    her_transitions = []
    augmented_transitions = []

    total_count = 0
    num_points = 20
    pbar = tqdm(total=num_points)

    info_dict = {'state': env.unwrapped.curr_pos, 'gripper_state': env.unwrapped.gripper_state,
                 'force': env.unwrapped.curr_force}
    listener_1 = keyboard.Listener(daemon=True, on_press=lambda event: on_space(event, info_dict=info_dict))
    listener_1.start()

    listener_2 = keyboard.Listener(on_press=on_esc, daemon=True)
    listener_2.start()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"ur5_test_{num_points}_demos_{uuid}_her.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    try:
        iter = 0
        # define goal position
        intersection_point = env.env.env.env.env.sample_goal_position()

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

                her_transitions, augmented_transitions = her.process_transitions(
                    transitions=transitions, 
                    last_obs=next_obs, 
                    goal_position=intersection_point,
                    her_transitions=her_transitions,
                    augmented_transitions=augmented_transitions
                    )
                
                # Reset transitions
                transitions = []
                iter += 1

                # sample new goal position
                intersection_point = env.env.env.env.env.sample_goal_position()
                
                total_count += 1
                print(
                    f"{rew}\tRecorded {iter}, {num_points} needed."
                )
                pbar.update(1)
                obs, _ = env.reset()

        with open(file_path, "wb") as f:
            augmented_transitions.extend(her_transitions)
            pkl.dump(augmented_transitions, f)
            print(f"saved {num_points} demos to {file_path}")
            
        with open (f"her_transitions_{uuid}.pkl", 'wb') as f:
            pkl.dump(her_transitions, f)

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