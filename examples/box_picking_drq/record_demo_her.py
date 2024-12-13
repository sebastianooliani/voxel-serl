import gymnasium as gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os
import threading
from pynput import keyboard

from ur_env.envs.relative_env import RelativeFrame, DualRelativeFrame
from ur_env.envs.wrappers import SpacemouseIntervention, TwoSpacemiceIntervention, DualQuat2MrpWrapper, Quat2MrpWrapper, ObservationRotationWrapper, SampleGoalPositionsWrapper

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleObservationWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

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
        
############################################################################################################

if __name__ == "__main__":
    env = gym.make("box_picking_camera_env_dual_robot",
                   camera_mode="rgb",
                   max_episode_length=100,
                   )
    
    DUAL = env.env.env.env.config.DUAL
    HER_EPISODE = env.env.env.env.config.HER
    
    env = SampleGoalPositionsWrapper(env) if HER_EPISODE else env
    env = SpacemouseIntervention(env) if not DUAL else TwoSpacemiceIntervention(env)
    env = RelativeFrame(env) if not DUAL else DualRelativeFrame(env)
    env = Quat2MrpWrapper(env) if not DUAL else DualQuat2MrpWrapper(env)
    env = ScaleObservationWrapper(env)
    # env = ObservationRotationWrapper(env)       # if it should be enabled
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    her = HER()
    transitions = []
    her_transitions = []
    augmented_transitions = []

    success_count = 0
    success_needed = 20
    total_count = 0
    pbar = tqdm(total=success_needed)

    info_dict = {'state': env.unwrapped.curr_pos, 'gripper_state': env.unwrapped.gripper_state,
                 'force': env.unwrapped.curr_force, 'reset_pose': env.unwrapped.curr_reset_pose}
    listener_1 = keyboard.Listener(daemon=True, on_press=lambda event: on_space(event, info_dict=info_dict))
    listener_1.start()

    listener_2 = keyboard.Listener(on_press=on_esc, daemon=True)
    listener_2.start()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"box_picking_{success_needed}_demos_{uuid}_her.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    try:
        running_reward = 0.

        iter = 0

        # Sample points in the intersection
        intersection_point = env.env.env.env.env.env.env.sample_goal_position()

        num_points = 20
        
        while iter < num_points:
            # define goal position
            env.env.env.env.env.goal_position = intersection_point
            
            if exit_program.is_set():
                raise KeyboardInterrupt  # stop program, but clean up before

            action = np.zeros((14,)) if DUAL else np.zeros((7,))
            next_obs, rew, done, truncated, info = env.step(action=action)
            actions = info["intervene_action"]

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

            obs = next_obs
            running_reward += rew

            if done or truncated:

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
                intersection_point = env.env.env.env.env.env.env.sample_goal_position()
                
                total_count += 1
                print(
                    f"{rew}\tRecorded {iter}, {num_points} needed."
                )
                pbar.update(1)
                obs, _ = env.reset()

        with open(file_path, "wb") as f:
            augmented_transitions.extend(her_transitions)
            pkl.dump(augmented_transitions, f)
            pkl.dump(her_transitions, f"her_transitions_{uuid}.pkl")
            print(f"saved {success_needed} demos to {file_path}")

        with open (f"her_transitions_{uuid}.pkl", 'wb') as f:
            pkl.dump(her_transitions, f)

    except KeyboardInterrupt as e:
        print(f'\nProgram was interrupted, cleaning up...  ', e.__str__())

    finally:
        pbar.close()
        env.close()
        listener_1.stop()
        listener_2.stop()
        print("Program ended.")