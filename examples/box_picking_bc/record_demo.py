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
from ur_env.envs.wrappers import SpacemouseIntervention, Quat2MrpWrapper, KeyboardInterventionWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SerlObsWrapperNoImages
from serl_launcher.wrappers.chunking import ChunkingWrapper

from gymnasium.wrappers import TransformReward
from ur_env.envs.relative_env import RelativeFrame

exit_program = threading.Event()


def on_space(key, info_dict):
    if key == keyboard.Key.space:
        for key, item in info_dict.items():
            print(f'{key}:  {item}', end='   ')
        print()


def on_esc(key):
    if key == keyboard.Key.esc:
        exit_program.set()

# dummy variable for debugging
SPACEMOUSE = True

if __name__ == "__main__":
    env = gym.make("box_picking_camera_env")
    if SPACEMOUSE:
        env = SpacemouseIntervention(env)
    else:
        # TODO: add a wrapper to use the keyboard for intervention or free-drive mode
        env = KeyboardInterventionWrapper(env) 
        pass
    env = RelativeFrame(env)
    env = Quat2MrpWrapper(env)
    env = SerlObsWrapperNoImages(env)
    # env = TransformReward(env, lambda r: 10. * r)
    # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    transitions = []
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
    file_name = f"ur5_test_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    try:
        while success_count < success_needed:
            if exit_program.is_set():
                raise KeyboardInterrupt  # stop program, but clean up before
            next_obs, rew, done, truncated, info = env.step(action=np.zeros((7,)))
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
            # pprint(transition)

            obs = next_obs

            if done:
                success_count += int(rew > 0.99)
                total_count += 1
                print(
                    f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
                )
                pbar.update(int(rew > 0.99))
                obs, _ = env.reset()

        with open(file_path, "wb") as f:
            pkl.dump(transitions, f)
            print(f"saved {success_needed} demos to {file_path}")

    except KeyboardInterrupt as e:
        print(f'\nProgram was interrupted, cleaning up...  ', e.__str__())

    finally:
        pbar.close()
        env.close()
        listener_1.stop()
        listener_2.stop()
