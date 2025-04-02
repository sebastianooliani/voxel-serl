import gymnasium as gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os
import threading
from pynput import keyboard
from absl import app, flags

from ur_env.envs.relative_env import RelativeFrame, DualRelativeFrame
from ur_env.envs.wrappers import SpacemouseIntervention, TwoSpacemiceIntervention, DualQuat2MrpWrapper, Quat2MrpWrapper, ObservationRotationWrapper

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleObservationWrapper, ScaleDualObservationWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

import ur_env

exit_program = threading.Event()


def on_space(key, info_dict):
    if key == keyboard.Key.space:
        for key, item in info_dict.items():
            print(f'{key}:  {item}', end='   ')
        print()


def on_esc(key):
    if key == keyboard.Key.esc:
        exit_program.set()

FLAGS = flags.FLAGS
flags.DEFINE_boolean("dual", True, "Whether to use dual spacemice or not.")
flags.DEFINE_string("camera_mode", "pointcloud", "Type of camera mode used.")
flags.DEFINE_integer("max_episode_length", 100, "Maximum length of trajectory.")

def main(_):
    env = gym.make("box_picking_camera_env_dual_robot_in_air_rotation",
                   camera_mode=FLAGS.camera_mode,
                   max_episode_length=FLAGS.max_episode_length)
    
    env = SpacemouseIntervention(env) if not FLAGS.dual else TwoSpacemiceIntervention(env)
    env = RelativeFrame(env) if not FLAGS.dual else DualRelativeFrame(env)
    env = Quat2MrpWrapper(env) if not FLAGS.dual else DualQuat2MrpWrapper(env)
    env = ScaleObservationWrapper(env) if not FLAGS.dual else ScaleDualObservationWrapper(env)
    # env = ObservationRotationWrapper(env)       # if it should be enabled
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    transitions = []
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
    file_name = f"box_picking_{success_needed}_demos_{uuid}_dual_reorient_pcd.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    try:
        running_reward = 0.
        while success_count < success_needed:
            if exit_program.is_set():
                raise KeyboardInterrupt  # stop program, but clean up before

            action = np.zeros((14,)) if FLAGS.dual else np.zeros((7,))
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
                success_count = env.unwrapped.config.SUCCESS_COUNT
                total_count += 1
                print(
                    f"Got {success_count} successes of {total_count} trials. {success_needed} successes needed."
                )
                pbar.update(int(env.unwrapped.success))
                obs, _ = env.reset()
                print(f"Running return: {running_reward}\n")
                running_reward = 0.

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
        print("Program ended.")

if __name__ == "__main__":
    app.run(main)
