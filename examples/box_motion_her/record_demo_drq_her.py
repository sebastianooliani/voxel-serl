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
from ur_env.envs.wrappers import SpacemouseIntervention, TwoSpacemiceIntervention, DualQuat2MrpWrapper, Quat2MrpWrapper, ObservationRotationWrapper, SampleGoalPositionsWrapper

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleDualObservationWrapper
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
        
FLAGS = flags.FLAGS
flags.DEFINE_boolean("dual", True, "Whether to use dual spacemice or not.")
flags.DEFINE_boolean("her", True, "Whether to use HER or not.")
flags.DEFINE_string("camera_mode", "pointcloud", "Type of camera mode used.")

############################################################################################################

def main(_):
    env = gym.make("box_picking_camera_env_dual_robot_motion_planning",
                   camera_mode=FLAGS.camera_mode,
                   max_episode_length=100,
                   )
    
    env = SampleGoalPositionsWrapper(env) if FLAGS.her else env
    env = SpacemouseIntervention(env) if not FLAGS.dual else TwoSpacemiceIntervention(env)
    env = RelativeFrame(env) if not FLAGS.dual else DualRelativeFrame(env)
    env = Quat2MrpWrapper(env) if not FLAGS.dual else DualQuat2MrpWrapper(env)
    env = ScaleDualObservationWrapper(env) if FLAGS.dual else env
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    her = HER(scale=True, trans=True, camera_mode=FLAGS.camera_mode)
    transitions = []
    her_transitions = []
    augmented_transitions = []
    all_transitions = []
    positive_transitions = []

    num_points = 20
    total_count = 0
    pbar = tqdm(total=num_points)

    info_dict = {'state': env.unwrapped.curr_pos, 'gripper_state': env.unwrapped.gripper_state,
                 'force': env.unwrapped.curr_force, 'reset_pose': env.unwrapped.curr_reset_pose}
    listener_1 = keyboard.Listener(daemon=True, on_press=lambda event: on_space(event, info_dict=info_dict))
    listener_1.start()

    listener_2 = keyboard.Listener(on_press=on_esc, daemon=True)
    listener_2.start()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"box_picking_{num_points}_demos_{uuid}_pcd_her.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    try:
        running_reward = 0.
        iter = 0

        # Sample points in the intersection
        intersection_point = env.env.env.env.env.env.env.sample_goal_position()
        
        while iter < num_points:
            # define goal position
            
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
                curr_reset_pose = env.unwrapped.curr_reset_pose

                her_transitions, augmented_transitions = her.process_transitions(
                    transitions=transitions, 
                    last_obs=next_obs, 
                    goal_position=intersection_point,
                    her_transitions=her_transitions,
                    augmented_transitions=augmented_transitions,
                    reset_pose=curr_reset_pose
                    )
                
                # Reset transitions
                transitions = []
                iter += 1
                positive_transitions.extend(her_transitions)
                all_transitions.extend(her_transitions)
                all_transitions.extend(augmented_transitions)

                # sample new goal position
                intersection_point = env.env.env.env.env.env.env.sample_goal_position()
                
                total_count += 1
                print(
                    f"{rew}\tRecorded {iter}, {num_points} needed."
                )
                pbar.update(1)
                obs, _ = env.reset()

        with open(file_path, "wb") as f:
            pkl.dump(all_transitions, f)
            print(f"saved {num_points} demos to {file_path}")

        with open(f"dual_{num_points}_her_transitions_{uuid}.pkl", 'wb') as f:
            pkl.dump(positive_transitions, f)

    except KeyboardInterrupt as e:
        print(f'\nProgram was interrupted, cleaning up...  ', e.__str__())

    finally:
        pbar.close()
        env.close()
        print("Environment closed.")
        listener_1.stop()
        listener_2.stop()
        print("Program ended.")

if __name__ == "__main__":
    app.run(main)