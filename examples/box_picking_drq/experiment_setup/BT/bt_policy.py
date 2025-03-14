import numpy as np
from BehaviorTree import BehaviorTree, DualBehaviorTree, DualBehaviorTreeReorientation, DualBehaviorTreeMotionPlanning

import copy
import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import pynput
import threading
import tqdm
from absl import app, flags
from flax.training import checkpoints
from datetime import datetime

import gymnasium as gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleObservationWrapper, ScaleDualObservationWrapper
from serl_launcher.wrappers.observation_statistics_wrapper import ObservationStatisticsWrapper, DualObservationStatisticsWrapper
from ur_env.envs.relative_env import RelativeFrame, DualRelativeFrame
from ur_env.envs.wrappers import Quat2MrpWrapper, ObservationRotationWrapper, DualQuat2MrpWrapper, SampleGoalPositionsWrapper

import ur_env

from serl_launcher.utils.launcher import make_wandb_logger
from serl_launcher.utils.sampling_utils import TemporalActionEnsemble

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "box_picking_camera_env_dual_robot", "Name of environment.")
flags.DEFINE_string("exp_name", "BT agent", "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of trajectories for evaluation.")
flags.DEFINE_boolean("dual", False, "Dual robot mode.")
flags.DEFINE_string("wandb_project", "bt", "Wandb project name.")
flags.DEFINE_boolean("debug", False, "Debug mode.")
flags.DEFINE_string("task", "lift", "Task to perform. Choices: lift, reorient, motion.")
flags.DEFINE_boolean("opposite_grasp", False, "Use opposite grasp for lift task.")

def main(_):
    env = gym.make(
        FLAGS.env,
        camera_mode="none",
        fake_env=False,
        max_episode_length=FLAGS.max_traj_length,
    )
    task = env.unwrapped.config.TASK
    if task in ["motion"]:
        env = SampleGoalPositionsWrapper(env)
    env = DualRelativeFrame(env) if FLAGS.dual else RelativeFrame(env)
    env = DualQuat2MrpWrapper(env) if FLAGS.dual else Quat2MrpWrapper(env)
    # env = ScaleDualObservationWrapper(env) if FLAGS.dual else ScaleObservationWrapper(env)  # scale obs space (after quat2mrp, but before serlobs)
    env = DualObservationStatisticsWrapper(env) if FLAGS.dual else ObservationStatisticsWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)

    

    if task in ["lift"]:
        agent = DualBehaviorTree(opposite_grasp=FLAGS.opposite_grasp) if FLAGS.dual else BehaviorTree()
    elif task in ["reorient"]:
        agent = DualBehaviorTreeReorientation(opposite_grasp=FLAGS.opposite_grasp, reorient=True)
    elif task in ["motion"]:
        agent = DualBehaviorTreeMotionPlanning(opposite_grasp=FLAGS.opposite_grasp)

    wandb_logger = make_wandb_logger(
        project=FLAGS.wandb_project,
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )
    action_ensemble = TemporalActionEnsemble(activated=False)
    success_counter = 0
    subsuccess_graps = 0
    subsuccess_lift = 0
    subsuccess_rot = 0
    subsuccess_motion = 0

    time_list = []
    trajectories = []
    traj_infos = []

    try:
        for episode in range(FLAGS.eval_n_trajs):
            trajectory = []
            obs, _ = env.reset()
            done = False
            action_ensemble.reset()

            if len(trajectories) == 0:
                input("ready? record robot view as well!")

            start_time = time.time()

            if task in ["motion"]:
                _ = env.env.env.env.env.env.env.sample_goal_position()
            
            while not done:
                actions = agent.sample_actions(
                    observations=obs,
                )

                ensembled_action = action_ensemble.sample(actions)  # will return actions if not activated
                next_obs, reward, done, truncated, info = env.step(ensembled_action)
                transition = dict(
                    observations=obs.copy(),  # do not save voxel grid or images
                    actions=ensembled_action,
                    next_observations=next_obs.copy(),
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done,
                )
                trajectory.append(transition)
                obs = next_obs

                if done:
                    success_counter = env.unwrapped.config.SUCCESS_COUNT
                    subsuccess_graps += float(env.unwrapped.config.SUBSUCCESS_GRASP)
                    subsuccess_lift += float(env.unwrapped.config.SUBSUCCESS_LIFT)
                    subsuccess_rot += float(env.unwrapped.config.SUBSUCCESS_ROT)
                    subsuccess_motion += float(env.unwrapped.config.SUBSUCCESS_MOTION)

                    dt = time.time() - start_time
                    time_list.append(dt)
                    running_reward = np.sum(np.asarray([t["rewards"] for t in trajectory]))
                    running_reward = max(running_reward, -100.)

                    print(f"{success_counter}/{episode + 1} ", end=' ')
                    print(f"time: {dt:.3f}s  running_rew: {running_reward:.2f}")

                    trajectories.append({"traj": trajectory, "time": dt, "success": (reward > 50.)})
                    infos = {
                        "running_reward": running_reward,
                        "time": dt,
                        "success_rate": success_counter / (episode + 1),
                        "action_cost": np.linalg.norm(np.asarray([t["actions"] for t in trajectory]), axis=1, ord=2).mean(),
                        "subsuccess_graps": subsuccess_graps / (episode + 1),
                        "subsuccess_lift": subsuccess_lift / (episode + 1),
                        "subsuccess_rot": subsuccess_rot / (episode + 1),
                        "subsuccess_motion": subsuccess_motion / (episode + 1),
                    }
                    traj_infos.append(infos)
                    wandb_logger.log(infos, step=episode)

                    # reset the subsuccess
                    env.unwrapped.config.SUBSUCCESS_GRASP = False
                    env.unwrapped.config.SUBSUCCESS_LIFT = False
                    env.unwrapped.config.SUBSUCCESS_ROT = False
                    env.unwrapped.config.SUBSUCCESS_MOTION = False

                    if task in ["motion"]:
                        _ = env.env.env.env.env.env.env.sample_goal_position()

        traj_infos = {k: [d[k] for d in traj_infos] for k in traj_infos[0]}  # list of dicts to dict of lists
        mean_infos = {"mean_" + key: np.mean(val) for key, val in traj_infos.items()}
        mean_infos["std_time"] = np.std(time_list)
        wandb_logger.log(mean_infos)
        for key, value in mean_infos.items():
            print(f"{key}: {value:.3f}")

        with open(f"trajectories {datetime.now().strftime('%m-%d %H%M')}.pkl", "wb") as f:
            import pickle
            pickle.dump(trajectories, f)

    except KeyboardInterrupt as e:
        print(f"Program was interrupted from keyboard, cleaning up...  {e.__str__}")
    finally:
        env.close()
        print("Program ended.")


if __name__ == "__main__":
    app.run(main)
