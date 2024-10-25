import numpy as np
from BehaviorTree import BehaviorTree

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
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleObservationWrapper
from serl_launcher.wrappers.observation_statistics_wrapper import ObservationStatisticsWrapper
from ur_env.envs.relative_env import RelativeFrame
from ur_env.envs.wrappers import Quat2MrpWrapper, ObservationRotationWrapper

import ur_env

from serl_launcher.utils.launcher import make_wandb_logger
from serl_launcher.utils.sampling_utils import TemporalActionEnsemble

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "box_picking_camera_env", "Name of environment.")
flags.DEFINE_string("exp_name", "BT agent", "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of trajectories for evaluation.")


def main(_):
    env = gym.make(
        FLAGS.env,
        camera_mode="none",
        fake_env=False,
        max_episode_length=FLAGS.max_traj_length,
    )
    env = RelativeFrame(env)
    env = Quat2MrpWrapper(env)
    env = ScaleObservationWrapper(env)  # scale obs space (after quat2mrp, but before serlobs)
    env = ObservationStatisticsWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)

    agent = BehaviorTree()

    wandb_logger = make_wandb_logger(
        project="paper_evaluation_unseen",
        description=FLAGS.exp_name or FLAGS.env,
        debug=True,
    )
    action_ensemble = TemporalActionEnsemble(activated=False)
    success_counter = 0

    trajectories = []
    traj_infos = []
    for episode in range(FLAGS.eval_n_trajs):
        trajectory = []
        obs, _ = env.reset()
        done = False
        action_ensemble.reset()

        if len(trajectories) == 0:
            input("ready? record robot view as well!")

        start_time = time.time()
        # Dict('state': Dict('action': Box(-1.0, 1.0, (7,), float32), 'gripper_state': Box(-1.0, 1.0, (2,), float32), 'tcp_force': Box(-inf, inf, (3,), float32), 'tcp_pose': Box(-inf, inf, (6,), float32), 'tcp_torque': Box(-inf, inf, (3,), float32), 'tcp_vel': Box(-inf, inf, (6,), float32)))
        # {'state': {'tcp_pose': array([-0.,  0., -0.,  0., -0.,  0.]), 'tcp_vel': array([ 0.0056, -0.0106,  0.0379,  0.0096,  0.0681,  0.0503],
        #       dtype=float32), 'gripper_state': array([0., 0.], dtype=float32), 'tcp_force': array([-1.0511,  0.1372,  0.1787]), 'tcp_torque': array([ 0.2035,  0.4531, -1.7272]), 'action': array([0., 0., 0., 0., 0., 0., 0.])}}


        # {'state': array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
        #  0.    ,  0.    , -0.8965, -0.1579, -0.3137, -0.    ,  0.    ,
        #  0.    ,  0.    ,  0.    ,  0.    ,  0.1384,  0.291 , -1.2326,
        # -0.0092,  0.0156,  0.0035,  0.0107,  0.0706,  0.0444]],
        #   dtype=float32)}

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

            if done or truncated:
                success_counter += (reward > 50.)
                dt = time.time() - start_time
                running_reward = np.sum(np.asarray([t["rewards"] for t in trajectory]))
                running_reward = max(running_reward, -100.)

                print(f"{success_counter}/{episode + 1} ", end=' ')
                print(f"time: {dt:.3f}s  running_rew: {running_reward:.2f}")

                trajectories.append({"traj": trajectory, "time": dt, "success": (reward > 50.)})
                infos = {
                    "running_reward": running_reward,
                    "time": dt,
                    "success_rate": float(reward > 50.),
                    "action_cost": np.linalg.norm(np.asarray([t["actions"] for t in trajectory]), axis=1, ord=2).mean()
                }
                traj_infos.append(infos)
                wandb_logger.log(infos, step=episode)

    traj_infos = {k: [d[k] for d in traj_infos] for k in traj_infos[0]}  # list of dicts to dict of lists
    mean_infos = {"mean_" + key: np.mean(val) for key, val in traj_infos.items()}
    wandb_logger.log(mean_infos)
    for key, value in mean_infos.items():
        print(f"{key}: {value:.3f}")

    with open(f"trajectories {datetime.now().strftime('%m-%d %H%M')}.pkl", "wb") as f:
        import pickle
        pickle.dump(trajectories, f)

    env.close()


if __name__ == "__main__":
    app.run(main)
