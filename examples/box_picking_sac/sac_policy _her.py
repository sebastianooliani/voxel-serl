#!/usr/bin/env python3

import time
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
from datetime import datetime

import gymnasium as gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers import TransformReward

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.data.data_store import populate_data_store

from serl_launcher.wrappers.chunking import ChunkingWrapper
from ur_env.envs.relative_env import RelativeFrame, DualRelativeFrame

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)

from serl_launcher.wrappers.serl_obs_wrappers import SerlObsWrapperNoImages
from ur_env.envs.wrappers import SpacemouseIntervention, Quat2MrpWrapper, DualQuat2MrpWrapper, TwoSpacemiceIntervention

import ur_env

from franka_env.utils.transformations import (
    construct_homogeneous_matrix
)
from fast_kinematics import FastKinematics
import math
from scipy.spatial.transform import Rotation as R

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "box_picking_basic_env", "Name of environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", "sac_drq_policy", "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", True, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("utd_ratio", 8, "UTD ratio.")
flags.DEFINE_integer("reward_scale", 1, "Reward Scale to help out SAC algorithm")

flags.DEFINE_integer("max_steps", 100000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 1000000, "Replay buffer capacity.")
flags.DEFINE_multi_string("demo_paths", None,
                          "paths to demos")

flags.DEFINE_integer("random_steps", 1000, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 1000, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 10, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 3, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("checkpoint_period", 10000, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", '/home/sebastiano/voxel-serl/examples/box_picking_sac/checkpoints',
                    "Path to save checkpoints.")

flags.DEFINE_integer("eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step")
flags.DEFINE_string("eval_checkpoint_path", None, "evaluate the policy from ckpt from this path")

flags.DEFINE_string("log_rlds_path", '/home/sebastiano/voxel-serl/examples/box_picking_sac/rlds',
                    "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging
flags.DEFINE_boolean("dual", False, "Dual robot mode.")

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################

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


def actor(agent: SACAgent, data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.eval_checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset()
            done = False
            start_time = time.time()
            while not done:
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))
                # print(actions)

                next_obs, reward, done, truncated, info = env.step(actions)
                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)

                    success_counter += int(reward > 0.99)
                    print(reward)
                    print(f"{success_counter}/{episode + 1}")

        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    obs, _ = env.reset()
    # print(f"obs:  {obs}")
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                # print("sampling randomly!")
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                    # deterministic=False,              # sample without argmax for more diverse actions
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)

            running_return += reward

            data_store.insert(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done or truncated,
                )
            )

            obs = next_obs
            if done or truncated:
                # print(f"running return: {running_return}   done:{done}  truncated:{truncated}")
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        if step % FLAGS.eval_period == 0 and step:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent: SACAgent, replay_buffer, replay_iterator, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    try:
        for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
            # Train the networks
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

            with timer.context("train"):
                if FLAGS.utd_ratio == 1:
                    agent, update_info = agent.update(batch=batch)  # try it without utd
                else:
                    agent, update_info = agent.update_high_utd(batch, utd_ratio=FLAGS.utd_ratio)
                agent = jax.block_until_ready(agent)

                # publish the updated network
                server.publish_network(agent.state.params)

            if update_steps % FLAGS.log_period == 0 and wandb_logger:
                wandb_logger.log(update_info, step=update_steps)
                wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)
                wandb_logger.log({"replay_buffer_size": len(replay_buffer)})

            if FLAGS.checkpoint_period and (update_steps + 1) % FLAGS.checkpoint_period == 0:
                assert FLAGS.checkpoint_path is not None
                checkpoints.save_checkpoint(
                    FLAGS.checkpoint_path, agent.state, step=update_steps + 1, keep=20
                )

            update_steps += 1
    finally:
        print("closing learner, clearning up...")
        del replay_buffer


##############################################################################

DUAL_SPACEMOUSE = True

def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)
    assert FLAGS.batch_size % num_devices == 0
    FLAGS.checkpoint_path = FLAGS.checkpoint_path + "_" + datetime.now().strftime("%m%d-%H:%M")

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    env = gym.make(
        FLAGS.env,
        fake_env=FLAGS.learner,
        max_episode_length=FLAGS.max_traj_length,
        camera_mode="rgb",
    )
    if FLAGS.actor:
        env = SpacemouseIntervention(env) if not DUAL_SPACEMOUSE else TwoSpacemiceIntervention(env)
    env = RelativeFrame(env) if not DUAL_SPACEMOUSE else DualRelativeFrame(env)
    env = Quat2MrpWrapper(env) if not DUAL_SPACEMOUSE else DualQuat2MrpWrapper(env)
    env = SerlObsWrapperNoImages(env)
    # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    # env = TransformReward(env, lambda r: FLAGS.reward_scale * r)
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
    agent: SACAgent = make_sac_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: SACAgent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = make_replay_buffer(
            env,
            capacity=FLAGS.replay_buffer_capacity,
            type="replay_buffer",
            rlds_logger_path=FLAGS.log_rlds_path,
            preload_rlds_path=FLAGS.preload_rlds_path,
        )

        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="dual_robot_top_sac",
            description=FLAGS.exp_name or FLAGS.env,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()

        if FLAGS.preload_rlds_path is None and FLAGS.demo_paths is not None:
            print(f"loaded demos from {FLAGS.demo_paths}")  # load demo trajectories the old way
            replay_buffer = populate_data_store(replay_buffer, FLAGS.demo_paths, reward_scaling=FLAGS.reward_scale)

        replay_iterator = replay_buffer.get_iterator(
            sample_args={
                "batch_size": FLAGS.batch_size * FLAGS.utd_ratio,
            },
            device=sharding.replicate(),
        )
        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            replay_iterator=replay_iterator,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        try:
            actor(agent, data_store, env, sampling_rng)
            print_green("actor loop finished")
        except KeyboardInterrupt:
            print_green("actor loop interrupted")
        finally:
            env.close()

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
