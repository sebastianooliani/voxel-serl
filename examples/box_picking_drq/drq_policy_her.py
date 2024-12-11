#!/usr/bin/env python3
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

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.sampling_utils import TemporalActionEnsemble
from serl_launcher.utils.train_utils import (
    print_agent_params,
    parameter_overview,
    plot_feature_kernel_histogram,
    find_zero_weights,
    plot_conv3d_kernels,
)

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_drq_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper, ScaleObservationWrapper, ScaleDualObservationWrapper
from serl_launcher.wrappers.observation_statistics_wrapper import ObservationStatisticsWrapper, DualObservationStatisticsWrapper
from ur_env.envs.relative_env import RelativeFrame, DualRelativeFrame
from ur_env.envs.wrappers import SpacemouseIntervention, Quat2MrpWrapper, ObservationRotationWrapper, DualQuat2MrpWrapper, SampleGoalPositionsWrapper
from serl_launcher.vision.data_augmentations import batched_random_rot90_state, batched_random_rot90_voxel, \
    batched_random_rot90_action


from franka_env.utils.transformations import (
    construct_homogeneous_matrix
)
from fast_kinematics import FastKinematics
import math
from scipy.spatial.transform import Rotation as R

# used to debug nan errors (also in jit-ed functions)
# jax.config.update("jax_debug_nans", True)

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "box_picking_camera_env", "Name of environment.")
flags.DEFINE_string("agent", "drq", "Name of agent.")
flags.DEFINE_string("exp_name", "DRQ agent", "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_string("camera_mode", "rgb", "Camera mode, one of (rgb, depth, both)")

flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("utd_ratio", 4, "UTD ratio.")

flags.DEFINE_string("state_mask", "no_ForceTorque",
                    "if all the states should be considered, possible: (all, none, no_ForceTorque, gripper, position_gripper)")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_integer("encoder_bottleneck_dim", 128, "bottleneck dimension of the encoder")
# flags.DEFINE_integer("proprio_latent_dim", 64,
#                     "the latent dimension for the state, will be concatenated with encoder bottleneck dim before being passed onward")
flags.DEFINE_multi_string("encoder_kwargs", None, "Encoder kwargs in the form ['dict key', 'dict value']")
flags.DEFINE_bool("enable_obs_rotation_wrapper", False,
                  "Whether to enable observation rotation wrapper (train in one quaternion)")
flags.DEFINE_bool("enable_obs_rotation_augmentation", False,
                  "Whether to enable observation rotation augmentation (90 deg)")
flags.DEFINE_bool("enable_temporal_ensemble_sampling", False,
                  "Whether to enable sampling the action from a temporal ensemble: action = 0.5*a0 + 0.3*a-1 + 0.2*a-2 + 0.1*a-3")

flags.DEFINE_integer("max_steps", 1000000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 10000,
                     "Replay buffer capacity.")  # quite low to forget demo trajectories

flags.DEFINE_integer("random_steps", 0, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 0, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 10, "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 1000, "Evaluation period in seconds")
flags.DEFINE_integer("eval_n_trajs", 10, "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("evaluation", False, "Evaluation mode.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", '/home/nico/real-world-rl/serl/examples/box_picking_drq/checkpoints',
                    "Path to save checkpoints.")
flags.DEFINE_string("load_checkpoint_path", '/home/nico/real-world-rl/serl/examples/box_picking_drq/checkpoints',
                    "Path to load previously saved checkpoints and start training from them.")

flags.DEFINE_integer("eval_checkpoint_step", 0, "evaluate the policy from ckpt at this step")
flags.DEFINE_string("log_rlds_path", '/home/nico/real-world-rl/serl/examples/box_picking_drq/rlds',
                    "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging
flags.DEFINE_boolean("dual", True, "Dual robot mode.")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))


PAUSE_EVENT_FLAG = threading.Event()
PAUSE_EVENT_FLAG.clear()  # clear() to continue the actor/learner loop, set() to pause


def pause_callback(key):
    """Callback for when a key is pressed"""
    global PAUSE_EVENT_FLAG
    try:
        # chosen a rarely used key to avoid conflicts. this listener is always on, even when the program is not in focus
        if not PAUSE_EVENT_FLAG.is_set() and key == pynput.keyboard.Key.pause:
            print("Requested pause training")
            # set the PAUSE FLAG to pause the actor/learner loop
            PAUSE_EVENT_FLAG.set()
    except AttributeError:
        # print(f'{key} pressed')
        pass


listener = pynput.keyboard.Listener(
    on_press=pause_callback
)  # to enable keyboard based pause
listener.start()


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

def actor(agent: DrQAgent, data_store, env, sampling_rng, dual=False):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    global PAUSE_EVENT_FLAG

    if FLAGS.eval_checkpoint_step and FLAGS.evaluation:
        wandb_logger = make_wandb_logger(
            project="drq_rgb_top",  # TODO only temporary
            description=FLAGS.exp_name or FLAGS.env,
            debug=FLAGS.debug,
        )
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            FLAGS.checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)
        find_zero_weights(agent.state.params, print_all=False)
        action_ensemble = TemporalActionEnsemble(activated=FLAGS.enable_temporal_ensemble_sampling)

        # examine model parameters if trajs==0
        if FLAGS.eval_n_trajs == 0:
            parameter_overview(agent)
            # plot_feature_kernel_histogram(agent)
            plot_conv3d_kernels(agent.state.params)

        trajectories = []
        traj_infos = []
        for episode in range(FLAGS.eval_n_trajs):
            trajectory = []
            obs, _ = env.reset()
            done = False
            action_ensemble.reset()
            start_time = time.time()

            while not done:
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))

                ensembled_action = action_ensemble.sample(actions)      # will return actions if not activated
                next_obs, reward, done, truncated, info = env.step(ensembled_action)
                transition = dict(
                    observations=obs["state"].copy(),  # do not save voxel grid or images
                    actions=ensembled_action,
                    next_observations=next_obs["state"].copy(),
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
                    running_reward = max(running_reward, -100.)     # -100 min value

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

            # if pause event is requested, pause the actor
            if PAUSE_EVENT_FLAG.is_set():
                print("Actor eval loop interrupted")
                response = input("Do you want to continue (c), or exit (e)? ")
                if response == "c":
                    # update PAUSE FLAG to continue training
                    PAUSE_EVENT_FLAG.clear()
                    print("Continuing")
                else:
                    print("Stopping actor eval")
                    break

        traj_infos = {k: [d[k] for d in traj_infos] for k in traj_infos[0]}     # list of dicts to dict of lists
        mean_infos = {"mean_" + key: np.mean(val) for key, val in traj_infos.items()}
        wandb_logger.log(mean_infos)
        for key, value in mean_infos.items():
            print(f"{key}: {value:.3f}")

        filename = f"trajectories {'temp_ens' if action_ensemble.is_activated() else ''} {datetime.now().strftime('%m-%d %H%M')}.pkl"
        with open(filename, "wb") as f:
            import pickle
            pickle.dump(trajectories, f)
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

    # training loop
    timer = Timer()
    running_return = 0.0

    transitions = []
    her_transitions = []
    augmented_transitions = []

    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        intersection_points = env.env.env.env.env.env.env.env.sample_goal_positions()

        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            elif not agent.config["activate_batch_rotation"]:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))
            else:
                sampling_rng, rot_rng, key = jax.random.split(sampling_rng, 3)

                rotated_obs = copy.deepcopy(obs)
                rotated_obs["state"] = batched_random_rot90_state(obs["state"], rot_rng)

                if not dual:
                    rotated_obs["wrist_pointcloud"] = batched_random_rot90_voxel(obs["wrist_pointcloud"], rot_rng)
                else:
                    rotated_obs["wrist_1_pointcloud"] = batched_random_rot90_voxel(obs["wrist_1_pointcloud"], rot_rng)
                    rotated_obs["wrist_2_pointcloud"] = batched_random_rot90_voxel(obs["wrist_2_pointcloud"], rot_rng)

                actions = agent.sample_actions(
                    observations=jax.device_put(rotated_obs),
                    seed=key,
                    deterministic=False,
                )
                for _ in range(3):
                    actions = batched_random_rot90_action(actions[None, ...], rot_rng)[0, ...]  # rotate back

                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action")

            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return = running_return * 0.99 + reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            # data_store.insert(transition)
            transitions.append(transition)

            obs = next_obs

            if done or truncated:
                last_obs = next_obs

                # HER transitions
                for trans in transitions:
                    # compute reward based on the new goal state
                    # concatenate the last observation to the current observation
                    # recompute the goal-box-position observation based on the reached point
                    her_transitions.append(
                        dict(
                            observations=np.concatenate(
                                [trans['observations'][:-6], last_obs[-3:] - trans['observations'][-3:], trans['observations'][-3:], last_obs[-3:]], 
                                axis=0
                                ),
                            actions=trans['actions'],
                            next_observations=np.concatenate(
                                [trans['next_observations'][:-6], last_obs[-3:] - trans['next_observations'][-3:], trans['next_observations'][-3:], last_obs[-3:]], 
                                axis=0
                                ), # TODO: should I recompute the goal_box_position observation?
                            # compute reward based on the new goal state
                            rewards=compute_reward_her(
                                obs=trans['observations'],
                                action=trans['actions'], 
                                goal_position=last_obs[-3:]
                                ), # TODO: implement this function
                            masks=trans['masks'],
                            dones=trans['dones'],
                        )
                    )
                    augmented_transitions.append(
                        dict(
                            observations=np.concatenate(
                                [trans['observations'], intersection_points], 
                                axis=0
                                ), 
                            actions=trans['actions'],
                            next_observations=np.concatenate(
                                [trans['next_observations'], intersection_points], 
                                axis=0
                                ),
                            rewards=trans['rewards'],
                            masks=trans['masks'],
                            dones=trans['dones'],
                        )
                    )

                transitions = []
                augmented_transitions.extend(her_transitions)

                data_store.insert(augmented_transitions)

                # sample new goal position
                intersection_points = env.env.env.env.env.env.env.env.sample_goal_positions()
                her_transitions = []
                augmented_transitions = []

                stats = {"train": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                print(f"running return: {running_return}")
                running_return = 0.0
                obs, _ = env.reset()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        timer.tock("total")

        if FLAGS.eval_period and step % FLAGS.eval_period == 0 and step:
            with timer.context("eval"):
                evaluate_info = evaluate(
                    policy_fn=partial(agent.sample_actions, argmax=True),
                    env=env,
                    num_episodes=FLAGS.eval_n_trajs,
                )
            stats = {"eval": evaluate_info}
            client.request("send-stats", stats)

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)

        if PAUSE_EVENT_FLAG.is_set():
            print_green("Actor loop interrupted")
            response = input(
                "Do you want to continue (c), save replay buffer and exit (s) or simply exit (e)? "
            )
            if response == "c":
                print("Continuing")
                PAUSE_EVENT_FLAG.clear()
            else:
                if response == "s":
                    print("Saving replay buffer")
                    data_store.save(
                        "replay_buffer_actor.npz"
                    )  # not yet supported for QueuedDataStore
                else:
                    print("Replay buffer not saved")
                print("Stopping actor client")
                client.stop()
                break


##############################################################################


def learner(rng, agent: DrQAgent, replay_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    # To track the step in the training loop
    update_steps = 0
    global PAUSE_EVENT_FLAG

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
    # TODO: load network from checkpoint
    if FLAGS.eval_checkpoint_step:
        print_yellow("loading checkpoint")
        ckpt = checkpoints.restore_checkpoint(
            FLAGS.load_checkpoint_path,
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)
        server.publish_network(agent.state.params)
        print_yellow("sent checkpoint network to actor")

    else:
        server.publish_network(agent.state.params)
        print_green("sent initial network to actor")

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True, desc="learner"):
        timer.tick("learner_total")

        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        for critic_step in range(FLAGS.utd_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(batch, )

        with timer.context("train"):
            batch = next(replay_iterator)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)

        timer.tock("learner_total")

        # publish the updated network
        if step > 0 and step % (FLAGS.steps_per_update) == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()}, step=update_steps)
            wandb_logger.log({"replay_buffer_size": len(replay_buffer)})

        update_steps += 1

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=update_steps, keep=100
            )

        if PAUSE_EVENT_FLAG.is_set():
            print("Learner loop interrupted")
            response = input(
                "Do you want to continue (c), save training state and exit (s) or simply exit (e)? "
            )
            if "c" in response:
                print("Continuing")
                PAUSE_EVENT_FLAG.clear()
            else:
                if response == "s":
                    print("Saving learner state")
                    agent_ckpt = checkpoints.save_checkpoint(
                        FLAGS.checkpoint_path, agent.state, step=update_steps, keep=100
                    )
                    replay_buffer.save(
                        "replay_buffer_learner.npz"
                    )  # not yet supported for QueuedDataStore
                    # TODO: save other parts of training state
                else:
                    print("Training state not saved")
                print("Stopping learner client")
                break

    server.stop()
    parameter_overview(agent)  # print end state


##############################################################################

def main(_):
    assert FLAGS.batch_size % num_devices == 0
    if FLAGS.checkpoint_path.split('/')[-1] == "checkpoints":
        FLAGS.checkpoint_path = FLAGS.checkpoint_path + " " + FLAGS.exp_name + " " + datetime.now().strftime(
            "%m%d-%H:%M")

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env and load dataset
    env = gym.make(
        FLAGS.env,
        camera_mode=FLAGS.camera_mode,
        fake_env=FLAGS.learner,
        max_episode_length=FLAGS.max_traj_length,
    )
    # if FLAGS.actor:
    #     env = SpacemouseIntervention(env)
    env = SampleGoalPositionsWrapper(env) if FLAGS.dual else env
    env = RelativeFrame(env) if not FLAGS.dual else DualRelativeFrame(env)
    env = Quat2MrpWrapper(env) if not FLAGS.dual else DualQuat2MrpWrapper(env)
    env = ScaleObservationWrapper(env) if not FLAGS.dual else ScaleDualObservationWrapper(env)  # scale obs space (after quat2mrp, but before serlobs)
    env = ObservationStatisticsWrapper(env) if not FLAGS.dual else DualObservationStatisticsWrapper(env)
    if FLAGS.enable_obs_rotation_wrapper:
        env = ObservationRotationWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    env = RecordEpisodeStatistics(env)

    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    print(f"image keys: {image_keys}")

    rng, sampling_rng = jax.random.split(rng)

    # assert FLAGS.encoder_kwargs is None or len(FLAGS.encoder_kwargs) % 2 == 0
    encoder_kwargs = {
        "bottleneck_dim": FLAGS.encoder_bottleneck_dim,
        **(dict(zip(*[iter(FLAGS.encoder_kwargs)] * 2)) if FLAGS.encoder_kwargs else {}),
    }
    encoder_kwargs = {k: (int(v) if str(v).isdigit() else v) for k, v in encoder_kwargs.items()}

    agent: DrQAgent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
        state_mask=FLAGS.state_mask,
        # proprio_latent_dim=FLAGS.proprio_latent_dim,
        encoder_kwargs=encoder_kwargs
    )

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent: DrQAgent = jax.device_put(
        jax.tree.map(jnp.array, agent), sharding.replicate()
    )

    # print useful info
    print_agent_params(agent, image_keys)
    parameter_overview(agent)
    # plot_conv3d_kernels(agent.state.params)

    agent.config["activate_batch_rotation"] = FLAGS.enable_obs_rotation_augmentation  # obs batch rotation control
    if FLAGS.enable_obs_rotation_augmentation:
        print("Batch Observation Rotation enabled!")
    assert not FLAGS.enable_obs_rotation_augmentation or not FLAGS.enable_obs_rotation_wrapper  # both is pointless

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
            image_keys=image_keys,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="drq_rgb_top",  # TODO only temporary
            description=FLAGS.exp_name or FLAGS.env,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()

        import pickle as pkl
        with open(FLAGS.demo_path, "rb") as f:
            trajs = pkl.load(f)

            # check which observations can be ignored for this run
            to_pop = []
            for obs_name in [i for i in trajs[0]["observations"].keys()]:
                if obs_name not in env.observation_space.spaces:
                    to_pop.append(obs_name)
            print(f"ignored {to_pop} observation in the demo trajectories")

            for traj in trajs:
                for obs_name in to_pop:
                    traj["observations"].pop(obs_name)
                    traj["next_observations"].pop(obs_name)

                # convert to grey here
                if FLAGS.camera_mode == "grey":
                    gray = np.array([0.2989, 0.5870, 0.1140])
                    traj["observations"]["wrist"] = np.dot(traj["observations"]["wrist"], gray)[..., None]
                    traj["next_observations"]["wrist"] = np.dot(traj["next_observations"]["wrist"], gray)[..., None]

                replay_buffer.insert(traj)
        print(f"replay buffer size: {len(replay_buffer)}")

        # learner loop
        print_green("starting learner loop")
        try:
            learner(
                sampling_rng,
                agent,
                replay_buffer=replay_buffer,
                wandb_logger=wandb_logger,
            )
        except KeyboardInterrupt:
            print_green("learner loop interrupted")
        finally:
            # Wrap up the learner loop
            env.close()
            print("Learner loop finished")

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor

        # actor loop
        print_green("starting actor loop")
        try:
            actor(agent, data_store, env, sampling_rng, dual=FLAGS.dual)
            print_green("actor loop finished")
        except (KeyboardInterrupt, RuntimeError) as e:
            print_green("actor loop interrupted: " + str(e))
        finally:
            env.close()

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
