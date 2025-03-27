# write a class similar to HER, but with curriculum learning, 
# i.e. recompute observations and rewards considering every time a different reward function. 
# Change transitions in the replay buffer accordingly.
import numpy as np
from scipy.spatial.transform import Rotation as R
from franka_env.utils.transformations import (
    construct_homogeneous_matrix,
    pose_2_homogeneous_matrix
)
import copy
from ur_env.envs.camera_env.config import UR5CameraConfigDualRobot as config
from serl_launcher.data.replay_buffer import ReplayBuffer
from serl_launcher.data.data_store import populate_data_store

class CurriculumLearning():
    def __init__(self, scale=False, trans=False, weights=None, camera_mode=None):        
        self.T_O1_O2=config.T_O1_O2
        self.T_EE_SC=config.T_EE_SC

        assert config.TASK in ["motion"], "Only motion task is supported by HER!"
        self.init_weigths = config.REWARD_DICT[config.TASK]

        assert weights is not None, "Weights must be provided!"
        self.weights = weights

        self.grasp_weights = {
            "step_weight": 0.2,
            "action_weight": 0.1,
            "action_diff_weight": 0.01,
            "orientation_weight": 10.,
            "position_weight": 10.,
            "distance_weight": 0.1,
            "grasping_weight": 0.75,
            "suction_weight": 0.75,
            "goal_weight": 0.,
            "penalty": 10,
            "safety_threshold": 0.1,
            "success_weight": 200.,
            "success_threshold": 0.05,
        }

        self.motion_weights = {
            "step_weight": 0.1,
            "action_weight": 0.1,
            "action_diff_weight": 0.01,
            "orientation_weight": 10.,
            "position_weight": 10.,
            "distance_weight": 0.1,
            "grasping_weight": 0.5,
            "suction_weight": 0.5,
            "goal_weight": 20,
            "penalty": 10,
            "success_weight": 200.,
            "success_threshold": 0.05,
            "safety_threshold": 0.1,
        }

        self.final_weights = {
            "step_weight": 0.1,
            "action_weight": 0.1,
            "action_diff_weight": 0.01,
            "orientation_weight": 10.,
            "position_weight": 10.,
            "distance_weight": 0.1,
            "grasping_weight": 0.3,
            "suction_weight": 0.75,
            "goal_weight": 0.5,
            "penalty": 10,
            "success_weight": 200.,
            "success_threshold": 0.05,
            "safety_threshold": 0.1,
        }
        
        self.last_action = np.zeros((14,))

        self.translation_scale=100.
        self.rotation_scale=10.
        self.force_scale=1.
        self.torque_scale=10.

        self.scale=scale
        self.trans=trans
        self.camera_mode=camera_mode

        self.R_1 = None
        self.R_2 = None

        self.count_grasps = 0
        self.count_motion = 0
        self.count_final = 0

        # measure how many steps the robot can hold the object
        self.seconds = 0

    def compute_reward_motion(self, 
                            obs, 
                            action, 
                            goal_position,
                            reset_pose,
                            weights,
                            ) -> float:
        """
        Compute the reward for the motion task.        
        """
        
        def convert_pose_2_7dim(pose):
            return np.concatenate([pose[:3], 
                                   R.from_mrp(pose[3:6]).as_quat(), 
                                   pose[6:9], 
                                   R.from_mrp(pose[9:]).as_quat()], 
                                   axis=0)
        
        def reached_goal_state(obs) -> bool:
            self.seconds += 1 if (0.1 < obs[14:18][0] < 1. and 0.1 < obs[14:18][2] < 1.) else self.seconds
            
            return np.linalg.norm(obs[57:60]) < self.weights["success_threshold"] \
                and 0.1 < obs[14:18][0] < 1. \
                    and 0.1 < obs[14:18][2] < 1.

        if self.scale:
            obs = self.unscale_obs(obs)
            self.init_box_position /= self.rotation_scale
            self.last_box_position /= self.rotation_scale
        
        # transform the observation
        if self.trans:
            obs = self.transform_obs(tcp_pose=obs[33:45], obs=obs, reset_pose=reset_pose)
            self.init_box_position = self.R_1 @ self.init_box_position
            self.last_box_position = self.R_1 @ self.last_box_position

        tcp_pose = obs[33:45]
        tcp_pose = convert_pose_2_7dim(tcp_pose)

        action_cost = self.weights["action_weight"] * np.sum(np.power(action, 2))
        action_diff_cost = self.weights["action_diff_weight"] * np.sum(np.power(obs[:14] - self.last_action, 2))
        self.last_action[:] = action
        
        # STEP: penalize each step
        step_cost = self.weights["step_weight"]

        # SUCTION: reward for successful grip and cost for unnecessary suctioning
        suction_reward = self.weights["grasping_weight"] * (float(obs[14:18][1] > 0.5) + float(obs[14:18][3] > 0.5))
        suction_cost = self.weights["suction_weight"] * (float(obs[14:18][1] < -0.5) + float(obs[14:18][3] < -0.5))

        # ORIENTATION: penalize deviating too much from the starting pose
        orientation_cost = 0
        orientation_cost = 1. - np.sum(tcp_pose[3:7] * reset_pose[3:7]) ** 2
        orientation_cost += 1. - np.sum(tcp_pose[10:] * reset_pose[10:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * self.weights["orientation_weight"]

        # POSITION: penalize deviating too much from the starting pose
        max_pose_diff = 0.05  # set to 5cm
        pos_diff = np.concatenate([tcp_pose[:2] - reset_pose[:2], tcp_pose[7:9] - reset_pose[7:9]])
        position_cost = self.weights["position_weight"] * np.sum(
            np.where(np.abs(pos_diff) > 0.5, np.abs(pos_diff - np.sign(pos_diff) * 0.5), 0.0) # larger movement allowed
        ) * (
            float(obs[14:18][1] > 0.5) + float(obs[14:18][3] > 0.5) # when is grasping
            ) + self.weights["position_weight"] * np.sum(
            np.where(np.abs(pos_diff) > 0.05, np.abs(pos_diff - np.sign(pos_diff) * 0.05), 0.0) # smaller movement allowed
        ) * (
            float(obs[14:18][1] < 0.5) + float(obs[14:18][3] < 0.5) # when is not grasping
            )

        actual_norm_pos = np.sum((obs[60:63] - self.init_box_position) * (obs[63:66] - self.init_box_position)) / np.sum(np.power(obs[63:66] - self.init_box_position, 2))
        prev_norm_pos = np.sum((self.last_box_position - self.init_box_position) * (obs[63:66] - self.init_box_position)) / np.sum(np.power(obs[63:66] - self.init_box_position, 2))
        goal_distance_reward = self.weights["goal_weight"] * (
            actual_norm_pos - prev_norm_pos
        ) * (
            float(obs[14:18][1] > 0.5) + float(obs[14:18][3] > 0.5)
            )
        self.last_tcp_pos = np.concatenate([obs[33:36], obs[42:45]], axis=0)

        # 3D DISTANCE: penalize the distance between the two robots' end-effectors
        # TODO: adjust reference frames and relative base positions
        if self.camera_mode is None:
            distance_cost = 0.
        else:
            T_O1_E1 = construct_homogeneous_matrix(tcp_pose[:7])
            T_O2_E2 = construct_homogeneous_matrix(tcp_pose[7:])
            T_O1_SC1 = T_O1_E1 @ self.T_EE_SC
            T_O1_SC2 = self.T_O1_O2 @ T_O2_E2 @ self.T_EE_SC
            distance_cost = self.weights["distance_weight"] / np.linalg.norm(T_O1_SC1[:3, 3] - T_O1_SC2[:3, 3])

        if reached_goal_state(obs):
            self.last_action[:] = 0.
            R_goal = weights["success_weights"]
            return R_goal - action_cost - orientation_cost - position_cost - action_diff_cost - distance_cost
        else:
            return 0. + suction_reward + goal_distance_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost - action_diff_cost - distance_cost
        
    def convert_pose_2_7dim(self, pose):
        return np.concatenate([pose[:3], 
                                R.from_mrp(pose[3:6]).as_quat(), 
                                pose[6:9], 
                                R.from_mrp(pose[9:]).as_quat()], 
                                axis=0)
        

    def unscale_obs(self, obs):
        """
        Unscale the observation before computing the episode reward.

        Args:
            obs: observation

        Returns:
            unscaled observation
        """
        obs[30:33] /= self.translation_scale

        obs[33:36] /= self.translation_scale
        obs[36:39] /= self.rotation_scale
        obs[39:42] /= self.translation_scale
        obs[42:45] /= self.rotation_scale

        obs[45:48] /= self.translation_scale
        obs[48:51] /= self.rotation_scale
        obs[51:54] /= self.translation_scale
        obs[54:57] /= self.rotation_scale

        obs[57:60] /= self.rotation_scale
        obs[60:63] /= self.rotation_scale
        obs[63:66] /= self.rotation_scale

        return obs

    def scale_obs(self, obs):
        """
        Scale the observation before saving the episode's transitions.

        Args:
            obs: observation

        Returns:
            scaled observation
        """
        obs[30:33] *= self.translation_scale

        obs[33:36] *= self.translation_scale
        obs[36:39] *= self.rotation_scale
        obs[39:42] *= self.translation_scale
        obs[42:45] *= self.rotation_scale

        obs[45:48] *= self.translation_scale
        obs[48:51] *= self.rotation_scale
        obs[51:54] *= self.translation_scale
        obs[54:57] *= self.rotation_scale

        obs[57:60] *= self.rotation_scale
        obs[60:63] *= self.rotation_scale
        obs[63:66] *= self.rotation_scale
        
        return obs
    
    def transform_obs(self, tcp_pose, obs, reset_pose):
        """
        Transform the observation before computing the episode reward.

        Args:
            tcp_pose: TCP pose (orientation in MRP)
            obs: observation
            reset_pose: reset pose (orientation in quaternion)
        """
        assert len(tcp_pose) == 12, "TCP pose must be of length 12"
        assert len(reset_pose) == 14, "Reset pose must be of length 14"

        self.T_1 = pose_2_homogeneous_matrix(tcp_pose[:6])
        self.T_2 = pose_2_homogeneous_matrix(tcp_pose[6:])
        # compute the relative pose
        self.T_1_temp = construct_homogeneous_matrix(reset_pose[:7]) @ np.linalg.inv(self.T_1)
        self.T_2_temp = construct_homogeneous_matrix(reset_pose[7:]) @ np.linalg.inv(self.T_2)

        # tcp_pose -> 33:45
        obs[33:36] = self.T_1_temp[:3, 3]
        obs[36:39] = R.from_matrix(self.T_1_temp[:3, :3]).as_mrp()
        obs[39:42] = self.T_2_temp[:3, 3]
        obs[42:45] = R.from_matrix(self.T_2_temp[:3, :3]).as_mrp()

        self.R_1 = self.T_1_temp[:3, :3]
        self.R_2 = self.T_2_temp[:3, :3]

        # action -> 0:14
        obs[0:3] = self.R_1.T @ obs[0:3]
        obs[3:6] = self.R_1.T @ obs[3:6]
        obs[7:10] = self.R_2.T @ obs[7:10]
        obs[10:13] = self.R_2.T @ obs[10:13]

        # tcp velocity -> 57:69
        obs[45:48] = self.R_1 @ obs[45:48]
        obs[48:51] = self.R_1 @ obs[48:51]
        obs[51:54] = self.R_2 @ obs[51:54]
        obs[54:57] = self.R_2 @ obs[54:57]

        # box position -> 60:63
        # obs[60:63] = self.R_1 @ obs[60:63]
        # obs[63:66] = self.R_1 @ obs[63:66]

        return obs.copy()
    
    def modify_replay_buffer(self, replay_buffer: ReplayBuffer, weights: dict, reward_scale: float) -> ReplayBuffer:
        """
        Modify the replay buffer by recomputing the observations and rewards.

        Args:
            replay_buffer: replay buffer
            weights: weights for the reward function
        """
        new_transition = {}
        new_transitions = []

        size = replay_buffer._size
        _, data = replay_buffer.download(from_idx=0, to_idx=size-1) # download all transitions, data is 'frozen_dict'

        for i, transition in enumerate(data):
            if i == 0:
                reset_pose = transition['observations'][39:51]
                reset_pose = self.convert_pose_2_7dim(reset_pose)

            new_transition["observations"] = transition["observations"].copy()
            new_transition["actions"] = transition["actions"].copy()
            new_transition["next_observations"] = transition["next_observations"].copy()
            new_transition["rewards"] = self.compute_reward_motion(
                obs=transition['observations'].copy(), 
                action=transition['actions'].copy(), 
                reset_pose=reset_pose,
                weights=weights,
            )
            new_transition["masks"] = transition["masks"]
            new_transition["dones"] = transition["dones"]

            new_transitions.append(new_transition)
            new_transition = {}

        replay_buffer = populate_data_store(replay_buffer, new_transitions, reward_scaling=reward_scale)

        return replay_buffer
        