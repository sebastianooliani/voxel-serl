import numpy as np
from scipy.spatial.transform import Rotation as R
from franka_env.utils.transformations import (
    construct_homogeneous_matrix,
    pose_2_homogeneous_matrix
)
import copy

from ur_env.envs.camera_env.config import UR5CameraConfigDualRobot as config

class HER():
    def __init__(self, scale=False, trans=False, camera_mode=None, weights_dict=None):        
        self.T_O1_O2=config.T_O1_O2
        self.T_EE_SC=config.T_EE_SC

        assert config.TASK in ["motion"], "Only motion task is supported by HER!"

        self.weights = config.REWARD_DICT[config.TASK] if weights_dict is None else weights_dict

        self.last_action = np.zeros((14,))

        self.translation_scale=100.
        self.rotation_scale=10.
        self.force_scale=1.
        self.torque_scale=10.

        self.scale=scale
        self.trans=trans
        self.camera_mode = camera_mode if camera_mode in ["pointcloud"] else None

        self.R_1 = None
        self.R_2 = None

        self.success = False
        self.add_to_buffer = False # add to buffer only if success is achieved
        self.last_box_position = None
        self.costs = {}

    ##########################################################################################
    #                               HER: her reward computation                              #
    ##########################################################################################
    # observation space ----------------------------------- after removing forces and torques
    # action -> 0:14                                        # 0:14
    # gripper state -> 14:18                                # 14:18
    # joint position -> 18:30                               # 18:30
    # tcp force -> 30:36                                    # -
    # tcp pos diff -> 36:39                                 # 30:33
    # tcp pose -> 39:51                                     # 33:45
    # tcp torque -> 51:57                                   # -
    # tcp velocity -> 57:69                                 # 45:57
    # goal box position -> 69:72                            # 57:60
    # box position -> 72:75                                 # 60:63
    # goal position -> 75:78                                # 63:66

    def compute_reward_her(self, 
                            obs, 
                            action, 
                            goal_position,
                            reset_pose,
                            ) -> float:
        
        def convert_pose_2_7dim(pose):
            return np.concatenate([pose[:3], 
                                   R.from_mrp(pose[3:6]).as_quat(), 
                                   pose[6:9], 
                                   R.from_mrp(pose[9:]).as_quat()], 
                                   axis=0)
        
        def reached_goal_state_her(obs) -> bool:
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
            # WARNINGS: boxes are not transformed anymore!
            # self.init_box_position = self.R_1 @ self.init_box_position
            # self.last_box_position = self.R_1 @ self.last_box_position

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
        orientation_cost = 1. - sum(tcp_pose[3:7] * reset_pose[3:7]) ** 2
        orientation_cost += 1. - sum(tcp_pose[10:] * reset_pose[10:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * self.weights["orientation_weight"]

        # POSITION: penalize deviating too much from the starting pose
        # max_pose_diff = 0.05  # set to 5cm
        # pos_diff = np.concatenate([tcp_pose[:2] - reset_pose[:2], tcp_pose[7:9] - reset_pose[7:9]])
        # position_cost = self.weights["position_weight"] * np.sum(
        #     np.where(np.abs(pos_diff) > 0.6, np.abs(pos_diff - np.sign(pos_diff) * 0.1), 0.0) # larger movement allowed
        # ) * (
        #     float(obs[14:18][1] > 0.5) + float(obs[14:18][3] > 0.5) # when is grasping
        #     ) + self.weights["position_weight"] * np.sum(
        #     np.where(np.abs(pos_diff) > 0.05, np.abs(pos_diff - np.sign(pos_diff) * 0.1), 0.0) # smaller movement allowed
        # ) * (
        #     float(obs[14:18][1] < 0.5) + float(obs[14:18][3] < 0.5) # when is not grasping
        #     )
        position_cost = self.weights["position_weight"] * np.linalg.norm(
            obs[57:60]
        )
        position_cost = 4. if position_cost > 4. else position_cost

        actual_norm_pos = np.sum((obs[60:63] - self.init_box_position) * (obs[63:66] - self.init_box_position)) / np.sum(np.power(obs[63:66] - self.init_box_position, 2))
        prev_norm_pos = np.sum((self.last_box_position - self.init_box_position) * (obs[63:66] - self.init_box_position)) / np.sum(np.power(obs[63:66] - self.init_box_position, 2))
        goal_distance_reward = self.weights["goal_weight"] * (
            actual_norm_pos - prev_norm_pos
        ) * (
            float(obs[14:18][1] > 0.5) + float(obs[14:18][3] > 0.5)
            )
        self.last_tcp_pos = np.concatenate([obs[33:36], obs[42:45]], axis=0) 
        goal_distance_reward = 0. if goal_distance_reward < 0. else goal_distance_reward
        goal_distance_reward = 4. if goal_distance_reward > 4. else goal_distance_reward

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

        costs = dict(
            action_cost=action_cost,
            step_cost=step_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            orientation_cost=orientation_cost,
            position_cost=position_cost,
            action_diff_cost=action_diff_cost,
            distance_cost=distance_cost,
            goal_distance_reward=goal_distance_reward,
            total_cost=-(-action_cost - step_cost + suction_reward + goal_distance_reward - suction_cost - orientation_cost - position_cost - action_diff_cost)
        )
        for key, info in costs.items():
            self.costs[key] = info + (0. if key not in self.costs else self.costs[key])

        if reached_goal_state_her(obs):
            R_goal = self.weights["success_weight"]
            self.success = True
            return R_goal - action_cost - orientation_cost - position_cost - action_diff_cost - distance_cost
        else:
            return 0. + suction_reward + goal_distance_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost - action_diff_cost - distance_cost
        
    def process_transitions(self, 
                            transitions, 
                            last_obs, 
                            goal_position, 
                            her_transitions, 
                            augmented_transitions, 
                            reset_pose):
        """
        Process transitions. Obtain HER and augment the standard transitions.
        
        Args:
            transitions: list of transitions
            last_obs: last observation
            goal_position: goal position
            her_transitions: list of HER transitions
            augmented_transitions: list of augmented transitions

        Returns:
            her_transitions: list of HER transitions
            augmented_transitions: list of augmented transitions
            add_to_buffer: boolean to add to buffer
        """
        # init
        her_transitions, augmented_transitions, self.add_to_buffer = [], [], False
        # scale goal position
        goal_position *= self.rotation_scale

        for i, trans in enumerate(transitions):
            # compute reward based on the new goal state
            # concatenate the last observation to the current observation
            # recompute the goal-box-position observation based on the reached position

            # initialize the last box position at the first transition
            if i == 0:
                self.init_box_position = trans['observations'][-6:-3].copy()
                self.last_box_position = self.init_box_position.copy()
                self.last_tcp_pos = np.concatenate([reset_pose[:3], reset_pose[7:10]], axis=0)

            her_dict = copy.deepcopy(
                dict(
                    observations=np.concatenate(
                        [trans['observations'][:-9], 
                            last_obs[-6:-3] - trans['observations'][-6:-3], 
                            trans['observations'][-6:-3], 
                            last_obs[-6:-3]], 
                        axis=0
                        ),
                    actions=trans['actions'],
                    next_observations=np.concatenate(
                        [trans['next_observations'][:-9], 
                            last_obs[-6:-3] - trans['next_observations'][-6:-3], 
                            trans['next_observations'][-6:-3], 
                            last_obs[-6:-3]], 
                        axis=0
                        ), # TODO: should I recompute the goal_box_position observation? YES
                    # compute reward based on the new goal state
                    rewards=self.compute_reward_her(
                        obs=np.concatenate(
                            [trans['observations'][:-9], 
                            last_obs[-6:-3] - trans['observations'][-6:-3], 
                            trans['observations'][-6:-3], 
                            last_obs[-6:-3]], 
                            axis=0
                            ), # use the new observation vector
                        action=trans['actions'], 
                        goal_position=last_obs[-6:-3],
                        reset_pose=reset_pose
                        ), # TODO: implement this function
                    masks=1.0 - self.success,
                    dones=self.success,
                )
            )
            her_transitions.append(her_dict)
            # store the last box position for the next transition
            self.last_box_position = trans['observations'][-6:-3].copy()

            augm_dict = copy.deepcopy(
                dict(
                    observations=np.concatenate(
                        [trans['observations'][:-3], goal_position], 
                        axis=0
                        ), 
                    actions=trans['actions'],
                    next_observations=np.concatenate(
                        [trans['next_observations'][:-3], goal_position], 
                        axis=0
                        ),
                    rewards=trans['rewards'],
                    masks=trans['masks'],
                    dones=trans['dones'],
                )
            )
            augmented_transitions.append(augm_dict)

            # cut episode length if success is achieved
            if self.success:
                self.add_to_buffer = True
                self.success = False
                break

        self.last_action[:] = 0.

        # print(f"Costs: {self.costs}")

        return her_transitions, augmented_transitions, self.add_to_buffer
    
    def process_transitions_drq(self,
                                transitions,
                                last_obs,
                                goal_position,
                                her_transitions,
                                augmented_transitions,
                                reset_pose):
        """
        Process transitions. Obtain HER and augment the standard transitions.
        
        Args:
            transitions: list of transitions
            last_obs: last observation
            goal_position: goal position
            her_transitions: list of HER transitions
            augmented_transitions: list of augmented transitions
        
        Returns:
            her_transitions: list of HER transitions
            augmented_transitions: list of augmented transitions
            add_to_buffer: boolean to add to buffer
        """
        # init
        her_transitions, augmented_transitions, self.add_to_buffer = [], [], False

        for i, trans in enumerate(transitions):
            # compute reward based on the new goal state
            # concatenate the last observation to the current observation
            # recompute the goal-box-position observation based on the reached position

            # initialize the last box position at the first transition
            if i == 0:
                self.init_box_position = trans['observations']['state'][0, -6:-3].copy()
                self.last_box_position = self.init_box_position.copy()
                self.last_tcp_pos = np.concatenate([reset_pose[:3], reset_pose[7:10]], axis=0)

            her_dict = copy.deepcopy(
                dict(
                    observations=dict(
                        state=np.concatenate([trans['observations']['state'][0, :-9], 
                            last_obs['state'][0, -6:-3] - trans['observations']['state'][0, -6:-3], 
                            trans['observations']['state'][0, -6:-3], 
                            last_obs['state'][0, -6:-3]], 
                            axis=0),
                            wrist_1_pointcloud=trans['observations']['wrist_1_pointcloud'],
                            wrist_2_pointcloud=trans['observations']['wrist_2_pointcloud'],
                            ),
                    actions=trans['actions'],
                    next_observations=dict(
                        state=np.concatenate(
                        [trans['next_observations']['state'][0, :-9], 
                            last_obs['state'][0, -6:-3] - trans['next_observations']['state'][0, -6:-3], 
                            trans['next_observations']['state'][0, -6:-3], 
                            last_obs['state'][0, -6:-3]], 
                        axis=0),
                        wrist_1_pointcloud=trans['next_observations']['wrist_1_pointcloud'],
                        wrist_2_pointcloud=trans['next_observations']['wrist_2_pointcloud'],
                        ),
                    # compute reward based on the new goal state
                    rewards=self.compute_reward_her(
                        obs=np.concatenate(
                            [trans['observations']['state'][0, :-9], 
                            last_obs['state'][0, -6:-3] - trans['observations']['state'][0, -6:-3], 
                            trans['observations']['state'][0, -6:-3], 
                            last_obs['state'][0, -6:-3]], 
                            axis=0
                            ), # use the new observation vector
                        action=trans['actions'], 
                        goal_position=last_obs['state'][0, -6:-3],
                        reset_pose=reset_pose
                        ), # TODO: implement this function
                    masks=1.0 - self.success,
                    dones=self.success,
                )
            )
            her_transitions.append(her_dict)

            # store the last box position for the next transition
            self.last_box_position = trans['observations']['state'][0, -6:-3].copy()

            augm_dict = copy.deepcopy(
                dict(
                    observations=dict(
                        state=np.concatenate(
                        [trans['observations']['state'][0, :-3], goal_position], axis=0
                        ),
                        wrist_1_pointcloud=trans['observations']['wrist_1_pointcloud'],
                        wrist_2_pointcloud=trans['observations']['wrist_2_pointcloud'],
                    ),
                    actions=trans['actions'],
                    next_observations=dict(
                        state=np.concatenate(
                        [trans['next_observations']['state'][0, :-3], goal_position], 
                        axis=0),
                        wrist_1_pointcloud=trans['next_observations']['wrist_1_pointcloud'],
                        wrist_2_pointcloud=trans['next_observations']['wrist_2_pointcloud'],
                    ),
                    rewards=trans['rewards'],
                    masks=trans['masks'],
                    dones=trans['dones'],
                )
            )
            augmented_transitions.append(augm_dict)

            # cut episode length if success is achieved
            if self.success:
                self.add_to_buffer = True
                self.success = False
                break

        self.last_action[:] = 0.

        # print(f"Costs: {self.costs}")

        return her_transitions, augmented_transitions, self.add_to_buffer

    def unscale_obs(self, obs):
        """
        Unscale the observation before computing the episode reward.

        Args:
            obs: observation
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
