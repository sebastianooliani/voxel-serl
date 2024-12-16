import numpy as np
from scipy.spatial.transform import Rotation as R
from franka_env.utils.transformations import (
    construct_homogeneous_matrix
)
import copy
from pprint import pprint
import pandas as pd

class HER():
    def __init__(self, scale=False):        
        self.T_O1_O2=np.array([[0., 1., 0., -0.945], 
                                [-1., 0., 0., -0.], 
                                [0., 0., 1., 0.01], 
                                [0., 0., 0., 1.]], dtype=np.float32)
        self.T_EE_SC=np.array([[1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., 0.130],
                                [0., 0., 0., 1.]], dtype=np.float32)
        
        self.last_action = np.zeros((14,))

        self.translation_scale=100.
        self.rotation_scale=10.
        self.force_scale=1.
        self.torque_scale=10.

        self.scale=scale

    ##########################################################################################
    #                               HER: her reward computation                              #
    ##########################################################################################
    # observation space
    # action -> 0:14
    # gripper state -> 14:18
    # joint position -> 18:30
    # tcp force -> 30:36
    # tcp pos diff -> 36:39
    # tcp pose -> 39:51
    # tcp torque -> 51:57
    # tcp velocity -> 57:69
    # goal box position -> 69:72
    # box position -> 72:75
    # goal position -> 75:78

    def compute_reward_her(self, 
                            obs, 
                            action, 
                            goal_position,
                            reset_pose,
                            ) -> float:
        
        def convert_pose_2_7dim(pose):
            return np.concatenate([pose[:3], 
                                   (R.from_mrp(pose[3:6])).as_quat(), 
                                   pose[6:9], 
                                   (R.from_mrp(pose[9:])).as_quat()], 
                                   axis=0)
        
        def reached_goal_state_her(obs) -> bool:
            return np.linalg.norm(obs[69:72]) < 0.05 \
                and 0.1 < obs[14:18][0] < 1. \
                    and 0.1 < obs[14:18][2] < 1.

        if self.scale:
            self.unscale_obs(obs)

        obs = self.transform_obs(tcp_pose=obs[39:51], obs=obs)
        
        tcp_pose = obs[39:51]
        tcp_pose = convert_pose_2_7dim(tcp_pose)

        action_cost = 0.1 * np.sum(np.power(action, 2))
        action_diff_cost = 0.1 * np.sum(np.power(obs[:14] - self.last_action, 2))
        self.last_action[:] = action
        
        # STEP: penalize each step
        step_cost = 0.1

        # SUCTION: reward for successful grip and cost for unnecessary suctioning
        suction_reward = 0.5 * 3 * (float(obs[14:18][1] > 0.5) + float(obs[14:18][3] > 0.5))
        suction_cost = 0.5 * 3. * (float(obs[14:18][1] < -0.5) + float(obs[14:18][3] < -0.5))

        # ORIENTATION: penalize deviating too much from the starting pose
        orientation_cost = 0
        orientation_cost = 0.5 - sum(tcp_pose[3:7] * reset_pose[3:7]) ** 2
        orientation_cost += 0.5 - sum(tcp_pose[10:] * reset_pose[10:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * 25.

        # POSITION: penalize deviating too much from the starting pose
        max_pose_diff = 0.05  # set to 5cm
        pos_diff = np.concatenate([tcp_pose[:2] - reset_pose[:2], tcp_pose[7:9] - reset_pose[7:9]])
        position_cost = 10. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, 
                     np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 
                     0.0)
        )

        # 3D DISTANCE: penalize the distance between the two robots' end-effectors
        # TODO: adjust reference frames and relative base positions
        T_O1_E1 = construct_homogeneous_matrix(tcp_pose[:7])
        T_O2_E2 = construct_homogeneous_matrix(tcp_pose[7:])
        T_O1_SC1 = T_O1_E1 @ self.T_EE_SC
        T_O1_SC2 = self.T_O1_O2 @ T_O2_E2 @ self.T_EE_SC
        distance_cost = 1. / np.linalg.norm(T_O1_SC1[:3, 3] - T_O1_SC2[:3, 3])

        # print(f"distance_cost: {distance_cost}, orientation_cost: {orientation_cost}, position_cost: {position_cost}, action_diff_cost: {action_diff_cost}, action_cost: {action_cost}, suction_cost: {suction_cost}, step_cost: {step_cost}, suction_reward: {suction_reward}")
        # with open('/home/sebastiano/voxel-serl/serl_robot_infra/ur_env/utils/her_costs.txt', 'a') as f:
        #     f.write(f"distance_cost: {distance_cost}, orientation_cost: {orientation_cost}, position_cost: {position_cost}, action_diff_cost: {action_diff_cost}, action_cost: {action_cost}, suction_cost: {suction_cost}, step_cost: {step_cost}, suction_reward: {suction_reward}\n")
        # with open('/home/sebastiano/voxel-serl/serl_robot_infra/ur_env/utils/her_obs.txt', 'a') as f:
        #     f.write(f"{obs}\n")

        if reached_goal_state_her(obs):
            self.last_action[:] = 0.
            R_goal = 100.
            return R_goal - action_cost - orientation_cost - position_cost - action_diff_cost - distance_cost
        else:
            return 0. + suction_reward - action_cost - orientation_cost - position_cost - \
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
        """

        for trans in transitions:
            # compute reward based on the new goal state
            # concatenate the last observation to the current observation
            # recompute the goal-box-position observation based on the reached position

            # observation space
            # action -> 0:14
            # gripper state -> 14:18
            # joint position -> 18:30
            # tcp force -> 30:36
            # tcp pos diff -> 36:39
            # tcp pose -> 39:51
            # tcp torque -> 51:57
            # tcp velocity -> 57:69
            # goal box position -> 69:72
            # box position -> 72:75
            # goal position -> 75:78

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
                    masks=trans['masks'],
                    dones=trans['dones'],
                )
            )
            her_transitions.append(her_dict)

            pprint(her_dict)
            # df = pd.DataFrame(her_transitions)
            # df.to_excel("her_dict.xlsx")
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
            # df = pd.DataFrame(augmented_transitions)
            # df.to_excel("augm_dict.xlsx")

        return her_transitions, augmented_transitions

    def unscale_obs(self, obs):
        """
        Unscale the observation before computing the episode reward.
        """
        obs[30:36] /= self.force_scale

        obs[36:39] /= self.translation_scale

        obs[39:42] /= self.translation_scale
        obs[42:45] /= self.rotation_scale
        obs[45:48] /= self.translation_scale
        obs[48:51] /= self.rotation_scale

        obs[51:57] /= self.torque_scale

        obs[57:60] /= self.translation_scale
        obs[60:63] /= self.rotation_scale
        obs[63:66] /= self.translation_scale
        obs[66:69] /= self.rotation_scale

        obs[69:72] /= self.translation_scale
        obs[72:75] /= self.translation_scale
        obs[75:78] /= self.translation_scale

    def scale_obs(self, obs):
        """
        Scale the observation before saving the episode's transitions.
        """
        obs[30:36] *= self.force_scale

        obs[36:39] *= self.translation_scale

        obs[39:42] *= self.translation_scale
        obs[42:45] *= self.rotation_scale
        obs[45:48] *= self.translation_scale
        obs[48:51] *= self.rotation_scale

        obs[51:57] *= self.torque_scale

        obs[57:60] *= self.translation_scale
        obs[60:63] *= self.rotation_scale
        obs[63:66] *= self.translation_scale
        obs[66:69] *= self.rotation_scale

        obs[69:72] *= self.translation_scale
        obs[72:75] *= self.translation_scale
        obs[75:78] *= self.translation_scale

    def transform_obs(self, tcp_pose, obs):
        """
        Transform the observation before computing the episode reward.
        """
        self.R_1 = R.from_mrp(tcp_pose[3:6]).as_matrix()
        self.R_2 = R.from_mrp(tcp_pose[9:]).as_matrix()

        # action -> 0:14
        obs[0:3] = self.R_1 @ obs[0:3]
        obs[3:6] = self.R_1 @ obs[3:6]
        obs[7:10] = self.R_2 @ obs[7:10]
        obs[10:13] = self.R_2 @ obs[10:13]

        # tcp force -> 30:36
        obs[30:33] = self.R_1 @ obs[30:33]
        obs[33:36] = self.R_2 @ obs[33:36]

        # tcp pose -> 39:51
        obs[39:42] = self.R_1 @ obs[39:42]
        obs[42:45] = self.R_1 @ obs[42:45]
        obs[45:48] = self.R_2 @ obs[46:49]
        obs[48:51] = self.R_2 @ obs[48:51]

        # tcp torque -> 51:57
        obs[51:54] = self.R_1 @ obs[51:54]
        obs[54:57] = self.R_2 @ obs[54:57]

        # tcp velocity -> 57:69
        obs[57:60] = self.R_1 @ obs[57:60]
        obs[60:63] = self.R_1 @ obs[60:63]
        obs[63:66] = self.R_2 @ obs[63:66]
        obs[66:69] = self.R_2 @ obs[66:69]

        return obs


