import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from franka_env.utils.transformations import (
    construct_homogeneous_matrix
)
from fast_kinematics import FastKinematics
import copy
from pprint import pprint

class HER():
    def __init__(self):
        self.file_name = "/home/sebastiano/voxel-serl/serl_robot_infra/robot_controllers/ur5.urdf"
        self. link = "ee_link"
        self.N = 1
        self.robot_model = FastKinematics(self.file_name, self.N, self.link)
        
        self.joint_positions = np.array([[- math.pi / 6. , -math.pi/2 + math.pi/24, math.pi/2 + math.pi/6, -math.pi/2 - math.pi/6 - math.pi/24, -math.pi/2, 0.,
                                math.pi + math.pi / 4, -math.pi/2 + math.pi/24, math.pi/2 + math.pi/6, -math.pi/2 - math.pi/6 - math.pi/24, -math.pi/2, 0.]], dtype=np.float32)
        
        # output of forward kinematics is position and quaternion
        self.curr_reset_pose = np.concatenate(
            [self.robot_model.forward_kinematics(self.joint_positions[0, :6].transpose()), 
             self.robot_model.forward_kinematics(self.joint_positions[0, 6:].transpose())], 
             axis=0)
        
        self.T_O1_O2=np.array([[0., 1., 0., -0.945], 
                                [-1., 0., 0., -0.], 
                                [0., 0., 1., 0.01], 
                                [0., 0., 0., 1.]], dtype=np.float32)
        self.T_EE_SC=np.array([[1., 0., 0., 0.],
                                [0., 1., 0., 0.],
                                [0., 0., 1., 0.130],
                                [0., 0., 0., 1.]], dtype=np.float32)
        
        self.last_action = np.zeros((14,))

    ################################################################################################
    #                                  HER: her reward computation                                 #
    ################################################################################################
    def compute_reward_her(self, 
                            obs, 
                            action, 
                            goal_position,
                            ) -> float:
        
        def convert_pose_2_7dim(pose):
            return np.concatenate([pose[:3], 
                                   (R.from_mrp(pose[3:6])).as_quat(), 
                                   pose[6:9], 
                                   (R.from_mrp(pose[9:])).as_quat()], 
                                   axis=0)
        
        def reached_goal_state_her(obs, goal_position) -> bool:
            return np.linalg.norm(goal_position - obs[69:72]) < 0.05 and 0.1 < obs[14:18][0] < 1. and 0.1 < obs[14:18][2] < 1.

        tcp_pose = obs[39:51]
        tcp_pose = convert_pose_2_7dim(tcp_pose)

        action_cost = 0.1 * np.sum(np.power(action, 2))
        action_diff_cost = 0.1 * np.sum(np.power(obs[:14] - self.last_action, 2))
        self.last_action[:] = action
        
        # STEP: penalize each step
        step_cost = 0.1

        # SUCTION: reward for successful grip and cost for unnecessary suctioning
        suction_reward = 5 * 0.3 * (float(obs[14:18][1] > 0.5) + float(obs[14:18][3] > 0.5))
        suction_cost = 0.5 * 3. * (float(obs[14:18][1] < -0.5) + float(obs[14:18][3] < -0.5))

        # ORIENTATION: penalize deviating too much from the starting pose
        orientation_cost = 0
        orientation_cost = 0.5 - sum(tcp_pose[3:7] * self.curr_reset_pose[3:7]) ** 2
        orientation_cost += 0.5 - sum(tcp_pose[10:] * self.curr_reset_pose[10:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * 25.

        # POSITION: penalize deviating too much from the starting pose
        max_pose_diff = 0.05  # set to 5cm
        pos_diff = np.concatenate([tcp_pose[:2] - self.curr_reset_pose[:2], tcp_pose[7:9] - self.curr_reset_pose[7:9]])
        position_cost = 10. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0)
        )

        # 3D DISTANCE: penalize the distance between the two robots' end-effectors
        # TODO: adjust reference frames and relative base positions
        T_O1_E1 = construct_homogeneous_matrix(tcp_pose[:7])
        T_O2_E2 = construct_homogeneous_matrix(tcp_pose[7:])
        T_O1_SC1 = T_O1_E1 @ self.T_EE_SC
        T_O1_SC2 = self.T_O1_O2 @ T_O2_E2 @ self.T_EE_SC
        distance_cost = 1. / np.linalg.norm(T_O1_SC1[:3, 3] - T_O1_SC2[:3, 3])
                
        if reached_goal_state_her(obs, goal_position):
            self.last_action[:] = 0.
            R_goal = 100.
            return R_goal - action_cost - orientation_cost - position_cost - action_diff_cost - distance_cost
        else:
            return 0. + suction_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost - action_diff_cost - distance_cost
        
    def process_transitions(self, transitions, last_obs, goal_position, her_transitions, augmented_transitions):
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
                        goal_position=last_obs[-6:-3]
                        ), # TODO: implement this function
                    masks=trans['masks'],
                    dones=trans['dones'],
                )
            )
            her_transitions.append(her_dict)
            pprint(her_dict)
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

        return her_transitions, augmented_transitions