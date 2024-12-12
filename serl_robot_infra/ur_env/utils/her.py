import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from franka_env.utils.transformations import (
    construct_homogeneous_matrix
)
from fast_kinematics import FastKinematics

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
        # curr_reset_pose = np.concatenate([curr_reset_pose[:3], (R.from_quat(curr_reset_pose[3:7])).as_mrp(), curr_reset_pose[7:10], (R.from_quat(curr_reset_pose[10:])).as_mrp()], axis=0)
        self.T_O1_O2=np.array([[0., 1., 0., -0.945], 
                                    [-1., 0., 0., -0.], 
                                    [0., 0., 1., 0.01], 
                                    [0., 0., 0., 1.]], dtype=np.float32)
        self.T_EE_SC=np.array([[1., 0., 0., 0.],
                                    [0., 1., 0., 0.],
                                    [0., 0., 1., 0.130],
                                    [0., 0., 0., 1.]], dtype=np.float32)

    ############################################################################################################
    #                                        HER: her reward computation                                       #
    ############################################################################################################
    def compute_reward_her(self, 
                            obs, 
                            action, 
                            goal_position,
                            last_action=np.zeros((14,)),
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
            last_action[:] = 0.
            R_goal = 100.
            return R_goal - action_cost - orientation_cost - position_cost - action_diff_cost - distance_cost
        else:
            return 0. + suction_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost - action_diff_cost - distance_cost