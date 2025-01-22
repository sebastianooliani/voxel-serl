import numpy as np
import copy
import asyncio
from scipy.spatial.transform import Rotation as R

from ur_env.envs.ur5_env import UR5Env, UR5DualRobotEnv
from ur_env.envs.camera_env.config import UR5CameraConfigFinal, UR5CameraConfigFinalTests, UR5CameraConfigFinalEvaluation, UR5CameraConfigDemo, UR5CameraConfigDualRobot

from franka_env.utils.transformations import (
    construct_homogeneous_matrix,
    orientation_difference_angle_axis,
)

class UR5CameraEnv(UR5Env):
    def __init__(self, load_config=True, **kwargs):
        if load_config:
            super().__init__(**kwargs, config=UR5CameraConfigFinal)
        else:
            super().__init__(**kwargs)

    def compute_reward(self, obs, action) -> float:
        action_cost = 0.1 * np.sum(np.power(action, 2))
        action_diff_cost = 0.1 * np.sum(np.power(obs["state"]["action"] - self.last_action, 2))
        self.last_action[:] = action
        step_cost = 0.1

        suction_reward = 0.3 * float(obs["state"]["gripper_state"][1] > 0.5)
        suction_cost = 3. * float(obs["state"]["gripper_state"][1] < -0.5)

        orientation_cost = 1. - sum(obs["state"]["tcp_pose"][3:] * self.curr_reset_pose[3:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * 25.

        max_pose_diff = 0.05  # set to 5cm
        pos_diff = obs["state"]["tcp_pose"][:2] - self.curr_reset_pose[:2]
        position_cost = 10. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0)
        )

        cost_info = dict(
            action_cost=action_cost,
            step_cost=step_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            orientation_cost=orientation_cost,
            position_cost=position_cost,
            action_diff_cost=action_diff_cost,
            total_cost=-(-action_cost - step_cost + suction_reward - suction_cost - orientation_cost - position_cost - action_diff_cost)
        )
        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])

        if self.reached_goal_state(obs):
            self.last_action[:] = 0.
            return 100. - action_cost - orientation_cost - position_cost - action_diff_cost
        else:
            return 0. + suction_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost - action_diff_cost

    def reached_goal_state(self, obs) -> bool:
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        state = obs["state"]
        # print(state['tcp_pose'][2] - self.curr_reset_pose[2])
        return 0.1 < state['gripper_state'][0] < 1. and state['tcp_pose'][2] > self.curr_reset_pose[2] + 0.05 # +1cm

    def close(self):
        super().close()

############################################################################################################

class UR5CameraEnvDualRobot(UR5DualRobotEnv):
    def __init__(self, load_config=True, **kwargs):
        if load_config:
            super().__init__(**kwargs, config=UR5CameraConfigDualRobot)
        else:
            super().__init__(**kwargs)
    
    def compute_reward(self, obs, action) -> float:
        # TODO: adjust actions dimensions
        # ACTION: penalize large action and action difference
        action_cost = self.config.ACTION_WEIGHT * np.sum(np.power(action, 2))
        action_diff_cost = self.config.ACTION_WEIGHT * np.sum(np.power(obs["state"]["action"] - self.last_action, 2))
        self.last_action[:] = action
        
        # STEP: penalize each step
        step_cost = self.config.STEP_WEIGHT

        # SUCTION: reward for successful grip and cost for unnecessary suctioning
        suction_reward = self.config.SUCTION_WEIGHT * (float(obs["state"]["gripper_state"][1] > 0.5) + float(obs["state"]["gripper_state"][3] > 0.5))
        suction_cost = self.config.SUCTION_WEIGHT * (float(obs["state"]["gripper_state"][1] < -0.5) + float(obs["state"]["gripper_state"][3] < -0.5))

        # ORIENTATION: penalize deviating too much from the starting pose
        orientation_cost = 0
        orientation_cost = 0.5 - sum(obs["state"]["tcp_pose"][3:7] * self.curr_reset_pose[3:7]) ** 2
        orientation_cost += 0.5 - sum(obs["state"]["tcp_pose"][10:] * self.curr_reset_pose[10:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * self.config.ORIENTATION_WEIGHT

        # POSITION: penalize deviating too much from the starting pose
        max_pose_diff = 0.05  # set to 5cm
        pos_diff = np.concatenate([obs["state"]["tcp_pose"][:2] - self.curr_reset_pose[:2], obs["state"]["tcp_pose"][7:9] - self.curr_reset_pose[7:9]])
        position_cost = self.config.POSITION_WEIGHT * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0)
        )

        # 3D DISTANCE: penalize the distance between the two robots' end-effectors
        # TODO: adjust reference frames and relative base positions
        if self.camera_mode in ["none"]:
            # when successfully runnning sac without wrist cameras, these were not used.
            distance_cost = 0
            grasp_reward = 0
        else:
            T_O1_E1 = construct_homogeneous_matrix(obs["state"]["tcp_pose"][:7])
            T_O2_E2 = construct_homogeneous_matrix(obs["state"]["tcp_pose"][7:])
            T_O1_SC1 = T_O1_E1 @ self.T_EE_SC
            T_O1_SC2 = self.T_O1_O2 @ T_O2_E2 @ self.T_EE_SC
            distance_cost = self.config.DISTANCE_WEIGHT / np.linalg.norm(T_O1_SC1[:3, 3] - T_O1_SC2[:3, 3])

            grasp_reward = self.config.GRASP_WEIGHT * (float(obs["state"]["gripper_state"][1] > 0.5) * np.max(obs["state"]["tcp_pose"][2], 0) +
                             float(obs["state"]["gripper_state"][3] > 0.5) * np.max(obs["state"]["tcp_pose"][9], 0))
            
            # suction_reward = 0.

        # TOTAL COST
        cost_info = dict(
            action_cost=action_cost,
            step_cost=step_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            orientation_cost=orientation_cost,
            position_cost=position_cost,
            action_diff_cost=action_diff_cost,
            distance_cost=distance_cost,
            grasp_reward=grasp_reward,
            total_cost=-(-action_cost - step_cost + suction_reward + grasp_reward - suction_cost - orientation_cost - position_cost - action_diff_cost - distance_cost),
        )
        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])
        
        if self.reached_goal_state(obs):
            print("\nSuccessfull lift!\n")
            self.last_action[:] = 0.
            R_goal = 100. if self.camera_mode in ["none"] else self.config.SUCCESS_WEIGHT
            return R_goal - action_cost - orientation_cost - position_cost - action_diff_cost
        else:
            return 0. + suction_reward + grasp_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost - action_diff_cost - distance_cost

    def reached_goal_state(self, obs) -> bool:
        # TODO: adjust this to dual robot
        state = obs["state"]
        # add condition for second robot
        return (0.1 < state['gripper_state'][0] < 1. and state['tcp_pose'][2] > self.curr_reset_pose[2] + 0.05) and \
            (0.1 < state['gripper_state'][2] < 1. and state['tcp_pose'][9] > self.curr_reset_pose[9] + 0.05) # +1cm for success
    
    def close(self):
        super().close()


############################################################################################################

class UR5CameraEnvDualRobotMotionPlanning(UR5DualRobotEnv):
    def __init__(self, load_config=True, **kwargs):
        if load_config:
            super().__init__(**kwargs, config=UR5CameraConfigDualRobot)
        else:
            super().__init__(**kwargs)

    def _get_obs(self, action) -> dict:
        # get image before state observation, so they match better in time

        images = None
        if self.camera_mode is not None:
            images = self.get_image()

        if self.pose_est:
            self._update_box_pos_estimate()
        else:
            self.box_position = np.array([0.5, 0.5, 0.5])

        self._update_currpos()
        state_observation = {
            "tcp_pose": self.curr_pos,
            "tcp_vel": self.curr_vel,
            "gripper_state": self.gripper_state,
            "tcp_force": self.curr_force,
            "tcp_torque": self.curr_torque,
            "action": action,
            # TODO: add my custom observations here
            "tcp_pos_diff": self.curr_pos[:3] - (self.T_O1_O2 @ np.concatenate([self.curr_pos[7:10], [1.]]))[:3],
            "joint_positions": self.curr_Q,
            # motion planning observations
            "goal_box_position": self.goal_position - self.box_position,
            "box_position": self.box_position,
            "goal_position": self.goal_position,
        }

        if images is not None:
            return copy.deepcopy(dict(images=images, state=state_observation))
        else:
            return copy.deepcopy(dict(state=state_observation))

    def compute_reward(self, obs, action) -> float:
        action_cost = 0.1 * np.sum(np.power(action, 2))
        action_diff_cost = 0.1 * np.sum(np.power(obs["state"]["action"] - self.last_action, 2))
        self.last_action[:] = action
        
        # STEP: penalize each step
        step_cost = 0.1

        # SUCTION: reward for successful grip and cost for unnecessary suctioning
        suction_reward = 5 * 0.3 * (float(obs["state"]["gripper_state"][1] > 0.5) + float(obs["state"]["gripper_state"][3] > 0.5))
        suction_cost = 0.5 * 3. * (float(obs["state"]["gripper_state"][1] < -0.5) + float(obs["state"]["gripper_state"][3] < -0.5))

        # ORIENTATION: penalize deviating too much from the starting pose
        orientation_cost = 0
        orientation_cost = 0.5 - sum(obs["state"]["tcp_pose"][3:7] * self.curr_reset_pose[3:7]) ** 2
        orientation_cost += 0.5 - sum(obs["state"]["tcp_pose"][10:] * self.curr_reset_pose[10:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * 25.

        # POSITION: penalize deviating too much from the starting pose
        max_pose_diff = 0.05  # set to 5cm
        pos_diff = np.concatenate([obs["state"]["tcp_pose"][:2] - self.curr_reset_pose[:2], obs["state"]["tcp_pose"][7:9] - self.curr_reset_pose[7:9]])
        position_cost = 10. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, 
                     np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 
                     0.0)
        )

        # 3D DISTANCE: penalize the distance between the two robots' end-effectors
        # TODO: adjust reference frames and relative base positions
        T_O1_E1 = construct_homogeneous_matrix(obs["state"]["tcp_pose"][:7])
        T_O2_E2 = construct_homogeneous_matrix(obs["state"]["tcp_pose"][7:])
        T_O1_SC1 = T_O1_E1 @ self.T_EE_SC
        T_O1_SC2 = self.T_O1_O2 @ T_O2_E2 @ self.T_EE_SC
        distance_cost = 1. / np.linalg.norm(T_O1_SC1[:3, 3] - T_O1_SC2[:3, 3])

        # TOTAL COST
        cost_info = dict(
            action_cost=action_cost,
            step_cost=step_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            orientation_cost=orientation_cost,
            position_cost=position_cost,
            action_diff_cost=action_diff_cost,
            distance_cost=distance_cost,
            total_cost=-(-action_cost - step_cost + suction_reward - suction_cost - orientation_cost - position_cost - action_diff_cost - distance_cost),
        )
        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])
        
        if self.reached_goal_state(obs):
            self.last_action[:] = 0.
            R_goal = 100.
            return R_goal - action_cost - orientation_cost - position_cost - action_diff_cost - distance_cost
        else:
            return 0. + suction_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost - action_diff_cost - distance_cost
    
    def reached_goal_state(self, obs) -> bool:
        state = obs['state']
        return np.linalg.norm(state['goal_box_position']) < 0.05 and 0.1 < state['gripper_state'][0] < 1. and 0.1 < state['gripper_state'][2] < 1.
    
############################################################################################################

class UR5CameraEnvDualRobotReorientation(UR5DualRobotEnv):
    def __init__(self, load_config=True, **kwargs):
        if load_config:
            super().__init__(**kwargs, config=UR5CameraConfigDualRobot)
            self.init = True # read and write the initial box orientation
        else:
            super().__init__(**kwargs)

    def _get_obs(self, action) -> dict:
        # get image before state observation, so they match better in time

        images = None
        if self.camera_mode is not None:
            images = self.get_image()

        if self.pose_est:
            self._update_box_orientation_estimate()
            if self.init:
                self.init_box_orientation = self.box_orientation
                self.init = False
                
        self._update_currpos()
        state_observation = {
            "tcp_pose": self.curr_pos,
            "tcp_vel": self.curr_vel,
            "gripper_state": self.gripper_state,
            "tcp_force": self.curr_force,
            "tcp_torque": self.curr_torque,
            "action": action,
            # TODO: add my custom observations here
            "tcp_pos_diff": self.curr_pos[:3] - (self.T_O1_O2 @ np.concatenate([self.curr_pos[7:10], [1.]]))[:3],
            "joint_positions": self.curr_Q,
            # TODO: reorientation observations
            "box_orientation": R.from_rotvec(self.box_orientation).as_mrp(), # in the neural network, orientation is represented in MRP
        }

        if images is not None:
            return copy.deepcopy(dict(images=images, state=state_observation))
        else:
            return copy.deepcopy(dict(state=state_observation))
        
    def compute_reward(self, obs, action) -> float:
        action_cost = 0.1 * np.sum(np.power(action, 2))
        action_diff_cost = 0.1 * np.sum(np.power(obs["state"]["action"] - self.last_action, 2))
        self.last_action[:] = action
        
        # STEP: penalize each step
        step_cost = 0.1

        # SUCTION: reward for successful grip and cost for unnecessary suctioning
        suction_reward = 5 * 0.3 * (float(obs["state"]["gripper_state"][1] > 0.5) + float(obs["state"]["gripper_state"][3] > 0.5))
        suction_cost = 0.5 * 3. * (float(obs["state"]["gripper_state"][1] < -0.5) + float(obs["state"]["gripper_state"][3] < -0.5))

        # ORIENTATION: penalize deviating too much from the starting pose
        orientation_cost = 0
        orientation_cost = 0.5 - sum(obs["state"]["tcp_pose"][3:7] * self.curr_reset_pose[3:7]) ** 2
        orientation_cost += 0.5 - sum(obs["state"]["tcp_pose"][10:] * self.curr_reset_pose[10:]) ** 2
        orientation_cost = max(orientation_cost - 0.005, 0.) * 25.

        # POSITION: penalize deviating too much from the starting pose
        max_pose_diff = 0.05
        pos_diff = np.concatenate([obs["state"]["tcp_pose"][:2] - self.curr_reset_pose[:2], obs["state"]["tcp_pose"][7:9] - self.curr_reset_pose[7:9]])
        position_cost = 10. * np.sum(
            np.where(np.abs(pos_diff) > max_pose_diff, 
                     np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 
                     0.0)
        )

        # 3D DISTANCE: penalize the distance between the two robots' end-effectors
        if self.camera_mode in ["none"]:
            distance_cost = 0
        else:
            T_O1_E1 = construct_homogeneous_matrix(obs["state"]["tcp_pose"][:7])
            T_O2_E2 = construct_homogeneous_matrix(obs["state"]["tcp_pose"][7:])
            T_O1_SC1 = T_O1_E1 @ self.T_EE_SC
            T_O1_SC2 = self.T_O1_O2 @ T_O2_E2 @ self.T_EE_SC
            distance_cost = 1. / np.linalg.norm(T_O1_SC1[:3, 3] - T_O1_SC2[:3, 3])

        # TOTAL COST
        cost_info = dict(
            action_cost=action_cost,
            step_cost=step_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            orientation_cost=orientation_cost,
            position_cost=position_cost,
            action_diff_cost=action_diff_cost,
            distance_cost=distance_cost,
            total_cost=-(-action_cost - step_cost + suction_reward - suction_cost - orientation_cost - position_cost - action_diff_cost - distance_cost),
        )
        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])
        
        if self.reached_goal_state(obs):
            self.last_action[:] = 0.
            R_goal = 100.
            return R_goal - action_cost - orientation_cost - position_cost - action_diff_cost - distance_cost
        else:
            return 0. + suction_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost - action_diff_cost - distance_cost
    
    def reached_goal_state(self, obs) -> bool:
        state = obs['state']
        # TODO: fix orientation error threshold
        # perform a 90° degrees rotation around the z-axis
        # convert obs from MRP to rotation vector
        rot_angle, _ = orientation_difference_angle_axis(self.init_box_orientation, R.from_mrp(state['box_orientation']).as_rotvec())
        # 0.09 rad = 5° tolerance
        return (np.abs(rot_angle) - np.pi/2) < 0.09 and 0.1 < state['gripper_state'][0] < 1. and 0.1 < state['gripper_state'][2] < 1.

############################################################################################################

class UR5CameraEnvTest(UR5CameraEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, load_config=False, config=UR5CameraConfigFinalTests)


class UR5CameraEnvEval(UR5CameraEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, load_config=False, config=UR5CameraConfigFinalEvaluation)

class UR5CameraEnvDemo(UR5CameraEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, load_config=False, config=UR5CameraConfigDemo)
