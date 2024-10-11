import numpy as np
from typing import Tuple

from ur_env.envs.ur5_env import UR5Env, UR5DualRobotEnv
from ur_env.envs.camera_env.config import UR5CameraConfigFinal, UR5CameraConfigFinalTests, UR5CameraConfigFinalEvaluation, UR5CameraConfigDemo, UR5CameraConfigDualRobot


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
        return 0.1 < state['gripper_state'][0] < 1. and state['tcp_pose'][2] > self.curr_reset_pose[2] + 0.01  # +1cm

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
            R_goal = 100.
            return R_goal - action_cost - orientation_cost - position_cost - action_diff_cost
        else:
            return 0. + suction_reward - action_cost - orientation_cost - position_cost - \
                suction_cost - step_cost - action_diff_cost

    def reached_goal_state(self, obs) -> bool:
        # TODO: adjust this to dual robot
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        state = obs["state"]
        return 0.1 < state['gripper_state'][0] < 1. and state['tcp_pose'][2] > self.curr_reset_pose[2] + 0.01  # +1cm

    def close(self):
        super().close()


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
