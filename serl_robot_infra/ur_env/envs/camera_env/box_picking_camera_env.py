import numpy as np
from typing import Tuple
import jax
import jax.numpy as jnp

from ur_env.envs.ur5_env import UR5Env, UR5DualRobotEnv
from ur_env.envs.camera_env.config import UR5CameraConfigFinal, UR5CameraConfigFinalTests, UR5CameraConfigFinalEvaluation, UR5CameraConfigDemo, UR5CameraConfigDualRobot

from scipy.spatial.transform import Rotation as R
from jax.scipy.spatial.transform import Rotation as jR


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
            self.T_O1_O2 = UR5CameraConfigDualRobot.T_O1_O2
            self.T_EE_SC = UR5CameraConfigDualRobot.T_EE_SC

            # initialize jit methods
            self._compute_end_effector_distance = jax.jit(self._compute_end_effector_distance_raw)
        else:
            super().__init__(**kwargs)

    @jax.jit
    def _compute_end_effector_distance_raw(self, target_pos: np.ndarray) -> float:
        """
        Jitted method to compute the distance between the two end effectors.
        
        Args:
            target_pos (np.ndarray): The target position of the end effectors.
            
        Returns:
            float: The distance between the two end effectors.
        """

        T_O1_E1 = np.eye(4)
        rotation = R.from_quat(target_pos[3:7]).as_matrix()
        translation = np.array(target_pos[:3])
        T_O1_E1[:3, :3] = rotation
        T_O1_E1[:3, 3] = translation
        
        T_O2_E2 = np.eye(4)
        rotation = R.from_quat(target_pos[10:13]).as_matrix()
        translation = np.array(target_pos[13:])
        T_O2_E2[:3, :3] = rotation
        T_O2_E2[:3, 3] = translation

        T_O1_SC1 = T_O1_E1 @ self.T_EE_SC
        T_O2_SC2 = T_O2_E2 @ self.T_EE_SC
        T_O1_SC2 = self.T_O1_O2 @ T_O2_SC2

        return np.sum(np.power(T_O1_SC1[:3, 3] - T_O1_SC2[:3, 3], 2))
    
    def compute_reward(self, obs, action) -> float:
        # TODO: adjust actions dimensions
        # ACTION: penalize large action and action difference
        action_cost = 0.1 * np.sum(np.power(action, 2))
        action_diff_cost = 0.1 * np.sum(np.power(obs["state"]["action"] - self.last_action, 2))
        self.last_action[:] = action
        
        # STEP: penalize each step
        step_cost = 0.1

        # SUCTION: reward for successful grip and cost for unnecessary suctioning
        suction_reward = 0.5 * 0.3 * (float(obs["state"]["gripper_state"][1] > 0.5) + float(obs["state"]["gripper_state"][3] > 0.5))
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
            np.where(np.abs(pos_diff) > max_pose_diff, np.abs(pos_diff - np.sign(pos_diff) * max_pose_diff), 0.0)
        )

        # 3D DISTANCE: penalize the distance between the two robots' end-effectors
        # TODO: adjust reference frames and relative base positions
        @jax.jit
        def compute_end_effector_distance_raw(target_pos, T_O1_O2, T_EE_SC):
            """
            Jitted method to compute the distance between the two end effectors.
            
            Args:
                target_pos (np.ndarray): The target position of the end effectors.
                
            Returns:
                float: The distance between the two end effectors.
            """

            def quat_to_matrix(quat):
                """
                Converts a quaternion to a rotation matrix.
                
                Args:
                    quat (jnp.ndarray): Quaternion [x, y, z, w].

                Returns:
                    jnp.ndarray: 3x3 rotation matrix.
                """
                x, y, z, w = quat
                xx, yy, zz = x*x, y*y, z*z
                xy, xz, yz = x*y, x*z, y*z
                wx, wy, wz = w*x, w*y, w*z

                return jnp.array([
                    [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                    [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                    [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
                ], dtype=jnp.float32)

            T_O1_E1 = jnp.eye(4, dtype=jnp.float32)
            rotation = jR.from_quat(target_pos[3:7]).as_matrix()
            translation = target_pos[:3]
            T_O1_E1 = T_O1_E1.at[:3, :3].set(rotation)
            T_O1_E1 = T_O1_E1.at[:3, 3].set(translation)
            
            T_O2_E2 = jnp.eye(4, dtype=jnp.float32)
            rotation = jR.from_quat(target_pos[7:11]).as_matrix()
            translation = target_pos[11:]
            T_O2_E2 = T_O2_E2.at[:3, :3].set(rotation)
            T_O2_E2 = T_O2_E2.at[:3, 3].set(translation)

            T_O1_SC1 = T_O1_E1 @ T_EE_SC
            T_O2_SC2 = T_O2_E2 @ T_EE_SC
            T_O1_SC2 = T_O1_O2 @ T_O2_SC2

            return jnp.sum(jnp.power(T_O1_SC1[:3, 3] - T_O1_SC2[:3, 3], 2))
        
        distance_cost = 1. * compute_end_effector_distance_raw(obs["state"]["tcp_pose"], self.T_O1_O2, self.T_EE_SC)

        # TOTAL COST
        cost_info = dict(
            action_cost=action_cost,
            step_cost=step_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            orientation_cost=orientation_cost,
            position_cost=position_cost,
            action_diff_cost=action_diff_cost,
            total_cost=-(-action_cost - step_cost + suction_reward - suction_cost - orientation_cost - position_cost - action_diff_cost),
            distance_cost=distance_cost
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
        # TODO: adjust this to dual robot
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        state = obs["state"]
        # add condition for second robot
        # print(f"{state['tcp_pose'][2] - self.curr_reset_pose[2]}, {state['tcp_pose'][9] - self.curr_reset_pose[9]}")
        return 0.1 < state['gripper_state'][0] < 1. and state['tcp_pose'][2] > self.curr_reset_pose[2] + 0.05 and 0.1 < state['gripper_state'][2] < 1. and state['tcp_pose'][9] > self.curr_reset_pose[9] + 0.05 # +1cm for success

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
