from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gym import Env
from robotiq_env.utils.transformations import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)


class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    # TODO delete base frame rotation afterwards
    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to desired frame
        transformed_action = self.transform_action(action)

        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)

        # s = f"pose: {obs['state']['tcp_pose']} vel: {obs['state']['tcp_vel']} force: {obs['state']['tcp_force']}"
        # s += f"torque: {obs['state']['tcp_torque']}"
        # print(s)

        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # obs['state']['tcp_pose'][:2] -= info['reset_shift']  # set rel pose to original reset pose (no random)

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action


class BaseFrameRotation(gym.Wrapper):
    def __init__(self, env: Env, base_frame_R=[0., 0., 0.]):
        super().__init__(env)
        base_frame_rotation_quat = R.from_euler("xyz", base_frame_R).as_quat()
        self.base_frame_pose = np.array([0., 0., 0., *base_frame_rotation_quat])
        self.adjoint_matrix = np.zeros((6, 6))

        self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        transformed_action = self.base_transform_action(action)

        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # we skip the intervene action transformation (since we do not want the i-action to be within the base frame)

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(self.base_frame_pose)

        # Transform observation to spatial frame
        transformed_obs = self.base_transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(self.base_frame_pose)
        # Update transformation matrix from the reset pose's relative frame to base frame
        self.T_r_o_inv = np.linalg.inv(
            construct_homogeneous_matrix(self.base_frame_pose)
        )

        # Transform observation to spatial frame
        return self.base_transform_observation(obs), info

    def base_transform_observation(self, obs):
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]

        T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
        T_b_r = self.T_r_o_inv @ T_b_o

        # Reconstruct transformed tcp_pose vector
        p_b_r = T_b_r[:3, 3]
        theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
        obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        force_torque = np.concatenate((obs["state"]["tcp_force"], obs["state"]["tcp_torque"]))
        force_torque_rotated = np.transpose(self.adjoint_matrix) @ force_torque

        # print(f'force torque rotated: {force_torque_rotated}')

        obs["state"]["tcp_force"] = force_torque_rotated[:3]
        obs["state"]["tcp_torque"] = force_torque_rotated[3:]      # TODO check if this is true

        return obs

    def base_transform_action(self, action: np.ndarray):
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action