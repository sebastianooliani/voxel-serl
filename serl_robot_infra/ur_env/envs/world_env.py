import numpy as np
import gymnasium as gym
from gym import Env
from scipy.spatial.transform import Rotation as R

from ur_env.envs.camera_env.config import UR5CameraConfigDualRobot

class WorldFrameEnv(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

        self.config = UR5CameraConfigDualRobot()

        self.T = self.config.T_O1_O2
        self.rot = self.T[:3, :3]
        self.t = self.T[:3, 3]
        self.quat = R.from_matrix(self.rot).as_quat()

        self.T_inv = np.linalg.inv(self.T)
        self.R_inv = self.T_inv[:3, :3]
        self.t_inv = self.T_inv[:3, 3]
        self.quat_inv = R.from_matrix(self.R_inv).as_quat()

        # in case it is gonna be needed
        self.task = self.config.TASK

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        return self.transform_obs_reset(obs), info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        transformed_obs = self.transform_obs(obs)

        return transformed_obs, reward, done, truncated, info
    
    def transform_obs(self, obs):
        obs["state"]["tcp_pos"][7:10] = (self.T @ np.concatenate([obs["state"]["tcp_pos"][7:10]], [1]))[:3]
        obs["state"]["tcp_pos"][10:] = R.from_matrix(R.from_quat(self.quat).as_matrix() @ R.from_quat(obs["state"]["tcp_pos"][10:]).as_matrix()).as_quat()

        obs["state"]["tcp_vel"][6:9] = (self.T @ np.concatenate([obs["state"]["tcp_vel"][6:9]], [1]))[:3]
        obs["state"]["tcp_vel"][9:] = self.rot @ obs["state"]["tcp_vel"][9:]

        obs["state"]["tcp_force"][3:] = (self.T @ np.concatenate([obs["state"]["tcp_force"][3:]], [1]))[:3]
        obs["state"]["tcp_torque"][3:] = (self.T @ np.concatenate([obs["state"]["tcp_torque"][3:]], [1]))[:3]

        return obs

    def transform_obs_reset(self, obs):
        obs["state"]["tcp_pos"][7:10] = (self.T_inv @ np.concatenate([obs["state"]["tcp_pos"][7:10]], [1]))[:3]
        obs["state"]["tcp_pos"][10:] = (np.linalg.inv(R.from_quat(self.quat_inv).as_matrix()) @ R.from_quat(obs["state"]["tcp_pos"][10:]).as_matrix()).as_quat()

        obs["state"]["tcp_vel"][6:9] = (self.T_inv @ np.concatenate([obs["state"]["tcp_vel"][6:9]], [1]))[:3]
        obs["state"]["tcp_vel"][9:] = self.R_inv @ obs["state"]["tcp_vel"][9:]

        obs["state"]["tcp_force"][3:] = (self.T_inv @ np.concatenate([obs["state"]["tcp_force"][3:]], [1]))[:3]
        obs["state"]["tcp_torque"][3:] = (self.T_inv @ np.concatenate([obs["state"]["tcp_torque"][3:]], [1]))[:3]

        return obs