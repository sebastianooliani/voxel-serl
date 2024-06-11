import gymnasium as gym
import numpy as np
from agentlace import action

from robotiq_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import time
from scipy.spatial.transform import Rotation as R

from robotiq_env.utils.rotations import quat_2_euler

sigmoid = lambda x: 1 / (1 + np.exp(-x))


class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, gripper_action_span=3):
        super().__init__(env)

        self.gripper_enabled = True

        self.expert = SpaceMouseExpert()
        self.last_intervene = 0
        self.left = np.array([False] * gripper_action_span, dtype=np.bool_)
        self.right = self.left.copy()

        self.invert_axes = [1, -1, 1, 1, 1, 1]
        self.deadspace = 0.15

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a = self.get_deadspace_action()

        if np.linalg.norm(expert_a) > 0.001 or self.left.any() or self.right.any():  # also read buttons with no movement
            self.last_intervene = time.time()

        if self.gripper_enabled:
            gripper_action = np.zeros((1,)) + int(self.left.any()) - int(self.right.any())
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if time.time() - self.last_intervene < 0.5:
            expert_a = self.adapt_spacemouse_output(expert_a)
            return expert_a

        return action

    def get_deadspace_action(self) -> np.ndarray:
        expert_a, buttons = self.expert.get_action()

        positive = np.clip((expert_a - self.deadspace) / (1. - self.deadspace), a_min=0.0, a_max=1.0)
        negative = np.clip((expert_a + self.deadspace) / (1. - self.deadspace), a_min=-1.0, a_max=0.0)
        expert_a = positive + negative

        self.left, self.right = np.roll(self.left, -1), np.roll(self.right, -1)     # shift them one to the left
        self.left[-1], self.right[-1] = tuple(buttons)

        return np.array(expert_a, dtype=np.float32)

    def adapt_spacemouse_output(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - expert_a: spacemouse raw output
        Output:
        - expert_a: spacemouse output adapted to force space (action)
        """

        # position = super().get_wrapper_attr("curr_pos")  # get position from robotiq_env
        position = self.unwrapped.curr_pos
        z_angle = np.arctan2(position[1], position[0])  # get first joint angle

        action[:6] *= self.invert_axes  # if some want to be inverted
        # z_rot = R.from_rotvec(np.array([0, 0, z_angle]))
        # action[:3] = z_rot.apply(action[:3])  # z rotation invariant translation
        # action[3:6] = z_rot.inv().apply(action[3:6])  # z rotation invariant rotation

        # rotate by the task frame definition
        task_frame_rotation = R.from_euler('zyx', [0., np.pi/4., 0.]).as_matrix()
        action[:3] = np.dot(task_frame_rotation, action[:3])

        return action

    def step(self, action):
        new_action = self.action(action)
        # print(f"new action: {new_action}")
        obs, rew, done, truncated, info = self.env.step(new_action)
        info["intervene_action"] = new_action
        info["left"] = self.left.any()
        info["right"] = self.right.any()
        return obs, rew, done, truncated, info


class ExperimentalFrameRotationWrapper(gym.Wrapper):
    def __init__(self, env, rotation):
        super().__init__(env)
        self.rotation = R.from_euler("xyz", rotation)

    def step(self, action):
        a = self._rotate_action(action)
        obs, rew, done, truncated, info = self.env.step(a)
        print(f"before: {obs['state']['tcp_pose']}")

        for info in ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque"]:
            obs["state"][info][:3] = np.dot(self.rotation.as_matrix(), obs["state"][info][:3])

        obs["state"]['tcp_pose'][3:] = (self.rotation * R.from_quat(obs["state"]['tcp_pose'][3:])).as_quat()
        obs["state"]['tcp_vel'][3:] = (self.rotation * R.from_rotvec(obs["state"]['tcp_vel'][3:])).as_rotvec()

        print(f"rotated observation: {obs['state']['tcp_pose']}")
        return obs, rew, done, truncated, info

    def _rotate_action(self, action: np.ndarray) -> np.ndarray:
        print(f"action: {action}")
        new_action = np.zeros_like(action)
        new_action[:3] = np.dot(self.rotation.as_matrix(), action[:3])
        new_action[3:] = action[3:]
        print(f"rotated action {new_action}")
        return new_action


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_euler(tcp_pose[3:]))
        )
        return observation


class BinaryRewardClassifierWrapper(gym.Wrapper):
    """
    Compute reward with custom binary reward classifier fn
    """

    def __init__(self, env: gym.Env, reward_classifier_func):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            logit = self.reward_classifier_func(obs).item()
            return (sigmoid(logit) >= 0.5) * 1
        return 0

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        return obs, rew, done, truncated, info
