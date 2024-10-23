import gymnasium as gym
import numpy as np
from agentlace import action
import pygame

from ur_env.spacemouse.spacemouse_expert import SpaceMouseExpert, TwoSpaceMiceExperts
import time
from scipy.spatial.transform import Rotation as R

from ur_env.utils.rotations import quat_2_euler, quat_2_mrp

ROT90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
ROT_GENERAL = np.array([np.eye(3), ROT90, ROT90 @ ROT90, ROT90.transpose()])


class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, gripper_action_span=3):
        # Initialize the base class with the provided environment.
        super().__init__(env)

        # Enable gripper control (assumes there's a gripper in the environment).
        self.gripper_enabled = True

        # Initialize the expert interface for the SpaceMouse control.
        self.expert = SpaceMouseExpert()

        # Tracks the last time an intervention was made using the SpaceMouse.
        self.last_intervene = 0

        # Arrays for tracking the state of the gripper buttons (left and right buttons on the SpaceMouse).
        self.left = np.array([False] * gripper_action_span, dtype=np.bool_)
        self.right = self.left.copy()

        # Invert certain axes from the SpaceMouse to match the robot's coordinate system.
        self.invert_axes = [-1, -1, 1, -1, -1, 1]

        # Define a deadspace threshold, within which SpaceMouse movements are ignored.
        self.deadspace = 0.15

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action (action proposed by the RL agent)
        Output:
        - action: spacemouse action if it's being used; else, policy action
        """
        # Get the current SpaceMouse input, considering the deadspace.
        expert_a = self.get_deadspace_action()

        # If the SpaceMouse is moved or buttons are pressed, update the last intervention time.
        if np.linalg.norm(expert_a) > 0.001 or self.left.any() or self.right.any():
            self.last_intervene = time.time()

        # Handle gripper action if gripper control is enabled.
        if self.gripper_enabled:
            # Create gripper action based on button states (left/right).
            gripper_action = np.zeros((1,)) + int(self.left.any()) - int(self.right.any())
            # Append the gripper action to the movement action.
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        # If the last intervention was within 0.5 seconds, adapt and return the SpaceMouse output.
        if time.time() - self.last_intervene < 0.5:
            expert_a = self.adapt_spacemouse_output(expert_a)
            return expert_a

        # If no recent intervention, return the original policy action.
        return action

    def get_deadspace_action(self) -> np.ndarray:
        """
        Get the SpaceMouse input, but ignore movements within the deadspace.
        """
        # Get the SpaceMouse expert's action and button states.
        expert_a, buttons = self.expert.get_action()

        # Apply deadspace filtering: ignore small movements below the deadspace threshold.
        positive = np.clip((expert_a - self.deadspace) / (1. - self.deadspace), a_min=0.0, a_max=1.0)
        negative = np.clip((expert_a + self.deadspace) / (1. - self.deadspace), a_min=-1.0, a_max=0.0)
        expert_a = positive + negative

        # Shift the button states by one position (for left and right buttons) to track their history.
        self.left, self.right = np.roll(self.left, -1), np.roll(self.right, -1)
        # Update the latest button states.
        self.left[-1], self.right[-1] = tuple(buttons)

        return np.array(expert_a, dtype=np.float64)

    def adapt_spacemouse_output(self, action: np.ndarray) -> np.ndarray:
        """
        Adjust the SpaceMouse output to align with the robot's action space, considering rotations.
        Input:
        - action: raw SpaceMouse output (position and orientation changes)
        Output:
        - action: transformed action for the robot's coordinate space
        """

        # Get the current position of the robot (e.g., end-effector).
        position = self.unwrapped.curr_pos

        # Calculate the z-axis rotation angle based on the robot's current position.
        z_angle = np.arctan2(position[1], position[0])

        # Create a rotation object for the z-axis.
        z_rot = R.from_rotvec(np.array([0, 0, z_angle]))

        # Invert certain axes of the SpaceMouse output, based on the configured axis inversions.
        action[:6] *= self.invert_axes

        # Apply the z-axis rotation to the translation components (first three values).
        action[:3] = z_rot.apply(action[:3])

        # Optionally: apply the z-axis rotation to the rotational components (next three values).
        action[3:6] = z_rot.apply(action[3:6])

        return action

    def step(self, action):
        """
        Step the environment, using either the SpaceMouse action (if intervening) or the policy action.
        """
        # Get the new action after considering potential SpaceMouse interventions.
        new_action = self.action(action)

        # Step the environment with the chosen action.
        obs, rew, done, truncated, info = self.env.step(new_action)

        # Add additional information to the info dictionary about the intervention.
        info["intervene_action"] = new_action
        info["left"] = self.left.any()  # Whether the left button is pressed.
        info["right"] = self.right.any()  # Whether the right button is pressed.

        # Return the observation, reward, done flag, truncation flag, and info dictionary.
        return obs, rew, done, truncated, info
    

class TwoSpacemiceIntervention_old(SpacemouseIntervention):
    def __init__(self, env, gripper_action_span=6):
        super(gym.Wrapper).__init__(env)

        self.gripper_enabled = True

        self.last_intervene = 0
        self.left = np.array([False] * gripper_action_span, dtype=np.bool_)
        self.right = self.left.copy()

        self.invert_axes = [-1, -1, 1, -1, -1, 1]
        self.deadspace = 0.15
        self.expert = TwoSpaceMiceExperts()

    def get_deadspace_action(self) -> np.ndarray:
        expert_a, buttons_a, expert_b, buttons_b = self.expert.get_action()

        positive = np.clip((expert_a - self.deadspace) / (1. - self.deadspace), a_min=0.0, a_max=1.0)
        negative = np.clip((expert_a + self.deadspace) / (1. - self.deadspace), a_min=-1.0, a_max=0.0)
        expert_a = positive + negative

        positive = np.clip((expert_b - self.deadspace) / (1. - self.deadspace), a_min=0.0, a_max=1.0)
        negative = np.clip((expert_b + self.deadspace) / (1. - self.deadspace), a_min=-1.0, a_max=0.0)
        expert_b = positive + negative

        self.left, self.right = np.roll(self.left, -2), np.roll(self.right, -2)
        self.left[-2], self.right[-2] = tuple(buttons_a)
        self.left[-1], self.right[-1] = tuple(buttons_b)

        return np.array(expert_a, dtype=np.float32), np.array(expert_b, dtype=np.float32)
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, expert_b = self.get_deadspace_action()
        policy_a = action[:6]
        policy_b = action[6:]

        if np.linalg.norm(
                expert_a) > 0.001 or self.left.any() or self.right.any():  # also read buttons with no movement
            self.last_intervene = time.time()

        if self.gripper_enabled:
            gripper_action = np.zeros((1,)) + int(self.left.any()) - int(self.right.any())
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if time.time() - self.last_intervene < 0.5:
            expert_a = self.adapt_spacemouse_output(expert_a)
            return expert_a

        return action
    
    def adapt_spacemouse_output(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - expert_a: spacemouse raw output
        Output:
        - expert_a: spacemouse output adapted to force space (action)
        """

        # position = super().get_wrapper_attr("curr_pos")  # get position from ur_env
        position = self.unwrapped.curr_pos
        z_angle_1 = np.arctan2(position[1], position[0])  # get first joint angle
        z_angle_2 = np.arctan2(position[7], position[6])

        z_rot = R.from_rotvec(np.array([0, 0, z_angle_1]))
        action[:6] *= self.invert_axes  # if some want to be inverted
        action[:3] = z_rot.apply(action[:3])  # z rotation invariant translation

        # TODO add tcp orientation to the equation (extract z rotation from tcp pose)
        action[3:6] = z_rot.apply(action[3:6])  # z rotation invariant rotation

        z_rot = R.from_rotvec(np.array([0, 0, z_angle_2]))
        action[6:] *= self.invert_axes  # if some want to be inverted
        action[6:9] = z_rot.apply(action[6:9])  # z rotation invariant translation

        # TODO add tcp orientation to the equation (extract z rotation from tcp pose)
        action[9:] = z_rot.apply(action[8:])  # z rotation invariant rotation

        return action
    

class TwoSpacemiceIntervention(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.expert_left = SpacemouseIntervention(env)
        self.expert_right = SpacemouseIntervention(env)

    def step(self, action):
        action_left = action[:7]
        action_right = action[7:]

        new_action_left = self.expert_left.action(action_left)
        new_action_left = self.expert_right.action(action_right)
        new_action = np.concatenate((new_action_left, new_action_left), axis=0)

        obs, rew, done, truncated, info = self.env.step(new_action)

        info["intervene_action"] = new_action
        info["left"] = self.expert_left.left.any() or self.expert_right.left.any()  # Whether the left button is pressed.
        info["right"] = self.expert_left.right.any() or self.expert_right.right.any()  # Whether the right button is pressed.

        return obs, rew, done, truncated, info



class Quat2EulerWrapper(gym.ObservationWrapper):  # not used anymore (stay away from euler angles!)
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


class Quat2MrpWrapper(gym.ObservationWrapper):
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
            (tcp_pose[:3], quat_2_mrp(tcp_pose[3:]))
        )
        return observation
    
class DualQuat2MrpWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles for Dual Arm setup.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = gym.spaces.Box(
            -np.inf, np.inf, shape=(14,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], quat_2_mrp(tcp_pose[3:7]), tcp_pose[7:10], quat_2_mrp(tcp_pose[10:]))
        )
        return observation


def rotate_state(state: np.ndarray, num_rot: int):
    assert len(state.shape) == 1 and state.shape[0] % 3 == 0
    state = state.reshape((-1, 3)).transpose()
    rotated = np.dot(ROT_GENERAL[num_rot % 4], state).transpose()
    return rotated.reshape((-1))


class ObservationRotationWrapper(gym.Wrapper):
    """
    Convert every observation into the first quadrant of the Relative Frame
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        print("Observation Rotation Wrapper enabled!")
        self.num_rot_quadrant = -1

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        obs = self.rotate_observation(obs, random=True)     # rotate initial state random
        return obs, info

    def step(self, action: np.ndarray):
        action = self.rotate_action(action=action)
        obs, reward, done, truncated, info = self.env.step(action)
        # print("\nquadrant: ", self.num_rot_quadrant)
        rotated_obs = self.rotate_observation(obs)
        return rotated_obs, reward, done, truncated, info

    def rotate_observation(self, observation, random=False):
        if not random:
            x, y = (observation["state"]["tcp_pose"][:2])
            self.num_rot_quadrant = int(x < 0.) * 2 + int(x * y < 0.)  # save quadrant info
        else:
            self.num_rot_quadrant = int(time.time_ns()) % 4  # do not mess with seeded np.random

        for state in observation["state"].keys():
            if state == "gripper_state":
                continue
            elif state == "action":
                observation["state"][state][:6] = rotate_state(observation["state"][state][:6], self.num_rot_quadrant)
            else:
                observation["state"][state][:] = rotate_state(observation["state"][state],
                                                              self.num_rot_quadrant)  # rotate

        if "images" in observation:
            for image_keys in observation["images"].keys():
                observation["images"][image_keys][:] = np.rot90(
                    observation["images"][image_keys],
                    axes=(0, 1),
                    k=self.num_rot_quadrant
                )
        return observation

    def rotate_action(self, action):
        rotated_action = action.copy()
        rotated_action[:6] = rotate_state(action[:6], 4 - self.num_rot_quadrant)  # rotate
        return rotated_action
    

class KeyboardInterventionWrapper(gym.Wrapper):
    """
    Not working at the moment.
    """
    def __init__(self, env):
        super().__init__(env)
        print("Keyboard Intervention Wrapper enabled!")
        pygame.init()
        self.action = np.zeros(self.env.action_space.shape)  # Default action (no movement)

    def process_keys(self, action):
        """ Map keyboard inputs to environment actions. """
        keys = pygame.key.get_pressed()

        # Modify the action vector based on key presses
        if keys[pygame.K_LEFT]:
            action[0] = -0.5  # Move left
        elif keys[pygame.K_RIGHT]:
            action[0] = 0.5   # Move right
        else:
            action[0] = 0   # Stop horizontal movement

        if keys[pygame.K_UP]:
            action[1] = 0.5   # Move up
        elif keys[pygame.K_DOWN]:
            action[1] = -0.5  # Move down
        else:
            action[1] = 0   # Stop vertical movement

        if keys[pygame.K_w]:
            action[2] = 0.5   # Move forward
        elif keys[pygame.K_s]:
            action[2] = -0.5  # Move backward
        else:
            action[2] = 0   # Stop forward/backward movement 

        return action
    
    def close(self):
        """Close the environment and pygame."""
        pygame.quit()
        self.env.close()

    def step(self, action):
        pygame.event.pump()
        new_action = self.process_keys(action)
        print(f"new action: {new_action}")
        obs, rew, done, truncated, info = self.env.step(new_action)
        info["intervene_action"] = new_action
        return obs, rew, done, truncated, info


class FreeDriveWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        print("\nFree Drive Wrapper enabled!\n")

        # self.activate_free_drive()

    def activate_free_drive(self):
        self.env.env.env.env.controller.ur_control.freedriveMode(free_axes=[1,1,1,1,1,1], feature=[0,0,0,0,0,0])

    def deactivate_free_drive(self):
        self.env.env.env.env.controller.ur_control.endFreedriveMode()

    def action(self, action):
        pose = np.array(self.env.env.env.env.controller.ur_receive.getActualTCPPose())
        action[:3] = pose[:3]
        action[3:] = R.from_rotvec(pose[3:]).as_quat()

        return action

    def step(self, action):
        # self.action = action
        new_action = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        info["intervene_action"] = new_action
        # self.deactivate_free_drive()
        return obs, rew, done, truncated, info
    
    def close(self):
        pass
        self.deactivate_free_drive()