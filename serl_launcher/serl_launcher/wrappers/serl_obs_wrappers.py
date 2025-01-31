import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, WrapperObsType
from gymnasium.spaces import flatten_space, flatten

from ur_env.envs.camera_env.config import UR5CameraConfigDualRobot

class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treats the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.env.observation_space["state"]),
                **(self.env.observation_space["images"] if "images" in self.env.observation_space.spaces else {}),

            }
        )

    def observation(self, obs):
        obs = {
            "state": flatten(self.env.observation_space["state"], obs["state"]),
            **(obs["images"] if "images" in obs else {}),
        }
        return obs


class SerlObsWrapperNoImages(gym.ObservationWrapper):
    """
    This observation wrapper treats the observation space as a flattened state
    space, if no images are present.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = flatten_space(self.env.observation_space["state"])

    def observation(self, obs):
        obs = flatten(self.env.observation_space["state"], obs["state"])
        return obs
    
class HERSerlObsWrapperNoImages(gym.ObservationWrapper):
    """
    This observation wrapper treats the observation space as a flattened state
    space, if no images are present but HER is used.
    """

    def __init__(self, env):
        super().__init__(env)
        # account for the bigger observation space
        self.observation_space = flatten_space(self.env.observation_space["state"])

    def observation(self, obs):
        obs = flatten(self.env.observation_space["state"], obs["state"])
        return obs


class ScaleObservationWrapper(gym.ObservationWrapper):
    """
    This observation wrapper scales the observations with the provided hyperparams
    (to somewhat normalize the observations space)
    """

    def __init__(self,
                 env,
                 translation_scale=100.,
                 rotation_scale=10.,
                 force_scale=1.,
                 torque_scale=10.
                 ):
        super().__init__(env)
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.force_scale = force_scale
        self.torque_scale = torque_scale

    def scale_wrapper_get_scales(self):
        return dict(
            translation_scale=self.translation_scale,
            rotation_scale=self.rotation_scale,
            force_scale=self.force_scale,
            torque_scale=self.torque_scale
        )

    def observation(self, obs):
        obs["state"]["tcp_pose"][:3] *= self.translation_scale
        obs["state"]["tcp_pose"][3:] *= self.rotation_scale
        obs["state"]["tcp_vel"][:3] *= self.translation_scale
        obs["state"]["tcp_vel"][3:] *= self.rotation_scale
        obs["state"]["tcp_force"] *= self.force_scale
        obs["state"]["tcp_torque"] *= self.torque_scale
        return obs

class ScaleDualObservationWrapper(gym.ObservationWrapper):
    """
    This observation wrapper scales the observations with the provided hyperparams
    (to somewhat normalize the observations space) for dual setups.
    """

    def __init__(self,
                 env,
                 translation_scale=100.,
                 rotation_scale=10.,
                 force_scale=1.,
                 torque_scale=10.
                 ):
        super().__init__(env) 
        self.translation_scale, self.rotation_scale, self.force_scale, self.torque_scale = translation_scale, rotation_scale, force_scale, torque_scale

        self.task = UR5CameraConfigDualRobot.TASK

    def scale_wrapper_get_scales(self):
        return dict(
            translation_scale=self.translation_scale,
            rotation_scale=self.rotation_scale,
            force_scale=self.force_scale,
            torque_scale=self.torque_scale
        )

    def observation(self, obs):
        obs["state"]["tcp_pose"][:3] *= self.translation_scale
        obs["state"]["tcp_pose"][3:6] *= self.rotation_scale
        obs["state"]["tcp_pose"][6:9] *= self.translation_scale
        obs["state"]["tcp_pose"][9:] *= self.rotation_scale

        obs["state"]["tcp_vel"][:3] *= self.translation_scale
        obs["state"]["tcp_vel"][3:6] *= self.rotation_scale
        obs["state"]["tcp_vel"][6:9] *= self.translation_scale
        obs["state"]["tcp_vel"][9:] *= self.rotation_scale

        obs["state"]["tcp_force"] *= self.force_scale
        obs["state"]["tcp_torque"] *= self.torque_scale

        obs['state']['tcp_pos_diff'] *= self.translation_scale
        # obs['state']['joint_position'] *= self.rotation_scale # TODO: Check if this is needed

        # TODO: write this in a better way, maybe using the config variable TASK
        if self.task in ["motion"]:
            obs['state']['goal_box_position'] *= self.translation_scale
            obs['state']['box_position'] *= self.translation_scale
            obs['state']['goal_position'] *= self.translation_scale
        elif self.task in ["reorient"]:
            obs['state']['box_orientation'] *= self.rotation_scale

        return obs