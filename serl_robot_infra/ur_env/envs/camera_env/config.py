from ur_env.envs.ur5_env import DefaultEnvConfig, DualRobotDefaultEnvConfig
import numpy as np
import math


class UR5CameraConfig(DefaultEnvConfig):
    RESET_Q = np.array([[1.3502, -1.2897, 1.9304, -2.2098, -1.5661, 1.4027]])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.00,)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.2, -0.4, 0.22, 3.2, 0.18, 3.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.2, -0.7, - 0.006, 2.8, -0.18, -3.2])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP: str = "192.168.1.66"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.1, 1., 1., 1.])

    REALSENSE_CAMERAS = {
        "wrist": "218622277164",
        "wrist_2": "218622279756"
    }


class UR5CameraConfigBox5(DefaultEnvConfig):
    RESET_Q = np.array([
        [1.3776, -1.0603, 1.6296, -2.1462, -1.5704, -0.2019],
        [0.9104, -0.9716, 1.3539, -1.9824, -1.545, -0.662],
        [0.4782, -1.4072, 2.1258, -2.3129, -1.5816, -1.1417],
        [1.2083, -1.656, 2.272, -2.202, -1.5828, -0.4231],
        [-0.0388, -1.754, 2.2969, -2.1271, -1.5423, -1.7011]
    ])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.05, 0.1, 0.22, 3.2, 0.18, 3.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.49, -0.75, -0.006, 2.8, -0.18, -3.2])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP: str = "192.168.1.66"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.1, 1., 1., 1.])

    REALSENSE_CAMERAS = {
        "wrist": "218622277164",
        # "shoulder": "218622279756"
    }


class UR5CameraConfigFinal(DefaultEnvConfig):  # config for 10 boxes
    RESET_Q = np.array([
        [2.6331, -1.5022, 2.1151, -2.183, -1.5664, -0.4762],
        [1.983, -1.2533, 1.9069, -2.2314, -1.5495, 0.4462],
        [1.8937, -0.8273, 1.2339, -1.9765, -1.5651, 0.3666],
        [1.4174, -1.6403, 2.2494, -2.179, -1.5666, -0.1286],
        [1.472, -0.8583, 1.2817, -1.9934, -1.5655, -0.0869],
        [1.1117, -0.7666, 1.0792, -1.8871, -1.5639, -0.443],
        [1.0242, - 1.3104, 2.0986, - 2.358, - 1.5664, - 2.0496],
        [0.8757, -1.1028, 1.6058, -2.2458, -1.8081, -0.7877],
        [0.4391, - 1.5926, 2.3356, - 2.3129, - 1.5668, - 1.1115],
        [0.1815, - 1.2945, 1.8964, - 2.1719, - 1.5658, - 1.3841],
    ])

    RANDOM_RESET = True
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_ROT_RANGE = (0.04,)
    ABS_POSE_LIMIT_HIGH = np.array([0.6, 0.1, 0.25, 0.05, 0.05, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.7, -0.85, -0.006, -0.05, -0.05, -0.2])
    ABS_POSE_RANGE_LIMITS = np.array([0.36, 0.83])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP: str = "192.168.1.66" #"172.22.22.2" # "192.168.1.66"
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.1  # faster but more vulnerable to crash...
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.5, 1., 1., 1.])

    REALSENSE_CAMERAS = {
        "wrist": "218622277164",
        "wrist_2": "218622279756"
    }

class UR5CameraConfigDemo(UR5CameraConfigFinal):
    RESET_Q = np.array([[0., -np.pi/2., np.pi/2., -np.pi/2., -np.pi/2., 0.]])
    ABS_POSE_LIMIT_HIGH = np.array([1., 1., 1., 0.1, 0.1, 0.3])
    ABS_POSE_LIMIT_LOW = np.array([-1., -1., -0.004, -0.1, -0.1, -0.3])

    ROBOT_IP: str = "192.168.1.66"


class UR5CameraConfigFinalTests(UR5CameraConfigFinal):
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_ROT_RANGE = (0.05,)

    RESET_Q = np.array([
        # [0.0421, -1.3161, 1.9649, -2.2358, -1.3221, -1.5237 + 0 * np.pi / 2.],  # schräge position
        # [0.1882, -1.2777, 1.9699, -2.2983, -1.5567, -1.384 + 2 * np.pi / 2],  # gerade pos
        [1.4843, -1.1314, 1.6531, -2.0676, -1.6014, 1.6402]
    ])


class UR5CameraConfigFinalEvaluation(UR5CameraConfigFinal):
    # config for the evaluation on 5 boxes the policy has never seen
    RANDOM_RESET = True
    RANDOM_XY_RANGE = (0.01,)
    RANDOM_ROT_RANGE = (0.05,)
    ABS_POSE_LIMIT_HIGH = np.array([0.6, 0.1, 0.25, 0.1, 0.1, 0.3])
    ABS_POSE_LIMIT_LOW = np.array([-0.7, -0.85, -0.006, -0.1, -0.1, -0.3])

    RESET_Q = np.array([
        [0.4102, -1.304, 1.9315, -2.1707, -1.5583, 2.0127],
        [0.9212, - 0.8757, 1.3325, - 2.0209, - 1.5508, 2.5185],
        [1.2869, - 0.9778, 1.476, - 2.0783, - 1.5458, 2.9585],
        [1.717, -1.1379, 1.7179, -2.4872, -1.4362, 2.5804],
        [2.2614, - 1.4378, 2.145, - 2.5039, - 1.7649, 2.2541],
    ])


class UR5CameraConfigDualRobot(DualRobotDefaultEnvConfig):
    RESET_Q = np.array([[0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.,
                         math.pi, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.]])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.00,)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.2, -0.4, 0.22, 3.2, 0.18, 3.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.2, -0.7, - 0.006, 2.8, -0.18, -3.2])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP_1: str = "192.168.1.66" # docker "172.17.0.2"
    ROBOT_IP_2: str = "192.168.1.33" # docker "172.17.0.3"
    ROBOT_PORT_1: int = 30004
    ROBOT_PORT_2: int = 40004
    CONTROLLER_HZ = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ERROR_DELTA: float = 0.05
    FORCEMODE_DAMPING: float = 0.0  # faster
    FORCEMODE_TASK_FRAME = np.zeros(6)
    FORCEMODE_SELECTION_VECTOR = np.ones(6, dtype=np.int8)
    FORCEMODE_LIMITS = np.array([0.5, 0.5, 0.1, 1., 1., 1.])

    REALSENSE_CAMERAS = {
        "shoulder": "",
        "wrist": "218622277164",
        "shoulder_2": "",
        "wrist_2": "218622279756"
    }