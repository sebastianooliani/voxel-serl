from ur_env.envs.camera_env.config import UR5CameraConfigDualRobot
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
import time


config = UR5CameraConfigDualRobot()
control = RTDEControlInterface(config.ROBOT_IP_2)
receive = RTDEReceiveInterface(config.ROBOT_IP_2)

orientation = receive.getActualTCPPose()[3:]

commands = []

for x in [config.ABS_POSE_LIMIT_HIGH[0], config.ABS_POSE_LIMIT_LOW[0]]:
    for y in [config.ABS_POSE_LIMIT_HIGH[1], config.ABS_POSE_LIMIT_LOW[1]]:
        for z in [config.ABS_POSE_LIMIT_HIGH[2], config.ABS_POSE_LIMIT_LOW[2]]:
            commands.append([x, y, z, *orientation])

converged = False

for command in commands:
    print(f"Moving to: {command}")
    while True:
        t_start = control.initPeriod()
        control.moveL(command, 0.1, 0.1)
        control.waitPeriod(t_start)
        if np.linalg.norm(np.array(receive.getActualTCPPose()[:3]) - np.array(command[:3])) < 0.01:
            time.sleep(0.5)
            break 