from ur_env.envs.camera_env.config import UR5CameraConfigDualRobot
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import numpy as np
import time


config = UR5CameraConfigDualRobot()
control = RTDEControlInterface(config.ROBOT_IP_2)
receive = RTDEReceiveInterface(config.ROBOT_IP_2)

orientation = receive.getActualTCPPose()[3:]

high = config.ABS_POSE_LIMIT_HIGH_ROBOT_2
low = config.ABS_POSE_LIMIT_LOW_ROBOT_2

commands = []

for x in [high[0], low[0]]:
    for y in [high[1], low[1]]:
        for z in [high[2], low[2]]:
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