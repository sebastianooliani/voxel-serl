import numpy as np
from queue import Queue
from ur_env.envs.camera_env.config import UR5CameraConfigDualRobot as config

from rtde_receive import RTDEReceiveInterface
from franka_env.utils.transformations import (
    pose_rotvec_2_homogeneous_matrix
)

class TreeState():
    """
    Commands for the robot are written in the tcp frame of the robot
    """
    def __init__(self):
        self.down = np.array([0., 0., 1., 0., 0., 0., 0.])
        self.up = -self.down
        self.suck = np.array([0., 0., 1., 0., 0., 0., 1.])
        self.random_direction = np.zeros_like(self.down)
        self.random_orientation = np.zeros_like(self.down)
        self.re_sample()

        self.current = np.zeros_like(self.down)

    def re_sample(self):
        rand = np.random.rand(2, 2) - 0.5
        self.random_direction[0:2] = rand[0] / np.linalg.norm(rand[0])
        self.random_orientation[3:5] = rand[1] / np.linalg.norm(rand[1])

    def reset(self):
        self.current = self.down

    def __call__(self, *args, **kwargs):
        return self.current.copy()
    
class DualTreeState():
    """
    Commands for the dual robot are written in the tcp frame of the robot
    """
    def __init__(self, opposite_grasp=False, reorient=False):
        self.down = np.array([0., 0., 1., 0., 0., 0., 0., 
                              0., 0., 1., 0., 0., 0., 0.])
        self.up = np.array([0., 1., 0., 0., 0., 0., 0.,
                            0., -1., 0., 0., 0., 0., 0.]) if opposite_grasp else -self.down
        self.forward = np.array([-1., -1., 0., 0., 0., 0., 1.,
                                 -1., -1., 0., 0., 0., 0., 1.])
        self.suck_old = np.array([0., 0., 1., 0., 0., 0., 1.,
                                  0., 0., 1., 0., 0., 0., 1.])
        self.suck = np.array([0., 1., 0., 0., 0., 0., 1.,
                              0., -1., 0., 0., 0., 0., 1.]) if opposite_grasp else self.suck_old
        
        self.random_direction = np.zeros_like(self.down)
        self.random_orientation = np.zeros_like(self.down)
        self.re_sample_xy()
        self.re_sample_xz()

        self.current = np.zeros_like(self.down)

        # new commands
        self.right = np.array([0., 1., 0., 0., 0., 0., 0., 
                               1., 0., 0., 0., 0., 0., 0.])
        self.left = -self.right
        # self.change_orientation = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def re_sample_xy(self):
        rand = np.random.rand(2, 2) - 0.5
        self.random_direction[0:2] = rand[0] / np.linalg.norm(rand[0])
        self.random_orientation[3:5] = rand[1] / np.linalg.norm(rand[1])

        rand = np.random.rand(2, 2) - 0.5
        self.random_direction[7:9] = rand[0] / np.linalg.norm(rand[0])
        self.random_orientation[10:12] = rand[1] / np.linalg.norm(rand[1])

    def re_sample_xz(self):
        rand = np.random.rand(2, 2) - 0.5
        self.random_direction[0] = rand[0][0] / np.linalg.norm(rand[0])
        self.random_direction[2] = rand[0][1] / np.linalg.norm(rand[0])
        self.random_orientation[3] = rand[1][0] / np.linalg.norm(rand[1])
        self.random_orientation[5] = rand[1][1] / np.linalg.norm(rand[1])

        rand = np.random.rand(2, 2) - 0.5
        self.random_direction[7] = rand[0][0] / np.linalg.norm(rand[0])
        self.random_direction[9] = rand[0][1] / np.linalg.norm(rand[0])
        self.random_orientation[10] = rand[1][0] / np.linalg.norm(rand[1])
        self.random_orientation[12] = rand[1][1] / np.linalg.norm(rand[1])

    def vert_reset(self):
        self.current = self.down

    def oriz_reset(self):
        self.current = self.left

    def __call__(self, *args, **kwargs):
        return self.current.copy()


class BehaviorTree():
    """
    simple behavior tree for picking boxes

    start: move down
    if force in z: suck
        if not successful:
            maybe orientation rot before? (test)
            move up, random direction xy, goto start
        if successful:
            move up, wait for end
    """

    def __init__(self):
        self.tree_state: TreeState = TreeState()
        self.queue = Queue()

    def reset(self):
        self.tree_state.reset()
        print("down")
        return self.tree_state()

    def sample_actions(self, observations):
        obs = observations["state"].reshape(-1)
        if not self.queue.empty():
            return self.queue.get()

        if obs[8] > 0.5:
            if np.all(self.tree_state.current == self.tree_state.up):
                pass
            else:
                print("go up")
                self.tree_state.current = self.tree_state.up

        elif obs[11] < -3.:  # force check
            if obs[8] < -0.5:  # if sucking
                print("do random direction")
                return self._fill_random_xy_queue()
            else:
                print("suck")
                self.tree_state.current = self.tree_state.suck
                return self._fill_suck_queue()
        else:
            self.tree_state.reset()

        return self.tree_state()

    def _fill_random_xy_queue(self):
        for _ in range(4):
            self.queue.put(self.tree_state.up)
        self.tree_state.re_sample()
        for _ in range(6):
            self.queue.put(self.tree_state.random_direction)

        return self.queue.get()

    def _fill_suck_queue(self):
        for _ in range(3):
            self.queue.put(self.tree_state.suck)
        return self.queue.get()
    

class DualBehaviorTree():
    """
    simple behavior tree for picking boxes from the upper side

    start: move down
    if force in z: suck
        if not successful:
            maybe orientation rot before? (test)
            move up, random direction xy, goto start
        if successful:
            move up, wait for end
    """

    def __init__(self, opposite_grasp=False):
        self.tree_state: DualTreeState = DualTreeState(opposite_grasp=opposite_grasp)
        self.queue = Queue()
        self.ur_receive_1 = RTDEReceiveInterface("192.168.1.66")
        self.ur_receive_2 = RTDEReceiveInterface("192.168.1.33")

    def reset(self):
        self.tree_state.vert_reset()
        print("down")
        return self.tree_state()

    def sample_actions(self, observations):
        obs = observations["state"].reshape(-1)
        if not self.queue.empty():
            return self.queue.get()

        force_1 = self.ur_receive_1.getActualTCPForce()
        force_2 = self.ur_receive_2.getActualTCPForce()
        # observation order in the dictionary
        # action, gripper, joint pos, force, pos diff, pose, torque, vel
        if obs[15] > 0.5 and obs[17] > 0.5:
            if np.all(self.tree_state.current == self.tree_state.up):
                pass
            else:
                print("go up")
                self.tree_state.current = self.tree_state.up

        elif -force_1[2] < -1. and -force_2[2] < -1.:  # force check
            if obs[15] < -0.5 and obs[17] < -0.5:  # if sucking
                print("do random direction")
                return self._fill_random_xy_queue()
            else:
                print("suck")
                self.tree_state.current = self.tree_state.suck
                return self._fill_suck_queue()
        else:
            self.tree_state.vert_reset()

        return self.tree_state()

    def _fill_random_xy_queue(self):
        for _ in range(4):
            self.queue.put(self.tree_state.up)
        self.tree_state.re_sample_xy()
        for _ in range(6):
            self.queue.put(self.tree_state.random_direction)

        return self.queue.get()

    def _fill_suck_queue(self):
        for _ in range(6):
            self.queue.put(self.tree_state.suck)
        return self.queue.get()
    

class DualBehaviorTreeReorientation():
    """
    simple behavior tree for rotating the box of 40 degrees
    
    start: move down
    if force in z: suck and move forward
        if not successful:
            maybe orientation rot before? (test)
            move up, random direction xy, goto start
        if successful:
            move up, wait for end
    """
    def __init__(self, opposite_grasp=False, reorient=True):
        self.tree_state: DualTreeState = DualTreeState(opposite_grasp=opposite_grasp, reorient=reorient)
        self.queue = Queue()
        self.ur_receive_1 = RTDEReceiveInterface("192.168.1.66")
        self.ur_receive_2 = RTDEReceiveInterface("192.168.1.33")

    def reset(self):
        self.tree_state.vert_reset()
        print("down")
        return self.tree_state()
    
    def sample_actions(self, observations):
        obs = observations["state"].reshape(-1)
        if not self.queue.empty():
            return self.queue.get()
        
        force_1 = self.ur_receive_1.getActualTCPForce()
        force_2 = self.ur_receive_2.getActualTCPForce()
        # observation order in the dictionary
        # action, gripper, joint pos, force, pos diff, pose, torque, vel
        if obs[15] > 0.5 and obs[17] > 0.5:
            if np.all(self.tree_state.current == self.tree_state.forward):
                pass
            else:
                print("go forward")
                self.tree_state.current = self.tree_state.forward

        elif -force_1[2] < -1. and -force_2[2] < -1.: # force check
            if obs[15] < -0.5 and obs[17] < -0.5: # if sucking
                print("do random direction")
                return self._fill_random_xy_queue()
            else:
                print("suck")
                self.tree_state.current = self.tree_state.suck
                return self._fill_suck_queue()
        else:
            self.tree_state.vert_reset()

        return self.tree_state()
    
    def _fill_random_xy_queue(self):
        for _ in range(4):
            self.queue.put(self.tree_state.up)
        self.tree_state.re_sample_xy()
        for _ in range(6):
            self.queue.put(self.tree_state.random_direction)

        return self.queue.get()
    
    def _fill_suck_queue(self):
        for _ in range(6):
            self.queue.put(self.tree_state.suck)
        return self.queue.get()
    

class DualBehaviorTreeMotionPlanning():
    """
    simple behavior tree for moving the box to a point in the xyz plane
    start: move down
    if force in z: suck and move to the point
        if not successful:
            maybe orientation rot before? (test)
            move up, random direction xy, goto start
        if successful:
            move up, wait for end
    """
    def __init__(self, opposite_grasp=False):
        self.tree_state: DualTreeState = DualTreeState(opposite_grasp=opposite_grasp)
        self.queue = Queue()
        self.command = np.zeros(14)
        self.command[6] = 1.
        self.command[13] = 1.
        self.ur_receive_1 = RTDEReceiveInterface("192.168.1.66")
        self.ur_receive_2 = RTDEReceiveInterface("192.168.1.33")

    def reset(self):
        self.tree_state.vert_reset()
        print("down")
        return self.tree_state()
    
    def sample_actions(self, observations):
        obs = observations["state"].reshape(-1)
        if not self.queue.empty():
            return self.queue.get()
        
        force_1 = self.ur_receive_1.getActualTCPForce()
        force_2 = self.ur_receive_2.getActualTCPForce()

        # observation order in the dictionary
        # action, gripper, joint pos, force, pos diff, pose, torque, vel
        if obs[15] > 0.5 and obs[17] > 0.5:
            self.compute_commands(obs)
            if np.all(self.tree_state.current == self.command):
                pass
            else:
                print("go to the goal")
                self.tree_state.current = self.command

        elif - force_1[2] < -1. and - force_2[2] < -1.:
            if obs[15] < - 0.5 and obs[17] < - 0.5:
                print("do random direction")
                return self._fill_random_xy_queue()
            else:
                print("suck")
                self.tree_state.current = self.tree_state.suck
                return self._fill_suck_queue()
        else:
            print("down")
            self.tree_state.vert_reset()

        return self.tree_state()
    
    def _fill_random_xy_queue(self):
        for _ in range(4):
            self.queue.put(self.tree_state.up)
        self.tree_state.re_sample_xy()
        for _ in range(6):
            self.queue.put(self.tree_state.random_direction)

        return self.queue.get()
    
    def _fill_suck_queue(self):
        for _ in range(6):
            self.queue.put(self.tree_state.suck)
        return self.queue.get()
    
    def compute_commands(self, obs):
        """ 
        compute commands based on the observations 
        """
        pose_1 = self.ur_receive_1.getActualTCPPose()
        pose_2 = self.ur_receive_2.getActualTCPPose()
        T_O1_O2 = config.T_O1_O2
        T_O2_O1 = np.linalg.inv(T_O1_O2)

        goal_box_pos = obs[57:60]

        T_1 = pose_rotvec_2_homogeneous_matrix(pose_1)
        T_2 = pose_rotvec_2_homogeneous_matrix(pose_2)
        R_1 = T_1[:3, :3]
        R_2 = T_2[:3, :3]

        distance_ee1 = 10 * (np.linalg.inv(R_1) @ goal_box_pos)
        distance_ee2 = 10 * (np.linalg.inv(R_2) @ (T_O2_O1[:3, :3] @ goal_box_pos))

        self.command[0:3] = distance_ee1
        self.command[7:10] = distance_ee2
    

class DualBehaviorTreeInAirRotation():
    def __init__(self, opposite_grasp=False, reorient=True):
        self.tree_state: DualTreeState = DualTreeState(opposite_grasp=opposite_grasp, reorient=reorient)
        self.queue = Queue()
        self.command = np.zeros(14)
        self.command[6] = 1.
        self.command[13] = 1.
        self.ur_receive_1 = RTDEReceiveInterface("192.168.1.66")
        self.ur_receive_2 = RTDEReceiveInterface("192.168.1.33")
        self.init_box_pos = None

    def reset(self):
        self.tree_state.vert_reset()
        print("down")
        return self.tree_state()
    
    def sample_actions(self, observations):
        obs = observations["state"].reshape(-1)
        if not self.queue.empty():
            return self.queue.get()
        
        force_1 = self.ur_receive_1.getActualTCPForce()
        force_2 = self.ur_receive_2.getActualTCPForce()
        # observation order in the dictionary
        # action, gripper, joint pos, force, pos diff, pose, torque, vel
        if obs[15] > 0.5 and obs[17] > 0.5:
            self.compute_commands(obs)
            if np.all(self.tree_state.current == self.command):
                pass
            else:
                print("go forward")
                self.tree_state.current = self.command

        elif -force_1[2] < -1. and -force_2[2] < -1.: # force check
            if obs[15] < -0.5 and obs[17] < -0.5: # if sucking
                print("do random direction")
                return self._fill_random_xy_queue()
            else:
                print("suck")
                self.tree_state.current = self.tree_state.suck
                return self._fill_suck_queue()
        else:
            self.tree_state.vert_reset()

        return self.tree_state()
    
    def _fill_random_xy_queue(self):
        for _ in range(4):
            self.queue.put(self.tree_state.up)
        self.tree_state.re_sample_xy()
        for _ in range(6):
            self.queue.put(self.tree_state.random_direction)

        return self.queue.get()
    
    def _fill_suck_queue(self):
        for _ in range(6):
            self.queue.put(self.tree_state.suck)
        return self.queue.get()
    
    def compute_commands(self, obs):
        """ 
        compute commands based on the observations 
        """
        if self.init_box_pos is None:
            self.init_box_pos = obs[-6:-3]

        if np.linalg.norm(obs[-6:-3] - self.init_box_pos) < 0.05:
            self.command[0:3] = np.array([0, 0, -1])
            self.command[7:10] = np.array([0, 0, -1])
        elif np.linalg.norm(obs[-6:-3] - self.init_box_pos) > 0.05:
            self.command[0:3] = np.array([0, 0, 0])
            self.command[7:10] = np.array([0, 0, -1])