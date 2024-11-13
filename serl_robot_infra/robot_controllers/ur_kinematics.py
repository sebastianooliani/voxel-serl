import numpy as np
import torch
import pytorch_kinematics as pk
from fast_kinematics import FastKinematics
import math

def evaluate_jacobian_determinant(file_name="/home/sebastiano/voxel-serl/serl_robot_infra/robot_controllers/ur5.urdf", link="ee_link", joint_pos=np.zeros(6), N=1000, d="cuda", dtype = torch.float32):
    """
    Evaluate the determinant of the Jacobian of a URDF file at a given link and joint position using
    the pytorch_kinematics library.    

    Args:
        file_name (str): URDF file name
        link (str): link name, note that any link can be chosen; it doesn't have to be a link with no children
        joint_pos (np.array): joint positions

    Returns:
        det (float): determinant of Jacobian
    """

    chain = pk.build_serial_chain_from_urdf(open(file_name).read(), N, link)
    joint_pos = torch.tensor(joint_pos, dtype=dtype, device=d)
    J = chain.jacobian(joint_pos)
    det = torch.det(J).item()

    return det

def fast_evaluate_jacobian_determinant(file_name="/home/sebastiano/voxel-serl/serl_robot_infra/robot_controllers/ur5.urdf", link="ee_link", joint_pos=np.zeros(6, np.float32), N=1, d="cuda", dtype = torch.float32):
    """
    Evaluate the determinant of the Jacobian of a URDF file at a given link and joint position using
    the fast_kinematics library.    

    Args:
        file_name (str): URDF file name
        link (str): link name, note that any link can be chosen; it doesn't have to be a link with no children
        joint_pos (np.array): joint positions

    Returns:
        det (float): determinant of Jacobian
    """
    joint_pos = np.array([math.radians(241.46), math.radians(-75.78), math.radians(107.78), math.radians(-38.43), math.radians(-24.73), math.radians(33.13)], dtype=np.float32)
    chain = FastKinematics(file_name, N, link)
    J = chain.jacobian_mixed_frame(joint_pos)
    J = J.reshape(N, 6, 6)
    J = torch.tensor(J).to(d)
    det = torch.det(J).item()

    return det

if __name__ == "__main__":
    print(fast_evaluate_jacobian_determinant())

# file_name = "ur5.urdf"

# # can convert Chain to SerialChain by choosing end effector frame
# chain = pk.build_chain_from_urdf(open(file_name).read())
# # print(chain) to see the available links for use as end effector
# print(f"\n{chain}\n")
# # note that any link can be chosen; it doesn't have to be a link with no children
# chain = pk.SerialChain(chain, "ee_link")

# chain = pk.build_serial_chain_from_urdf(open(file_name).read(), "ee_link")
# th = torch.tensor([math.radians(-43.80), math.radians(-55.76), math.radians(102.76), math.radians(-45.15), math.radians(-37.34), math.radians(1.50)])
# # (1,6,7) tensor, with 7 corresponding to the DOF of the robot
# J = chain.jacobian(th)
# print(f"Jacobian: \n{J}\n")
# # determinant of Jacobian
# det = torch.det(J)
# print(f"\nDeterminant: {det}\n")

# # get Jacobian in parallel and use CUDA if available
# N = 1000
# d = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"\nUsing device: {d}\n")
# dtype = torch.float64

# chain = chain.to(dtype=dtype, device=d)
# # Jacobian calculation is differentiable
# # th = torch.rand(N, 6, dtype=dtype, device=d, requires_grad=True)
# th = th.to(d)
# # (N,6,7)
# J = chain.jacobian(th)

# print(f"\n{J}\n")

# # can get Jacobian at a point offset from the end effector (location is specified in EE link frame)
# # by default location is at the origin of the EE frame
# loc = torch.rand(N, 3, dtype=dtype, device=d)
# th = th.to(d)
# loc = loc.to(d)
# # J = chain.jacobian(th, locations=loc, device=d)
# # print(f"\n{J}\n")