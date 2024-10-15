import math
import torch
import pytorch_kinematics as pk

file_name = "ur5.urdf"

# can convert Chain to SerialChain by choosing end effector frame
chain = pk.build_chain_from_urdf(open(file_name).read())
# print(chain) to see the available links for use as end effector
print(f"\n{chain}\n")
# note that any link can be chosen; it doesn't have to be a link with no children
chain = pk.SerialChain(chain, "wrist_3_link")

chain = pk.build_serial_chain_from_urdf(open(file_name).read(), "wrist_3_link")
th = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0])
# (1,6,7) tensor, with 7 corresponding to the DOF of the robot
J = chain.jacobian(th)
print(f"\n{J}\n")

# get Jacobian in parallel and use CUDA if available
N = 1000
d = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {d}\n")
dtype = torch.float64

chain = chain.to(dtype=dtype, device=d)
# Jacobian calculation is differentiable
# th = torch.rand(N, 6, dtype=dtype, device=d, requires_grad=True)
th = th.to(d)
# (N,6,7)
J = chain.jacobian(th)

print(f"\n{J}\n")

# can get Jacobian at a point offset from the end effector (location is specified in EE link frame)
# by default location is at the origin of the EE frame
loc = torch.rand(N, 3, dtype=dtype, device=d)
th = th.to(d)
loc = loc.to(d)
# J = chain.jacobian(th, locations=loc, device=d)
# print(f"\n{J}\n")