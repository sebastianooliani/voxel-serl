from scipy.spatial.transform import Rotation as R
import numpy as np


def construct_adjoint_matrix(tcp_pose, base_frame_rotation):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    if type(base_frame_rotation) != R:
        base_frame_rotation = R.from_euler("xyz", base_frame_rotation)
    rotation = (R.from_quat(tcp_pose[3:]) * base_frame_rotation).as_matrix()
    translation = np.array(tcp_pose[:3])
    skew_matrix = np.array(
        [
            [0, -translation[2], translation[1]],
            [translation[2], 0, -translation[0]],
            [-translation[1], translation[0], 0],
        ]
    )
    adjoint_matrix = np.zeros((6, 6))
    adjoint_matrix[:3, :3] = rotation
    adjoint_matrix[3:, 3:] = rotation
    adjoint_matrix[3:, :3] = skew_matrix @ rotation
    return adjoint_matrix


def construct_homogeneous_matrix(tcp_pose, base_frame_rotation):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    if type(base_frame_rotation) != R:
        base_frame_rotation = R.from_euler("xyz", base_frame_rotation)

    rotation = (R.from_quat(tcp_pose[3:]) * base_frame_rotation).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.zeros((4, 4))
    T[:3, :3] = rotation
    T[:3, 3] = translation
    T[3, 3] = 1
    return T
