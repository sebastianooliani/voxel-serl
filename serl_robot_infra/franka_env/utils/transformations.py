from scipy.spatial.transform import Rotation as R
import numpy as np


def construct_adjoint_matrix(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    :args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    rotation = R.from_quat(tcp_pose[3:]).as_matrix()
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


def construct_rotation_matrix(tcp_pose):
    """
    Construct the adjoint matrix for a spatial velocity vector
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    return R.from_quat(tcp_pose[3:]).as_matrix()


def construct_homogeneous_matrix(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose.
    args: tcp_pose: (x, y, z, qx, qy, qz, qw)
    """
    T = np.eye(4)
    T[:3, :3] = R.from_quat(tcp_pose[3:]).as_matrix()
    T[:3, 3] = np.array(tcp_pose[:3])
    return T

def pose_2_homogeneous_matrix(tcp_pose):
    """
    Construct the homogeneous transformation matrix from given pose with orientation
    represented with Modified Rodriguez Parameters.
    args: tcp_pose: (x, y, z, qx, qy, qz)
    """
    rotation = R.from_mrp(tcp_pose[3:]).as_matrix()
    translation = np.array(tcp_pose[:3])
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def orientation_difference_angle_axis(angle_axis1, angle_axis2):
    """
    Compute the orientation difference between two angle-axis representations.

    Args:
        angle_axis1 (array-like): First angle-axis representation (3 elements: axis * angle).
        angle_axis2 (array-like): Second angle-axis representation (3 elements: axis * angle).

    Returns:
        tuple: (angle_difference, axis_difference)
            - angle_difference (float): Angle of rotation difference in radians.
            - axis_difference (numpy.ndarray): Axis of the relative rotation (unit vector).
    """
    # Convert angle-axis to scipy Rotation objects
    r1 = R.from_rotvec(angle_axis1)
    r2 = R.from_rotvec(angle_axis2)
    
    # Compute the relative rotation
    r_rel = r2 * r1.inv()
    
    # Extract the angle-axis representation of the relative rotation
    angle_axis_rel = r_rel.as_rotvec()
    angle_difference = np.linalg.norm(angle_axis_rel)  # Magnitude of rotation
    axis_difference = angle_axis_rel / angle_difference if angle_difference > 1e-6 else np.array([0, 0, 0])
    
    return angle_difference, axis_difference

def quaternion_multiplication(q_ab: np.array, q_bc: np.array):
    """
    Compute the multiplication of two quaternions, i.e. q_ac = q_ab * q_bc. 
    Quaternions are represented with the scalar component as last element.

    Args:
        q_ab (np.array): First quaternion (4 elements).
        q_bc (np.array): Second quaternion (4 elements).

    Returns:
        np.array: Resultant quaternion (4 elements).
    """
    # Extract scalar and vector components
    s_bc, v_bc = q_bc[-1], q_bc[:3]
    
    # matrix
    M = np.array([s_bc, -v_bc[0], -v_bc[1], -v_bc[2]],
                 [v_bc[0], s_bc, v_bc[2], -v_bc[1]],
                 [v_bc[1], -v_bc[2], s_bc, v_bc[0]],
                 [v_bc[2], v_bc[1], -v_bc[0], s_bc])
    
    return M @ q_ab
                  
    