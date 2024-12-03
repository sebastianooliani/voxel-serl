import numpy as np
from scipy.spatial.transform import Rotation as R

"""
UR5 represents the orientation in axis angle representation
"""


def rotvec_2_quat(rotvec):
    return R.from_rotvec(rotvec).as_quat()


def quat_2_rotvec(quat):
    return R.from_quat(quat).as_rotvec()


def quat_2_euler(quat):
    return R.from_quat(quat).as_euler('xyz')


def quat_2_mrp(quat):
    return R.from_quat(quat).as_mrp()


def euler_2_quat(euler):
    return R.from_euler(euler).as_quat()


def pose2quat(rotvec_pose) -> np.ndarray:
    return np.concatenate((rotvec_pose[:3], rotvec_2_quat(rotvec_pose[3:])))


def pose2rotvec(quat_pose) -> np.ndarray:
    return np.concatenate((quat_pose[:3], quat_2_rotvec(quat_pose[3:])))

def exp_coord_2_mrp(omega):
    """
    Convert exponential coordinates to Modified Rodriguez Parameters (MRP).
    
    Parameters:
    -----------
    omega : array-like
        3D exponential coordinates (rotation vector)
        The direction represents the axis of rotation
        The magnitude represents the angle of rotation in radians
    
    Returns:
    --------
    numpy.ndarray
        Modified Rodriguez Parameters (3D vector)
    """
    # Compute the angle of rotation (magnitude of the vector)
    theta = np.linalg.norm(omega)
    
    # Handle the case of near-zero rotation
    if np.isclose(theta, 0):
        return np.zeros(3)
    
    # Normalize the axis of rotation
    axis = omega / theta
    
    # Compute tan(θ/2)
    tan_half_theta = np.tan(theta / 2)
    
    # MRP is axis * tan(θ/2)
    mrp = axis * tan_half_theta
    
    return mrp

def mrp_2_exp_coord(mrp):
    """
    Convert Modified Rodriguez Parameters back to exponential coordinates.
    
    Parameters:
    -----------
    mrp : array-like
        Modified Rodriguez Parameters (3D vector)
    
    Returns:
    --------
    numpy.ndarray
        Exponential coordinates (rotation vector)
    """
    # Compute the magnitude of MRP
    mrp_mag = np.linalg.norm(mrp)
    
    # Handle near-zero rotation case
    if np.isclose(mrp_mag, 0):
        return np.zeros(3)
    
    # Compute angle
    # θ = 4 * arctan(|s|)
    theta = 4 * np.arctan(mrp_mag)
    
    # Normalize the rotation axis
    axis = mrp / mrp_mag
    
    # Exponential coordinates is axis * angle
    return axis * theta
