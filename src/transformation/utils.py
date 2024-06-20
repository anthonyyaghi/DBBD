import numpy as np

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles to a rotation matrix.
    
    :param roll: Rotation angle around the X-axis in radians.
    :param pitch: Rotation angle around the Y-axis in radians.
    :param yaw: Rotation angle around the Z-axis in radians.
    :return: 3x3 rotation matrix as a NumPy array.
    """
    # Compute individual rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combine the rotation matrices
    R = R_z @ R_y @ R_x
    return R
