import numpy as np
from scipy.spatial.transform import Rotation
from mujoco.viewer import Handle
import mujoco

def base_configuration(mjData):
    """Robot base configuration (homogenous transformation matrix) in world reference frame."""
    com_pos = mjData.qpos[0:3]  # world frame
    quat_wxyz = mjData.qpos[3:7]  # world frame (wxyz) mujoco convention
    quat_xyzw = np.roll(quat_wxyz, -1)  # SciPy convention (xyzw)
    X_B = np.eye(4)
    X_B[0:3, 0:3] = Rotation.from_quat(quat_xyzw).as_matrix()
    X_B[0:3, 3] = com_pos
    return X_B


def base_lin_vel(mjData, frame='world'):
    """Returns the base linear velocity (3,) in the specified frame."""
    if frame == 'world':
        return mjData.qvel[0:3]
    elif frame == 'base':
        R = base_configuration(mjData)[0:3, 0:3]
        return R.T @ mjData.qvel[0:3]
    else:
        raise ValueError(f"Invalid frame: {frame} != 'world' or 'base'")
    

def base_ang_vel(mjData, frame='world'):
    """Returns the base angular velocity (3,) in the specified frame."""
    if frame == 'base':
        return mjData.qvel[3:6]
    elif frame == 'world':
        R = base_configuration(mjData)[0:3, 0:3]
        return R @ mjData.qvel[3:6]
    else:
        raise ValueError(f"Invalid frame: {frame} != 'world' or 'base'")


def base_ori_euler_xyz(mjData):
    """Returns the base orientation in Euler XYZ angles (roll, pitch, yaw) in the world reference frame."""
    quat_wxyz = mjData.qpos[3:7]
    quat_xyzw = np.roll(quat_wxyz, -1)
    return Rotation.from_quat(quat_xyzw).as_euler('xyz')


def heading_orientation_SO3(mjData):
    """Returns a SO(3) matrix that aligns with the robot's base heading orientation and the world z axis."""
    X_B = base_configuration(mjData)
    R_B = X_B[0:3, 0:3]
    euler_xyz = Rotation.from_matrix(R_B).as_euler('xyz')
    # Rotation aligned with the base orientation and the vertical axis
    R_B_horizontal = Rotation.from_euler('xyz', euler_xyz * [0, 0, 1]).as_matrix()
    return R_B_horizontal


def base_pos(mjData):
    """Returns the base position (3,) in the world reference frame."""
    return mjData.qpos[0:3]


def target_base_vel(mjData, ref_base_lin_vel_H, ref_base_ang_yaw_dot, frame='world') -> tuple[np.ndarray, np.ndarray]:
    """Returns the target base linear (3,) and angular (3,) velocity in the world reference frame."""
    if ref_base_lin_vel_H is None:
        return np.zeros(3), np.zeros(3)
    R_B_heading = heading_orientation_SO3(mjData)
    ref_base_lin_vel = (R_B_heading @ ref_base_lin_vel_H.reshape(3, 1)).squeeze()
    ref_base_ang_vel = np.array([0.0, 0.0, ref_base_ang_yaw_dot])
    if frame == 'world':
        return ref_base_lin_vel, ref_base_ang_vel
    elif frame == 'base':
        R = base_configuration(mjData)[0:3, 0:3]
        return R.T @ ref_base_lin_vel, R.T @ ref_base_ang_vel
    

def render_sphere(viewer: Handle, position: np.ndarray, diameter: float, color: np.ndarray, geom_id: int = -1) -> int:
    """Function to render a sphere in the Mujoco viewer.

    Args:
        viewer (Handle): The Mujoco viewer.
        position (np.ndarray): The position of the sphere.
        diameter (float): The diameter of the sphere.
        color (np.ndarray): The color of the sphere.
        geom_id (int, optional): The id of the geometry. Defaults to -1.

    Returns:
        int: The id of the geometry.
    """
    if viewer is None:
        return -1

    if geom_id < 0 or geom_id is None:
        # Instantiate a new geometry
        geom = mujoco.MjvGeom()
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1

    geom = viewer.user_scn.geoms[geom_id]

    # Initialize the geometry
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.asarray([diameter / 2] * 3),  # Radius is half the diameter
        mat=np.eye(3).flatten(),
        pos=position,
        rgba=color,
    )

    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1

    return geom_id