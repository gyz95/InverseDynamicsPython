import numpy as np
from scipy.spatial.transform import Rotation

def axes_align(angle,source):
    if source == 'xsens':
        R1 = [[0,1,0], [0,0,1], [1,0,0]]
    else:
        R1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    inv_R1 = np.linalg.inv(R1)
    new_angle = np.matmul((np.matmul(R1, angle)), inv_R1)
    return new_angle

def quaternion_to_axis_angle(quaternion):
    rotation = Rotation.from_quat(quaternion)
    axis_angle = rotation.as_rotvec(degrees=False)
    return axis_angle

def mat_to_quat(matrix):
    rotation = Rotation.from_matrix(matrix)
    return rotation.as_quat()

def quat_to_mat(quaternion):
    rotation = Rotation.from_quat(quaternion)
    return rotation.as_matrix()

def related_rotation(father,child):
    father = [father[1], father[2], father[3], father[0]]
    child = [child[1], child[2], child[3], child[0]]
    father_rotm = Rotation.from_quat(father).as_matrix()
    child_rotm = Rotation.from_quat(child).as_matrix()
    angles_raw = np.linalg.inv(father_rotm) @ child_rotm
    return angles_raw

def quat2rotm(quat):
    quat = [quat[1], quat[2], quat[3], quat[0]]
    rotm = Rotation.from_quat(quat).as_matrix()
    return rotm