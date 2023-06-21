import numpy as np
import torch
from scipy.spatial.transform import Rotation
from Load import LoadMvnx, LoadVideo, LoadOptitrack
from Anthropometric import Estimator
from InverseDynamic import InverseDynamicsSolver
from scipy.spatial.transform import Rotation
from SMPL_Vis import SMPLVis, smpl4vis
from Preprocess import KinematicsProcesser,center_of_mass_acceleration
import cv2
import keyboard
import pandas as pd

if __name__ == "__main__":
    smpl_segments = ['L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
                     'L_Foot', 'R_Foot',
                     'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
                     'L_Wrist',
                     'R_Wrist', 'L_Hand', 'R_Hand']
    MvnxLoader = LoadMvnx('C:/Users\gyz95\OneDrive\Desktop\Docs\Test-006.mvnx')
    MvnxOutput, length_xs = MvnxLoader.extract_info()
    smpl_poses_xs, global_rots_xs = MvnxLoader.xsens2smpl()

    OpLoader = LoadOptitrack('C:/Users\gyz95\OneDrive\Desktop\Docs\InverseDynamics\picking_0615.csv',subject_id=0)
    OpOutput, length_op = OpLoader.extract_info()
    smpl_poses_op, global_rots_op = OpLoader.op2smpl()

    op_tcs = OpLoader.get_times()
    xsens_data = {}
    op_data = {}

    beta = [[-0.8738482 , -0.74419683,  0.26313874,  0.06141186,  0.04800983, 0.13762029, -0.06491265,  0.02792327, -0.01361502, -0.04689167]]
    vis = SMPLVis(beta, 'male')


    for seg in smpl_segments:
        xsens_data[seg+'_x'] = []
        xsens_data[seg+'_y'] = []
        xsens_data[seg+'_z'] = []
        xsens_data['Timestamp'] = []

        op_data[seg+'_x'] = []
        op_data[seg+'_y'] = []
        op_data[seg+'_z'] = []
        op_data['Timestamp'] = []

    for i in range(length_xs): #length_xs
        xsenstc = MvnxLoader.get_time_single(index=i)
        op_index = (min(range(len(op_tcs)), key=lambda j: abs(op_tcs[j] - xsenstc)))
        if abs(op_tcs[op_index] - xsenstc) > 0.005: continue

        poses_xs = smpl_poses_xs[i]
        global_rot_xs = global_rots_xs[i]

        poses_op = smpl_poses_op[op_index]
        global_rot_op = global_rots_op[op_index]

        poses_xs, global_rot_xs = smpl4vis(poses_xs,global_rot_xs,source='xsens')
        poses_op, global_rot_op = smpl4vis(poses_op,global_rot_op,source='op')
        poses_xs_list = poses_xs.tolist()
        poses_op_list = poses_op.tolist()
        poses_op_euler = []
        poses_xs_euler = []

        for j in range(len(poses_xs_list[0])):
            op_j = list(Rotation.from_rotvec(poses_op_list[0][j]).as_euler('XYZ',degrees=True))
            xsens_j = list(Rotation.from_rotvec(poses_xs_list[0][j]).as_euler('XYZ',degrees=True))
            poses_op_euler.append(op_j)
            poses_xs_euler.append(xsens_j)

            xsens_data[smpl_segments[j]+'_x'].append(xsens_j[0])
            xsens_data[smpl_segments[j]+'_y'].append(xsens_j[1])
            xsens_data[smpl_segments[j]+'_z'].append(xsens_j[2])
            op_data[smpl_segments[j]+'_x'].append(op_j[0])
            op_data[smpl_segments[j]+'_y'].append(op_j[1])
            op_data[smpl_segments[j]+'_z'].append(op_j[2])
        op_data['Timestamp'].append(xsenstc)
        xsens_data['Timestamp'].append(xsenstc)

        #smpl_image_xs = vis.get_image(poses_xs,global_rot_xs)
        #smpl_image_op = vis.get_image(poses_op,global_rot_op)
        #image = cv2.hconcat([smpl_image_xs, smpl_image_op])

        #cv2.imshow('Vis',image)
        #cv2.waitKey(1)

    op_data = pd.DataFrame.from_dict(op_data)
    xsens_data = pd.DataFrame.from_dict(xsens_data)
    op_data.to_csv('optitrack_data.csv',index=False)
    xsens_data.to_csv('xsens_data.csv',index=False)
