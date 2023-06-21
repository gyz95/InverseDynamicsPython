import numpy as np
import torch
from scipy.spatial.transform import Rotation
from Load import LoadMvnx, LoadVideo, LoadOptitrack
from Anthropometric import Estimator
from InverseDynamic import InverseDynamicsSolver
from scipy.spatial.transform import Rotation
from SMPL_Vis import SMPLVis, smpl4vis
from Preprocess import KinematicsProcesser,center_of_mass_acceleration
from GroundReaction import GroundReactionEstimator
import cv2
import keyboard

if __name__ == "__main__":
    MvnxLoader = LoadMvnx('C:/Users\gyz95\PycharmProjects\InverseDynamicsPython\data\Test-006.mvnx')
    MvnxOutput, length = MvnxLoader.extract_info()
    smpl_poses, global_rots = MvnxLoader.xsens2smpl()
    KinematicsProcesser = KinematicsProcesser(MvnxOutput)
    Acc,AngularVel,AngularAcc,CenterPos,RLG,Vel,RightToeVel,LeftToeVel= KinematicsProcesser.extract()
    AnthropometricEstimator = Estimator(BodyWeight = 90, BodyHeight = 1.7, Gender='male')
    segment_mass, segment_length, centerofmass, seg_ori, inertiatensor = AnthropometricEstimator.get()
    CM_acc = center_of_mass_acceleration(Acc,AngularVel,AngularAcc,centerofmass,RLG)
    joint_force, joint_moment = InverseDynamicsSolver(centerofmass,seg_ori,inertiatensor,segment_mass,RLG,CM_acc,AngularVel,AngularAcc).get_force_moment()
    GRFM = GroundReactionEstimator(Vel, RightToeVel, LeftToeVel)
    #OpLoader = LoadOptitrack('C:/Users\gyz95\OneDrive\Desktop\Docs\InverseDynamics\picking_0615.csv',subject_id=0)
    #OpOutput, length = OpLoader.extract_info()
    #smpl_poses, global_rots = OpLoader.op2smpl()
    #beta = [list(np.random.rand(10))]
    beta = [[-0.8738482 , -0.74419683,  0.26313874,  0.06141186,  0.04800983, 0.13762029, -0.06491265,  0.02792327, -0.01361502, -0.04689167]]
    #beta = [[0.05921802, -0.36343077,  0.09036009,  0.10710271,  0.02527221, 0.08302245, -0.07464162,  0.0130268 , -0.01174876, -0.04193153]]
    VideoLoader = LoadVideo('C:/Users\gyz95\PycharmProjects\InverseDynamicsPython\data\GX011768.MP4','00:57:55:17')
    VideoTc = VideoLoader.get_times()
    vis = SMPLVis(beta, 'male')
    out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640*2, 480))

    for i in range(length):
        if i%2 != 0: continue
        #x=joint_moment[0][0][i]
        #y=joint_moment[1][0][i]
        #z=joint_moment[2][0][i]
        #print([x,y,z])
        mocaptc = MvnxLoader.get_time_single(index=i)
        video_index = (min(range(len(VideoTc)), key=lambda j: abs(VideoTc[j] - mocaptc)))
        if abs(VideoTc[video_index] - mocaptc) > 0.05: continue
        video_image = cv2.resize(VideoLoader.extract_frame(video_index),(640,480))
        poses = smpl_poses[i]
        global_rot = global_rots[i]
        poses, global_pose = smpl4vis(poses,global_rot,source='xsens')
        smpl_image = vis.get_image(poses,global_pose)
        image = cv2.hconcat([video_image, smpl_image])
        rightcontact, leftcontact = GRFM.Get_Single_Contact(index=i)
        image = cv2.putText(image, 'RightFootContact:'+str(rightcontact), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)
        image = cv2.putText(image, 'LeftFootContact:'+str(leftcontact), (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Vis',image)
        cv2.waitKey(1)
        out.write(image)
        if keyboard.is_pressed('q'): break
    # When everything done, release the video capture and video write objects
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()
