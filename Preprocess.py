import numpy as np
import torch
from scipy.spatial.transform import Rotation
from Anthropometric import Estimator
from InverseDynamic import InverseDynamicsSolver
from GroundReaction import GroundReactionEstimator
from scipy.spatial.transform import Rotation

def get_joint_position(data,segment):
    output = []
    for i in range(len(data)):
        output.append(data[i]['position']['segment'])
    return  output

def center_of_mass_acceleration(acceleration, omega, alpha, r, R_L_G):
    acceleration = np.transpose(acceleration, (2, 0, 1))
    omega = np.transpose(omega, (2, 0, 1))
    alpha = np.transpose(alpha, (2, 0, 1))
    r = np.transpose(r, (1,0))
    R_L_G = np.transpose(R_L_G, (2,3,1,0))
    CM_acceleration = np.zeros(acceleration.shape)
    for frame in range(acceleration.shape[1] - 1):
        for segment in range(15):
            r_cm = np.dot(R_L_G[:, :, segment, frame], r[:,segment])  # center of mass location away from segment's origin (proximal_joint) in global CS
            # center of mass acceleration
            CM_acceleration[:, frame, segment] = acceleration[:, frame, segment] + np.cross(alpha[:, frame, segment], r_cm) + np.cross(omega[:, frame, segment], np.cross(omega[:, frame, segment], r_cm))
    return CM_acceleration

class KinematicsProcesser:
    def __init__(self, data):
        self.data = data
        self.segments = ['Pelvis', 'L5', 'Neck', 'RightUpperArm', 'RightForeArm', 'RightHand', 'LeftUpperArm', 'LeftForeArm', 'LeftHand', 'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot']
        self.R = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, -1, 0]])
        self.inv_R = np.transpose(self.R)
        self.R_L_G = []
        self.Acc = []
        self.Vel = []
        self.AngularVel = []
        self.AngularAcc = []
        self.CenterPos = []
        self.RightToeVel = []
        self.LeftToeVel = []
        self.conversion()

    def quat2rotm(self,rot_quat):
        rot_quat = [rot_quat[1],rot_quat[2],rot_quat[3],rot_quat[0]]
        rotation = Rotation.from_quat(rot_quat)
        rotation_matrix = rotation.as_matrix()
        return rotation_matrix

    def conversion(self):
        data = list(self.data.values())
        for i in range(len(self.data)):
            RLG_i = []
            Acc_i = []
            Vel_i = []
            AngularVel_i = []
            AngularAcc_i = []
            self.CenterPos.append(self.R @ data[i]['position']['L5'])
            self.RightToeVel.append(self.R @ data[i]['velocity']['RightToe'])
            self.LeftToeVel.append(self.R @ data[i]['velocity']['LeftToe'])
            for seg in self.segments:
                RLG_i.append(self.R @ self.quat2rotm(data[i]['orientation'][seg]) @ self.inv_R)
                Acc_i.append(self.R @ data[i]['acceleration'][seg])
                Vel_i.append(self.R @ data[i]['velocity'][seg])
                AngularVel_i.append(self.R @ data[i]['angular velocity'][seg])
                AngularAcc_i.append(self.R @ data[i]['angular acceleration'][seg])
            self.Acc.append(Acc_i)
            self.R_L_G.append(RLG_i)
            self.Vel.append(Vel_i)
            self.AngularVel.append(AngularVel_i)
            self.AngularAcc.append(AngularAcc_i)
        for j in range(len(self.Acc)):
            for k in range(len(self.Acc[0])):
                self.Acc[j][k][1] += 9.807

    def extract(self):
        return self.Acc,self.AngularVel,self.AngularAcc,self.CenterPos,self.R_L_G,self.Vel,self.RightToeVel,self.LeftToeVel

if __name__ == "__main__":
    pass
    #loader = LoadMvnx('C:/Users\gyz95\OneDrive\Desktop\Docs\Guoyang-heavylift-4.mvnx')
    #output = loader.extract_info()
    #KinematicsProcesser = KinematicsProcesser(loader.extract_info())
    #Acc,AngularVel,AngularAcc,CenterPos,RLG,Vel,RightToeVel,LeftToeVel= KinematicsProcesser.extract()
    #GroundReaction = GroundReactionEstimator(Vel,RightToeVel,LeftToeVel)
    #AnthropometricEstimator = Estimator(BodyWeight = 90, BodyHeight = 1.7, Gender='male')
    #segment_mass, segment_length, centerofmass, seg_ori, inertiatensor = AnthropometricEstimator.get()
    #CM_acc = center_of_mass_acceleration(Acc,AngularVel,AngularAcc,centerofmass,RLG)
    #joint_force, joint_moment = InverseDynamicsSolver(centerofmass,seg_ori,inertiatensor,segment_mass,RLG,CM_acc,AngularVel,AngularAcc).get_force_moment()
    #    poses = [[element for sublist in smpl_poses[i] for element in sublist]]
    #    poses = torch.tensor(poses,dtype=torch.float32)
    #    global_orient = torch.zeros([1, 3], dtype=torch.float32)
    #    image = vis.show(poses,global_orient)
    #    cv2.imshow('Vis',image)
    #    cv2.waitKey(10)

    #DoubleOptimization(joint_moment[:,0,:],joint_force[:,0,:],RLG[:,:,0,:],)