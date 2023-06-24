import numpy as np
import math
from scipy.optimize import minimize

class GroundReactionEstimator:
    def __init__(self, Vel, RightToeVel, LeftToeVel, XsensContact, Use_Xsens=True):
        self.pd_pairs = self.build_pair_pd()
        self.segments = ['Pelvis', 'L5', 'Neck', 'RightUpperArm', 'RightForeArm', 'RightHand', 'LeftUpperArm', 'LeftForeArm', 'LeftHand', 'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot']
        self.RightFootVel = np.asarray(Vel)[:,11,:]
        self.LeftFootVel = np.asarray(Vel)[:,14,:]
        self.RightToeVel = np.asarray(RightToeVel)
        self.LeftToeVel = np.asarray(LeftToeVel)
        self.RightFootNorm = self.Vel_Norm(self.RightFootVel)
        self.RightFootAcc = np.diff(self.RightFootNorm)
        self.LeftFootNorm = self.Vel_Norm(self.LeftFootVel)
        self.LeftFootAcc = np.diff(self.LeftFootNorm)
        self.RightToeNorm = self.Vel_Norm(self.RightToeVel)
        self.LeftToeNorm = self.Vel_Norm(self.LeftToeVel)
        self.Detect_Th = 1.2
        self.RightFoot, self.LeftFoot = self.Contact_detection(XsensContact,Use_Xsens)
        self.contact = np.asarray([self.RightFoot, self.LeftFoot]).transpose(1,0)



    def build_pair_pd(self):
        pd_pairs = {}
        pd_pairs['LeftForeArm'] = ['LeftHand'] #9-8
        pd_pairs['LeftUpperArm'] = ['LeftForeArm'] #8-7
        pd_pairs['RightForeArm'] = ['RightHand'] #5-6
        pd_pairs['RightUpperArm'] = ['RightForeArm'] #4-5
        pd_pairs['L5'] = ['Neck','RightUpperArm','LeftUpperArm']
        pd_pairs['Pelvis'] = ['L5','LeftUpperLeg','RightUpperLeg']
        pd_pairs['LeftLowerLeg'] = ['LeftFoot']
        pd_pairs['LeftUpperLeg'] = ['LeftLowerLeg']
        pd_pairs['RightLowerLeg'] = ['RightFoot']
        pd_pairs['RightUpperLeg'] = ['RightLowerLeg']
        return pd_pairs

    def cost_function(self, r_l_GRF_M, r,zeta,I,M,R_L_G,A,omega,alpha, frame, external_forces, external_moments, external_point_of_action):
        # Initialize joint force and joint moment for this frame only, for joints from 8:13 only (segment 9:14)
        J_F = np.zeros((3, 14))
        J_M = np.zeros((3, 14))

        external_forces[:3, frame, 11] = external_forces[:3, frame, 11] + r_l_GRF_M[:3]
        external_moments[:3, frame, 11] = external_moments[:3, frame, 11] + r_l_GRF_M[3:6]
        external_forces[:3, frame, 14] = external_forces[:3, frame, 14] + r_l_GRF_M[6:9]
        external_moments[:3, frame, 14] = external_moments[:3, frame, 14] + r_l_GRF_M[9:12]

        for segment in range(14, 9, -1):
            R_S_G = R_L_G[:, :, segment, frame]
            I_segment = np.dot(np.dot(R_S_G.T, I[2, 2, segment]), R_S_G)
            r_CM = np.dot(R_S_G, r[:,segment])
            r_external = np.dot(R_S_G,external_point_of_action[:3, frame, segment])
            F_inertia = np.dot(-M[segment], A[:, frame, segment])
            M_inertia = -(I_segment @ alpha[:3, frame, segment] + np.cross(omega[:3, frame, segment], I_segment @ omega[:3, frame, segment]))
            F_external = external_forces[:3, frame, segment]
            M_external = external_moments[:3, frame, segment] + np.cross((r_external - r_CM), F_external)

            #distal_segments = np.where(L_B_A == segment)[0]
            F_distal_joints = np.zeros(3)
            M_distal_joints = np.zeros(3)
            if 'Foot' not in self.segments[segment] and 'Hand' not in self.segments[segment] and 'Neck' not in self.segments[segment]:
                distal_joints = self.pd_pairs[self.segments[segment]]
            else:
                distal_joints = []

            for distal in distal_joints:
                id = self.segments.index(distal)
                r_distal_joint = R_S_G @ zeta[:, id]
                F_distal_joints -= J_F[:3, id - 1]
                M_distal_joints -= J_M[:3, id - 1] + np.cross((r_distal_joint - r_CM), -J_F[:3,id - 1])

            J_F[:3, segment - 1] = -F_inertia - F_distal_joints - F_external
            J_M[:3, segment - 1] = -M_inertia - M_distal_joints - np.cross(-r_CM, J_F[:3, segment - 1]) - M_external

        l_knee_moment = R_L_G[:, :, 12, frame] @ J_M[:, 12]
        l_ankle_moment = R_L_G[:, :, 13, frame] @ J_M[:, 13]
        r_knee_moment = R_L_G[:, :, 9, frame] @ J_M[:, 9]
        r_ankle_moment = R_L_G[:, :, 10, frame] @ J_M[:, 10]

        joint_moments_squared = np.sum(np.square(J_M), axis=0)
        sum_of_moments = np.sum(joint_moments_squared[8:14]) + l_ankle_moment[0] ** 4 + r_ankle_moment[0] ** 4 + \
                         l_ankle_moment[1] ** 4 + r_ankle_moment[1] ** 4
        return sum_of_moments

    def GRFM_Estimation(self, r, zeta, I, M, R_L_G, A, omega, alpha):
        M = np.asarray(M)
        A = np.asarray(A)
        omega = np.transpose(omega, (2, 0, 1))
        alpha = np.transpose(alpha, (2, 0, 1))
        r = np.transpose(r, (1, 0))
        R_L_G = np.transpose(R_L_G, (2, 3, 1, 0))
        zeta = np.transpose(zeta, (1, 0))
        I = np.transpose(I,(1,2,0))

        frames = np.shape(A)[1]
        external_forces = np.zeros((3,frames,15))
        external_moments = np.zeros((3,frames,15))
        external_point_of_action = np.zeros((3,frames,15))

        Net_GRF = np.zeros((3, frames))
        Net_GRM = np.zeros((3, frames))
        GRF_r = np.zeros((3, frames))
        GRF_l = np.zeros((3, frames))
        GRM_r = np.zeros((3, frames))
        GRM_l = np.zeros((3, frames))
        R_Foot_origin = np.zeros((3, frames))
        L_Foot_origin = np.zeros((3, frames))

        for frame in range(frames):
            print(frame)
            # Net GRF and GRM
            Net_GRF[:, frame] = 0
            Net_GRM[:, frame] = 0
            right_lower_limb_net_force = np.zeros(3)
            right_lower_limb_net_moment = np.zeros(3)
            left_lower_limb_net_force = np.zeros(3)
            left_lower_limb_net_moment = np.zeros(3)
            for segment in range(14, -1, -1):
                R_S_G = R_L_G[:, :, segment, frame]
                I_segment = np.dot(np.dot(R_S_G.T, I[2,2,segment]),R_S_G)
                r_CM = np.dot(R_S_G,r[:,segment])

                F_inertia = np.dot(-M[segment] , A[:,frame,segment])
                M_inertia = -(I_segment @ alpha[:, frame, segment] + np.cross(omega[:, frame, segment], I_segment @ omega[:, frame, segment]))

                segment_origin = np.zeros(3)
                proximal_segment = None
                for seg in list(self.pd_pairs.keys()):
                    if self.segments[segment] in self.pd_pairs[seg]:
                        proximal_segment = self.segments.index(seg)
                current_segment = segment

                if proximal_segment != None:
                    while proximal_segment >= 1:
                        R_PS_G = R_L_G[:, :, proximal_segment, frame]
                        zeta_current = zeta[:, current_segment]
                        segment_origin = segment_origin + np.dot(R_PS_G, zeta_current)
                        current_segment = proximal_segment
                        for seg in list(self.pd_pairs.keys()):
                            if self.segments[proximal_segment] in self.pd_pairs[seg]:
                                proximal_segment = self.segments.index(seg)

                Net_GRF[:, frame] = Net_GRF[:,frame] - F_inertia #-F_external
                Net_GRM[:, frame] = Net_GRM[:,frame] - M_inertia + np.cross(r_CM + segment_origin, F_inertia) #- np.cross((r_external+segment_origin),F_external)
                if segment == 11:
                    R_Foot_origin[:, frame] = segment_origin
                elif segment == 14:
                    L_Foot_origin[:, frame] = segment_origin

                if 9 <= segment <= 11:
                    right_lower_limb_net_force = right_lower_limb_net_force - F_inertia
                    right_lower_limb_net_moment = right_lower_limb_net_moment - M_inertia + np.cross(r_CM + segment_origin, F_inertia)
                elif 12 <= segment <= 14:
                    left_lower_limb_net_force = left_lower_limb_net_force - F_inertia
                    left_lower_limb_net_moment = left_lower_limb_net_moment - M_inertia + np.cross(r_CM + segment_origin, F_inertia)

                if self.contact[frame, 0] == 0 and self.contact[frame, 1] == 0:
                    GRF_r[:, frame] = 0
                    GRF_l[:, frame] = 0
                    GRM_r[:, frame] = 0
                    GRM_l[:, frame] = 0
                elif self.contact[frame, 0] == 1 and self.contact[frame, 1] == 0:
                    GRF_r[:, frame] = Net_GRF[:, frame]
                    GRM_r[:, frame] = Net_GRM[:, frame] + np.cross(-R_Foot_origin[:, frame], Net_GRF[:, frame])
                elif self.contact[frame, 0] == 0 and self.contact[frame, 1] == 1:
                    GRF_l[:, frame] = Net_GRF[:, frame]
                    GRM_l[:, frame] = Net_GRM[:, frame] + np.cross(-L_Foot_origin[:, frame], Net_GRF[:, frame])
                else:
                    if frame != 0:
                        GRFM_first_iteration = np.concatenate((GRF_r[:, frame - 1], GRM_r[:, frame - 1],
                                                               Net_GRF[:, frame - 1] - GRF_r[:, frame - 1],
                                                               GRM_l[:, frame - 1]), axis=0)
                    else:
                        right_GRM_firstiteration = right_lower_limb_net_moment + (Net_GRM[:,
                                                                                  frame] - left_lower_limb_net_moment - right_lower_limb_net_moment) / 2 + np.cross(
                            -R_Foot_origin[:, frame], Net_GRF[:, frame] / 2)
                        left_GRM_firstiteration = left_lower_limb_net_moment + (Net_GRM[:,
                                                                                frame] - left_lower_limb_net_moment - right_lower_limb_net_moment) / 2 + np.cross(
                            -L_Foot_origin[:, frame], Net_GRF[:, frame] / 2)
                        GRFM_first_iteration = np.concatenate((Net_GRF[:, frame] / 2, right_GRM_firstiteration,
                                                               Net_GRF[:, frame] / 2, left_GRM_firstiteration), axis=0)
                    #print(GRFM_first_iteration)
                    # Equality Constraints
                    Aeq = np.zeros((6, 12))
                    beq = np.zeros(6)
                    Aeq[0, 0] = 1
                    Aeq[0, 6] = 1
                    beq[0] = Net_GRF[0, frame]
                    Aeq[1, 1] = 1
                    Aeq[1, 7] = 1
                    beq[1] = Net_GRF[1, frame]
                    Aeq[2, 2] = 1
                    Aeq[2, 8] = 1
                    beq[2] = Net_GRF[2, frame]
                    Aeq[3, 1] = -R_Foot_origin[2, frame]
                    Aeq[3, 2] = R_Foot_origin[1, frame]
                    Aeq[3, 7] = -L_Foot_origin[2, frame]
                    Aeq[3, 8] = L_Foot_origin[1, frame]
                    Aeq[3, 3] = 1
                    Aeq[3, 9] = 1
                    beq[3] = Net_GRM[0, frame]
                    Aeq[4, 0] = R_Foot_origin[2, frame]
                    Aeq[4, 2] = R_Foot_origin[0, frame]
                    Aeq[4, 7] = L_Foot_origin[2, frame]
                    Aeq[4, 8] = -L_Foot_origin[1, frame]
                    Aeq[4, 4] = 1
                    Aeq[4, 10] = 1
                    beq[4] = Net_GRM[1, frame]
                    Aeq[5, 0] = -R_Foot_origin[1, frame]
                    Aeq[5, 1] = R_Foot_origin[0, frame]
                    Aeq[5, 7] = -L_Foot_origin[1, frame]
                    Aeq[5, 8] = L_Foot_origin[0, frame]
                    Aeq[5, 5] = 1
                    Aeq[5, 11] = 1
                    beq[5] = Net_GRM[2, frame]

                    # Upper and Lower Bounds
                    lb = [(-1000, 0, -1000, -1000, -25, -1000, -1000, 0, -1000, -1000, -25, -1000)]
                    ub = [(1000, 10000, 1000, 1000, 25, 1000, 1000, 10000, 1000, 1000, 25, 1000)]
                    bounds = [(-1000, 1000),
                              (0, 10000),
                              (-1000, 1000),
                              (-1000, 1000),
                              (-25, 25),
                              (-1000, 1000),
                              (-1000, 1000),
                              (0, 10000),
                              (-1000,1000),
                              (-1000,1000),
                              (-25,25),
                              (-1000,1000)]

                    Aineq = None
                    bineq = None
                    # Create a lambda function to use in the minimize call
                    cost_fn = lambda x: self.cost_function(x, r,zeta,I,M,R_L_G,A,omega,alpha, frame, external_forces, external_moments, external_point_of_action)

                    # Set the constraints as a dictionary
                    #ineq_constr = {'type': 'ineq', 'fun': lambda x: Aineq @ x - bineq}
                    eq_constr = {'type': 'eq', 'fun': lambda x: Aeq @ x - beq}

                    # Perform the minimization
                    res = minimize(cost_fn, GRFM_first_iteration, method='SLSQP', bounds=bounds, constraints=[eq_constr])
                    GRF_r[:, frame] = res.x[:3]
                    GRM_r[:, frame] = res.x[3:6]
                    GRF_l[:, frame] = res.x[6:9]
                    GRM_l[:, frame] = res.x[9:12]
            print(GRF_l[:,frame])
            print(GRF_r[:,frame])
            print('-----------------------------------------')

        return GRF_r, GRF_l, GRM_r, GRM_l

    def Get_Single_Contact(self,index):
        return self.RightFoot[index], self.LeftFoot[index]

    def Vel_Norm(self, Vel):
        Vel_norm = []
        for i in range(len(Vel)):
            norm = math.sqrt(Vel[i][0]**2+Vel[i][1]**2+Vel[i][2]**2)
            Vel_norm.append(norm)
        return Vel_norm

    def Contact_detection(self,XsensContact,Use_Xsens):
        if Use_Xsens == False:
            contact_r = np.zeros(len(self.RightFootVel))
            contact_l = np.zeros(len(self.LeftFootVel))
            for i in range(len(self.RightFootVel)):
                if i == 0:
                    if self.RightToeNorm[0] <= self.Detect_Th: contact_r[i] = 1
                    else: contact_r[i] = 0
                    if self.LeftToeNorm[0] <= self.Detect_Th: contact_l[i] = 1
                    else: contact_l[i] = 0
                else:
                    if contact_r[i-1] == 1:
                        if self.RightToeNorm[i] > self.Detect_Th and self.RightFootAcc[i] <= 0: contact_r[i] = 0
                        else: contact_r[i] = 1
                    else:
                        if self.RightToeNorm[i] > self.Detect_Th: contact_r[i] = 0
                        else: contact_r[i] = 1

                    if contact_l[i-1] == 1:
                        if self.LeftToeNorm[i] > self.Detect_Th and self.LeftFootAcc[i] <= 0: contact_l[i] = 0
                        else: contact_l[i] = 1
                    else:
                        if self.LeftToeNorm[i] > self.Detect_Th: contact_l[i] = 0
                        else: contact_l[i] = 1

                    if contact_l[i] == 0 and contact_r[i] == 0: contact_r[i] = 1
        else:
            contact_r = np.zeros(len(XsensContact))
            contact_l = np.zeros(len(XsensContact))
            for i in range(len(XsensContact)):
                left = [XsensContact[i][0],XsensContact[i][1]]
                right = [XsensContact[i][2],XsensContact[i][3]]
                if 1 in left:
                    contact_l[i] = 1
                if 1 in right:
                    contact_r[i] = 1
        return contact_r, contact_l







