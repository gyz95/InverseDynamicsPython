import numpy as np

class InverseDynamicsSolver:
    def __init__(self,r,zeta,I,M,R_L_G,ComACC,omega,alpha):
        # r: The location of each segment's Center of mass in segment's CS. 3x15 matrix. 3D vector for each segment.
        # zeta: The location of each segment's CS origin in the preceding segment's CS. 3x15 matrix. 3D vector for each segment.
        # M: The mass of each segment. Vector of 15 elements.
        # I: The inertia tensor of the segment about its center of mass, in the segment's CS. 3x3x15 matrix. 3x3 dyadic for each segment.
        # R_L_G: Rotational matrix from local segment CS to global CS (reference frame). 3x3x15xframes matrix. 3x3 matrix for each segment for each frame (15 segments).
        # ComACC: Center of mass acceleration of each segment. 3xnx15 matrix. 3D vector for each frame for each segment.
        # omega: Angular velocity of each segment. 3xnx15 matrix. 3D vector for each frame for each segment.
        # alpha: Angular acceleration of each segment. 3xnx15 matrix. 3D vector for each frame for each segment.
        # External Forces: The force on each segment for each frame in the reference frame. 3xnx15 matrix. 3D vector for each frame for each segment.
        # External Moments: The moment on each segment for each frame in the reference frame. 3xnx15 matrix. 3D vector for each frame for each segment.
        # External point of action: The location of the point of action of external force on each segment for each frame in the segment's CS. 3xnx15 matrix. 3D vector for each frame for each segment.
        self.ComACC = ComACC
        self.omega = np.transpose(omega, (2, 0, 1))
        self.alpha = np.transpose(alpha, (2, 0, 1))
        self.r = np.transpose(r, (1, 0))
        self.R_L_G = np.transpose(R_L_G, (2, 3, 1, 0))
        self.segmentmass = M
        self.zeta = np.transpose(zeta, (1, 0))
        self.inertiatensor = np.transpose(I,(1,2,0))
        self.segments = ['Pelvis', 'L5', 'Neck', 'RightUpperArm', 'RightForeArm', 'RightHand', 'LeftUpperArm', 'LeftForeArm', 'LeftHand', 'RightUpperLeg', 'RightLowerLeg', 'RightFoot', 'LeftUpperLeg', 'LeftLowerLeg', 'LeftFoot']
        self.pd_pairs = self.build_pair_pd()
        self.joint_force, self.joint_moment = self.solver()

    def get_force_moment(self):
        return self.joint_force, self.joint_moment

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

    def solver(self):
        joint_force = np.zeros((3,15,len(self.omega[1])-1))
        joint_moment = np.zeros((3,15,len(self.omega[1])-1))
        for frame in range(len(self.omega[1])-1):
            for segment in range(8, -1, -1):
                RSG = self.R_L_G[:,:,segment,frame]
                I_segment = np.dot(np.dot(RSG.T, self.inertiatensor[2,2,segment]),RSG)
                r_cm = np.dot(RSG,self.r[:,segment])
                F_inertia = np.dot(-self.segmentmass[segment] , self.ComACC[:,frame,segment])
                M_inertia = -(I_segment @ self.alpha[:, frame, segment] + np.cross(self.omega[:, frame, segment], I_segment @ self.omega[:, frame, segment]))
                if 'Foot' not in self.segments[segment] and 'Hand' not in self.segments[segment] and 'Neck' not in self.segments[segment]:
                    distal_joints = self.pd_pairs[self.segments[segment]]
                else:
                    distal_joints = []
                F_distal_joints = np.zeros(3)
                M_distal_joints = np.zeros(3)
                for distal_joint in distal_joints:
                    id = self.segments.index(distal_joint)
                    r_distal_joint = np.dot(RSG,self.zeta[:, id])
                    F_distal_joints = F_distal_joints - joint_force[:,id-1,frame]
                    M_distal_joints = M_distal_joints - joint_moment[:,id-1,frame] + np.cross((r_distal_joint - r_cm), -joint_force[:,id-1,frame])
                if segment != 0:
                    joint_force[:,segment-1,frame] = -F_inertia - F_distal_joints
                    joint_moment[:,segment-1,frame] = -M_inertia - M_distal_joints - np.cross(-r_cm, joint_force[:,segment-1,frame])
                else:
                    joint_force[:,14,frame] = -F_inertia - F_distal_joints
                    joint_moment[:,14,frame] = -M_inertia - M_distal_joints - np.cross(-r_cm, joint_force[:,14,frame])
                '''
                if segment == 1:
                    print('F_inertia:',F_inertia)
                    print('M_inertia:',M_inertia)
                    print('RSG:',RSG)
                    print('I_segment',I_segment)
                    print('r_cm:',r_cm)
                    print('Distal Joints:',distal_joints)
                    print('Force Distal Joints:',F_distal_joints)
                    print('Moment Distal Joints:', M_distal_joints)
                    print('Alpha:',self.alpha[:, frame, segment])
                    print('Omega:',self.omega[:, frame, segment])
                    print('Segment Mass',self.segmentmass[segment])
                    print('ComAcc',self.ComACC[:,frame,segment])
                    print('---------------------')
                '''
        return joint_force, joint_moment


