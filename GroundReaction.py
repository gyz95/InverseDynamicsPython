import numpy as np
import math

class GroundReactionEstimator:
    def __init__(self, Vel, RightToeVel, LeftToeVel):
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
        self.contact = []
        self.Contact_detection()

    def Vel_Norm(self, Vel):
        Vel_norm = []
        for i in range(len(Vel)):
            norm = math.sqrt(Vel[i][0]**2+Vel[i][1]**2+Vel[i][2]**2)
            Vel_norm.append(norm)
        return Vel_norm

    def Contact_detection(self):
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
        print(contact_r)
        print(contact_l)






