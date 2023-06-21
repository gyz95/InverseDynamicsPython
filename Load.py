import mvnx
import numpy as np
from scipy.signal import butter, filtfilt
from Tools import related_rotation, quat2rotm
import cv2
from threading import Thread
import time
import pandas as pd
from timecode import Timecode

class LoadOptitrack:
    def __init__(self, filename, subject_id=0):
        self.rawdata = pd.read_csv(filename,skiprows=2, header=[0,1,3,4],low_memory=False)
        self.BoneNames, self.SubjectNames = self.KeyHeaders()
        self.Subject = self.SubjectNames[subject_id]
        self.Data = self.process_raw()
        self.Timecode = self.rawdata[self.rawdata.columns[2]].tolist()
        self.TC_Mocap = self.MocapTCProcessed(self.Timecode.copy())
        self.Data[self.Subject+'MetaInfo','Time','Processed'] = self.TC_Mocap
        self.Data[self.Subject+'MetaInfo','Time','Raw'] = self.Timecode
        self.Data = self.Data.dropna()
        self.Data_dict = self.ToDict()

    def get_time_single(self,raw=False,index=0):
        if raw:
            times = self.Data[self.Subject+'MetaInfo']['Time']['Raw'].tolist()[index]
        else:
            times = self.Data[self.Subject+'MetaInfo']['Time']['Processed'].tolist()[index]
        return times

    def get_times(self,raw=False):
        if raw:
            times = self.Data[self.Subject+'MetaInfo']['Time']['Raw'].tolist()
        else:
            times = self.Data[self.Subject+'MetaInfo']['Time']['Processed'].tolist()
        return times

    def extract_info(self):
        return self.Data_dict, len(self.Data)

    def process_raw(self):
        data = self.rawdata['Bone']
        col_names = []
        for bone in self.BoneNames:
            if bone == 'Hip': bone = self.Subject
            col_names.append(self.Subject+':'+bone)
        return(data.loc[:, col_names])

    def op2smpl(self):
        smpl_op_pair = {}
        smpl_segments = ['L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
                         'L_Foot', 'R_Foot',
                         'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
                         'L_Wrist',
                         'R_Wrist', 'L_Hand', 'R_Hand']
        smpl_op_pair['L_Hip'] = ['Hip-LThigh']
        smpl_op_pair['R_Hip'] = ['Hip-RThigh']
        smpl_op_pair['Spine1'] = ['Hip-Ab']
        smpl_op_pair['L_Knee'] = ['LThigh-LShin']
        smpl_op_pair['R_Knee'] = ['RThigh-RShin']
        smpl_op_pair['Spine2'] = []
        smpl_op_pair['L_Ankle'] = ['LShin-LFoot']
        smpl_op_pair['R_Ankle'] = ['RShin-RFoot']
        smpl_op_pair['Spine3'] = ['Ab-Chest']
        smpl_op_pair['L_Foot'] = ['LFoot-LToe']
        smpl_op_pair['R_Foot'] = ['RFoot-RToe']
        smpl_op_pair['Neck'] = ['Chest-Neck']
        smpl_op_pair['L_Collar'] = ['Chest-LShoulder']
        smpl_op_pair['R_Collar'] = ['Chest-RShoulder']
        smpl_op_pair['Head'] = ['Neck-Head']
        smpl_op_pair['L_Shoulder'] = ['LShoulder-LUArm']
        smpl_op_pair['R_Shoulder'] = ['RShoulder-RUArm']
        smpl_op_pair['L_Elbow'] = ['LUArm-LFArm']
        smpl_op_pair['R_Elbow'] = ['RUArm-RFArm']
        smpl_op_pair['L_Wrist'] = ['LFArm-LHand']
        smpl_op_pair['R_Wrist'] = ['RFArm-RHand']
        smpl_op_pair['L_Hand'] = []
        smpl_op_pair['R_Hand'] = []
        smpl_poses = []
        global_rot = []
        for i in range(len(self.rawdata)):
            frame_smpl_poses = []
            for smpl_seg in smpl_segments:
                matched_op = smpl_op_pair[smpl_seg]
                if len(matched_op) == 1:
                    father = matched_op[0].split('-')[0]
                    child = matched_op[0].split('-')[1]
                    fx = self.Data_dict[father]['Rot_x'][i]
                    fy = self.Data_dict[father]['Rot_y'][i]
                    fz = self.Data_dict[father]['Rot_z'][i]
                    fw = self.Data_dict[father]['Rot_w'][i]

                    cx = self.Data_dict[child]['Rot_x'][i]
                    cy = self.Data_dict[child]['Rot_y'][i]
                    cz = self.Data_dict[child]['Rot_z'][i]
                    cw = self.Data_dict[child]['Rot_w'][i]

                    father_rot = [fw,fx,fy,fz]
                    child_rot = [cw,cx,cy,cz]
                    angles = related_rotation(father_rot,child_rot)
                else:
                    angles = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                frame_smpl_poses.append(angles)
            gx = self.Data_dict['Hip']['Rot_x'][i]
            gy = self.Data_dict['Hip']['Rot_y'][i]
            gz = self.Data_dict['Hip']['Rot_z'][i]
            gw = self.Data_dict['Hip']['Rot_w'][i]
            global_rot.append(quat2rotm([gw,gx,gy,gz]))
            smpl_poses.append(frame_smpl_poses)
        return smpl_poses,global_rot

    def ToDict(self):
        data_dict = {}
        for bone in self.BoneNames:
            bone_each = {}
            if bone == 'Hip': header = self.Subject + ':' + self.Subject
            else: header = self.Subject + ':' + bone
            bone_each['Pos_x'] = self.Data[header]['Position']['X'].tolist()
            bone_each['Pos_y'] = self.Data[header]['Position']['Y'].tolist()
            bone_each['Pos_z'] = self.Data[header]['Position']['Z'].tolist()
            bone_each['Rot_x'] = self.Data[header]['Rotation']['X'].tolist()
            bone_each['Rot_y'] = self.Data[header]['Rotation']['Y'].tolist()
            bone_each['Rot_z'] = self.Data[header]['Rotation']['Z'].tolist()
            bone_each['Rot_w'] = self.Data[header]['Rotation']['W'].tolist()
            data_dict[bone] = bone_each
        return data_dict

    def MocapTCProcessed(self, MocapTCCopy):
        for i in range(len(MocapTCCopy)):
            tc = MocapTCCopy[i]
            subframe = int(tc.split('.')[1])
            hours, minutes, seconds, frames = map(int, tc.split('.')[0].split(':'))
            time_elased = hours * 60 * 60 + minutes * 60 + seconds + frames * (1 / 30) + subframe * (1 / 8) * (
                        1 / 30)
            MocapTCCopy[i] = time_elased
        return MocapTCCopy

    def KeyHeaders(self):
        headers = self.rawdata['Bone'].columns.tolist()
        bone_names = []
        subject_names = []
        for i in range(len(headers)):
            bone_name = headers[i][0].split(':')[1]
            subject_name = headers[i][0].split(':')[0]
            if bone_name == subject_name:
                if subject_name not in subject_names:
                    subject_names.append(subject_name)
                if 'Hip' not in bone_names:
                    bone_names.append('Hip')
                    continue
                else:
                    continue

            if bone_name in bone_names:
                continue
            else:
                bone_names.append(bone_name)
        return bone_names, subject_names

class LoadVideo:
    def __init__(self, filename,Start_Tc):
        self.video = cv2.VideoCapture(filename)
        No_Frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.VideoTc = self.Process_VideoTc(Start_Tc,No_Frame)

    def Process_VideoTc(self,StartTc, No_Frames):
        Tc_Video = Timecode(30, StartTc).float
        Converted_Tc = []
        for i in range(No_Frames):
            Converted_Tc.append(Tc_Video)
            Tc_Video += (1 / 59.94)
        return Converted_Tc

    def get_time_single(self, index=0):
        return self.VideoTc[index]

    def get_times(self):
        return self.VideoTc

    def set_index(self,index):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, index)  # optional

    def extract_frame(self,index):
        self.set_index(index)
        success, self.image = self.video.read()
        return self.image

class LoadMvnx:
    def __init__(self, filename):
        self.data = mvnx.load(filename)
        self.theta_h = self.data.jointAngleErgo[:,17]
        self.frameRate = 60
        self.segments = list(self.data.segments.values())
        self.raw_tc = list(self.data.timecode)
        self.processed_tc = self.Process_XsensTc()
        self.pc_tc = list(self.data.ms)
        self.overall_output = {}
        self.run()

    def get_foot_contacts(self):
        return self.data.footContacts

    def get_time_single(self,raw=False,index=0):
        if raw:
            return self.raw_tc[index]
        else:
            return self.processed_tc[index]

    def get_times(self,raw=False):
        if raw:
            return self.raw_tc
        else:
            return self.processed_tc

    def Process_XsensTc(self):
        Tc_Xsens = Timecode(60, self.raw_tc[0]).float
        Converted_Tc = []
        for i in range(len(self.raw_tc)):
            Converted_Tc.append(Tc_Xsens)
            Tc_Xsens += (1 / 59.94)
        return Converted_Tc

    def run(self):
        for i in range(len(self.raw_tc)):
            output = {}
            output['smpte_tc']=self.raw_tc[i]
            output['pc_tc']=self.pc_tc[i]
            output['orientation'] = self.get_seg_info(i,'orientation')
            output['position'] = self.get_seg_info(i,'position')
            output['velocity'] = self.get_seg_info(i,'velocity')
            output['acceleration'] = self.get_seg_info(i,'acceleration')
            output['angular acceleration'] = self.get_seg_info(i,'angular acceleration')
            output['angular velocity'] = self.get_seg_info(i,'angular velocity')
            self.overall_output['frame_'+str(i)] = output

    def filter(self,data):
        Fc = 6  # Desired effective cutoff frequency. (Winter, 2009)
        p = 2  # Number of filter passes
        Wc = Fc / np.power((2 ** (1 / p)) - 1,
                           1 / 4)  # Actual cutoff frequency to be applied after accounting for number of passes (Winter, 2009)
        Wn = Wc / (self.frameRate / 2)  # Cutoff frequency over half the original frequency
        b, a = butter(2, Wn, 'low')  # Butterworth filter coefficients
        return filtfilt(b, a, data)  # Applying Butterworth filter on acceleration

    def get_seg_info(self,frame,info):
        if info == 'orientation': data = self.data.orientation
        elif info == 'position': data = self.data.position
        elif info == 'velocity': data = self.data.velocity
        elif info == 'acceleration': data = self.data.acceleration
        elif info == 'angular velocity': data = self.data.angularVelocity
        elif info == 'angular acceleration': data = self.data.angularAcceleration
        else: return None
        info_dict = {}
        #data = self.filter(data)
        if info == 'orientation': info_arr = np.reshape(data, (len(data), 23, 4))
        else: info_arr = np.reshape(data, (len(data), 23, 3))
        index = 0
        for segment in self.segments:
            info_dict[segment] = list(info_arr[frame,index,:])
            index+=1
        return info_dict

    def extract_info(self):
        return self.overall_output, len(self.overall_output)

    def extract_segment_names(self):
        return self.segments

    def xsens2smpl(self):
        smpl_xsens_pair = {}
        smpl_segments = ['L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3',
                         'L_Foot', 'R_Foot',
                         'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
                         'L_Wrist',
                         'R_Wrist', 'L_Hand', 'R_Hand']
        smpl_xsens_pair['L_Hip'] = ['Pelvis-LeftUpperLeg']
        smpl_xsens_pair['R_Hip'] = ['Pelvis-RightUpperLeg']
        smpl_xsens_pair['Spine1'] = ['Pelvis-L5', 'L5-L3']
        smpl_xsens_pair['L_Knee'] = ['LeftUpperLeg-LeftLowerLeg']
        smpl_xsens_pair['R_Knee'] = ['RightUpperLeg-RightLowerLeg']
        smpl_xsens_pair['Spine2'] = ['L3-T12']
        smpl_xsens_pair['L_Ankle'] = ['LeftLowerLeg-LeftFoot']
        smpl_xsens_pair['R_Ankle'] = ['RightLowerLeg-RightFoot']
        smpl_xsens_pair['Spine3'] = ['T12-T8']
        smpl_xsens_pair['L_Foot'] = ['LeftFoot-LeftToe']
        smpl_xsens_pair['R_Foot'] = ['RightFoot-RightToe']
        smpl_xsens_pair['Neck'] = ['T8-Neck']
        smpl_xsens_pair['L_Collar'] = ['T8-LeftShoulder']
        smpl_xsens_pair['R_Collar'] = ['T8-RightShoulder']
        smpl_xsens_pair['Head'] = ['Neck-Head']
        smpl_xsens_pair['L_Shoulder'] = ['LeftShoulder-LeftUpperArm']
        smpl_xsens_pair['R_Shoulder'] = ['RightShoulder-RightUpperArm']
        smpl_xsens_pair['L_Elbow'] = ['LeftUpperArm-LeftForeArm']
        smpl_xsens_pair['R_Elbow'] = ['RightUpperArm-RightForeArm']
        smpl_xsens_pair['L_Wrist'] = ['LeftForeArm-LeftHand']
        smpl_xsens_pair['R_Wrist'] = ['RightForeArm-RightHand']
        smpl_xsens_pair['L_Hand'] = []
        smpl_xsens_pair['R_Hand'] = []
        smpl_poses = []
        global_rot = []
        data =  list(self.overall_output.values())
        for i in range(len(self.overall_output)):
            frame_smpl_poses = []
            for smpl_seg in smpl_segments:
                matched_xsens = smpl_xsens_pair[smpl_seg]
                if len(matched_xsens) == 1:
                    father = matched_xsens[0].split('-')[0]
                    child = matched_xsens[0].split('-')[1]
                    father_ori = data[i]['orientation'][father]
                    child_ori = data[i]['orientation'][child]
                    angles = related_rotation(father_ori, child_ori)
                elif matched_xsens == []:
                    angles = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                else:
                    holder = []
                    for item in matched_xsens:
                        father = item.split('-')[0]
                        child = item.split('-')[1]
                        father_ori = data[i]['orientation'][father]
                        child_ori = data[i]['orientation'][child]
                        holder.append(related_rotation(father_ori, child_ori))
                    angles = holder[0] @ holder[1]
                frame_smpl_poses.append(angles)
            global_rot.append(quat2rotm(data[i]['orientation']['Pelvis']))
            smpl_poses.append(frame_smpl_poses)
        return smpl_poses, global_rot

if __name__ == "__main__":

    #loader = LoadMvnx('C:/Users\gyz95\OneDrive\Desktop\Docs\Guoyang-heavylift-4.mvnx')
    loader = LoadOptitrack('C:/Users\gyz95\OneDrive\Desktop\Docs\InverseDynamics\Load_0614.csv')
    #output = loader.extract_info()
    #print(loader.extract_segment_names())
    #print(list(output.values())[1]['angular velocity']['Head'])
    #print(output['frame_0']['orientation'])