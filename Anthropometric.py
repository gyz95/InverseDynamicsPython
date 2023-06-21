import numpy as np

class Estimator:
    def __init__(self, BodyWeight, BodyHeight, Gender='male'):
        self.bodymass = float(BodyWeight)
        self.bodyheight = float(BodyHeight)
        self.gender = Gender
        self.segment_mass = []
        self.segment_length = []
        self.centerofmass = []
        self.seg_ori = []
        self.inertiatensor = []
        self.estimate_segmass()
        self.estimate_seglen()
        self.estimate_com()
        self.estimate_ori()
        self.estimate_inertiatensor()

    def get(self):
        return self.segment_mass, self.segment_length, self.centerofmass, self.seg_ori, self.inertiatensor

    def estimate_segmass(self):
        if self.gender == 'male':
            self.segment_mass.append(self.bodymass * 0.142)
            self.segment_mass.append(self.bodymass * 0.333)
            self.segment_mass.append(self.bodymass * 0.067)
            self.segment_mass.append(self.bodymass * 0.024)
            self.segment_mass.append(self.bodymass * 0.017)
            self.segment_mass.append(self.bodymass * 0.006)
            self.segment_mass.append(self.bodymass * 0.024)
            self.segment_mass.append(self.bodymass * 0.017)
            self.segment_mass.append(self.bodymass * 0.006)
            self.segment_mass.append(self.bodymass * 0.123)
            self.segment_mass.append(self.bodymass * 0.048)
            self.segment_mass.append(self.bodymass * 0.012)
            self.segment_mass.append(self.bodymass * 0.123)
            self.segment_mass.append(self.bodymass * 0.048)
            self.segment_mass.append(self.bodymass * 0.012)
        else:
            self.segment_mass.append(self.bodymass * 0.146)
            self.segment_mass.append(self.bodymass * 0.304)
            self.segment_mass.append(self.bodymass * 0.067)
            self.segment_mass.append(self.bodymass * 0.022)
            self.segment_mass.append(self.bodymass * 0.013)
            self.segment_mass.append(self.bodymass * 0.005)
            self.segment_mass.append(self.bodymass * 0.022)
            self.segment_mass.append(self.bodymass * 0.013)
            self.segment_mass.append(self.bodymass * 0.005)
            self.segment_mass.append(self.bodymass * 0.146)
            self.segment_mass.append(self.bodymass * 0.045)
            self.segment_mass.append(self.bodymass * 0.010)
            self.segment_mass.append(self.bodymass * 0.146)
            self.segment_mass.append(self.bodymass * 0.045)
            self.segment_mass.append(self.bodymass * 0.010)

    def estimate_seglen(self):
        if self.gender == 'male':
            self.segment_length.append(0.094 * self.bodyheight / 1.77)
            self.segment_length.append(0.477 * self.bodyheight / 1.77)
            self.segment_length.append(0.244 * self.bodyheight / 1.77)
            self.segment_length.append(0.271 * self.bodyheight / 1.77)
            self.segment_length.append(0.283 * self.bodyheight / 1.77)
            self.segment_length.append(0.080 * self.bodyheight / 1.77)
            self.segment_length.append(0.271 * self.bodyheight / 1.77)
            self.segment_length.append(0.283 * self.bodyheight / 1.77)
            self.segment_length.append(0.080 * self.bodyheight / 1.77)
            self.segment_length.append(0.432 * self.bodyheight / 1.77)
            self.segment_length.append(0.433 * self.bodyheight / 1.77)
            self.segment_length.append(0.183 * self.bodyheight / 1.77)
            self.segment_length.append(0.432 * self.bodyheight / 1.77)
            self.segment_length.append(0.433 * self.bodyheight / 1.77)
            self.segment_length.append(0.183 * self.bodyheight / 1.77)
        else:
            self.segment_length.append(0.107 * self.bodyheight / 1.61)
            self.segment_length.append(0.429 * self.bodyheight / 1.61)
            self.segment_length.append(0.221 * self.bodyheight / 1.61)
            self.segment_length.append(0.241 * self.bodyheight / 1.61)
            self.segment_length.append(0.247 * self.bodyheight / 1.61)
            self.segment_length.append(0.071 * self.bodyheight / 1.61)
            self.segment_length.append(0.243 * self.bodyheight / 1.61)
            self.segment_length.append(0.247 * self.bodyheight / 1.61)
            self.segment_length.append(0.071 * self.bodyheight / 1.61)
            self.segment_length.append(0.379 * self.bodyheight / 1.61)
            self.segment_length.append(0.388 * self.bodyheight / 1.61)
            self.segment_length.append(0.165 * self.bodyheight / 1.61)
            self.segment_length.append(0.379 * self.bodyheight / 1.61)
            self.segment_length.append(0.388 * self.bodyheight / 1.61)
            self.segment_length.append(0.165 * self.bodyheight / 1.61)

    def estimate_com(self):
        if self.gender == 'male':
            self.centerofmass.append(np.dot(self.segment_length[0] / 100 , [2.8, -28.0, -0.6]))
            self.centerofmass.append(np.dot(self.segment_length[1] / 100 , [-3.6, 58, -0.2]))
            self.centerofmass.append(np.dot(self.segment_length[2] / 100 , [-6.2, 55.5, 0.1]))
            self.centerofmass.append(np.dot(self.segment_length[3] / 100 , [1.7, -45.2, -2.6]))
            self.centerofmass.append(np.dot(self.segment_length[4] / 100 , [1, -41.7, 1.4]))
            self.centerofmass.append(np.dot(self.segment_length[5] / 100 , [8.2, -83.9, 7.4]))
            self.centerofmass.append(np.dot(self.segment_length[6] / 100 , [1.7, -45.2, -2.6]))
            self.centerofmass.append(np.dot(self.segment_length[7] / 100 , [1, -41.7, 1.4]))
            self.centerofmass.append(np.dot(self.segment_length[8] / 100 , [8.2, -83.9, 7.4]))
            self.centerofmass.append(np.dot(self.segment_length[9] / 100 , [-4.1, -42.9, 3.3]))
            self.centerofmass.append(np.dot(self.segment_length[10] / 100 , [-4.8, -41.0, 0.7]))
            self.centerofmass.append(np.dot(self.segment_length[11] / 100 , [38.2, -15.1, 2.6]))
            self.centerofmass.append(np.dot(self.segment_length[12] / 100 , [-4.1, -42.9, 3.3]))
            self.centerofmass.append(np.dot(self.segment_length[13] / 100 , [-4.8, -41.0, 0.7]))
            self.centerofmass.append(np.dot(self.segment_length[14] / 100 , [38.2, -15.1, 2.6]))
        else:
            self.centerofmass.append(np.dot(self.segment_length[0] / 100 , [-0.9, -23.2, 0.2]))
            self.centerofmass.append(np.dot(self.segment_length[1] / 100 , [-1.6, 56.4, -0.6]))
            self.centerofmass.append(np.dot(self.segment_length[2] / 100 , [-7.0, 59.7, 0]))
            self.centerofmass.append(np.dot(self.segment_length[3] / 100 , [-7.3, -45.4, -2.8]))
            self.centerofmass.append(np.dot(self.segment_length[4] / 100 , [2.1, -41.1, 1.9]))
            self.centerofmass.append(np.dot(self.segment_length[5] / 100 , [7.7, -76.8, 4.8]))
            self.centerofmass.append(np.dot(self.segment_length[6] / 100 , [-7.3, -45.4, -2.8]))
            self.centerofmass.append(np.dot(self.segment_length[7] / 100 , [2.1, -41.1, 1.9]))
            self.centerofmass.append(np.dot(self.segment_length[8] / 100 , [7.7, -76.8, 4.8]))
            self.centerofmass.append(np.dot(self.segment_length[9] / 100 , [-7.7, -37.7, 0.9]))
            self.centerofmass.append(np.dot(self.segment_length[10] / 100 , [-4.9, -40.4, 3.1]))
            self.centerofmass.append(np.dot(self.segment_length[11] / 100 , [27.0, -21.8, 3.9]))
            self.centerofmass.append(np.dot(self.segment_length[12] / 100 , [-7.7, -37.7, 0.9]))
            self.centerofmass.append(np.dot(self.segment_length[13] / 100 , [-4.9, -40.4, 3.1]))
            self.centerofmass.append(np.dot(self.segment_length[14] / 100 , [27.0, -21.8, 3.9]))

    def estimate_ori(self):
        if self.gender == 'male':
            self.seg_ori.append([0,0,0]) #1
            self.seg_ori.append([0,0,0]) #2
            self.seg_ori.append([0,self.segment_length[1],0]) #3
            self.seg_ori.append([0.021,self.segment_length[1]-0.073,0.209]) #4
            self.seg_ori.append([0,-self.segment_length[3],0]) #5
            self.seg_ori.append([0,-self.segment_length[4],0]) #6
            self.seg_ori.append([0.021,self.segment_length[1]-0.073,-0.209]) #7
            self.seg_ori.append([0,-self.segment_length[6],0]) #8
            self.seg_ori.append([0,-self.segment_length[7],0]) #9
            self.seg_ori.append([0.056, -0.075, 0.081])
            self.seg_ori.append([0,-self.segment_length[9],0])
            self.seg_ori.append([0,-self.segment_length[10],0])
            self.seg_ori.append([0.056, -0.075, -0.081])
            self.seg_ori.append([0,-self.segment_length[12],0])
            self.seg_ori.append([0,-self.segment_length[13],0])
        else:
            self.seg_ori.append([0,0,0]) #1
            self.seg_ori.append([0,0,0]) #2
            self.seg_ori.append([0,self.segment_length[1],0]) #3
            self.seg_ori.append([0.021,self.segment_length[1]-0.073,0.209]) #4
            self.seg_ori.append([0,-self.segment_length[3],0]) #5
            self.seg_ori.append([0,-self.segment_length[4],0]) #6
            self.seg_ori.append([0.021,self.segment_length[1]-0.073,-0.209]) #7
            self.seg_ori.append([0,-self.segment_length[6],0]) #8
            self.seg_ori.append([0,-self.segment_length[7],0]) #9
            self.seg_ori.append([0.056, -0.075, 0.081])
            self.seg_ori.append([0,-self.segment_length[9],0])
            self.seg_ori.append([0,-self.segment_length[10],0])
            self.seg_ori.append([0.056, -0.075, -0.081])
            self.seg_ori.append([0,-self.segment_length[12],0])
            self.seg_ori.append([0,-self.segment_length[13],0])

    def estimate_inertiatensor(self):
        if self.gender == 'male':
            I1_arr = np.array([[101, 25j, 12j], [25j, 106, 8j], [12j, 8j, 95]])
            I2_arr = np.array([[27, 18, 2], [18, 25, 4j], [2, 4j, 28]])
            I3_arr = np.array([[31, 9j, 2j], [9j, 25, 3], [2j, 3, 33]])
            I4_arr = np.array([[31, 6, 5], [6, 14, 2], [5, 2, 32]])
            I5_arr = np.array([[28, 3, 2], [3, 11, 8j], [2, 8j, 27]])
            I6_arr = np.array([[61, 22, 15], [22, 38, 20j], [15, 20j, 56]])
            I7_arr = np.array([[31, 6, 5], [6, 14, 2], [5, 2, 32]])
            I8_arr = np.array([[28, 3, 2], [3, 11, 8j], [2, 8j, 27]])
            I9_arr = np.array([[61, 22, 15], [22, 38, 20j], [15, 20j, 56]])
            I10_arr = np.array([[29, 7, 2j], [7, 15, 7j], [2j, 7j, 30]])
            I11_arr = np.array([[28, 4j, 2j], [4j, 10, 5], [2j, 5, 28]])
            I12_arr = np.array([[17, 13, 8j], [13, 37, 0], [8j, 0, 36]])
            I13_arr = np.array([[29, 7, 2j], [7, 15, 7j], [2j, 7j, 30]])
            I14_arr = np.array([[28, 4j, 2j], [4j, 10, 5], [2j, 5, 28]])
            I15_arr = np.array([[17, 13, 8j], [13, 37, 0], [8j, 0, 36]])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[0]/100, I1_arr))*self.segment_mass[0])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[1]/100, I2_arr))*self.segment_mass[1])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[2]/100, I3_arr))*self.segment_mass[2])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[3]/100, I4_arr))*self.segment_mass[3])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[4]/100, I5_arr))*self.segment_mass[4])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[5]/100, I6_arr))*self.segment_mass[5])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[6]/100, I7_arr))*self.segment_mass[6])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[7]/100, I8_arr))*self.segment_mass[7])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[8]/100, I9_arr))*self.segment_mass[8])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[9]/100, I10_arr))*self.segment_mass[9])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[10]/100, I11_arr))*self.segment_mass[10])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[11]/100, I12_arr))*self.segment_mass[11])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[12]/100, I13_arr))*self.segment_mass[12])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[13]/100, I14_arr))*self.segment_mass[13])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[14]/100, I15_arr))*self.segment_mass[14])
        else:
            I1_arr = np.array([[91, 34j, 1j], [34j, 100, 1j], [1j, 1j, 79]])
            I2_arr = np.array([[29, 22, 5], [22, 27, 5j], [5, 5j, 29]])
            I3_arr = np.array([[32, 6j, 1], [6j, 27, 1j], [2j, 3, 33]])
            I4_arr = np.array([[33, 3, 5], [3, 17, 14], [5j, 14, 33]])
            I5_arr = np.array([[26, 10, 4], [10, 14, 13j], [4, 13j, 25]])
            I6_arr = np.array([[63, 29, 23], [29, 43, 28j], [23, 28j, 58]])
            I7_arr = np.array([[33, 3, 5], [3, 17, 14], [5j, 14, 33]])
            I8_arr = np.array([[26, 10, 4], [10, 14, 13j], [4, 13j, 25]])
            I9_arr = np.array([[63, 29, 23], [29, 43, 28j], [23, 28j, 58]])
            I10_arr = np.array([[31, 7, 2j], [7, 19, 7j], [2j, 7j, 32]])
            I11_arr = np.array([[28, 2, 1], [2, 10, 6], [1, 6, 28]])
            I12_arr = np.array([[17, 10j, 6], [10j, 36, 4j], [6, 4j, 35]])
            I13_arr = np.array([[31, 7, 2j], [7, 19, 7j], [2j, 7j, 32]])
            I14_arr = np.array([[28, 2, 1], [2, 10, 6], [1, 6, 28]])
            I15_arr = np.array([[17, 10j, 6], [10j, 36, 4j], [6, 4j, 35]])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[0]/100, I1_arr))*self.segment_mass[0])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[1]/100, I2_arr))*self.segment_mass[1])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[2]/100, I3_arr))*self.segment_mass[2])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[3]/100, I4_arr))*self.segment_mass[3])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[4]/100, I5_arr))*self.segment_mass[4])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[5]/100, I6_arr))*self.segment_mass[5])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[6]/100, I7_arr))*self.segment_mass[6])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[7]/100, I8_arr))*self.segment_mass[7])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[8]/100, I9_arr))*self.segment_mass[8])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[9]/100, I10_arr))*self.segment_mass[9])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[10]/100, I11_arr))*self.segment_mass[10])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[11]/100, I12_arr))*self.segment_mass[11])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[12]/100, I13_arr))*self.segment_mass[12])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[13]/100, I14_arr))*self.segment_mass[13])
            self.inertiatensor.append(np.square(np.dot(self.segment_length[14]/100, I15_arr))*self.segment_mass[14])

if __name__ == "__main__":
    AnthropometricEstimator = Estimator(BodyWeight = 90, BodyHeight = 1.7, Gender='male')
