import numpy as np
from scipy.optimize import linprog

def L4L5_LinearOptimization_Bean_Schultz(L5S1_Moment, L5S1_Force, R_Pelvic_Global, Theta_H):
    # Find Moment components
    L4L5_Moment_local = np.zeros((3, np.shape(L5S1_Moment)[1]))
    L5S1_Force_local = np.zeros((3, np.shape(L5S1_Moment)[1]))

    for i in range(np.shape(L5S1_Moment)[1]):
        L4L5_Moment_local[:, i] = np.dot(R_Pelvic_Global[:, :, i].T, L5S1_Moment[:, i])
        L5S1_Force_local[:, i] = np.dot(R_Pelvic_Global[:, :, i].T, L5S1_Force[:, i])

    L4L5_Coronal_Moment = L4L5_Moment_local[0, :]
    L4L5_Torque = L4L5_Moment_local[1, :]
    L4L5_Saggital_Moment = L4L5_Moment_local[2, :]

    L4L5_Anterior_force = L5S1_Force_local[0, :]
    L4L5_Normal_force = L5S1_Force_local[1, :]
    L4L5_Lateral_force = L5S1_Force_local[2, :]

    # Muscle properties
    gender = 1  # male

    if gender == 1:  # if male
        # Physiological cross-sectional area in cm^2
        A_ES = 31  # Erector Spinae
        A_LD = 3  # Latissimus Dorsi
        A_RA = 13  # Rectus Abdominus
        A_IO = 5  # Internal Oblique
        A_EO = 5  # External Oblique

        # Coronal Moment Arm in m
        r_Coronal_ES = 5.4 * 10 ** -2  # Erector Spinae
        r_Coronal_LD = 6.3 * 10 ** -2  # Latissimus Dorsi
        r_Coronal_RA = 3.6 * 10 ** -2  # Rectus Abdominis
        r_Coronal_IO = 13.5 * 10 ** -2  # Internal Oblique
        r_Coronal_EO = 13.5 * 10 ** -2  # External Oblique

        # Saggital Moment Arm in m
        r_Saggital_ES = 4.4 * 10 ** -2  # Erector Spinae
        r_Saggital_LD = 5.6 * 10 ** -2  # Latissimus Dorsi
        r_Saggital_RA = -10.8 * 10 ** -2  # Rectus Abdominis
        r_Saggital_IO = -3.8 * 10 ** -2  # Internal Oblique
        r_Saggital_EO = -3.8 * 10 ** -2  # External Oblique

        # Line of action angle to the disk normal (in degrees)
        Theta_ES = 0  # Erector Spinae in the Saggital plane
        Theta_LD = 45  # Latissimus Dorsi in the Coronal Plane
        Theta_RA = 0  # Rectus Abdominis in the Saggital Plane
        Theta_IO = 45  # Internal Oblique in the Saggital Plane
        Theta_EO = -45  # External Oblique in the Saggital Plane
        Theta_AP = 0  # Abdominal Pressure in the Saggital Plane

        # Diaphragm area affected by the abdominal pressure in m^2
        A_ab = 465 * 10 ** -4
    else:  # female
        # Physiological cross-sectional area in in cm^2
        A_ES = 31  # Erector Spinae
        A_LD = 3  # Latissimus Dorsi
        A_RA = 13  # Rectus Abdominus
        A_IO = 5  # Internal Oblique
        A_EO = 5  # External Oblique

        # Coronal Moment Arm in m
        r_Coronal_ES = 5.4 * 10 ** -2  # Erector Spinae
        r_Coronal_LD = 6.3 * 10 ** -2  # Latissimus Dorsi
        r_Coronal_RA = 3.6 * 10 ** -2  # Rectus Abdominis
        r_Coronal_IO = 13.5 * 10 ** -2  # Internal Oblique
        r_Coronal_EO = 13.5 * 10 ** -2  # External Oblique

        # Saggital Moment Arm in m
        r_Saggital_ES = 4.4 * 10 ** -2  # Erector Spinae
        r_Saggital_LD = 5.6 * 10 ** -2  # Latissimus Dorsi
        r_Saggital_RA = -10.8 * 10 ** -2  # Rectus Abdominis
        r_Saggital_IO = -3.8 * 10 ** -2  # Internal Oblique
        r_Saggital_EO = -3.8 * 10 ** -2  # External Oblique

        # Line of action angle to the disk normal (in degrees)
        Theta_ES = 0  # Erector Spinae in the Saggital plane
        Theta_LD = 45  # Latissimus Dorsi in the Coronal Plane
        Theta_RA = 0  # Rectus Abdominis in the Saggital Plane
        Theta_IO = 45  # Internal Oblique in the Saggital Plane
        Theta_EO = -45  # External Oblique in the Saggital Plane
        Theta_AP = 0  # Abdominal Pressure in the Saggital Plane

        # Diaphragm area affected by the abdominal pressure in m^2
        A_ab = 465 * 10 ** -4

    # Abdominal Pressure Force and Moment Arm
    # from Morris 1961 and Chaffin's book
    P_A_mmHg = np.zeros(np.shape(L5S1_Moment)[1])
    P_A = np.zeros(np.shape(L5S1_Moment)[1])
    F_AP = np.zeros(np.shape(L5S1_Moment)[1])
    r_Saggital_AP = np.zeros(np.shape(L5S1_Moment)[1])

    for i in range(np.shape(L5S1_Moment)[1]):
        P_A_mmHg[i] = (43 - 0.36 * Theta_H[i]) * np.power(np.abs(L4L5_Saggital_Moment[i]), 1.8) / 10000  # Abdominal Pressure in mmHg
        P_A[i] = P_A_mmHg[i] * 133.322368  # Abdominal Pressure in Pa
        F_AP[i] = P_A[i] * A_ab  # Abdominal Pressure force in N
        r_Saggital_AP[i] = (7 + 8 * np.sin(np.radians(Theta_H[i]))) * 10 ** -2  # Abdominal Pressure's force moment arm in the saggital plane

    # First Linear Programming, the objective function is the maximum muscle intensity
    L4L5Compression = np.zeros(np.shape(L5S1_Moment)[1])
    L4L5AnteriorShear = np.zeros(np.shape(L5S1_Moment)[1])
    L4L5LateralShear = np.zeros(np.shape(L5S1_Moment)[1])
    I = np.zeros(np.shape(L5S1_Moment)[1])
    muscles_forces = np.zeros((13, np.shape(L5S1_Moment)[1]))

    options = {"disp": False}

    for i in range(np.shape(L5S1_Moment)[1]):
        Aeq = np.array([[0, 0, -np.sin(np.radians(Theta_LD)), np.sin(np.radians(Theta_LD)), 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [np.sin(np.radians(Theta_ES)), np.sin(np.radians(Theta_ES)), 0, 0, np.sin(np.radians(Theta_RA)), np.sin(np.radians(Theta_RA)),
                         np.sin(np.radians(Theta_IO)), np.sin(np.radians(Theta_IO)), np.sin(np.radians(Theta_EO)), np.sin(np.radians(Theta_EO)), 0, 0, 1, 0],
                        [-np.cos(np.radians(Theta_ES)), -np.cos(np.radians(Theta_ES)), -np.cos(np.radians(Theta_LD)), -np.cos(np.radians(Theta_LD)),
                         -np.cos(np.radians(Theta_RA)), -np.cos(np.radians(Theta_RA)), -np.cos(np.radians(Theta_IO)), -np.cos(np.radians(Theta_IO)),
                         -np.cos(np.radians(Theta_EO)), -np.cos(np.radians(Theta_EO)), 1, 0, 0, 0],
                        [r_Saggital_ES * np.cos(np.radians(Theta_ES)), r_Saggital_ES * np.cos(np.radians(Theta_ES)),
                         r_Saggital_LD * np.cos(np.radians(Theta_LD)), r_Saggital_LD * np.cos(np.radians(Theta_LD)),
                         r_Saggital_RA * np.cos(np.radians(Theta_RA)), r_Saggital_RA * np.cos(np.radians(Theta_RA)),
                         r_Saggital_IO * np.cos(np.radians(Theta_IO)), r_Saggital_IO * np.cos(np.radians(Theta_IO)),
                         r_Saggital_EO * np.cos(np.radians(Theta_EO)), r_Saggital_EO * np.cos(np.radians(Theta_EO)), 0, 0, 0, 0],
                        [r_Coronal_ES * np.cos(np.radians(Theta_ES)), -r_Coronal_ES * np.cos(np.radians(Theta_ES)),
                         r_Coronal_LD * np.cos(np.radians(Theta_LD)), -r_Coronal_LD * np.cos(np.radians(Theta_LD)),
                         r_Coronal_RA * np.cos(np.radians(Theta_RA)), -r_Coronal_RA * np.cos(np.radians(Theta_RA)),
                         r_Coronal_IO * np.cos(np.radians(Theta_IO)), -r_Coronal_IO * np.cos(np.radians(Theta_IO)),
                         r_Coronal_EO * np.cos(np.radians(Theta_EO)), -r_Coronal_EO * np.cos(np.radians(Theta_EO)), 0, 0, 0, 0],
                        [r_Coronal_ES * np.sin(np.radians(Theta_ES)), -r_Coronal_ES * np.sin(np.radians(Theta_ES)),
                         r_Saggital_LD * np.sin(np.radians(Theta_LD)), -r_Saggital_LD * np.sin(np.radians(Theta_LD)),
                         r_Coronal_RA * np.sin(np.radians(Theta_RA)), -r_Coronal_RA * np.sin(np.radians(Theta_RA)),
                         r_Coronal_IO * np.sin(np.radians(Theta_IO)), -r_Coronal_IO * np.sin(np.radians(Theta_IO)),
                         r_Coronal_EO * np.sin(np.radians(Theta_EO)), -r_Coronal_EO * np.sin(np.radians(Theta_EO)), 0, 0, 0, 0]])

        beq = np.array([L4L5_Lateral_force[i], L4L5_Anterior_force[i] - F_AP[i] * np.sin(np.radians(Theta_AP)),
                        L4L5_Normal_force[i] - F_AP[i] * np.cos(np.radians(Theta_AP)), L4L5_Saggital_Moment[i] - F_AP[i] * np.cos(np.radians(Theta_AP)) * r_Saggital_AP[i],
                        L4L5_Coronal_Moment[i], L4L5_Torque[i]])

        Aineq = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -A_ES],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -A_ES],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -A_LD],
                          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -A_LD],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -A_RA],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -A_RA],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -A_IO],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -A_IO],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -A_EO],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -A_EO]])

        bineq = np.zeros((10, 1))

        lb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -50000, -50000, -50000, 0]

        ub = [50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000]

        f = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        res = linprog(f, A_ub=Aineq, b_ub=bineq, A_eq=Aeq, b_eq=beq, bounds=list(zip(lb, ub)), options=options)

        if res.success:
            I[i] = res.x[-1]
        else:
            continue

    # Second linear programming, the objective function is the summation of muscle forces
    for i in range(np.shape(L5S1_Moment)[1]):
        Aeq = np.array([[0, 0, -np.sin(np.radians(Theta_LD)), np.sin(np.radians(Theta_LD)), 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [np.sin(np.radians(Theta_ES)), np.sin(np.radians(Theta_ES)), 0, 0, np.sin(np.radians(Theta_RA)), np.sin(np.radians(Theta_RA)),
                         np.sin(np.radians(Theta_IO)), np.sin(np.radians(Theta_IO)), np.sin(np.radians(Theta_EO)), np.sin(np.radians(Theta_EO)), 0, 0, 1],
                        [-np.cos(np.radians(Theta_ES)), -np.cos(np.radians(Theta_ES)), -np.cos(np.radians(Theta_LD)), -np.cos(np.radians(Theta_LD)),
                         -np.cos(np.radians(Theta_RA)), -np.cos(np.radians(Theta_RA)), -np.cos(np.radians(Theta_IO)), -np.cos(np.radians(Theta_IO)),
                         -np.cos(np.radians(Theta_EO)), -np.cos(np.radians(Theta_EO)), 1, 0, 0],
                        [r_Saggital_ES * np.cos(np.radians(Theta_ES)), r_Saggital_ES * np.cos(np.radians(Theta_ES)),
                         r_Saggital_LD * np.cos(np.radians(Theta_LD)), r_Saggital_LD * np.cos(np.radians(Theta_LD)),
                         r_Saggital_RA * np.cos(np.radians(Theta_RA)), r_Saggital_RA * np.cos(np.radians(Theta_RA)),
                         r_Saggital_IO * np.cos(np.radians(Theta_IO)), r_Saggital_IO * np.cos(np.radians(Theta_IO)),
                         r_Saggital_EO * np.cos(np.radians(Theta_EO)), r_Saggital_EO * np.cos(np.radians(Theta_EO)), 0, 0, 0],
                        [r_Coronal_ES * np.cos(np.radians(Theta_ES)), -r_Coronal_ES * np.cos(np.radians(Theta_ES)),
                         r_Coronal_LD * np.cos(np.radians(Theta_LD)), -r_Coronal_LD * np.cos(np.radians(Theta_LD)),
                         r_Coronal_RA * np.cos(np.radians(Theta_RA)), -r_Coronal_RA * np.cos(np.radians(Theta_RA)),
                         r_Coronal_IO * np.cos(np.radians(Theta_IO)), -r_Coronal_IO * np.cos(np.radians(Theta_IO)),
                         r_Coronal_EO * np.cos(np.radians(Theta_EO)), -r_Coronal_EO * np.cos(np.radians(Theta_EO)), 0, 0, 0],
                        [r_Coronal_ES * np.sin(np.radians(Theta_ES)), -r_Coronal_ES * np.sin(np.radians(Theta_ES)),
                         r_Saggital_LD * np.sin(np.radians(Theta_LD)), -r_Saggital_LD * np.sin(np.radians(Theta_LD)),
                         r_Coronal_RA * np.sin(np.radians(Theta_RA)), -r_Coronal_RA * np.sin(np.radians(Theta_RA)),
                         r_Coronal_IO * np.sin(np.radians(Theta_IO)), -r_Coronal_IO * np.sin(np.radians(Theta_IO)),
                         r_Coronal_EO * np.sin(np.radians(Theta_EO)), -r_Coronal_EO * np.sin(np.radians(Theta_EO)), 0, 0, 0]])

        beq = np.array([L4L5_Lateral_force[i], L4L5_Anterior_force[i] - F_AP[i] * np.sin(np.radians(Theta_AP)),
                        L4L5_Normal_force[i] - F_AP[i] * np.cos(np.radians(Theta_AP)), L4L5_Saggital_Moment[i] - F_AP[i] * np.cos(np.radians(Theta_AP)) * r_Saggital_AP[i],
                        L4L5_Coronal_Moment[i], L4L5_Torque[i]])

        lb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -50000, -50000, -50000]

        ub = I[i] * np.array([A_ES, A_ES, A_LD, A_LD, A_RA, A_RA, A_IO, A_IO, A_EO, A_EO, 50000, 50000, 50000])

        f = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]


        res = linprog(f, A_eq=Aeq, b_eq=beq, bounds=list(zip(lb, ub)), options=options)

        if res.success:
            L4L5Compression[i] = res.x[10]
            L4L5LateralShear[i] = res.x[11]
            L4L5AnteriorShear[i] = res.x[12]
            muscles_forces[:, i] = res.x[:13]
            print(res.x[10])

    return L4L5Compression, L4L5LateralShear, L4L5AnteriorShear, muscles_forces

if __name__ == "__main__":
    L5S1_Moment = np.random.rand(3,157)
    L5S1_Force = np.random.rand(3,157)
    R_Pelvic_Global = np.random.rand(3,3,157)
    Theta_H = np.random.rand(157)
    print(np.shape(L5S1_Moment)[1])
    a,b,c,d = L4L5_LinearOptimization_Bean_Schultz(L5S1_Moment, L5S1_Force, R_Pelvic_Global, Theta_H)
