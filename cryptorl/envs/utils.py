import numpy as np

def norm(H):
  return 1/np.linalg.norm(H) * H
H = np.array([[1],[0]])
H = norm(H)
V = np.array([[0],[1]])
V = norm(V)
D = np.array([[1],[1]])
D = norm(D)
A = np.array([[1],[-1]])
A = norm(A)
R = np.array([[1],[1j]])
R = norm(R)
L = np.array([[1],[-1j]])
L = norm(L)


def PhaseShift(phi):
  return np.array([[np.exp( 1j*phi), 0],
                   [0,               1]])


def Rotator(thetta):
  matr = np.array([[np.cos(thetta), -np.sin(thetta)],
                   [np.sin(thetta),  np.cos(thetta)]])
  return matr


R_clock = Rotator(-np.pi/4)
R_anticlock = Rotator(np.pi/4)

def GetAliceMatrix(phi_1, phi_a):
    phase_a = np.array([[np.exp(1j * phi_1), 0],
                        [0, 1]])

    pm_a = np.array([[np.exp(1j * phi_a), 0],
                     [0, 1]])

    return pm_a @ phase_a @ R_anticlock


def GetBobMatrix(U0, U1, U2, U3, phi_b, phi_3, phi_pc, thetta_pc_b, thetta_pc_a):
    Act_0 = R_clock @ np.array([[np.exp(-1j * (U0)), 0],
                                [0, 1]]) @ R_anticlock

    Act_1 = np.array([[np.exp(1j * (U1)), 0],
                      [0, 1]])  # Здесь знак + в экспоненте, чтобы положительные напряжения крутили вправо

    Act_2 = R_clock @ np.array([[np.exp(-1j * (U2)), 0],
                                [0, 1]]) @ R_anticlock

    Act_3 = np.array([[np.exp(1j * (U3)), 0],
                      [0, 1]])  # Здесь знак + в экспоненте, чтобы положительные напряжения крутили вправо

    PC = Act_3 @ Act_2 @ Act_1 @ Act_0

    U_PC_PM = Rotator(thetta_pc_a) @ PhaseShift(phi_pc) @ Rotator(thetta_pc_b)

    PM_b = np.array([[np.exp(1j * phi_b), 0],
                     [0, 1]])

    U_PM_PBS = PhaseShift(phi_3)

    return R_clock @ U_PM_PBS @ PM_b @ U_PC_PM @ PC


class Environment():
    def __init__(self):
        self.phi_1 = 0
        self.phi_2 = 0
        self.phi_3 = 0
        self.phi_pc = 0

        self.thetta_2_b = 0
        self.thetta_2_a = 0
        self.thetta_pc_b = 0
        self.thetta_pc_a = 0

    def InitRandom(self):
        self.phi_1 = np.random.uniform(0, 2*np.pi)
        self.phi_2 = np.random.uniform(0, 2*np.pi)
        self.phi_3 = np.random.uniform(0, 2*np.pi)
        self.phi_pc = np.random.uniform(0, 2*np.pi)

        self.thetta_2_b = np.random.uniform(0, 2*np.pi)
        self.thetta_2_a = np.random.uniform(0, 2*np.pi)
        self.thetta_pc_b = np.random.uniform(0, 2*np.pi)
        self.thetta_pc_a = np.random.uniform(0, 2*np.pi)

    def GetTotalMatrix(self, phi_a, phi_b, U0, U1, U2, U3):
        Alice = GetAliceMatrix(self.phi_1, phi_a)
        Bob = GetBobMatrix(U0, U1, U2, U3, phi_b, self.phi_3, self.phi_pc, self.thetta_pc_b, self.thetta_pc_a)

        U_fiber = Rotator(self.thetta_2_a) @ PhaseShift(self.phi_2) @ Rotator(self.thetta_2_b)
        return Bob @ U_fiber @ Alice

    def Step(self, phi_a, phi_b, U0, U1, U2, U3):
        out = self.GetTotalMatrix(phi_a, phi_b, U0, U1, U2, U3)
        return abs(out @ H)**2