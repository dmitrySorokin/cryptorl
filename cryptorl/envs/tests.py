import numpy as np
from .utils import *


envTest = Environment()

# One detector click check
out = envTest.Step(0, 0, 0, 0, 0, 0)
assert(out[1] < 1e-16)

out = envTest.Step(np.pi/2, np.pi/2, 0, 0, 0, 0)
assert(out[0] < 1e-16)

out = envTest.Step(np.pi, 0, 0, 0, 0, 0)
assert(out[0] < 1e-16)

out = envTest.Step(3*np.pi/2, np.pi/2, 0, 0, 0, 0)
assert(out[1] < 1e-16)


# Both detector click check
out = envTest.Step(np.pi/2, 0, 0, 0, 0, 0)
assert(abs(out[1] - out[0])< 1e-15)

out = envTest.Step(np.pi/2, np.pi, 0, 0, 0, 0)
assert(abs(out[1] - out[0])< 1e-15)

out = envTest.Step(3*np.pi/2, 0, 0, 0, 0, 0)
assert(abs(out[1] - out[0]) < 1e-15)

out = envTest.Step(3*np.pi/2, np.pi , 0, 0, 0, 0)
assert(abs(out[1] - out[0]) < 1e-15)


##################

# Identity check
out = envTest.GetTotalMatrix(0, 0, 0, 0, 0, 0)
assert(np.linalg.norm( out - np.eye(2)) < 1e-16)

# Actuator rotation check
out =  R_anticlock @ envTest.GetTotalMatrix(0, 0, np.pi/4, 0, 0, 0) @ R_clock @  V
outTrue = L*np.cos(np.pi/8) - R*np.sin(np.pi/8)
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )

out =  R_anticlock @ envTest.GetTotalMatrix(0, 0, 0, np.pi/4, 0, 0) @ R_clock @  V
outTrue = V
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )

out =  R_anticlock @ envTest.GetTotalMatrix(0, 0, 0, 0, 0, np.pi/4) @ R_clock @  H
outTrue = H
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )

out = R_anticlock @ envTest.GetTotalMatrix(0, 0, 0, 0, np.pi/4, 0) @ R_clock @  D
outTrue = D
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )

out = R_anticlock @ envTest.GetTotalMatrix(0, 0, 0, -np.pi/4, np.pi/4, np.pi/2) @ R_clock @  D
outTrue = L*np.cos(np.pi/8) - np.exp(1j*np.pi/4)*R*np.sin(np.pi/8)
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )


out = R_anticlock @ envTest.GetTotalMatrix(0, 0, 3*np.pi/4, 0, 3*np.pi/4, 0) @ R_clock @ H
outTrue = L
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )


# Basis choise check
out = envTest.GetTotalMatrix(np.pi, 0, 0, 0, 0, 0) @ H
outTrue = V
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )


out = envTest.GetTotalMatrix(np.pi, 0, 0, 0, 0, 0) @ D
outTrue = D
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )

out = envTest.GetTotalMatrix(np.pi, 0, 0, 0, 0, 0) @ D
outTrue = D
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )




out = envTest.GetTotalMatrix(np.pi/2, 0, 0, 0, 0, 0) @ norm(L + 1/2*R + 0.5*D)
outTrue = R_clock @ np.array([[np.exp(-1j*(-np.pi/2)), 0],[0,1]]) @ R_anticlock @ norm(L + 1/2*R + 0.5*D)
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )

out = envTest.GetTotalMatrix(0, np.pi/4, 0, 0, 0, 0) @ H
outTrue = norm(L+H)
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )

out = envTest.GetTotalMatrix(np.pi/2, np.pi/2, 0, 0, 0, 0) @ H
outTrue = V
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )



envTest.phi_1 = np.random.normal(0, np.pi, 1)[0]
envTest.phi_2 = np.random.normal(0, np.pi, 1)[0]
envTest.phi_3 = np.pi - envTest.phi_1 - envTest.phi_2

out = envTest.GetTotalMatrix(0, 0, 0, 0, 0, 0) @ H
outTrue = V
assert(abs(abs(out.T.conj() @ outTrue) - np.linalg.norm(out)* np.linalg.norm(outTrue)) < 1e-16 )