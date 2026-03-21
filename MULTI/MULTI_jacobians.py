# Libraries
import numpy as np
from sympy import symbols, sqrt, atan2, cos, sin, Matrix, lambdify, simplify, asin, trigsimp # type: ignore

# Units of measure:
# - LENGTHS: [mm]
# - ANGLES: [rad]

# Note: the following code is adopted when considering the entire robot, while for a single section
# (assuming it to be rotated properly) the SINGLE_jacobians.py library can be used.

# q2tendon(): from q to DELTAL
def q2tendon(COUPLING=False):
    """
    Jacobians of tendons ([Dx1 Dy1 Dl1 Dx2 Dy2 Dl2 Dx3 Dy3 Dl3] to [DELTAL1 DELTAL2 DELTAL3]).
    """

    # Components
    Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3 = symbols("Dx1 Dy1 Dl1 Dx2 Dy2 Dl2 Dx3 Dy3 Dl3")

    # Positions of tendons (assuming origin is coherent here): they are wrt the base,
    # with geometries defined according to the real robot. With respect to x:
    # - Tendon 1: 0°
    # - Tendon 2: 60°
    # - Tendon 3: 150°
    delta1 = Matrix([np.radians(0), np.radians(120), np.radians(-120)])
    delta2 = Matrix([np.radians(60), np.radians(120+60), np.radians(-120+60)])
    delta3 = Matrix([np.radians(150), np.radians(120+150), np.radians(-120+150)])

    # Variations of lengths [deltaL1,j; deltaL2,j; deltaL3,j]; they are wrt the global
    # reference frame, defining variations wrt q.

    DELTAL1 = Matrix([Dl1 - Dx1 * cos(delta_i) - Dy1 * sin(delta_i) for delta_i in delta1])
    DELTAL2 = Matrix([Dl2 - Dx2 * cos(delta_i) - Dy2 * sin(delta_i) for delta_i in delta2])
    DELTAL3 = Matrix([Dl3 - Dx3 * cos(delta_i) - Dy3 * sin(delta_i) for delta_i in delta3])

    # Combine
    DELTAL = Matrix.vstack(DELTAL1, DELTAL2, DELTAL3)

    # Define variables for computation
    variables_vector = Matrix([Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3])

    # Jacobian
    J = DELTAL.jacobian(variables_vector)

    return lambdify((Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3), J, modules=['numpy'])

# q2coordinates(): from q to the CNN-FCN output
def q2coordinates():
    """
    Jacobian of the helyx ([Dx1,Dy1,Dl1,Dx2,Dy2,Dl2,Dx3,Dy3,Dl3]) to [x,y,z,yaw,pitch,roll].
    Note that the matrix is non-invertible in a classical way, since there are differences.
    """

    # Components
    Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3 = symbols("Dx1 Dy1 Dl1 Dx2 Dy2 Dl2 Dx3 Dy3 Dl3")
    diameter = 60       # tuned base diameter [mm]
    d = diameter/2      # constant section
    
    # Initial lengths of 145 mm (measured from real robot)
    L0_1 = 145
    L0_2 = 145
    L0_3 = 145

    # Curvatures
    D1 = sqrt(Dx1**2 + Dy1**2 + 1e-12)   # curvature (1)
    D2 = sqrt(Dx2**2 + Dy2**2 + 1e-12)   # curvature (2)
    D3 = sqrt(Dx3**2 + Dy3**2 + 1e-12)   # curvature (3)

    # Robot kinematics (from paper):
    # - Rotation matrices, based on curvature decomposition
    def R_MATRIX(Dx, Dy, D):
        return Matrix([
            [1 + Dx**2/D**2 * (cos(D/d) - 1),   Dx*Dy/D**2 * (cos(D/d) - 1),   Dx/D * sin(D/d)],
            [Dx*Dy/D**2 * (cos(D/d) - 1),       1 + Dy**2/D**2 * (cos(D/d) - 1), Dy/D * sin(D/d)],
            [-Dx/D * sin(D/d),                   -Dy/D * sin(D/d),               cos(D/d)]
    ])

    # R for each section and global
    R1 = R_MATRIX(Dx1, Dy1, D1)
    R2 = R_MATRIX(Dx2, Dy2, D2)
    R3 = R_MATRIX(Dx3, Dy3, D3)
    R = R1 * R2 * R3

    # - Translation vector (center of the section)
    def t_vector(Dx, Dy, D, Dl, L0):
        return Matrix([
            [(d*(L0+Dl)/D**2)*Dx*(1-cos(D/d))],
            [(d*(L0+Dl)/D**2)*Dy*(1-cos(D/d))],
            [(d*(L0+Dl)/D**2)*D*sin(D/d)]
    ])

    # t for each section and global
    t1 = t_vector(Dx1, Dy1, D1, Dl1, L0_1)
    t2 = t_vector(Dx2, Dy2, D2, Dl2, L0_2)
    t3 = t_vector(Dx3, Dy3, D3, Dl3, L0_3)
    t = t1 + R1 * t2 + R1 * R2 * t3

    # Convert rotation matrix to Yaw-Pitch-Roll (ZYX, intrinsic)
    yaw = atan2(R[1, 0], R[0, 0] + 1e-12)    # LaValle confirms
    pitch = asin(- R[2, 0])                 # LaValle confirms
    roll = atan2(R[2, 1], R[2, 2] + 1e-12)   # LaValle confirms

    # Vector with [translations; quaternions]
    T = Matrix.vstack(t, Matrix([yaw, pitch, roll]))

    # Define variables for computation
    variables_vector = Matrix([Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3])

    # Jacobian
    J = T.jacobian(variables_vector)

    return lambdify((Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3), J, modules=['numpy'])

# q2xyz(): from q to the xyz output
def q2xyz():
    """
    Jacobian of the helyx ([Dx1,Dy1,Dl1,Dx2,Dy2,Dl2,Dx3,Dy3,Dl3]) to [x1, y1, z1, x2, y2, z2, x3, y3, z3].
    """

    # Components
    Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3 = symbols("Dx1 Dy1 Dl1 Dx2 Dy2 Dl2 Dx3 Dy3 Dl3")
    diameter = 60       # tuned base diameter [mm]
    d = diameter/2      # constant section
    
    # Initial lengths of 145 mm (measured from real robot)
    L0_1 = 145
    L0_2 = 145
    L0_3 = 145

    # Curvatures
    D1 = sqrt(Dx1**2 + Dy1**2 + 1e-12)   # curvature (1)
    D2 = sqrt(Dx2**2 + Dy2**2 + 1e-12)   # curvature (2)
    D3 = sqrt(Dx3**2 + Dy3**2 + 1e-12)   # curvature (3)

    # Robot kinematics (from paper):
    # - Rotation matrices, based on curvature decomposition
    def R_MATRIX(Dx, Dy, D):
        return Matrix([
            [1 + Dx**2/D**2 * (cos(D/d) - 1),   Dx*Dy/D**2 * (cos(D/d) - 1),   Dx/D * sin(D/d)],
            [Dx*Dy/D**2 * (cos(D/d) - 1),       1 + Dy**2/D**2 * (cos(D/d) - 1), Dy/D * sin(D/d)],
            [-Dx/D * sin(D/d),                   -Dy/D * sin(D/d),               cos(D/d)]
    ])

    # R for first and second section
    R1 = R_MATRIX(Dx1, Dy1, D1)
    R2 = R_MATRIX(Dx2, Dy2, D2)

    def t_vector(Dx, Dy, D, Dl, L0):
        D_reg = sqrt(D**2 + 1e-12)  # Regularized curvature
        return Matrix([
            [(d*(L0+Dl)/D_reg**2)*Dx*(1-cos(D_reg/d))],
            [(d*(L0+Dl)/D_reg**2)*Dy*(1-cos(D_reg/d))],
            [(d*(L0+Dl)/D_reg**2)*D_reg*sin(D_reg/d)]
        ])

    # t for each section and global
    t1 = t_vector(Dx1, Dy1, D1, Dl1, L0_1)
    t2 = t_vector(Dx2, Dy2, D2, Dl2, L0_2)
    t3 = t_vector(Dx3, Dy3, D3, Dl3, L0_3)
    t1_global = t1
    t2_global = t1 + R1 * t2
    t3_global = t1 + R1 * t2 + R1 * R2 * t3

    T = Matrix.vstack(t1_global, t2_global, t3_global)
    variables_vector = Matrix([Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3])

    # Jacobian
    J = T.jacobian(variables_vector)

    return lambdify((Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3), J, modules=['numpy'])
