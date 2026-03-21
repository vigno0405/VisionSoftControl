# Libraries
import numpy as np
from sympy import symbols, sqrt, atan2, cos, sin, Matrix, lambdify, simplify, asin # type: ignore
from MULTI_jacobians import q2tendon, q2coordinates, q2xyz

def tendon2coordinates(DELTAL0, DxDyDl0, coords0, d_DELTAL, step_size=1):
    """
    Compute the new configuration from tendon length variations with adaptive discretization.

    Output:
    - DELTAL   <- tendon lengths of the new configuration (9):
                [DELTAL1, DELTAL2, DELTAL3]
    - DxDyDl   <- curvature of the new configuration (9):
                [Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3]
    - coords   <- (x,y,z,yaw,pitch,roll) of the new configuration

    Input:
    - DELTAL0  <- tendon lengths of the old configuration (9)
    - DxDyDl0  <- curvature of the old configuration (9)
    - coords0  <- (x,y,z,yaw,pitch,roll) of the old configuration
    - d_DELTAL <- tendon length variations (DELTAL - DELTAL0) (9)
    - step_size <- step granularity for discretization (default: 1)
    """

    # Prevent numerical instability when the initial curvature is close to zero
    if np.linalg.norm(DxDyDl0) < 1e-4:
        DxDyDl0 = np.full(9, 1e-4)

    # Compute the number of steps based on d_DELTAL magnitude
    max_change = np.linalg.norm(d_DELTAL)
    N = max(int(np.ceil(max_change / step_size)), 5)    # Ensure at least 5 steps

    # Compute step-wise variations
    d_DELTAL_step = d_DELTAL / N

    # Initialize variables
    DELTAL = np.copy(DELTAL0)
    DxDyDl = np.copy(DxDyDl0)
    coords = np.copy(coords0)

    # Initialize Jacobians
    Q2TENDON = q2tendon()
    Q2COORDINATES = q2coordinates()

    # Iterative update with Jacobian recomputation
    for _ in range(N):
        J_tendon = Q2TENDON(*DxDyDl)        # 9x9
        J_coords = Q2COORDINATES(*DxDyDl)   # 6x9

        # Compute the incremental variation in curvature (fully defined)
        d_DxDyDl = np.linalg.pinv(J_tendon) @ d_DELTAL_step

        # Update curvature
        DxDyDl = DxDyDl + d_DxDyDl

        # Compute the incremental variation in coordinates (fully defined)
        d_coords = J_coords @ d_DxDyDl

        # Update coordinates and tendon lengths
        coords = coords + d_coords
        DELTAL = DELTAL + d_DELTAL_step

    return DELTAL, DxDyDl, coords

def coordinates2tendon(DELTAL0, DxDyDl0, coords0, d_coords, step_size=1, control=False):
    """
    Compute the new tendon lengths from coordinate variations with adaptive discretization.

    Output:
    - DELTAL   <- tendon lengths of the new configuration (9):
                [DELTAL1, DELTAL2, DELTAL3]
    - DxDyDl   <- curvature of the new configuration (9):
                [Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3]
    - coords   <- (x,y,z,yaw,pitch,roll) of the new configuration

    Input:
    - DELTAL0  <- tendon lengths of the old configuration (9)
    - DxDyDl0  <- curvature of the old configuration (9)
    - coords0  <- (x,y,z,yaw,pitch,roll) of the old configuration
    - d_coords <- coordinate variations (in x,y,z,yaw,pitch,roll) (6)
    - step_size <- step granularity for discretization
    - control   <- if True, check motion boundaries and return the old configuration if violated

    NOTES: not possible to be usen for control, then the data-driven split should be used,
    as the system is over-actuated -> decomposition.
    """

    # Prevent numerical instability when the initial curvature is close to zero
    if np.linalg.norm(DxDyDl0) < 1e-4:
        DxDyDl0 = np.full(9, 1e-4)

    # Compute the number of steps based on d_xyz magnitude
    max_change = np.linalg.norm(d_coords)
    N = max(int(np.ceil(max_change / step_size)), 5)    # Ensure at least 5 steps

    # Compute step-wise variations
    d_coords_step = d_coords / N

    # Initialize variables
    DELTAL = np.copy(DELTAL0)
    DxDyDl = np.copy(DxDyDl0)
    coords = np.copy(coords0)

    # Initialize Jacobians
    Q2COORDINATES = q2coordinates()
    Q2TENDON = q2tendon()

    # Iterative update with Jacobian recomputation
    for _ in range(N):
        J_tendon = Q2TENDON(*DxDyDl)        # 9x9
        J_coords = Q2COORDINATES(*DxDyDl)   # 6x9   
        
        # Compute incremental variation in curvature (minimizes spectral norm, due to underactuation)
        d_DxDyDl = np.linalg.pinv(J_coords) @ d_coords_step

        # Update curvature
        DxDyDl = DxDyDl + d_DxDyDl
        coords = coords + d_coords_step

        # Compute tendon Jacobian and incremental tendon variation (fully defined, according to the curvature)
        d_DELTAL = J_tendon @ d_DxDyDl

        # Update tendon lengths
        DELTAL = DELTAL + d_DELTAL

        # Check motion boundaries for each section
        def D_Dl(Dx, Dy, Dl):
            return np.sqrt(Dx**2 + Dy**2), Dl

        D1, Dl1 = D_Dl(DxDyDl[0], DxDyDl[1], DxDyDl[2])
        D2, Dl2 = D_Dl(DxDyDl[3], DxDyDl[4], DxDyDl[5])
        D3, Dl3 = D_Dl(DxDyDl[6], DxDyDl[7], DxDyDl[8])

        # Measures (defined for the single section, extended here)
        Dl_min = -30
        D_max = 35

        if (Dl1 < Dl_min or Dl2 < Dl_min or Dl3 < Dl_min or
            D1 > D_max or D2 > D_max or D3 > D_max) and control:
            return DELTAL0, DxDyDl0, coords0    # boundaries
        
    return DELTAL, DxDyDl, coords

def xyz2tendon(DELTAL0, DxDyDl0, xyz0, d_xyz, step_size=1, control=False):
    """
    Compute the new tendon lengths from tip coordinates variations with adaptive discretization.

    Output:
    - DELTAL    <- tendon lengths of the new configuration (9):
                [DELTAL1, DELTAL2, DELTAL3]
    - DxDyDl    <- curvature of the new configuration (9):
                [Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3]
    - xyz       <- (x,y,z) of all sections for the new configuration

    Input:
    - DELTAL0   <- tendon lengths of the old configuration (9)
    - DxDyDl0   <- curvature of the old configuration (9)
    - xyz0      <- (x,y,z) of all sections for the old configuration
    - d_xyz     <- tip coordinates variations (in x,y,z) (9)
    - step_size <- step granularity for discretization
    - control   <- if True, check motion boundaries and return the old configuration if violated
    """

    # Prevent numerical instability when the initial curvature is close to zero
    if np.linalg.norm(DxDyDl0) < 1e-4:
        DxDyDl0 = np.full(9, 1e-4)

    # Compute the number of steps based on d_xyz magnitude
    d_xyz_reshaped = d_xyz.reshape(3, 3)
    distances = np.array([
        np.linalg.norm(d_xyz_reshaped[0]),
        np.linalg.norm(d_xyz_reshaped[1]),
        np.linalg.norm(d_xyz_reshaped[2])
    ])

    # Use maximum distance to determine waypoints
    max_distance = np.max(distances)
    N = max(int(np.ceil(max_distance / step_size)), 5)

    # Compute step-wise variations
    d_xyz_step = d_xyz / N

    # Initialize variables
    DELTAL = np.copy(DELTAL0)
    DxDyDl = np.copy(DxDyDl0)
    xyz = np.copy(xyz0)

    # Initialize Jacobians
    Q2XYZ = q2xyz()
    Q2TENDON = q2tendon()

    # Iterative update with Jacobian recomputation
    for _ in range(N):
        J_tendon = Q2TENDON(*DxDyDl)        # 9x9
        J_xyz = Q2XYZ(*DxDyDl)              # 9x9
        reg = 1e-3 * np.eye(9)
        J_xyz_pinv = np.linalg.inv(J_xyz.T @ J_xyz + reg) @ J_xyz.T

        # Compute incremental variation in curvature
        d_DxDyDl = J_xyz_pinv @ d_xyz_step

        # Update curvature
        DxDyDl = DxDyDl + d_DxDyDl
        xyz = xyz + d_xyz_step

        # Compute tendon Jacobian
        d_DELTAL = J_tendon @ d_DxDyDl

        # Update tendon lengths
        DELTAL = DELTAL + d_DELTAL

        # Check motion boundaries for each section
        def D_Dl(Dx, Dy, Dl):
            return np.sqrt(Dx**2 + Dy**2), Dl
        
        D1, Dl1 = D_Dl(DxDyDl[0], DxDyDl[1], DxDyDl[2])
        D2, Dl2 = D_Dl(DxDyDl[3], DxDyDl[4], DxDyDl[5])
        D3, Dl3 = D_Dl(DxDyDl[6], DxDyDl[7], DxDyDl[8])

        # Measures (defined for the single section, extended here)
        Dl_min = -30
        D_max = 35

        if (Dl1 < Dl_min or Dl2 < Dl_min or Dl3 < Dl_min or
            D1 > D_max or D2 > D_max or D3 > D_max) and control:
            return DELTAL0, DxDyDl0, xyz0
        
    return DELTAL, DxDyDl, xyz

def tendon2xyz(DELTAL0, DxDyDl0, xyz0, d_DELTAL, step_size=1, control=False):
    """
    Compute the new tendon lengths from tip coordinates variations with adaptive discretization.

    Output:
    - DELTAL    <- tendon lengths of the new configuration (9):
                [DELTAL1, DELTAL2, DELTAL3]
    - DxDyDl    <- curvature of the new configuration (9):
                [Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3]
    - xyz       <- (x,y,z) of all sections for the new configuration

    Input:
    - DELTAL0   <- tendon lengths of the old configuration (9)
    - DxDyDl0   <- curvature of the old configuration (9)
    - xyz0      <- (x,y,z) of all sections for the old configuration
    - d_DELTAL  <- DELTAL variations (9)
    - step_size <- step granularity for discretization
    - control   <- if True, check motion boundaries and return the old configuration if violated
    """

    # Prevent numerical instability when the initial curvature is close to zero
    if np.linalg.norm(DxDyDl0) < 1e-4:
        DxDyDl0 = np.full(9, 1e-4)

    # Compute the number of steps based on d_DELTAL magnitude
    d_DELTAL_reshaped = d_DELTAL.reshape(3, 3)
    distances = np.array([
        np.linalg.norm(d_DELTAL_reshaped[0]),
        np.linalg.norm(d_DELTAL_reshaped[1]),
        np.linalg.norm(d_DELTAL_reshaped[2])
    ])

    # Use maximum distance to determine waypoints
    max_distance = np.max(distances)
    N = max(int(np.ceil(max_distance / step_size)), 5)

    # Compute step-wise variations
    d_DELTAL_step = d_DELTAL / N

    # Initialize variables
    DELTAL = np.copy(DELTAL0)
    DxDyDl = np.copy(DxDyDl0)
    xyz = np.copy(xyz0)

    # Initialize Jacobians
    Q2XYZ = q2xyz()
    Q2TENDON = q2tendon()

    # Iterative update with Jacobian recomputation
    for _ in range(N):
        J_tendon = Q2TENDON(*DxDyDl)        # 9x9
        J_xyz = Q2XYZ(*DxDyDl)              # 9x9
        reg = 1e-3 * np.eye(9)
        J_tendon_pinv = np.linalg.inv(J_tendon.T @ J_tendon + reg) @ J_tendon.T

        # Compute incremental variation in curvature
        d_DxDyDl = J_tendon_pinv @ d_DELTAL_step

        # Update curvature
        DxDyDl = DxDyDl + d_DxDyDl
        DELTAL = DELTAL + d_DELTAL_step

        # Compute xyz Jacobian
        d_xyz = J_xyz @ d_DxDyDl

        # Update xyz
        xyz = xyz + d_xyz

        # Check motion boundaries for each section
        def D_Dl(Dx, Dy, Dl):
            return np.sqrt(Dx**2 + Dy**2), Dl
        
        D1, Dl1 = D_Dl(DxDyDl[0], DxDyDl[1], DxDyDl[2])
        D2, Dl2 = D_Dl(DxDyDl[3], DxDyDl[4], DxDyDl[5])
        D3, Dl3 = D_Dl(DxDyDl[6], DxDyDl[7], DxDyDl[8])

        # Measures (defined for the single section, extended here)
        Dl_min = -30
        D_max = 35

        if (Dl1 < Dl_min or Dl2 < Dl_min or Dl3 < Dl_min or
            D1 > D_max or D2 > D_max or D3 > D_max) and control:
            return DELTAL0, DxDyDl0, xyz0
        
    return DELTAL, DxDyDl, xyz

def xyz2coordinates(xyz0, DxDyDl0, coords0, d_xyz, step_size=1):
    """
    Compute the new configuration from cartesian coordinates variations with adaptive discretization.

    Output:
    - coords   <- (x,y,z,yaw,pitch,roll) of the new configuration
    - DxDyDl   <- curvature of the new configuration (9):
                [Dx1, Dy1, Dl1, Dx2, Dy2, Dl2, Dx3, Dy3, Dl3]
    - xyz   <- (x,y,z) for all sections of the new configuration

    Input:
    - xyz0     <- (x,y,z) of the old configuration (9)
    - DxDyDl0  <- curvature of the old configuration (9)
    - coords0  <- (x,y,z,yaw,pitch,roll) of the old configuration
    - d_DELTAL <- tendon length variations (DELTAL - DELTAL0) (9)
    - step_size <- step granularity for discretization (default: 1)
    """

    # Prevent numerical instability when the initial curvature is close to zero
    if np.linalg.norm(DxDyDl0) < 1e-4:
        DxDyDl0 = np.full(9, 1e-4)

    # Compute the number of steps based on d_xyz magnitude
    d_xyz_reshaped = d_xyz.reshape(3, 3)
    distances = np.array([
        np.linalg.norm(d_xyz_reshaped[0]),
        np.linalg.norm(d_xyz_reshaped[1]),
        np.linalg.norm(d_xyz_reshaped[2])
    ])

    # Use maximum distance to determine waypoints
    max_distance = np.max(distances)
    N = max(int(np.ceil(max_distance / step_size)), 5)

    # Compute step-wise variations
    d_xyz_step = d_xyz / N

    # Initialize variables
    coords = np.copy(coords0)
    DxDyDl = np.copy(DxDyDl0)
    xyz = np.copy(xyz0)

    # Initialize Jacobians
    Q2XYZ = q2xyz()
    Q2COORDINATES = q2coordinates()

    # Iterative update with Jacobian recomputation
    for _ in range(N):
        J_xyz = Q2XYZ(*DxDyDl)
        J_coords = Q2COORDINATES(*DxDyDl)   # 6x9

        # Compute the incremental variation in curvature (fully defined)
        d_DxDyDl = np.linalg.pinv(J_xyz) @ d_xyz_step

        # Update curvature
        DxDyDl = DxDyDl + d_DxDyDl

        # Compute the incremental variation in coordinates (fully defined)
        d_coords = J_coords @ d_DxDyDl

        # Update coordinates and tendon lengths
        coords = coords + d_coords
        xyz = xyz + d_xyz_step

    return coords, DxDyDl, xyz

if __name__ == "__main__":
   # Suppress scientific notation
   np.set_printoptions(suppress=True, precision=6)

   # Initial conditions (rest position)
   DELTAL0 = np.full(9, 1e-4)
   DxDyDl0 = np.full(9, 1e-4)
   coords0 = np.array([0, 0, 145 * 3, 1e-4, 1e-4, 1e-4])
   xyz0 = np.array([0, 0, 145, 0, 0, 145 * 2, 0, 0, 145 * 3])

   # xyz_initial = np.array([2, -3, 145, 20, 10, 120 + 145, 30, -40, 145 + 120 * 2])
   # DELTAL_initial, _, _ = xyz2tendon(DELTAL0, DxDyDl0, xyz0, xyz_initial - xyz0)
   # print(f"Initial DELTAL: {DELTAL_initial}")

   d_DELTAL = np.array([-20, -2, -2, -10, -1, -10, 0, 0, 0])
   _, _, coords = tendon2coordinates(DELTAL0, DxDyDl0, coords0, d_DELTAL)
   print(f"Coords: {coords}")
