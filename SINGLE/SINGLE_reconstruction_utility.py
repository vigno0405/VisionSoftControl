import numpy as np
import casadi as ca
from scipy import integrate
from scipy.linalg import null_space
from numpy.linalg import lstsq
from scipy.spatial.transform import Rotation as R

class SPCModel:
    """
    Polynomial Curvature Model for Soft Robotic Shape Estimation
    
    This class implements a constant, linear, and quadratic polynomial 
    curvature model for soft robotics.
    """
    
    def __init__(self):
        pass
    
    def curvature(self, s, c0, c1, c2):
        """
        Affine curvature model
        
        Parameters:
        s: trunk length
        c0, c1, c2: Coefficients for the Affine function
        
        Returns:
        C: Curvature
        """
        return c0 + c1*s + c2*s**2
    
    def orientation(self, s, c0, c1, c2):
        """
        Transform the curvature to the local orientation
        
        Parameters:
        s: trunk length
        c0, c1, c2: Coefficients for the Affine function
        
        Returns:
        theta: Orientation angle
        """
        return c0*s + (1/2)*c1*s**2 + (1/3)*c2*s**3
    
    def theta2R(self, s, c0, c1, c2, phi):
        """
        Get the AC model end effector posture (rotation)
        
        Parameters:
        s: trunk length
        c0, c1, c2: Coefficients for the Affine function
        phi: input parameter for transformation matrix
        
        Returns:
        R: Rotation matrix
        """
        # calculate theta
        theta = self.orientation(s, c0, c1, c2)
        
        # Francesco Stella's paper formula
        # Compute the rotation matrix R
        R = np.array([
            [(np.cos(phi))**2 * (np.cos(theta) - 1) + 1, np.sin(phi) * np.cos(phi) * (np.cos(theta) - 1), np.cos(phi) * np.sin(theta)],
            [np.sin(phi) * np.cos(phi) * (np.cos(theta) - 1), (np.sin(phi))**2 * (np.cos(theta) - 1) + 1, np.sin(phi) * np.sin(theta)],
            [-np.cos(phi) * np.sin(theta), -np.sin(phi) * np.sin(theta), np.cos(theta)]
        ])
        
        return R
    
    def theta2t(self, s, c0, c1, c2, phi):
        """
        Get the AC model end effector posture (translation)
        
        Parameters:
        s: trunk length
        c0, c1, c2: Coefficients for the Affine function
        phi: input parameter for transformation matrix
        
        Returns:
        t: Translation vector
        """
        # Define integrand functions
        def t1_integrand(v):
            return np.sin(c0*v + (1/2)*c1*v**2 + (1/3)*c2*v**3)
        
        def t2_integrand(v):
            return np.sin(c0*v + (1/2)*c1*v**2 + (1/3)*c2*v**3)
        
        def t3_integrand(v):
            return np.cos(c0*v + (1/2)*c1*v**2 + (1/3)*c2*v**3)
        
        # Compute integrals
        t1 = np.cos(phi) * integrate.quad(t1_integrand, 0, s)[0]
        t2 = np.sin(phi) * integrate.quad(t2_integrand, 0, s)[0]
        t3 = integrate.quad(t3_integrand, 0, s)[0]
        
        return np.array([t1, t2, t3])
    
    def theta2X(self, s, c0, c1, c2, phi):
        """
        Get the AC model end effector posture (rotation and translation)
        
        Parameters:
        s: trunk length
        c0, c1, c2: Coefficients for the Affine function
        phi: input parameter for transformation matrix
        
        Returns:
        R: Rotation matrix
        t: Translation vector
        """
        R = self.theta2R(s, c0, c1, c2, phi)
        t = self.theta2t(s, c0, c1, c2, phi)
        
        return R, t
    
    def curvature_reconstruction(self, S, Rbase_1, Rbase_2, Rbase_3, mode='quadratic'):
        """
        Reconstruct curvature parameters from measured rotations
        
        Parameters:
        S: the trunk length of the whole section
        Rbase_1: The orientation of the segment1 tip relative to the base
        Rbase_2: The orientation of the segment2 tip relative to the base
        Rbase_3: The orientation of the segment3 tip relative to the base
        mode:
        'quadratic' (default)
        'affine'
        'constant'
        
        Returns:
        c0, c1, c2: Reconstructed curvature parameters
        phi: Mean phi value
        """
        
        theta1 = np.arccos(Rbase_1[2, 2])
        phi1 = np.arctan2(-Rbase_1[2, 1]/np.sin(theta1), -Rbase_1[2, 0]/np.sin(theta1))
        
        theta2 = np.arccos(Rbase_2[2, 2])
        phi2 = np.arctan2(-Rbase_2[2, 1]/np.sin(theta2), -Rbase_2[2, 0]/np.sin(theta2))
        
        theta3 = np.arccos(Rbase_3[2, 2])
        phi3 = np.arctan2(-Rbase_3[2, 1]/np.sin(theta3), -Rbase_3[2, 0]/np.sin(theta3))
        
        # Casadi optimization
        opti = ca.Opti()

        # Cases (constant, linear, quadratic)
        if mode == 'constant':
            c0 = opti.variable()
            c0_estimate = theta3 / S
            opti.set_initial(c0, c0_estimate)
            c1 = opti.parameter()
            opti.set_value(c1, 0.0)
            c2 = opti.parameter()
            opti.set_value(c2, 0.0)
        elif mode == 'affine':
            c0 = opti.variable()
            c1 = opti.variable()
            c0_estimate = (9*theta1 - theta3)/(2*S)
            c1_estimate = (3*theta3 - 9*theta1)/(S**2)
            opti.set_initial(c0, c0_estimate)
            opti.set_initial(c1, c1_estimate)
            c2 = opti.parameter()
            opti.set_value(c2, 0.0)
        else:
            c0 = opti.variable()
            c1 = opti.variable()
            c2 = opti.variable()
            c0_estimate = (2*theta3 - 9*theta2 + 18*theta1)/(2*S)
            c1_estimate = (-9*theta3 + 36*theta2 - 45*theta1)/(S**2)
            c2_estimate = (27*theta3 - 81*theta2 + 81*theta1)/(2*S**3)
            opti.set_initial(c0, c0_estimate)
            opti.set_initial(c1, c1_estimate)
            opti.set_initial(c2, c2_estimate)
            
        # Rotation matrix (depending on c0, c1, c2)
        def R_hat(s, phi):
            theta = c0*s + (1/2)*c1*s**2 + (1/3)*c2*s**3
            R = ca.vertcat(
                ca.horzcat((ca.cos(phi))**2 * (ca.cos(theta) - 1) + 1, ca.sin(phi) * ca.cos(phi) * (ca.cos(theta) - 1), ca.cos(phi) * ca.sin(theta)),
                ca.horzcat(ca.sin(phi) * ca.cos(phi) * (ca.cos(theta) - 1), (ca.sin(phi))**2 * (ca.cos(theta) - 1) + 1, ca.sin(phi) * ca.sin(theta)),
                ca.horzcat(-ca.cos(phi) * ca.sin(theta), -ca.sin(phi) * ca.sin(theta), ca.cos(theta))
            )
            return R
        
        # Estimated rotation matrices
        R1_hat = R_hat(S/3, phi1)
        R2_hat = R_hat(2*S/3, phi2)
        R3_hat = R_hat(S, phi3)
        
        # Measured rotation matrices
        R1_meas = ca.DM(Rbase_1)
        R2_meas = ca.DM(Rbase_2)
        R3_meas = ca.DM(Rbase_3)
        
        # Cost function
        cost = ca.sum1(ca.sum2((R1_hat - R1_meas)**2)) + \
               ca.sum1(ca.sum2((R2_hat - R2_meas)**2)) + \
               ca.sum1(ca.sum2((R3_hat - R3_meas)**2))
        
        # Cost minimization
        opti.minimize(cost)
        
        # Options
        p_opts = {"expand": True}
        s_opts = {"max_iter": 1000, "tol": 1e-6, "print_level": 1}
        opti.solver("ipopt", p_opts, s_opts)

        # Optimization
        sol = opti.solve()
        c0_opt = sol.value(c0)
        c1_opt = sol.value(c1)
        c2_opt = sol.value(c2)

        # Angle
        phi = np.mean([phi1, phi2, phi3])

        return c0_opt, c1_opt, c2_opt, phi
    
class SPRegModel:
    """
    Polynomial Regression Model for Soft Robotic Shape Estimation
    
    This class implements a polynomial in the form A8*z**8 + A6*z**6 + A4*z**4 + A2*z**2
    for soft robotics, with variable number of parameters.
    """

    def __init__(self, num_coeffs=3):
        self.num_coeffs = num_coeffs

    def curvature(self, points, angles):
        """
        Fit polynomial y(z) with two hard constraints at tip (y, dy),
        and least squares on the remaining intermediate points.
        Degree = 2 * num_coeffs
        """

        base, tip = points[0], points[3]
        theta = np.arctan2(tip[1] - base[1], tip[0] - base[0])

        # Local bending plane
        v = tip - base
        z_axis = np.array([0.0, 0.0, 1.0])
        n_pi = np.cross(v, z_axis)
        if np.linalg.norm(n_pi) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
            y_axis = np.cross(z_axis, x_axis)
        else:
            x_axis = n_pi / np.linalg.norm(n_pi)
            y_axis = np.cross(z_axis, x_axis)
        R_plane = np.vstack([x_axis, y_axis, z_axis]).T

        # Project to bending plane
        rotated = (points - base) @ R_plane
        zs = rotated[:, 2]
        ys = rotated[:, 1]
        z0, z1, z2, z3 = zs
        y0, y1, y2, y3 = ys
        H = z3 - z0

        # Tip derivative
        yaw, pitch, roll = angles
        R_tip = R.from_euler('ZYX', [yaw, pitch, roll])
        x_tip = R_tip.apply([1, 0, 0])
        y_tip = R_tip.apply([0, 1, 0])
        n_tip = np.cross(x_tip, y_tip)
        n_tip /= np.linalg.norm(n_tip)
        n_tip_in_pi = R_plane.T @ n_tip
        dy3 = n_tip_in_pi[1] / n_tip_in_pi[2]

        # Build constraint matrix (2 rows: value + derivative at z=H)
        C = np.zeros((2, self.num_coeffs))
        dC = np.zeros((2, self.num_coeffs))

        for i in range(self.num_coeffs):
            deg = 2 * (i + 1)
            C[0, i] = H**deg             # y(H)
            C[1, i] = deg * H**(deg - 1) # y'(H)

        Yc = np.array([y3, dy3])

        # Solve constraints: reduce A = [Ac | Af]
        # Here: Af = A[2:], Ac = A[:2] if num_coeffs = 4
        # Find nullspace of constraints to parametrize solution
        N = null_space(C)
        particular, *_ = lstsq(C, Yc, rcond=None)

        # Build system for least-squares: y1, y2
        Z = np.array([
            [z1**(2*(i+1)) for i in range(self.num_coeffs)],
            [z2**(2*(i+1)) for i in range(self.num_coeffs)]
        ])
        Y = np.array([y1, y2])

        # Reduced LS problem: Z * (N x) + particular ≈ Y
        Z_reduced = Z @ N
        Y_residual = Y - Z @ particular

        x, *_ = lstsq(Z_reduced, Y_residual, rcond=None)
        A = particular + N @ x

        return theta, H, A
    
    def PCC(self, tip):
        theta = np.arctan2(tip[1], tip[0])
        z_axis = np.array([0.0, 0.0, 1.0])
        n_pi = np.cross(tip, z_axis)
        if np.linalg.norm(n_pi) < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
            y_axis = np.cross(z_axis, x_axis)
        else:
            x_axis = n_pi / np.linalg.norm(n_pi)
            y_axis = np.cross(z_axis, x_axis)
        R_plane = np.vstack([x_axis, y_axis, z_axis]).T
        rotated = (tip @ R_plane).reshape(-1)
        z3 = rotated[2]
        y3 = rotated[1]
        H = z3
        a = (z3**2 + y3**2) / (2*y3)
        radius = a
        center_2d = np.array([a, 0])

        return theta, H, center_2d, radius

if __name__ == "__main__":

    s = 0.578466        # trunk length
    c0 = 0.142355
    c1 = 0.533245
    c2 = 0.144636
    phi = 1.455674

    model = SPCModel()

    # Test single functions
    curvature = model.curvature(s, c0, c1, c2)
    orientation = model.orientation(s, c0, c1, c2)
    R_tip = model.theta2R(s, c0, c1, c2, phi)
    t = model.theta2t(s, c0, c1, c2, phi)
    X = model.theta2X(s, c0, c1, c2, phi)

    # Test curvature_reconstruction function
    R1 = model.theta2R(s/3, c0, c1, c2, phi)
    R2 = model.theta2R(2*s/3, c0, c1, c2, phi)
    R3 = model.theta2R(s, c0, c1, c2, phi)

    # Curvature reconstruction (to be done in the code)
    c0, c1, c2, phi = model.curvature_reconstruction(S=s, Rbase_1=R1, Rbase_2=R2, Rbase_3=R3,
                                                     mode='quadratic')
    
    # Sample 4 points along the trunk
    s_vals = np.linspace(0, s, 4)
    points = []
    for si in s_vals:
        _, ti = model.theta2X(si, c0, c1, c2, phi)
        points.append(ti)
    points = np.stack(points, axis=0)
    r = R.from_matrix(R_tip)
    angles = r.as_euler('zyx')

    # Polynomial reconstruction
    spreg = SPRegModel(num_coeffs=3)
    theta, H, A = spreg.curvature(points, angles)

    # PCC reconstruction
    theta_PCC, H_PCC, center_2d_PCC, radius_PCC = spreg.PCC(points[-1])

    # Print results
    print("Curvature parameters:")
    print(c0, c1, c2, phi)
    print("Polynomial parameters:")
    print(H, A, theta)
    print("PCC parameters:")
    print(H_PCC, center_2d_PCC, radius_PCC, theta_PCC)