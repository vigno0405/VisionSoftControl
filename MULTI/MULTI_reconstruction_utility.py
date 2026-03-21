import numpy as np
import casadi as ca # type: ignore
from scipy import integrate # type: ignore
from scipy.linalg import null_space # type: ignore
from numpy.linalg import lstsq
from scipy.spatial.transform import Rotation as R # type: ignore
import torch # type: ignore

class SPCModel:
    """
    Polynomial Curvature Model for Soft Robotic Shape Estimation
    
    This class implements a constant, linear, and quadratic polynomial 
    curvature model for soft robotics.
    """
    
    def __init__(self, modeAnn='noMode'):

        # GPU in each case
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modeAnn = modeAnn

        # Select the model
        if self.modeAnn=='quadratic':
            self.model = torch.jit.load("models/curvature/quadratic_for_MULTI_model.pt", map_location=self.device)
            self.model.eval()
        elif self.modeAnn=='affine':
            self.model = torch.jit.load("models/curvature/affine_for_MULTI_model.pt", map_location=self.device)
            self.model.eval()
        elif self.modeAnn=='constant':
            self.model = torch.jit.load("models/curvature/constant_for_MULTI_model.pt", map_location=self.device)
            self.model.eval()
        else:
            pass
    
    def curvature(self, s, c0, c1, c2):
        """
        Quadratic curvature model
        
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
    
    def ComputeParameters(self, S, R11, R12, R1, R21, R22, R2, R31, R32, R3, mode='quadratic'):
        """
        Reconstructs curvature parameters from rotation matrices of the multi-segment robot.
        Parameters:
        S: vector of three trunk lengths of the whole sections (np.array)
        R11: backbone_11 rotation matrix (absolute measure)
        R12: backbone 12 rotation matrix (absolute measure)
        R1: tip1 rotation matrix
        R21: backbone_21 rotation matrix (absolute measure)
        R22: backbone_22 rotation matrix (absolute measure)
        R2: tip2 rotation matrix
        R31: backbone_31 rotation matrix (absolute measure)
        R32: backbone_32 rotation matrix (absolute measure)
        R3: tip3 rotation matrix
        mode:
        'quadratic' (default)
        'affine'
        'constant'

        Returns:
        c01, c11, c21, c02, c12, c22, c03, c13, c23: reconstructed curvature parameters
        phi1, phi2, phi3: mean phi values (as there is no torsion)
        """

        # Compute relative rotation matrices (in the reference systems)
        R1_1 = R11 
        R2_1 = R12
        R3_1 = R1
        R1_2 = R1.T @ R21
        R2_2 = R1.T @ R22
        R3_2 = R1.T @ R2
        R1_3 = R2.T @ R31
        R2_3 = R2.T @ R32
        R3_3 = R2.T @ R3

        # Apply curvature reconstruction
        c01, c11, c21, phi1 = self.curvature_reconstruction(
            S=S[0],
            Rbase_1=R1_1,
            Rbase_2=R2_1,
            Rbase_3=R3_1,
            mode=mode
        )
        c02, c12, c22, phi2 = self.curvature_reconstruction(
            S=S[1],
            Rbase_1=R1_2,
            Rbase_2=R2_2,
            Rbase_3=R3_2,
            mode=mode
        )
        c03, c13, c23, phi3 = self.curvature_reconstruction(
            S=S[2],
            Rbase_1=R1_3,
            Rbase_2=R2_3,
            Rbase_3=R3_3,
            mode=mode
        )
        return c01, c11, c21, phi1, c02, c12, c22, phi2, c03, c13, c23, phi3
    
    def ComputeANNParameters(self, mocap1, mocap2, mocap3):
        """
        Reconstructs curvature parameters from rotation matrices of the multi-segment robot.
        Parameters:
        mocap1: tip1 xyzZYX
        mocap2: tip2 xyzZYX
        mocap3: tip3 xyzZYX

        Returns
        c01, c11, c21, c02, c12, c22, c03, c13, c23: reconstructed curvature parameters
        phi1, phi2, phi3: mean phi values (as there is no torsion)
        """
    
        # Extract rotation matrices and translations
        R1 = R.from_euler('zyx', mocap1[3:]).as_matrix()
        R2 = R.from_euler('zyx', mocap2[3:]).as_matrix()
        R3 = R.from_euler('zyx', mocap3[3:]).as_matrix()
        t1 = mocap1[:3]
        t2 = mocap2[:3]
        t3 = mocap3[:3]
    
        # Compute relative transformations
        R01 = R1
        ZYX_01 = R.from_matrix(R01).as_euler('zyx')
        R12 = R1.T @ R2
        ZYX_12 = R.from_matrix(R12).as_euler('zyx')
        R23 = R2.T @ R3
        ZYX_23 = R.from_matrix(R23).as_euler('zyx')
    
        t01 = t1
        t12 = R1.T @ (t2 - t1)
        t23 = R2.T @ (t3 - t2)
    
        # Prepare inputs for neural network
        input1 = np.concatenate([t01, ZYX_01])
        input2 = np.concatenate([t12, ZYX_12])
        input3 = np.concatenate([t23, ZYX_23])
    
        # Convert to tensors
        input1_tensor = torch.tensor(input1, dtype=torch.float32).unsqueeze(0).to(self.device)
        input2_tensor = torch.tensor(input2, dtype=torch.float32).unsqueeze(0).to(self.device)
        input3_tensor = torch.tensor(input3, dtype=torch.float32).unsqueeze(0).to(self.device)
    
        with torch.no_grad():
            if self.modeAnn == 'quadratic':
                # Model outputs: c0, c1, c2, phi (4 outputs)
                output1 = self.model(input1_tensor).cpu().numpy().squeeze()
                output2 = self.model(input2_tensor).cpu().numpy().squeeze()
                output3 = self.model(input3_tensor).cpu().numpy().squeeze()
            
                s1, c01, c11, c21, phi1 = output1
                s2, c02, c12, c22, phi2 = output2
                s3, c03, c13, c23, phi3 = output3
            
            elif self.modeAnn == 'affine':
                # Model outputs: c0, c1, phi (3 outputs, c2 is zero)
                output1 = self.model(input1_tensor).cpu().numpy().squeeze()
                output2 = self.model(input2_tensor).cpu().numpy().squeeze()
                output3 = self.model(input3_tensor).cpu().numpy().squeeze()
            
                s1, c01, c11, phi1 = output1
                s2, c02, c12, phi2 = output2
                s3, c03, c13, phi3 = output3
                c21 = c22 = c23 = 0.0
            
            elif self.modeAnn == 'constant':
                # Model outputs: c0, phi (2 outputs, c1 and c2 are zero)
                output1 = self.model(input1_tensor).cpu().numpy().squeeze()
                output2 = self.model(input2_tensor).cpu().numpy().squeeze()
                output3 = self.model(input3_tensor).cpu().numpy().squeeze()
            
                s1, c01, phi1 = output1
                s2, c02, phi2 = output2
                s3, c03, phi3 = output3
                c11 = c12 = c13 = 0.0
                c21 = c22 = c23 = 0.0
    
        return s1, c01, c11, c21, phi1, s2, c02, c12, c22, phi2, s3, c03, c13, c23, phi3