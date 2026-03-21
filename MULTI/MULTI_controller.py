# Libraries
import numpy as np # type: ignore
import sys
import time
import rospy
import os
from datetime import datetime
from dynamixel_sdk import *
from geometry_msgs.msg import PoseStamped
import cv2 # type: ignore
from scipy.spatial.transform import Rotation as R # type: ignore
from scipy.spatial.transform import Slerp # type: ignore
import threading
from ctypes import c_uint32
import matplotlib.pyplot as plt # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore
import termios
import tty
import torch # type: ignore

# User-defined kinematics
from MULTI_jacobians import *
from MULTI_kinematics import *
from MULTI_cameras_utility import *

# AI libraries
import os
import glob
import random
import warnings
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# Torch
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import torch.nn.functional as F # type: ignore
import plotly.graph_objects as go # type: ignore
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau # type: ignore
import torchvision.transforms as T # type: ignore
import torchvision.transforms.functional as TF # type: ignore

import matplotlib.pyplot as plt # type: ignore
import random
import sys
from torch.amp import GradScaler, autocast # type: ignore

warnings.filterwarnings("ignore")
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Controller class
class MULTIController:
    """
    Controller for the MULTI section robot.
    Exploits:
    - Data-driven method
    - Model-based method
    - Null-space controller
    Includes:
    - Initialization (motors, camera, models)
    - Camera search and definition
    - Calibration
    - Sensing
    - Selection of the optimal actuation
    - Motion planning and execution
    E.g.:
    ...
    controller = MULTIController(DELTAL_mean, DATADRIVEN=True, discr_wp=1)
    trajectory, waypoints = controller.CONTROL(pose_target)
    controler.CleanEnv()
    """

    def __init__(self, DELTAL_mean, DATADRIVEN=True, DEVICENAME='/dev/ttyUSB0',
                 MOTOR_IDs=np.array([11, 12, 13, 14, 15, 16, 17, 18, 19]),
                 discr_wp=1, CheckNullSpace=False, convergence=10):
        """
        Initializes the Controller parameters and sets up the robot.
        """

        # Initialize parameters
        self.DEVICENAME = DEVICENAME
        self.MOTOR_IDs = MOTOR_IDs
        self.MOTOR_Num = self.MOTOR_IDs.shape[0]
        self.BAUDRATE = 57600
        self.D_pulley = 6
        self.unit_scale = 4096 / (np.pi * self.D_pulley)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.locks = [threading.Lock(), threading.Lock(), threading.Lock()]
        self.latest_frames = [None, None, None]
        self.TIMEOUT = 3
        self.TOLERANCE = 30
        self.DELTAL_mean = DELTAL_mean

        # Check Null Space
        self.CheckNullSpace = CheckNullSpace

        # Find camera IDs
        cams = probe_and_assign_cameras()
        self.camera_ids = [cams.get("camera_1"), cams.get("camera_2"), cams.get("camera_3")]

        # Controller specifications
        self.data_driven = DATADRIVEN
        self.Q2TENDON = q2tendon()
        self.Q2COORDS = q2coordinates()
        self.discr_wp = discr_wp
        self.convergence = convergence

        # Initialize DELTAL integral
        self.DELTAL_integral = np.zeros(9, dtype=np.float32)

        # Initialize and calibrate the motors
        self._init_dynamixel()
        self.calibrate_motors()

        # Initialize and check the cameras
        self.camera_system = ThreadedCameraSystem(self.camera_ids)
        self.camera_system.initialize_cameras()
        self.camera_system.start_capture_threads()

        # Load the pre-trained models
        self._load_models()
        self.calibrate_motors()

        # Define starting positions of the motors
        self.starting_positions = np.array(
            [self.read_position_float(mid) for mid in self.MOTOR_IDs],
            dtype=np.float32
        )

        # Define reference DELTAL (i.e. initial position, encoder abs. coordinates)
        DELTAL0 = np.full(9, 1e-4)
        DxDyDl0 = np.full(9, 1e-4)
        xyz0 = np.array([1e-4, 1e-4, 145, 1e-4, 1e-4, 145 * 2, 1e-4, 1e-4, 145 * 3])
        _, PositionEstimated = self.CameraSensor()
        DELTAL_initial, _, _ = xyz2tendon(DELTAL0, DxDyDl0, xyz0,
                                        PositionEstimated - xyz0)
        self.DELTAL_reference = self.read_DELTAL_from_motors() - DELTAL_initial
    
    def _init_dynamixel(self):
        """
        Initializes the Dynamixel motors.
        """

        # PortHandler, PacketHandler, port, baudrate
        self.port_handler = PortHandler(self.DEVICENAME)
        self.packet_handler = Protocol2PacketHandler()
        if not self.port_handler.openPort():
            raise RuntimeError("Failed to open port")
        if not self.port_handler.setBaudRate(self.BAUDRATE):
            raise RuntimeError("Failed to set baudrate")
        
        # Set Dynamixel control table addresses and values
        self.ADDR_OPERATING_MODE = 11
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.ADDR_PROFILE_VELOCITY = 112
        self.TORQUE_ENABLE = 1
        self.VELOCITY_LIMIT = 40
        self.EXPOSITION_MODE = 4

        # Set motors to exposition mode and enable torque
        for motor_id in self.MOTOR_IDs:
            self._write_register(motor_id, self.ADDR_OPERATING_MODE, self.EXPOSITION_MODE)
            self._write_register(motor_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
            self._write_register(motor_id, self.ADDR_PROFILE_VELOCITY, self.VELOCITY_LIMIT, size=4)

    def _write_register(self, motor_id, address, value, size=1):
        """
        Writes a value to a specific register of a motor.
        """

        if size == 1:
            self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, address, value)
        elif size == 2:
            self.packet_handler.write2ByteTxRx(self.port_handler, motor_id, address, value)
        elif size == 4:
            self.packet_handler.write4ByteTxRx(self.port_handler, motor_id, address, value)

    def read_position_float(self, motor_id):
        """
        Reads the current position of the motors in float format.
        """

        raw, _, _ = self.packet_handler.read4ByteTxRx(self.port_handler, motor_id, self.ADDR_PRESENT_POSITION)
        signed = np.array(raw, dtype=np.uint32).view(np.int32)
        return float(signed)
    
    def read_DELTAL_from_motors(self):
        """
        Reads the DELTAL values from the motors.
        """

        positions = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
        DELTAL = positions / self.unit_scale
        return DELTAL
    
    def get_key(self):
        """
        Read a single character from keyboard without requiring Enter.
        """
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def move_motors(self, d_DELTAL, TOLERANCE=20):
        """
        Moves motors by d_DELTAL increment and returns the actual delta achieved.
        """

        start_positions = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
        goal_positions = start_positions + (self.unit_scale * d_DELTAL).astype(np.float32)

        groupSyncWrite = GroupSyncWrite(self.port_handler, self.packet_handler, self.ADDR_GOAL_POSITION, 4)
        for motor_id, pos in zip(self.MOTOR_IDs, goal_positions):
            pos_int = int(round(pos))
            param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                     DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
            groupSyncWrite.addParam(motor_id, param)
        groupSyncWrite.txPacket()
        groupSyncWrite.clearParam()

        # Wait for completion or timeout
        tic = time.time()
        while time.time() - tic < 1.5:
            current = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
            if np.all(np.abs(current - goal_positions) <= TOLERANCE):
                break
            time.sleep(0.05)

        end_positions = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
        d_tick = end_positions - start_positions
        d_DELTAL_real = d_tick / self.unit_scale
        return d_DELTAL_real

    def calibrate_motors(self):
        """
        Interactive calibration for each motor.
        Press:
          - 'w' to move forward (with adaptive increment)
          - 's' to move backward (with adaptive increment)
          - 'h' to confirm and go to the next motor
        """

        print("--- Calibration started ---")
        base_step = np.pi * self.D_pulley / 12
        max_multiplier = 10
        accel_threshold = 0.4

        for idx, motor_id in enumerate(self.MOTOR_IDs):
            print(f"\n Motor {motor_id} selected. Press 'h' to skip, 'w' to loosen, 's' to tend.")
            last_key = None
            last_time = time.time()
            step_multiplier = 1

            while True:
                key = self.get_key()
                now = time.time()

                if key != last_key or (now - last_time) > accel_threshold:
                    step_multiplier = 1
                else:
                    step_multiplier = min(step_multiplier + 1, max_multiplier)

                last_time = now
                last_key = key

                if key == 'h':
                    break
                elif key in ['w', 's']:
                    d_DEL = np.zeros(self.MOTOR_Num, dtype=np.float32)
                    direction = 1 if key == 'w' else -1
                    step = direction * base_step * step_multiplier
                    d_DEL[idx] = step
                    self.move_motors(d_DEL)
                else:
                    print("\n Invalid input. Use 'h', 'w', or 's'.")

        print("\n --- Calibration completed ---")
                    
    def capture_images(self):
        """
        Captures images from all three cameras.
        Returns tuple of (gray_1, gray_2, gray_3)
        """

        frames = self.camera_system.get_all_frames()
        frame1 = frames[0]
        frame2 = frames[1]
        frame3 = frames[2]
        gray_1 = color2gray(frame1)
        gray_2 = color2gray(frame2)
        gray_3 = color2gray(frame3)

        return gray_1, gray_2, gray_3

    def BWGen(self, gray_1, gray_2, gray_3):
        """
        Converts three BW frames to stacked BW format,
        to match the CNN input.
        Inputs:
        - gray_1, gray_2, gray_3: BW frames from cameras 1, 2, 3
        Ouputs:
        - torch.Tensor: shape [1, 3, H, W] ready for model input
        """

        # Stack grayscale images into 3-channel format ([H, W, 3] → [1, 3, H, W])
        stacked_image = np.stack([gray_1, gray_2, gray_3], axis=-1) # Shape: [H, W, 3]
        image_tensor = torch.from_numpy(stacked_image).permute(2, 0, 1).unsqueeze(0).contiguous()

        return image_tensor
    
    def _load_models(self):
        """
        Loads the pre-trained models for:
        - CNN pose estimation
        - Data-driven (FK) method
        - Data-driven (IK) method
        """

        self.model = torch.jit.load("models/vision/helyx_model.pt", map_location=self.device)
        self.model.eval()
        self.FK = torch.jit.load("models/kinematics/FK_model.pt", map_location=self.device)
        self.FK.eval()
        self.IK = torch.jit.load("models/kinematics/IK_model.pt", map_location=self.device)
        self.IK.eval()

    def CameraSensor(self):
        """
        Captures images and defines two outputs:
        - Tip pose (x, y, z, Z, Y, X), third section
        - (x, y, z) positions of each of the three tips
        """

        frame1, frame2, frame3 = self.capture_images()
        if frame1 is None or frame2 is None or frame3 is None:
            raise RuntimeError("Failed to capture images from all cameras")
        input_tensor = self.BWGen(frame1, frame2, frame3).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)

        full_output = output[0].detach().cpu().numpy()  # Shape: [18]

        # Extract tip pose (third section: last 6 values)
        tip_pose = full_output[12:18]

        # Extract positions of all three tips
        tip_positions = np.array([
            full_output[0:3],   # tip1: x, y, z
            full_output[6:9],   # tip2: x, y, z  
            full_output[12:15]  # tip3: x, y, z
        ]).flatten()            # shape: [9]

        return tip_pose, tip_positions

    def JacobianSensor(self, pose):
        """
        Computes the Jacobian pseudoinverse matrix for the given tips pose.
        Input:
        - pose: numpy array of shape [6]
        Output:
        - J_pseudo: Jacobian numpy matrix of shape [6, 9] between pose and DELTAL
        """

        pose_tensor = torch.tensor(pose, dtype=torch.float32, device=self.device, requires_grad=True)
        pose_batched = pose_tensor.unsqueeze(0)
        jacobian = torch.autograd.functional.jacobian(self.IK, pose_batched)
        return jacobian.squeeze().cpu().numpy()

    def MotorWriter(self, DELTAL):
        """
        Writes integral DELTAL to the motors.
        """
        
        goal = self.starting_positions + (self.unit_scale * DELTAL).astype(np.float32)
        groupSyncWrite = GroupSyncWrite(self.port_handler, self.packet_handler, self.ADDR_GOAL_POSITION, 4)
        for motor_id, pos in zip(self.MOTOR_IDs, goal):
            pos_int = int(round(pos))
            param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                     DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
            groupSyncWrite.addParam(motor_id, param)
        groupSyncWrite.txPacket()
        groupSyncWrite.clearParam()

        # Wait until motors reach goal or timeout, updating position based on difference
        tic = time.time()
        while time.time() - tic < self.TIMEOUT:
            current = np.array(
                [self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32
            )
            
            # Calculate difference
            diff = goal - current
            
            # Check if within tolerance
            if np.all(np.abs(diff) <= self.TOLERANCE):
                break
            
            # Update goal positions based on current difference
            groupSyncWrite = GroupSyncWrite(self.port_handler, self.packet_handler, self.ADDR_GOAL_POSITION, 4)
            for motor_id, pos, in zip(self.MOTOR_IDs, goal):
                pos_int = int(round(pos))
                param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                        DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
                groupSyncWrite.addParam(motor_id, param)
            groupSyncWrite.txPacket()
            groupSyncWrite.clearParam()
            
            # Small delay to avoid overwhelming the motors
            time.sleep(0.05)

    def kinematics(self, DELTAL):
        """
        Returns the estimated curvature parameters from DELTAL.
        Inputs:
        - DELTAL: np array of tendon lengths [9]
        Outputs:
        - DxDyDl: np array of curvature parameters [9] for all 3 sections
        """

        # Delta angles for each section (same formulation pattern)
        deltas = [
            np.array([np.radians(0), np.radians(120), np.radians(-120)]),
            np.array([np.radians(60), np.radians(120+60), np.radians(-120+60)]),
            np.array([np.radians(150), np.radians(120+150), np.radians(-120+150)])
        ]

        DxDyDl = []

        for i in range(3):
            DELTAL_section = DELTAL[i*3:(i+1)*3]
            delta = deltas[i]
        
            # Compute curvature parameters
            Dl = np.mean(DELTAL_section)
            Dx = (2/3) * np.sum(-DELTAL_section * np.cos(delta))
            Dy = (2/3) * np.sum(-DELTAL_section * np.sin(delta))
            DxDyDl.extend([Dx, Dy, Dl])

        return np.array(DxDyDl)
    
    def AnalyticJacobians(self, DELTAL_current):
        """
        Computes the Jacobians for the current DELTAL with PCC model.
        Input:
        - DELTAL_current: numpy array of current tendon lengths [9]
        Output:
        - J: Jacobian numpy matrix of shape [9, 6] between DELTAL and coords
        """

        DxDyDl_current = self.kinematics(DELTAL_current)
        J_q2coordinates = self.Q2COORDS(*DxDyDl_current)
        J_q2tendon = self.Q2TENDON(*DxDyDl_current)
        return J_q2coordinates @ np.linalg.pinv(J_q2tendon)
    
    def PseudoInverse(self, J, k=0.1):
        """
        Returns pseudo-inverse matrix (J*) using damped least squares.
        """

        return J.T @ np.linalg.pinv(J @ J.T + k**2 * np.eye(J.shape[0]))
    
    def NullSpaceProj(self, J, J_pseudo):
        """
        Returns null space projection matrix (N).
        """

        return np.eye(J.shape[1]) - J_pseudo @ J

    def ComputeDELTALintegral(self, pose, pose_ref, DELTAL_current):
        """
        Computes the change in DELTAL to reach the reference positions pose_ref from current one.
        Input:
        - pose: current pose of tip3 [6]
        - pose_ref: reference pose of tip3 [6]
        - DELTAL_current: current tendon lengths [9]
        Output:
        - self.DELTAL_integral: updated integral of DELTAL
        """

        if self.data_driven==True:

            J_pseudo = self.JacobianSensor(pose)
            d_DELTAL = J_pseudo @ (pose_ref - pose)

            # Debugging
            print(f"DELTAL: {np.linalg.norm(J_pseudo @ (pose_ref - pose))}")

            self.DELTAL_integral += d_DELTAL

        else:

            J = self.AnalyticJacobians(DELTAL_current)
            J_pseudo = self.PseudoInverse(J)
            delta_pose = pose_ref - pose
            N = self.NullSpaceProj(J, J_pseudo)
            delta_DELTAL0 = self.DELTAL_mean - DELTAL_current
            d_DELTAL = J_pseudo @ delta_pose + N @ delta_DELTAL0

            # Debugging
            print(f"Pseudo component: {np.linalg.norm(J_pseudo @ delta_pose)}")
            print(f"Null space component: {np.linalg.norm(N @ delta_DELTAL0)}")

            self.DELTAL_integral += d_DELTAL

    def _angular_distance(self, current_euler, goal_euler):
        """
        Computes angular distance given current and goal euler angles ('zyx').
        This is done according to the rotation along the principal axis.
        """

        r_current = R.from_euler('zyx', current_euler)
        r_goal = R.from_euler('zyx', goal_euler)
        r_error = r_goal * r_current.inv()
        axis_angle = r_error.as_rotvec()
        return np.linalg.norm(axis_angle)

    def _get_waypoints(self, current_pose, target_pose):
        """
        Gets waypoints between current pose and target pose.
        """

        curr_pos, curr_euler = current_pose[:3], current_pose[3:]
        targ_pos, targ_euler = target_pose[:3], target_pose[3:]
        
        trans_dist = np.linalg.norm(targ_pos - curr_pos)
        
        rot_dist_rad = self._angular_distance(curr_euler, targ_euler)
        rot_dist_mm = np.rad2deg(rot_dist_rad)
        
        total_distance = trans_dist + rot_dist_mm / 5.0
        N_waypoints = int(np.ceil(total_distance / self.discr_wp)) + 1
        
        t_values = np.linspace(0, 1, N_waypoints)
        positions = np.linspace(curr_pos, targ_pos, N_waypoints)
        
        key_rotations = R.from_euler('zyx', [curr_euler, targ_euler])
        slerp = Slerp([0, 1], key_rotations)
        orientations = slerp(t_values).as_euler('zyx')
        
        waypoints_with_current = np.column_stack([positions, orientations])
        waypoints = waypoints_with_current[1:]
        
        return waypoints, waypoints_with_current
    
    def _compute_pose_error(self, pose1, pose2):
        """
        Computes combined pose error (position + orientation).
        """

        pos_error = np.linalg.norm(pose1[:3] - pose2[:3])
        rot_error = self._angular_distance(pose1[3:], pose2[3:])
        rot_error_deg = np.rad2deg(rot_error)
        total_error = pos_error + rot_error_deg / 5.0
        return total_error

    def CONTROL(self, pose_target):
        """
        Controls the robot to reach the target pose. Uses null-space control.
        Uses either data-driven or analytic Jacobian method.
        Input:
        - pose_target: target pose as numpy array [x, y, z, Z, Y, X]
        Output:
        - pose_trajectory: numpy array of shape [N, 6] with trajectory followed by the tip
        - waypoints_with_current: numpy array of shape [N, 6] with waypoints
        """

        current_pose, _ = self.CameraSensor()
        pose_trajectory = current_pose.reshape(1, 6)

        while self._compute_pose_error(current_pose, pose_target) > self.convergence:

            # 1. Set waypoints
            waypoints, _ = self._get_waypoints(current_pose, pose_target)
            print(f"Waypoints: {len(waypoints)}")

            # 2. Compute DELTAL_integral
            DELTAL_current = self.read_DELTAL_from_motors() - self.DELTAL_reference
            self.ComputeDELTALintegral(current_pose, waypoints[0], DELTAL_current)

            # 3. Write motors movement
            self.MotorWriter(self.DELTAL_integral)

            # 4. Capture new position
            current_pose, _ = self.CameraSensor()
            pose_trajectory = np.vstack((pose_trajectory, current_pose))

        return pose_trajectory
    
    def CleanEnv(self):
        """
        Cleans up resources when the control task is done.
        """
    
        self.camera_system.stop_capture()
        self.port_handler.closePort()

# Controller
class MULTIOpenLoop():
    """
    Controller for the MULTI section robot.
    In open-loop, uses:
    - Data-driven method
    - Jacobian-based method
    Includes:
    - Initialization (motors, camera, models)
    - Camera search and definition
    - Calibration
    - Sensing
    - Selection of the optimal actuation
    - Motion planning and execution
    E.g.:
    ...
    controller = MULTI_controller(DELTAL_mean, DATADRIVEN=True, discr_wp=1)
    trajectory, waypoints = controller.CONTROL(pose_target)
    controler.CleanEnv()
    """

    def __init__(self, DELTAL_mean, DATADRIVEN=True, DEVICENAME='/dev/ttyUSB0',
                 MOTOR_IDs=np.array([11, 12, 13, 14, 15, 16, 17, 18, 19]),
                 discr_wp=1, xyz_initial=None):
        """
        Initializes the Controller parameters and sets up the robot.
        """

        # Initialize parameters
        self.DEVICENAME = DEVICENAME
        self.MOTOR_IDs = MOTOR_IDs
        self.MOTOR_Num = self.MOTOR_IDs.shape[0]
        self.BAUDRATE = 57600
        self.D_pulley = 6
        self.unit_scale = 4096 / (np.pi * self.D_pulley)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TIMEOUT = 0.1
        self.TOLERANCE = 50
        self.DELTAL_mean = DELTAL_mean

        # Controller specifications
        self.data_driven = DATADRIVEN
        self.Q2TENDON = q2tendon()
        self.Q2COORDS = q2coordinates()
        self.discr_wp = discr_wp

        # Initialize DELTAL integral
        self.DELTAL_integral = np.zeros(9, dtype=np.float32)

        # Initialize and calibrate the motors
        self._init_dynamixel()
        self.calibrate_motors()

        # Load and use the pre-trained models
        self._load_models()
        self.calibrate_motors()

        # Define starting positions of the motors
        self.starting_positions = np.array(
            [self.read_position_float(mid) for mid in self.MOTOR_IDs],
            dtype=np.float32
        )

        # Define reference DELTAL (i.e. initial position, encoder abs. coordinates)
        DELTAL0 = np.full(9, 1e-4)
        DxDyDl0 = np.full(9, 1e-4)
        xyz0 = np.array([1e-4, 1e-4, 145, 1e-4, 1e-4, 145 * 2, 1e-4, 1e-4, 145 * 3])
        coords0 = np.array([1e-4, 1e-4, 145*3, 1e-4, 1e-4, 1e-4])
        self.xyz_initial = xyz_initial
        self.DELTAL_initial, _, _ = xyz2tendon(DELTAL0, 
                                                DxDyDl0,
                                                xyz0,
                                                self.xyz_initial - xyz0)
        _, _, self.coords_initial = tendon2coordinates(DELTAL0,
                                                  DxDyDl0,
                                                  coords0,
                                                  self.DELTAL_initial - DELTAL0)        
        self.DELTAL_reference = self.read_DELTAL_from_motors() - self.DELTAL_initial
    
    def _init_dynamixel(self):
        """
        Initializes the Dynamixel motors.
        """

        # PortHandler, PacketHandler, port, baudrate
        self.port_handler = PortHandler(self.DEVICENAME)
        self.packet_handler = Protocol2PacketHandler()
        if not self.port_handler.openPort():
            raise RuntimeError("Failed to open port")
        if not self.port_handler.setBaudRate(self.BAUDRATE):
            raise RuntimeError("Failed to set baudrate")
        
        # Set Dynamixel control table addresses and values
        self.ADDR_OPERATING_MODE = 11
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.ADDR_PROFILE_VELOCITY = 112
        self.TORQUE_ENABLE = 1
        self.VELOCITY_LIMIT = 40
        self.EXPOSITION_MODE = 4

        # Set motors to exposition mode and enable torque
        for motor_id in self.MOTOR_IDs:
            self._write_register(motor_id, self.ADDR_OPERATING_MODE, self.EXPOSITION_MODE)
            self._write_register(motor_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
            self._write_register(motor_id, self.ADDR_PROFILE_VELOCITY, self.VELOCITY_LIMIT, size=4)

    def _write_register(self, motor_id, address, value, size=1):
        """
        Writes a value to a specific register of a motor.
        """

        if size == 1:
            self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, address, value)
        elif size == 2:
            self.packet_handler.write2ByteTxRx(self.port_handler, motor_id, address, value)
        elif size == 4:
            self.packet_handler.write4ByteTxRx(self.port_handler, motor_id, address, value)

    def read_position_float(self, motor_id):
        """
        Reads the current position of the motors in float format.
        """

        raw, _, _ = self.packet_handler.read4ByteTxRx(self.port_handler, motor_id, self.ADDR_PRESENT_POSITION)
        signed = np.array(raw, dtype=np.uint32).view(np.int32)
        return float(signed)
    
    def read_DELTAL_from_motors(self):
        """
        Reads the DELTAL values from the motors.
        """

        positions = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
        DELTAL = positions / self.unit_scale
        return DELTAL
    
    def get_key(self):
        """
        Read a single character from keyboard without requiring Enter.
        """
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def move_motors(self, d_DELTAL, TOLERANCE=20):
        """
        Moves motors by d_DELTAL increment and returns the actual delta achieved.
        """

        start_positions = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
        goal_positions = start_positions + (self.unit_scale * d_DELTAL).astype(np.float32)

        groupSyncWrite = GroupSyncWrite(self.port_handler, self.packet_handler, self.ADDR_GOAL_POSITION, 4)
        for motor_id, pos in zip(self.MOTOR_IDs, goal_positions):
            pos_int = int(round(pos))
            param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                     DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
            groupSyncWrite.addParam(motor_id, param)
        groupSyncWrite.txPacket()
        groupSyncWrite.clearParam()

        # Wait for completion or timeout
        tic = time.time()
        while time.time() - tic < self.TIMEOUT:
            current = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
            if np.all(np.abs(current - goal_positions) <= self.TOLERANCE):
                break
            time.sleep(0.05)

        end_positions = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
        d_tick = end_positions - start_positions
        d_DELTAL_real = d_tick / self.unit_scale
        return d_DELTAL_real

    def calibrate_motors(self):
        """
        Interactive calibration for each motor.
        Press:
          - 'w' to move forward (with adaptive increment)
          - 's' to move backward (with adaptive increment)
          - 'h' to confirm and go to the next motor
        """

        print("--- Calibration started ---")
        base_step = np.pi * self.D_pulley / 12
        max_multiplier = 10
        accel_threshold = 0.4

        for idx, motor_id in enumerate(self.MOTOR_IDs):
            print(f"\n Motor {motor_id} selected. Press 'h' to skip, 'w' to loosen, 's' to tend.")
            last_key = None
            last_time = time.time()
            step_multiplier = 1

            while True:
                key = self.get_key()
                now = time.time()

                if key != last_key or (now - last_time) > accel_threshold:
                    step_multiplier = 1
                else:
                    step_multiplier = min(step_multiplier + 1, max_multiplier)

                last_time = now
                last_key = key

                if key == 'h':
                    break
                elif key in ['w', 's']:
                    d_DEL = np.zeros(self.MOTOR_Num, dtype=np.float32)
                    direction = 1 if key == 'w' else -1
                    step = direction * base_step * step_multiplier
                    d_DEL[idx] = step
                    self.move_motors(d_DEL)
                else:
                    print("\n Invalid input. Use 'h', 'w', or 's'.")

        print("\n --- Calibration completed ---")

    def MotorWriter(self, DELTAL):
        """
        Writes integral DELTAL to the motors.
        """
        
        goal = self.starting_positions + (self.unit_scale * DELTAL).astype(np.float32)
        groupSyncWrite = GroupSyncWrite(self.port_handler, self.packet_handler, self.ADDR_GOAL_POSITION, 4)
        for motor_id, pos in zip(self.MOTOR_IDs, goal):
            pos_int = int(round(pos))
            param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                     DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
            groupSyncWrite.addParam(motor_id, param)
        groupSyncWrite.txPacket()
        groupSyncWrite.clearParam()

        # Wait until motors reach goal or timeout, updating position based on difference
        tic = time.time()
        while time.time() - tic < self.TIMEOUT:
            current = np.array(
                [self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32
            )
            
            # Calculate difference
            diff = goal - current
            
            # Check if within tolerance
            if np.all(np.abs(diff) <= self.TOLERANCE):
                break
            
            # Update goal positions based on current difference
            groupSyncWrite = GroupSyncWrite(self.port_handler, self.packet_handler, self.ADDR_GOAL_POSITION, 4)
            for motor_id, pos, in zip(self.MOTOR_IDs, goal):
                pos_int = int(round(pos))
                param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                        DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
                groupSyncWrite.addParam(motor_id, param)
            groupSyncWrite.txPacket()
            groupSyncWrite.clearParam()
            
            # Small delay to avoid overwhelming the motors
            time.sleep(0.05)

    def kinematics(self, DELTAL):
        """
        Returns the estimated curvature parameters from DELTAL.
        Inputs:
        - DELTAL: np array of tendon lengths [9]
        Outputs:
        - DxDyDl: np array of curvature parameters [9] for all 3 sections
        """

        # Delta angles for each section (same formulation pattern)
        deltas = [
            np.array([np.radians(0), np.radians(120), np.radians(-120)]),
            np.array([np.radians(60), np.radians(120+60), np.radians(-120+60)]),
            np.array([np.radians(150), np.radians(120+150), np.radians(-120+150)])
        ]

        DxDyDl = []

        for i in range(3):
            DELTAL_section = DELTAL[i*3:(i+1)*3]
            delta = deltas[i]
        
            # Compute curvature parameters
            Dl = np.mean(DELTAL_section)
            Dx = (2/3) * np.sum(-DELTAL_section * np.cos(delta))
            Dy = (2/3) * np.sum(-DELTAL_section * np.sin(delta))
            DxDyDl.extend([Dx, Dy, Dl])

        return np.array(DxDyDl)

    def _load_models(self):
        """
        Loads the pre-trained models for:
        - Data-driven (FK) jacobian
        - Data-driven (IK) jacobian
        """

        self.FK = torch.jit.load("models/kinematics/FK_model.pt", map_location=self.device)
        self.FK.eval()
        self.IK = torch.jit.load("models/kinematics/IK_model.pt", map_location=self.device)
        self.IK.eval()

    def AnalyticJacobians(self, DELTAL_current):
        """
        Computes the Jacobians for the current DELTAL with PCC model.
        Input:
        - DELTAL_current: numpy array of current tendon lengths [9]
        Output:
        - J: Jacobian numpy matrix of shape [9, 6] between DELTAL and coords
        """

        DxDyDl_current = self.kinematics(DELTAL_current)
        J_q2coordinates = self.Q2COORDS(*DxDyDl_current)
        J_q2tendon = self.Q2TENDON(*DxDyDl_current)
        return J_q2coordinates @ np.linalg.pinv(J_q2tendon)

    def PseudoInverse(self, J, k=0.1):
        """
        Returns pseudo-inverse matrix (J*) using damped least squares.
        """

        return J.T @ np.linalg.pinv(J @ J.T + k**2 * np.eye(J.shape[0]))
    
    def NullSpaceProj(self, J):
        """
        Returns null space projection matrix (N).
        """

        # Moore-Penrose inverse
        J_pseudo = self.PseudoInverse(J)
        return np.eye(J.shape[1]) - J_pseudo @ J

    def JacobianSensor(self, DELTAL, pose):
        """
        Computes the Jacobian and pseudo Jacobian matrix for the given tips positions.
        Input:
        - DELTAL: numpy array of shape [9]
        Output:
        - J: Jacobian numpy matrix of shape [9, 6] between DELTAL and coords
        - J_pseudo: pseudo Jacobian numpy matrix of shape [6, 9] between coords and DELTAL
        """

        pose_tensor = torch.tensor(pose, dtype=torch.float32, device=self.device, requires_grad=True)
        pose_batched = pose_tensor.unsqueeze(0)
        jacobian_pseudo = torch.autograd.functional.jacobian(self.IK, pose_batched)

        deltal_tensor = torch.tensor(DELTAL, dtype=torch.float32, device=self.device, requires_grad=True)
        deltal_batched = deltal_tensor.unsqueeze(0)
        jacobian = torch.autograd.functional.jacobian(self.FK, deltal_batched)
        return jacobian.squeeze().cpu().numpy(), jacobian_pseudo.squeeze().cpu().numpy()

    def _angular_distance(self, current_euler, goal_euler):
        """
        Computes angular distance given current and goal euler angles ('zyx').
        This is done according to the rotation along the principal axis.
        """

        r_current = R.from_euler('zyx', current_euler)
        r_goal = R.from_euler('zyx', goal_euler)
        r_error = r_goal * r_current.inv()
        axis_angle = r_error.as_rotvec()
        return np.linalg.norm(axis_angle)

    def _get_waypoints(self, current_pose, target_pose):
        """
        Gets waypoints between current pose and target pose.
        """

        curr_pos, curr_euler = current_pose[:3], current_pose[3:]
        targ_pos, targ_euler = target_pose[:3], target_pose[3:]
        
        trans_dist = np.linalg.norm(targ_pos - curr_pos)
        
        rot_dist_rad = self._angular_distance(curr_euler, targ_euler)
        rot_dist_mm = np.rad2deg(rot_dist_rad)
        
        total_distance = trans_dist + rot_dist_mm / 5.0
        N_waypoints = int(np.ceil(total_distance / self.discr_wp)) + 1
        
        t_values = np.linspace(0, 1, N_waypoints)
        positions = np.linspace(curr_pos, targ_pos, N_waypoints)
        
        key_rotations = R.from_euler('zyx', [curr_euler, targ_euler])
        slerp = Slerp([0, 1], key_rotations)
        orientations = slerp(t_values).as_euler('zyx')
        
        waypoints = np.column_stack([positions, orientations])[1:]

        return waypoints

    def Pose2Tendon(self, DELTAL0, starting_pose, pose_target):
        """
        According to null space control, defines the parameters to reach the target pose.
        Uses either data-driven or analytic method.
        Inputs:
        - DELTAL0: numpy array of shape [9] with estimated starting DELTAL
        - starting_pose: starting pose as numpy array [6]
        - pose_target: target pose as numpy array [6]
        Output:
        - DELTAL: numpy array of shape [9] with estimated final DELTAL
        - J: computed Jacobian [9, 6]
        """

        if self.data_driven==True:

            J, J_pseudo = self.JacobianSensor(DELTAL0, starting_pose)
            d_DELTAL = J_pseudo @ (pose_target - starting_pose)

            # Debugging
            print(f"DELTAL: {np.linalg.norm(d_DELTAL)}")

        else:

            J = self.AnalyticJacobians(DELTAL0)
            J_pseudo = self.PseudoInverse(J)
            delta_pose = pose_target - starting_pose
            N = self.NullSpaceProj(J)
            delta_DELTAL0 = self.DELTAL_mean - DELTAL0
            d_DELTAL = J_pseudo @ delta_pose + N @ delta_DELTAL0

            # Debugging
            print(f"Pseudo component: {np.linalg.norm(J_pseudo @ delta_pose)}")
            print(f"Null space component: {np.linalg.norm(N @ delta_DELTAL0)}")

        return DELTAL0 + d_DELTAL, J

    def CONTROL(self, starting_pose, pose_target, DELTAL0):
        """
        Controls the robot to reach the target pose.
        Uses either data-driven or analytic method.
        Input:
        - starting pose: starting pose as numpy array [x, y, z, Z, Y, X]
        - pose_target: target pose as numpy array [x, y, z, Z, Y, X]
        - DELTAL0: numpy array of shape [9] with starting DELTAL
        Output:
        - DELTAL: numpy array of shape [9] with estimated DELTAL
        """

        # Define waypoints
        waypoints = self._get_waypoints(starting_pose, pose_target)

        # Define DELTAL_init and pose
        DELTAL_init = DELTAL0.copy()
        pose = starting_pose.copy()

        # Motion
        for i, pose_ref in enumerate(waypoints):

            # Print
            print(f"{i+1}/{len(waypoints)}")

            # 1. Shift DELTAL of the needed value
            DELTAL, J = self.Pose2Tendon(DELTAL_init, pose, pose_ref)
            self.DELTAL_integral += DELTAL - DELTAL_init
            self.MotorWriter(self.DELTAL_integral)

            # 2. Update current pose and DELTAL_init
            delta = J @ (DELTAL - DELTAL_init)
            pose = pose + delta
            DELTAL_init = DELTAL

        return DELTAL
        
    def CleanEnv(self):
        """
        Cleans up resources when the control task is done.
        """

        self.port_handler.closePort()

# Null space controller class
class NullSpaceCheck:
    """
    Given an arbitrary DELTAL, shows the behavior of the Null Space Controller.
    Exploits:
    - Data-driven method
    - Model-based method
    E.g.:
    ...
    check = NullSpaceCheck(DATADRIVEN=True)
    check.MOVE(DELTAL_imposed=DELTAL)
    check.CleanEnv()
    ...
    """

    def __init__(self, DEVICENAME='/dev/ttyUSB0',
                 discr_wp=1, MOTOR_IDs=np.array([11, 12, 13, 14, 15, 16, 17, 18, 19]),
                 xyz_initial=None):
        """
        Initializes the parameters and sets up the robot.
        """

        # Initialize parameters
        self.DEVICENAME = DEVICENAME
        self.MOTOR_IDs = MOTOR_IDs
        self.MOTOR_Num = self.MOTOR_IDs.shape[0]
        self.BAUDRATE = 57600
        self.D_pulley = 6
        self.unit_scale = 4096 / (np.pi * self.D_pulley)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TIMEOUT = 5
        self.TOLERANCE = 20

        # Controller specifications
        self.Q2TENDON = q2tendon()
        self.Q2COORDS = q2coordinates()
        self.discr_wp = discr_wp

        # Initialize DELTAL integral
        self.DELTAL_integral = np.zeros(9, dtype=np.float32)

        # Initialize and calibrate the motors
        self._init_dynamixel()
        self.calibrate_motors()

        # Load the pre-trained model
        self.calibrate_motors()

        # Define starting positions of the motors
        self.starting_positions = np.array(
            [self.read_position_float(mid) for mid in self.MOTOR_IDs],
            dtype=np.float32
        )

        # Define reference DELTAL (i.e. initial position, encoder abs. coordinates)
        DELTAL0 = np.full(9, 1e-4)
        DxDyDl0 = np.full(9, 1e-4)
        xyz0 = np.array([1e-4, 1e-4, 145, 1e-4, 1e-4, 145 * 2, 1e-4, 1e-4, 145 * 3])
        coords0 = np.array([1e-4, 1e-4, 145*3, 1e-4, 1e-4, 1e-4])
        self.xyz_initial = xyz_initial
        self.DELTAL_initial, _, _ = xyz2tendon(DELTAL0, 
                                                DxDyDl0,
                                                xyz0,
                                                self.xyz_initial - xyz0)
        _, _, self.coords_initial = tendon2coordinates(DELTAL0,
                                                  DxDyDl0,
                                                  coords0,
                                                  self.DELTAL_initial - DELTAL0)        
        self.DELTAL_reference = self.read_DELTAL_from_motors() - self.DELTAL_initial

    def _init_dynamixel(self):
        """
        Initializes the Dynamixel motors.
        """

        # PortHandler, PacketHandler, port, baudrate
        self.port_handler = PortHandler(self.DEVICENAME)
        self.packet_handler = Protocol2PacketHandler()
        if not self.port_handler.openPort():
            raise RuntimeError("Failed to open port")
        if not self.port_handler.setBaudRate(self.BAUDRATE):
            raise RuntimeError("Failed to set baudrate")

        # Set Dynamixel control table addresses and values
        self.ADDR_OPERATING_MODE = 11
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.ADDR_PROFILE_VELOCITY = 112
        self.TORQUE_ENABLE = 1
        self.VELOCITY_LIMIT = 40
        self.EXPOSITION_MODE = 4

        # Set motors to exposition mode and enable torque
        for motor_id in self.MOTOR_IDs:
            self._write_register(motor_id, self.ADDR_OPERATING_MODE, self.EXPOSITION_MODE)
            self._write_register(motor_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE)
            self._write_register(motor_id, self.ADDR_PROFILE_VELOCITY, self.VELOCITY_LIMIT, size=4)

    def _write_register(self, motor_id, address, value, size=1):
        """
        Writes a value to a specific register of a motor.
        """

        if size == 1:
            self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, address, value)
        elif size == 2:
            self.packet_handler.write2ByteTxRx(self.port_handler, motor_id, address, value)
        elif size == 4:
            self.packet_handler.write4ByteTxRx(self.port_handler, motor_id, address, value)

    def read_position_float(self, motor_id):
        """
        Reads the current position of the motors in float format.
        """

        raw, _, _ = self.packet_handler.read4ByteTxRx(self.port_handler, motor_id, self.ADDR_PRESENT_POSITION)
        signed = np.array(raw, dtype=np.uint32).view(np.int32)
        return float(signed)    
        
    def read_DELTAL_from_motors(self):
        """
        Reads the DELTAL values from the motors.
        """

        positions = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
        DELTAL = positions / self.unit_scale
        return DELTAL
    
    def get_key(self):
        """
        Read a single character from keyboard without requiring Enter.
        """
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def move_motors(self, d_DELTAL, TOLERANCE=20):
        """
        Moves motors by d_DELTAL increment and returns the actual delta achieved.
        """

        start_positions = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
        goal_positions = start_positions + (self.unit_scale * d_DELTAL).astype(np.float32)

        groupSyncWrite = GroupSyncWrite(self.port_handler, self.packet_handler, self.ADDR_GOAL_POSITION, 4)
        for motor_id, pos in zip(self.MOTOR_IDs, goal_positions):
            pos_int = int(round(pos))
            param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                     DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
            groupSyncWrite.addParam(motor_id, param)
        groupSyncWrite.txPacket()
        groupSyncWrite.clearParam()

        # Wait for completion or timeout
        tic = time.time()
        while time.time() - tic < self.TIMEOUT:
            current = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
            if np.all(np.abs(current - goal_positions) <= self.TOLERANCE):
                break
            time.sleep(0.05)

        end_positions = np.array([self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32)
        d_tick = end_positions - start_positions
        d_DELTAL_real = d_tick / self.unit_scale
        return d_DELTAL_real

    def calibrate_motors(self):
        """
        Interactive calibration for each motor.
        Press:
          - 'w' to move forward (with adaptive increment)
          - 's' to move backward (with adaptive increment)
          - 'h' to confirm and go to the next motor
        """

        print("--- Calibration started ---")
        base_step = np.pi * self.D_pulley / 12
        max_multiplier = 10
        accel_threshold = 0.4

        for idx, motor_id in enumerate(self.MOTOR_IDs):
            print(f"\n Motor {motor_id} selected. Press 'h' to skip, 'w' to loosen, 's' to tend.")
            last_key = None
            last_time = time.time()
            step_multiplier = 1

            while True:
                key = self.get_key()
                now = time.time()

                if key != last_key or (now - last_time) > accel_threshold:
                    step_multiplier = 1
                else:
                    step_multiplier = min(step_multiplier + 1, max_multiplier)

                last_time = now
                last_key = key

                if key == 'h':
                    break
                elif key in ['w', 's']:
                    d_DEL = np.zeros(self.MOTOR_Num, dtype=np.float32)
                    direction = 1 if key == 'w' else -1
                    step = direction * base_step * step_multiplier
                    d_DEL[idx] = step
                    self.move_motors(d_DEL)
                else:
                    print("\n Invalid input. Use 'h', 'w', or 's'.")

        print("\n --- Calibration completed ---")

    def MotorWriter(self, DELTAL):
        """
        Writes integral DELTAL to the motors.
        """
        
        goal = self.starting_positions + (self.unit_scale * DELTAL).astype(np.float32)
        groupSyncWrite = GroupSyncWrite(self.port_handler, self.packet_handler, self.ADDR_GOAL_POSITION, 4)
        for motor_id, pos in zip(self.MOTOR_IDs, goal):
            pos_int = int(round(pos))
            param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                     DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
            groupSyncWrite.addParam(motor_id, param)
        groupSyncWrite.txPacket()
        groupSyncWrite.clearParam()

        # Wait until motors reach goal or timeout, updating position based on difference
        tic = time.time()
        while time.time() - tic < self.TIMEOUT:
            current = np.array(
                [self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32
            )
            
            # Calculate difference
            diff = goal - current
            
            # Check if within tolerance
            if np.all(np.abs(diff) <= self.TOLERANCE):
                break
            
            # Update goal positions based on current difference
            groupSyncWrite = GroupSyncWrite(self.port_handler, self.packet_handler, self.ADDR_GOAL_POSITION, 4)
            for motor_id, pos, in zip(self.MOTOR_IDs, goal):
                pos_int = int(round(pos))
                param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                        DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
                groupSyncWrite.addParam(motor_id, param)
            groupSyncWrite.txPacket()
            groupSyncWrite.clearParam()
            
            # Small delay to avoid overwhelming the motors
            time.sleep(0.05)

    def kinematics(self, DELTAL):
        """
        Returns the estimated curvature parameters from DELTAL.
        Inputs:
        - DELTAL: np array of tendon lengths [9]
        Outputs:
        - DxDyDl: np array of curvature parameters [9] for all 3 sections
        """

        # Delta angles for each section (same formulation pattern)
        deltas = [
            np.array([np.radians(0), np.radians(120), np.radians(-120)]),
            np.array([np.radians(60), np.radians(120+60), np.radians(-120+60)]),
            np.array([np.radians(150), np.radians(120+150), np.radians(-120+150)])
        ]

        DxDyDl = []

        for i in range(3):
            DELTAL_section = DELTAL[i*3:(i+1)*3]
            delta = deltas[i]
        
            # Compute curvature parameters
            Dl = np.mean(DELTAL_section)
            Dx = (2/3) * np.sum(-DELTAL_section * np.cos(delta))
            Dy = (2/3) * np.sum(-DELTAL_section * np.sin(delta))
            DxDyDl.extend([Dx, Dy, Dl])

        return np.array(DxDyDl)

    def AnalyticJacobians(self, DELTAL_current):
        """
        Computes the Jacobians for the current DELTAL with PCC model.
        Input:
        - DELTAL_current: numpy array of current tendon lengths [9]
        Output:
        - J: Jacobian numpy matrix of shape [9, 6] between DELTAL and coords
        """

        DxDyDl_current = self.kinematics(DELTAL_current)
        J_q2coordinates = self.Q2COORDS(*DxDyDl_current)
        J_q2tendon = self.Q2TENDON(*DxDyDl_current)
        return J_q2coordinates @ np.linalg.pinv(J_q2tendon)

    def PseudoInverse(self, J, k=1.0):
        """
        Returns pseudo-inverse matrix (J*) using damped least squares.
        """

        return J.T @ np.linalg.pinv(J @ J.T + k**2 * np.eye(J.shape[0]))
    
    def NullSpaceProj(self, J):
        """
        Returns null space projection matrix (N).
        """

        # Moore-Penrose inverse
        J_pseudo = self.PseudoInverse(J)
        return np.eye(J.shape[1]) - J_pseudo @ J

    def Pose2Tendon(self, DELTAL0, DELTAL_imposed):
        """
        According to null space control, defines the parameters to reach the target pose.
        Uses either data-driven or analytic method.
        Inputs:
        - DELTAL0: numpy array of shape [9] with estimated starting DELTAL
        - DELTAL_imposed: numpy array of shape [9] with imposed DELTAL
        Output:
        - DELTAL: numpy array of shape [9] with estimated final DELTAL
        """

        J = self.AnalyticJacobians(DELTAL0)

        # Define delta_pose and J*
        N = self.NullSpaceProj(J)

        # Define motion according to null space control
        d_DELTAL = N @ DELTAL_imposed

        return DELTAL0 + d_DELTAL
    
    def MOVE(self, DELTAL_imposed):
        """
        Controls the robot in the null space.
        Uses either data-driven or analytic method.
        Input:
        - DELTAL_imposed: numpy array of shape [9] with imposed DELTAL
        """

        # Define DELTAL0
        DELTAL0 = self.read_DELTAL_from_motors() - self.DELTAL_reference

        # Motion
        DELTAL = self.Pose2Tendon(DELTAL0, DELTAL_imposed)
        self.DELTAL_integral += DELTAL - DELTAL0
        self.MotorWriter(self.DELTAL_integral)
    
    def CleanEnv(self):
        """
        Cleans up resources when the control task is done.
        """

        self.port_handler.closePort()

# Main script
if __name__ == "__main__":

    # Optitrack
    print("OPTITRACK:")
    print("roslaunch optitrack_ros_communication optitrack_nodes.launch")

    # Cameras visualization
    print("CAMERAS:")
    print("v4l2-ctl --list-devices")