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
import threading
from ctypes import c_uint32
import matplotlib.pyplot as plt # type: ignore
from mpl_toolkits.mplot3d import Axes3D # type: ignore
import termios
import tty
import torch # type: ignore

# User-defined kinematics
from SINGLE_jacobians import *
from SINGLE_kinematics import *

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
import shutil
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

# Controller
class SINGLEController:
    """
    Controller for the SINGLE section robot. Uses:
    - Data-driven method
    - Jacobian-based method
    Includes:
    - Initialization (motors, camera, models)
    - Calibration
    - Sensing
    - Motion planning and execution
    E.g.:
    ...
    controller = SINGLEController(DATADRIVEN=True, CAMERA_ID=2, discr_wp=1)
    trajectory, waypoints = controller.CONTROL(x_target)
    controller.CleanEnv()
    ...
    """

    def __init__(self, DATADRIVEN=True, CAMERA_ID=2, 
                 MOTOR_IDs=np.array([1, 2, 3]), DEVICENAME='/dev/ttyUSB0',
                 discr_wp=1, convergence=3):
        """
        Initializes the Controller parameters and sets up the robot.
        """

        # Initialize parameters
        self.DEVICENAME = DEVICENAME
        self.MOTOR_IDs = MOTOR_IDs
        self.CAMERA_ID = CAMERA_ID
        self.MOTOR_Num = self.MOTOR_IDs.shape[0]
        self.BAUDRATE = 57600
        self.D_pulley = 6
        self.unit_scale = 4096 / (np.pi * self.D_pulley)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()
        self.latest_frame = None
        self.TIMEOUT = 1.5
        self.TOLERANCE = 20

        # Controller specifications
        self.data_driven = DATADRIVEN
        self.Q2COORDINATES = q2coordinates()
        self.Q2TENDON = q2tendon()
        self.discr_wp = discr_wp
        self.convergence = convergence

        # Initialize DELTAL integral
        self.DELTAL_integral = np.zeros(3, dtype=np.float32)
        
        # Initialize and calibrate the motors
        self._init_dynamixel()
        self.calibrate_motors()

        # Initialize and check the camera
        self._init_camera()
        self.display_internal_image()

        # Load and use the pre-trained models
        self._load_models()
        self.show_Pose_Jacobian()
        self.calibrate_motors()

        # Define starting positions of the motors
        self.starting_positions = np.array(
            [self.read_position_float(mid) for mid in self.MOTOR_IDs],
            dtype=np.float32
        )        

        # Define reference DELTAL (i.e. initial position, encoder abs. coordinates)
        DELTAL0 = np.array([1e-4, 1e-4, 1e-4])
        DxDyDl0 = np.array([1e-4, 1e-4, 1e-4])
        coords0 = np.array([1e-4, 1e-4, 150, 1e-4, 1e-4, 1e-4])
        DELTAL_initial, _, _ = coordinates2tendon(DELTAL0, DxDyDl0, coords0,
                                        self.CameraSensor() - coords0[:3])
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
        self.VELOCITY_LIMIT = 75
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

    def _init_camera(self):
        """
        Initializes the camera for capturing images.
        """

        self.cap = cv2.VideoCapture(self.CAMERA_ID, cv2.CAP_V4L2)
        time.sleep(2)
        thread = threading.Thread(target=self._update_camera, daemon=True)
        thread.start()
        while self.latest_frame is None:
            pass    # wait for the first frame to be captured

    def _update_camera(self):
        """
        Continuously updates the latest frame from the camera.
        """

        while True:
            ret, frame = self.cap.read()
            if ret:     # adjust latest frame variable
                with self.lock:
                    self.latest_frame = frame

    def capture_image(self):
        """
        Captures the latest image from the camera.
        """

        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
            
    def display_internal_image(self):
        """
        Display the internal image from the camera.
        """

        image = self.capture_image()
        if image is not None:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Initial Frame", image_gray)
            cv2.waitKey(2000)   # wait for 2 seconds
            cv2.destroyAllWindows()
        else:
            print("No image captured.")
            time.sleep(2)
            
    def _load_models(self):
        """
        Loads the pre-trained models for data-driven and kinematics methods.
        """

        self.model = torch.jit.load("models/vision/helyx_model.pt", map_location=self.device)
        self.model.eval()
        self.model_jac = torch.jit.load("models/kinematics/kinematics_model.pt", map_location=self.device)
        self.model_jac.eval()

    def CameraSensor(self):
        """
        Captures image and defines the tip position (x, y, z).
        """

        img = self.capture_image()
        if img is None:
            raise ValueError("No image captured.")
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        input = torch.from_numpy(image_gray).unsqueeze(0).contiguous()  # [1, H, W]
        input_tensor = input.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output[0, :3].detach().cpu().numpy()
    
    def JacobianSensor(self, x):
        """
        Computes the Jacobian matrix for the given input x,
        between x and the motor DELTAL.
        """

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(self.model_jac, x_tensor)
        jac = jacobian[0, :, 0, :].cpu().numpy()
        return jac
    
    def show_Pose_Jacobian(self):
        """
        Displays initial Pose and Jacobian matrix.
        """

        try:
            pose = self.CameraSensor()
            print("Current Position:")
            print(f"x: {pose[0]:.2f}")
            print(f"y: {pose[1]:.2f}")
            print(f"z: {pose[2]:.2f}")
            jac = self.JacobianSensor(pose)
            print("Jacobian:")
            print(np.array2string(jac, formatter={'float_kind': lambda x: f"{x:6.3f}"}))
        except Exception as e:
            print(f"Error in showing Pose or Jacobian: {e}")

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

        # Wait until motors reach goal or timeout
        tic = time.time()
        while time.time() - tic < self.TIMEOUT:
            current = np.array(
                [self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32
            )
            if np.all(np.abs(current - goal) <= self.TOLERANCE):
                break

    def kinematics(self, DELTAL):
        """
        Returns the estimated curvature parameters from DELTAL.
        Inputs:
        - DELTAL: np array of tendon lengths.
        Outputs:
        - DxDyDl: np array of [Dx, Dy, Dl] parameters.
        """

        Dl = np.mean(DELTAL)
        delta = np.array([0, 2*np.pi/3, -2*np.pi/3])
        Dx = (2/3) * np.sum(- DELTAL * np.cos(delta))
        Dy = (2/3) * np.sum(- DELTAL * np.sin(delta))
        return np.array([Dx, Dy, Dl])
    
    def AnalyticJacobians(self, DELTAL_current):
        """
        Computes the Jacobians for the current DELTAL.
        Input:
        - DELTAL_current: current tendon lengths (DELTAL) for the 3 motors
        Output:
        - J_q2L: Jacobian for tendon lengths
        - J_q2x: Jacobian for coordinates
        """

        DxDyDl_current = self.kinematics(DELTAL_current)
        J_coords = self.Q2COORDINATES(*DxDyDl_current)
        J_q2L = self.Q2TENDON(*DxDyDl_current)
        J_q2x = J_coords[0:3, :]
        return J_q2L, J_q2x
    
    def ComputeDELTALintegral(self, x, x_ref, DELTAL_current=None):
        """
        Computes the change in DELTAL to reach the reference position x_ref from current position x.
        Input:
        - x: current position
        - x_ref: reference position to reach                 
        - DELTAL_current: current tendon lengths (DELTAL) for the 3 motors (used if data_driven=False)
        Output:
        - self.DELTAL_integral: updated integral of DELTAL
        """

        delta_x = x_ref - x
        if self.data_driven==True:
            d_DELTAL = self.JacobianSensor(x) @ delta_x
        else:
            J_q2L, J_q2x = self.AnalyticJacobians(DELTAL_current)
            reg = 1e-3 * np.eye(3)
            J_pinv = np.linalg.inv(J_q2x.T @ J_q2x + reg) @ J_q2x.T
            d_DELTAL = J_q2L @ J_pinv @ delta_x

        # Adjust DELTAL_integral
        self.DELTAL_integral += d_DELTAL

    def CONTROL(self, x_target):
        """
        Controls the robot to reach the target position x_target.
        Uses either data-driven or Jacobian-based method.
        Input:
        - x_target: target position in the form of np array [x, y, z]
        Output:
        - trajectory: np array of the trajectory followed by the robot
        - waypoints: np array of the waypoints used for the movement
        """

        # Set waypoints
        A = self.CameraSensor()     # current position
        B = x_target                # target position
        N_waypoints = int(np.ceil(np.linalg.norm(B - A) / self.discr_wp)) + 1
        waypoints_with_A = np.linspace(A, B, N_waypoints)

        # Initialized followed trajectory
        trajectory = A.reshape(1, 3)

        while np.linalg.norm(A - B) > self.convergence:

            # 1. Discretization step
            N_waypoints = int(np.ceil(np.linalg.norm(B - A) / self.discr_wp)) + 1
            waypoints = np.linspace(A, B, N_waypoints)[1:]

            # 2. Move to next waypoint
            if self.data_driven==True:
                self.ComputeDELTALintegral(A, waypoints[0])
            else:
                DELTAL_current = self.read_DELTAL_from_motors() - self.DELTAL_reference
                self.ComputeDELTALintegral(A, waypoints[0], DELTAL_current)
            self.MotorWriter(self.DELTAL_integral)

            # 3. Capture new position
            A = self.CameraSensor()
            trajectory = np.vstack((trajectory, A))

        return trajectory, waypoints_with_A
    
    def CleanEnv(self):
        """
        Cleans up resources when the control task is done.
        """

        self.cap.release()
        cv2.destroyAllWindows()
        self.port_handler.closePort()

# Controller
class SINGLEOpenLoop:
    """
    Controller for the SINGLE section robot. Uses:
    - Data-driven method
    - Jacobian-based method
    Includes:
    - Initialization (motors, camera, models)
    - Calibration
    - Sensing
    - Motion planning and execution
    # E.g.:
    ...
    controller = SINGLEOpenLoop(DATADRIVEN=True)
    ... Keeping track of the previous state ...
    controller.CleanEnv()
    ...
    """

    def __init__(self, DATADRIVEN=True,
                 MOTOR_IDs=np.array([1, 2, 3]), DEVICENAME='/dev/ttyUSB0',
                 discr_wp=1, x_initial=None):
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()
        self.latest_frame = None
        self.TIMEOUT = 1.5
        self.TOLERANCE = 20

        # Controller specifications
        self.data_driven = DATADRIVEN
        self.Q2COORDINATES = q2coordinates()
        self.Q2TENDON = q2tendon()
        self.discr_wp = discr_wp

        # Initialize DELTAL integral
        self.DELTAL_integral = np.zeros(3, dtype=np.float32)
        
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
        DELTAL0 = np.array([1e-4, 1e-4, 1e-4])
        DxDyDl0 = np.array([1e-4, 1e-4, 1e-4])
        coords0 = np.array([1e-4, 1e-4, 150, 1e-4, 1e-4, 1e-4])
        self.x_initial = x_initial
        self.DELTAL_initial, self.DxDyDl_initial, _ = coordinates2tendon(DELTAL0, DxDyDl0, coords0, self.x_initial[:3] - coords0[:3], step_size=1)
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
        self.VELOCITY_LIMIT = 75
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
            
    def _load_models(self):
        """
        Loads the pre-trained models for data-driven and kinematics methods.
        """

        self.model = torch.jit.load("models/vision/helyx_model.pt", map_location=self.device)
        self.model.eval()
        self.model_jac = torch.jit.load("models/kinematics/kinematics_model.pt", map_location=self.device)
        self.model_jac.eval()

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

        # Wait until motors reach goal or timeout
        tic = time.time()
        while time.time() - tic < self.TIMEOUT:
            current = np.array(
                [self.read_position_float(mid) for mid in self.MOTOR_IDs], dtype=np.float32
            )
            if np.all(np.abs(current - goal) <= self.TOLERANCE):
                break
    
    def CONTROL(self, x0, x_target, DELTAL0, DxDyDl0=None):
        """
        Computes the change in DELTAL to reach the reference position.
        Controls to the final position.
        """
        
        delta_x = x_target - x0[:3]
        N_waypoints = int(np.ceil(np.linalg.norm(delta_x) / self.discr_wp)) + 1
        waypoints_with_x0 = np.linspace(x0[:3], x_target, N_waypoints)
        waypoints = waypoints_with_x0[1:]

        # Initialize
        DELTAL = DELTAL0.copy()
        if self.data_driven==False:
            DxDyDl = DxDyDl0.copy()
        x = x0.copy()

        for x_ref in waypoints:
            DELTAL0 = DELTAL
            if self.data_driven==False:
                DELTAL, DxDyDl, x = coordinates2tendon(DELTAL, DxDyDl, x, x_ref - x[:3])
                d_DELTAL = DELTAL - DELTAL0
            else:
                with torch.no_grad():
                    x_tensor = torch.tensor(x_ref, dtype=torch.float32, device=self.device).unsqueeze(0)
                    DELTAL = self.model_jac(x_tensor)[0].cpu().numpy()
                d_DELTAL = DELTAL - DELTAL0          

            # Adjust DELTAL_integral
            self.DELTAL_integral += d_DELTAL

            # Move
            self.MotorWriter(self.DELTAL_integral)

        if self.data_driven==False:
            return x, DELTAL, DxDyDl
        else:
            return x_target, DELTAL
        
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