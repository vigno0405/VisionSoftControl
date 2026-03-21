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
import threading
import queue
from dataclasses import dataclass

# User-defined
from MULTI_jacobians import *
from MULTI_kinematics import *
from MULTI_reconstruction_utility import SPCModel
from MULTI_cameras_utility import *

# At the beginning, for the optitrack:
# roslaunch optitrack_ros_communication optitrack_nodes.launch

# For the cameras:
# v4l2-ctl --list-devices

# ======================================================================================
#                                    Functions
# ======================================================================================

# Write register
def write_register(motor_id, address, value, size=1):
    """
    Writes a single register.
    """
    
    if size == 1:
        packet_handler.write1ByteTxRx(port_handler, motor_id, address, value)
    elif size == 2:
        packet_handler.write2ByteTxRx(port_handler, motor_id, address, value)
    elif size == 4:
        packet_handler.write4ByteTxRx(port_handler, motor_id, address, value)

def get_key():
    """
    Read a single character without requiring Enter.
    """

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Pose callback for positions and quaternions
def pose_callback(msg, topic_name):
    """
    Callback function to process messages from each topic.
    """

    global positions, quaternions, angles
    
    # Extract position and orientation
    position = msg.pose.position
    orientation = msg.pose.orientation

    # Update stored data
    positions[topic_name] = [position.x, position.y, position.z]
    quaternions[topic_name] = [orientation.x, orientation.y, orientation.z, orientation.w]  # scalar last

    # Convert quaternion to Euler angles (roll, pitch, yaw)
    angles[topic_name] = R.from_quat(quaternions[topic_name]).as_euler('zyx')

# Move motors
def move_motors(d_DELTAL, TOLERANCE=20):
    """
    Moves motors by d_DELTAL increment and returns the actual delta achieved.
    Encoder values are read as uint32 and explicitly reinterpreted as int32 (Dynamixel standard).
    All computations are in float32 to avoid overflow.
    """

    # Read position as signed int32 (from raw uint32) and convert to float
    def read_position_float(motor_id):
        raw, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        signed = np.array(raw, dtype=np.uint32).view(np.int32)
        return float(signed)

    # Initial positions
    start_positions = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)

    # Target positions
    goal_positions = start_positions + (unit_scale * d_DELTAL).astype(np.float32)

    # Send goal to motors
    groupSyncWrite = GroupSyncWrite(port_handler, packet_handler, ADDR_GOAL_POSITION, 4)
    for motor_id, pos in zip(MOTOR_IDs, goal_positions):
        pos_int = int(round(pos))  # Cast to int for sync write
        param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                 DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
        groupSyncWrite.addParam(motor_id, param)
    groupSyncWrite.txPacket()
    groupSyncWrite.clearParam()

    # Wait for completion or timeout
    tic = time.time()
    while time.time() - tic < 1.5:
        current = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)
        if np.all(np.abs(current - goal_positions) <= TOLERANCE):
            break
        time.sleep(0.05)

    # Final positions
    end_positions = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)

    # Compute delta in mm
    d_tick = end_positions - start_positions
    d_DELTAL_real = d_tick / unit_scale

    return d_DELTAL_real

# Calibration function
def calibrate_motors():
    """
    Interactive calibration for each motor.
    Press:
      - 'w' to move forward (with adaptive increment)
      - 's' to move backward (with adaptive increment)
      - 'h' to confirm and go to the next motor
    """

    print("--- Calibration started ---")
    base_step = np.pi * D_pulley / 12  # [mm]
    max_multiplier = 10
    accel_threshold = 0.4  # seconds between presses for acceleration

    for idx, motor_id in enumerate(MOTOR_IDs):
        print(f"\n Motor {motor_id} selected. Press 'h' to skip, 'w' to loosen, 's' to tend.")
        last_key = None
        last_time = time.time()
        step_multiplier = 1

        while True:
            key = get_key()
            now = time.time()

            # Reset multiplier if too slow or switched key
            if key != last_key or (now - last_time) > accel_threshold:
                step_multiplier = 1
            else:
                step_multiplier = min(step_multiplier + 1, max_multiplier)

            last_time = now
            last_key = key

            if key == 'h':
                break
            elif key in ['w', 's']:
                d_DEL = np.zeros(MOTOR_Num, dtype=np.float32)
                direction = 1 if key == 'w' else -1
                step = direction * base_step * step_multiplier
                d_DEL[idx] = step
                move_motors(d_DEL)
            else:
                print("\n Invalid input. Use 'h', 'w', or 's'.")

    print("\n --- Calibration completed ---")

# Mapping from mocap to robot frame:
# x -> x, y -> -z, z -> y
R_corr = np.array([
    [1, 0,  0],
    [0, 0, -1],
    [0, 1,  0]
])

# Relative coordinates functions (for all tips and backbones)
def relative_coordinates(x_tip, y_tip, z_tip, yaw_tip, pitch_tip, roll_tip,
                         x_base, y_base, z_base, yaw_base, pitch_base, roll_base,
                         R_corr=R_corr):
    """
    Compute the position and orientation of the tip relative to the base,
    both given in mocap global frame. Output is in the robot frame.
    """

    # Absolute positions
    p_tip_global = np.array([x_tip, y_tip, z_tip])
    p_base_global = np.array([x_base, y_base, z_base])

    # Position difference in mocap frame
    p_rel_global = p_tip_global - p_base_global

    # Apply mocap → robot frame
    p_rel_robot = R_corr @ p_rel_global

    # Rotation matrices from mocap (ZYX)
    R_tip_global = R.from_euler('zyx', [yaw_tip, pitch_tip, roll_tip]).as_matrix()
    R_base_global = R.from_euler('zyx', [yaw_base, pitch_base, roll_base]).as_matrix()

    # Rotation from base to tip
    R_rel = R_base_global.T @ R_tip_global

    # Convert to ZYX Euler angles
    euler_rel = R.from_matrix(R_rel).as_euler('zyx')

    return np.concatenate((p_rel_robot * 1000, euler_rel))  # mm + radians

# Relative point function
def relative_point(points, base, R_corr=R_corr):
    """
    Parameters:
        points: np.ndarray of shape (N, 3)
            Array of 3D points in global mocap frame.
        base: np.ndarray of shape (3,)
            Base point in global mocap frame.
        R_corr: np.ndarray of shape (3, 3)
            Rotation matrix from mocap frame to robot frame.

    Returns:
        np.ndarray of shape (N, 3)
            Points expressed in robot frame (in mm).
    """

    rel_global = points - base  # shape (N, 3)
    rel_robot = rel_global @ R_corr.T  # shape (N, 3)
    return rel_robot * 1000  # convert to mm

# Sampling function
def SamplingSolver(delta_old, deltaL_old, delta_new):
    """
    Solve the linear system:
    [cos(s1)  sin(s1)  -1] [Dx]   [- delta_l1]
    [cos(s2)  sin(s2)  -1] [Dy] = [- delta_l2]
    [cos(s3)  sin(s3)  -1] [De]   [- delta_l3]
    
    Inputs:
    - delta_old: array of 3 angles [s1, s2, s3] in radians (old config.)
    - delta_new: array of 3 angles [d1, d2, d3] in radians (new config.)
    - deltaL_old: array of 3 values [delta_l1, delta_l2, delta_l3] (old config.)
    Outputs:
    - deltaL_new: array [delta_l1, delta_l2, delta_l3] (new config.)
    """

    s1, s2, s3 = delta_old
    d1, d2, d3 = delta_new
    
    # Build coefficient matrix A_old
    A_old = np.array([
        [np.cos(s1), np.sin(s1), -1],
        [np.cos(s2), np.sin(s2), -1],
        [np.cos(s3), np.sin(s3), -1]
    ])
    
    # Right-hand side vector b
    b = - deltaL_old
    
    # Solve A_old x = b
    solution = np.linalg.solve(A_old, b)

    # Build new matrix
    A_new = np.array([
        [np.cos(d1), np.sin(d1), -1],
        [np.cos(d2), np.sin(d2), -1],
        [np.cos(d3), np.sin(d3), -1]
    ])

    # Write A_new x = b
    deltaL_new = - A_new @ solution
    
    return deltaL_new

# ======================================================================================
#                                   Dynamixel Setting
# ======================================================================================

DEVICENAME      = '/dev/ttyUSB0'
BAUDRATE        = 57600
PROTOCOL_VERSION = 2.0
EXPOSITION_MODE = 4			# position + current (to be changed)

# Motors according to how they are placed
MOTOR_IDs = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19])	# 9 motors (as in the setup)
MOTOR_Num = MOTOR_IDs.shape[0]	# number of motors

D_pulley = 6				# [mm]
unit_scale = 4096/(np.pi*D_pulley)	# motor position/DELTAL

# ======================================================================================
#                                   Dynamixel Setup
# ======================================================================================

# Initialize PortHandler
port_handler = PortHandler(DEVICENAME)

# Initialize PacketHandler
packet_handler = Protocol2PacketHandler()

# Open port
if not port_handler.openPort():
    print("Failed to open port")
    sys.exit(1)
print("Port opened successfully")

# Set baudrate
if not port_handler.setBaudRate(BAUDRATE):
    print("Failed to set baudrate")
    sys.exit(1)
print("Baudrate set successfully")

# Define Dynamixel control table addresses
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
ADDR_PROFILE_VELOCITY = 112

# Set operating mode for all motors
for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_OPERATING_MODE, EXPOSITION_MODE, size=1)

# Enable torque for all motors
for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_TORQUE_ENABLE, 1, size=1)

VELOCITY_LIMIT = 30     # velocity limitation

for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_PROFILE_VELOCITY, VELOCITY_LIMIT, size=4)

def read_DELTAL_from_motors():
    def read_position_float(motor_id):
        raw, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        signed = np.array(raw, dtype=np.uint32).view(np.int32)
        return float(signed)

    positions = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)
    DELTAL = positions / unit_scale
    return DELTAL

print("--- Reference positions set to zero ---")

# ======================================================================================
#                                Calibrate The Motors
# ======================================================================================

calibrate_motors()

# ======================================================================================
#                                   Motion Capture Setup
# ======================================================================================

# Initialize ROS node
rospy.init_node('optitrack_listener')

# Dictionaries to store positions and angles informations
positions = {}
quaternions = {}
angles = {}

# Define the TOPICS to subscribe to:
# - Base (static):  BASE
# - Tips (dynamic): TIP1, TIP2, TIP3
# Define the TOPICS to subscribe to:
helyx_topics = [
    ('/optitrack/BASE/pose', 'BASE'),
    ('/optitrack/TIP1/pose', 'TIP1'),
    ('/optitrack/TIP2/pose', 'TIP2'),
    ('/optitrack/TIP3/pose', 'TIP3')
]

# Create subscribers for the topics
subscribers = []
for topic_name, key in helyx_topics:
    sub = rospy.Subscriber(topic_name, PoseStamped, pose_callback, callback_args=key)   # use pose_callback
    subscribers.append(sub)

# Allow messages to start arriving
time.sleep(1)

# Initial conditions (rest position)
DELTAL0 = np.full(9, 1e-4)
DxDyDl0 = np.full(9, 1e-4)
coords0 = np.array([0, 0, 145 * 3, 1e-4, 1e-4, 1e-4])
xyz0 = np.array([1e-4, 1e-4, 145, 1e-4, 1e-4, 145 * 2, 1e-4, 1e-4, 145 * 3])

# ======================================================================================
#                                   Camera Reading
# ======================================================================================

cams = probe_and_assign_cameras()
first_camera_N = cams.get("camera_1")
second_camera_N = cams.get("camera_2")
third_camera_N = cams.get("camera_3")
camera_ids = [first_camera_N, second_camera_N, third_camera_N]

# Create camera system
camera_system = ThreadedCameraSystem(camera_ids)
camera_system.initialize_cameras()
camera_system.start_capture_threads()

# Display cameras for 10 seconds
if False:
    display_cameras(camera_system, duration=10)

time.sleep(2)

# ======================================================================================
#                               Samples Cycles
# ======================================================================================

# Calibrate motors
print("--- Press 'y' to calibrate the motors, ENTER to go on ---")
key = get_key()
if key == 'y':
    calibrate_motors()
    print("\n")

# Sampling movements (datasets)
diameter = 60       # tuned base diameter [mm]
d = diameter / 2    # constant section
L0 = 145*3      # rest length [mm]

# Print initial position
def get_mocap_pose(name):
    """
    Generalized pose acquisition.
    """

    x_tip, y_tip, z_tip = positions.get(name)
    yaw_tip, pitch_tip, roll_tip = angles.get(name)
    return x_tip, y_tip, z_tip, yaw_tip, pitch_tip, roll_tip

# Acquire
x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1 = get_mocap_pose("TIP1")
x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2 = get_mocap_pose("TIP2")
x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")

# Relative coordinates
mocap1 = relative_coordinates(x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1,
                              x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
mocap2 = relative_coordinates(x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2,
                              x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
mocap3 = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                              x_base, y_base, z_base, yaw_base, pitch_base, roll_base)

# Define initial position
xyz_initial = np.concatenate([mocap1[:3], mocap2[:3], mocap3[:3]])
DELTAL_initial, _, _ = xyz2tendon(DELTAL0, DxDyDl0, xyz0, xyz_initial - xyz0)
DELTAL_reference = read_DELTAL_from_motors()

print(f"DELTAL_initial = {DELTAL_initial}")
print(f"Initial xyz = {xyz_initial}")

time.sleep(1)

START_BATCH = 1
NUM_BATCHES = 100
BATCH_SIZE = 20
TOTAL_SAMPLES = NUM_BATCHES * BATCH_SIZE

print("--- Acquisition started ---")

# Curvature model
curvature_model_quad = SPCModel(modeAnn='quadratic')
curvature_model_affine = SPCModel(modeAnn='affine')
curvature_model_const = SPCModel(modeAnn='constant')

# Function to iterate over the batch
@dataclass
class AcquisitionSample:
    timestamp: float
    mocap1: np.ndarray
    mocap2: np.ndarray  
    mocap3: np.ndarray
    gray_1: np.ndarray
    gray_2: np.ndarray
    gray_3: np.ndarray
    DELTAL1: np.ndarray
    quadratic: np.ndarray
    affine: np.ndarray
    constant: np.ndarray

class SimpleDataAcquisition:
    def __init__(self, frequency_hz=1.0):
        self.frequency_hz = frequency_hz
        self.interval = 1.0 / frequency_hz
        self.running = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.acquisition_thread = None
    
    def acquire_sample(self):
        try:
            # Get mocap poses
            x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1 = get_mocap_pose("TIP1")
            x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2 = get_mocap_pose("TIP2")
            x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
            x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")

            # Relative coordinates
            mocap1 = relative_coordinates(x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1,
                                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
            mocap2 = relative_coordinates(x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2,
                                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
            mocap3 = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)

            # Cameras
            frames = camera_system.get_all_frames()
            if frames[0] is None or frames[1] is None or frames[2] is None:
                raise RuntimeError("Camera frame(s) missing - acquisition failed!")
            
            gray_1 = color2gray(frames[0])
            gray_2 = color2gray(frames[1])
            gray_3 = color2gray(frames[2])

            # Motor positions
            DELTAL1 = read_DELTAL_from_motors() + DELTAL_initial - DELTAL_reference

            # Curvature parameters
            s1_quad, c01_quad, c11_quad, c21_quad, phi1_quad, \
            s2_quad, c02_quad, c12_quad, c22_quad, phi2_quad, \
            s3_quad, c03_quad, c13_quad, c23_quad, phi3_quad = curvature_model_quad.ComputeANNParameters(
                mocap1=mocap1, mocap2=mocap2, mocap3=mocap3)

            s1_affine, c01_affine, c11_affine, c21_affine, phi1_affine, \
            s2_affine, c02_affine, c12_affine, c22_affine, phi2_affine, \
            s3_affine, c03_affine, c13_affine, c23_affine, phi3_affine = curvature_model_affine.ComputeANNParameters(
                mocap1=mocap1, mocap2=mocap2, mocap3=mocap3)

            s1_constant, c01_constant, c11_constant, c21_constant, phi1_constant, \
            s2_constant, c02_constant, c12_constant, c22_constant, phi2_constant, \
            s3_constant, c03_constant, c13_constant, c23_constant, phi3_constant = curvature_model_const.ComputeANNParameters(
                mocap1=mocap1, mocap2=mocap2, mocap3=mocap3)

            quadratic = np.array([
                s1_quad, c01_quad, c11_quad, c21_quad, phi1_quad,
                s2_quad, c02_quad, c12_quad, c22_quad, phi2_quad,
                s3_quad, c03_quad, c13_quad, c23_quad, phi3_quad
            ])

            affine = np.array([
                s1_affine, c01_affine, c11_affine, phi1_affine,
                s2_affine, c02_affine, c12_affine, phi2_affine,
                s3_affine, c03_affine, c13_affine, phi3_affine
            ])

            constant = np.array([
                s1_constant, c01_constant, phi1_constant,
                s2_constant, c02_constant, phi2_constant,
                s3_constant, c03_constant, phi3_constant
            ])

            return AcquisitionSample(
                timestamp=time.time(),
                mocap1=mocap1,
                mocap2=mocap2,
                mocap3=mocap3,
                gray_1=gray_1,
                gray_2=gray_2,
                gray_3=gray_3,
                DELTAL1=DELTAL1,
                quadratic=quadratic,
                affine=affine,
                constant=constant
            )
        except:
            raise RuntimeError("Acquisition failed!")
    
    def acquisition_loop(self):
        while self.running:
            start_time = time.time()
            sample = self.acquire_sample()
            
            if sample:
                try:
                    self.data_queue.put_nowait(sample)
                except queue.Full:
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put_nowait(sample)
                    except queue.Empty:
                        pass
            
            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed)
            time.sleep(sleep_time)
    
    def start(self):
        self.running = True
        self.acquisition_thread = threading.Thread(target=self.acquisition_loop, daemon=True)
        self.acquisition_thread.start()
    
    def stop(self):
        self.running = False
        if self.acquisition_thread:
            self.acquisition_thread.join(timeout=2.0)
    
    def get_samples(self, count=None):
        samples = []
        target_count = count if count is not None else float('inf')
        while len(samples) < target_count and not self.data_queue.empty():
            try:
                samples.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return samples

# Replace your batch loop with this:
acquisition = SimpleDataAcquisition(frequency_hz=1.0)
acquisition.start()

batch = START_BATCH
while batch < START_BATCH + NUM_BATCHES:
    print(f"Collecting dataset {batch}...")

    for i in range(BATCH_SIZE):
        idx = (batch - START_BATCH) * BATCH_SIZE + i + 1
        
        # Your existing motor configuration logic
        config = np.random.randint(0, 3)
        delta1 = [np.radians(0), np.radians(120), np.radians(-120)]
        delta2 = [np.radians(60), np.radians(120+60), np.radians(-120+60)]
        delta3 = [np.radians(150), np.radians(120+150), np.radians(-120+150)]

        if config==0:
            DELTAL_first = np.random.uniform(low=-25, high=10, size=3)
            DELTAL_first_2 = SamplingSolver(delta1, DELTAL_first, delta2)
            DELTAL_second = DELTAL_first_2
            DELTAL_first_3 = SamplingSolver(delta1, DELTAL_first, delta3)
            DELTAL_third = DELTAL_first_3
        elif config==1:
            DELTAL_first = np.random.uniform(low=-25, high=10, size=3)
            DELTAL_first_2 = SamplingSolver(delta1, DELTAL_first, delta2)
            DELTAL_second = DELTAL_first_2
            DELTAL_third = np.random.uniform(low=-35, high=10, size=3)
        elif config==2:
            DELTAL_first = np.random.uniform(low=-25, high=10, size=3)
            DELTAL_second = np.random.uniform(low=-35, high=10, size=3)
            DELTAL_third = np.random.uniform(low=-40, high=10, size=3)

        DELTAL_sample = np.concatenate([DELTAL_first, DELTAL_second, DELTAL_third])
        d_DELTAL = DELTAL_sample - (read_DELTAL_from_motors() - DELTAL_reference)
        
        move_motors(d_DELTAL)
        print(f"Datum {idx}")
        time.sleep(3)

    # Get 20 samples from acquisition
    samples = acquisition.get_samples()
    
    if len(samples) < BATCH_SIZE:
        print(f"Warning: Only got {len(samples)} samples")
    
    # Convert to your format
    tip_1 = [s.mocap1 for s in samples]
    tip_2 = [s.mocap2 for s in samples] 
    tip_3 = [s.mocap3 for s in samples]
    X_images_1 = [s.gray_1 for s in samples]
    X_images_2 = [s.gray_2 for s in samples]
    X_images_3 = [s.gray_3 for s in samples]
    X_DELTAL = [s.DELTAL1 for s in samples]
    X_quad = [s.quadratic for s in samples]
    X_affine = [s.affine for s in samples]
    X_const = [s.constant for s in samples]

    flag = input("Correct batch?\n")
    if flag!='n':
        # Your existing save code
        tip_1_np = np.array(tip_1, dtype=np.float32)
        tip_2_np = np.array(tip_2, dtype=np.float32)
        tip_3_np = np.array(tip_3, dtype=np.float32)
        X_DELTAL_np = np.array(X_DELTAL, dtype=np.float32)
        X_quad_np = np.array(X_quad, dtype=np.float32)
        X_affine_np = np.array(X_affine, dtype=np.float32)
        X_const_np = np.array(X_const, dtype=np.float32)

        X_images_np = np.stack([
            np.stack(X_images_1, axis=0),
            np.stack(X_images_2, axis=0),
            np.stack(X_images_3, axis=0)
        ], axis=-1)

        X_images_torch = torch.from_numpy(X_images_np).permute(0, 3, 1, 2).contiguous()
        tip_1_torch = [torch.tensor(v, dtype=torch.float32) for v in tip_1_np]
        tip_2_torch = [torch.tensor(v, dtype=torch.float32) for v in tip_2_np]
        tip_3_torch = [torch.tensor(v, dtype=torch.float32) for v in tip_3_np]
        X_DELTAL_torch = [torch.tensor(v, dtype=torch.float32) for v in X_DELTAL_np]
        X_quad_torch = [torch.tensor(v, dtype=torch.float32) for v in X_quad_np]
        X_affine_torch = [torch.tensor(v, dtype=torch.float32) for v in X_affine_np]
        X_const_torch = [torch.tensor(v, dtype=torch.float32) for v in X_const_np]

        data_dict = {
            "X_images": X_images_torch,
            "X_DELTAL": X_DELTAL_torch,
            "X_quad": X_quad_torch,
            "X_affine": X_affine_torch,
            "X_const": X_const_torch,
            "tip_1": tip_1_torch,
            "tip_2": tip_2_torch,
            "tip_3": tip_3_torch
        }

        os.makedirs("data", exist_ok=True)
        torch.save(data_dict, f"data/DatasetMulti{batch}.pt")
        print(f"[OK] Saved: data/DatasetMulti{batch}.pt ({len(tip_1)} samples)")
        batch += 1

acquisition.stop()

# =======================================================================================
#                              Close the Camera
# =======================================================================================

# Closing with cleanup function
camera_system.stop_capture()