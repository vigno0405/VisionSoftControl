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
def calibrate_motors(fast=False):
    """
    Interactive calibration for each motor.
    Press:
      - 'w' to move forward (with adaptive increment)
      - 's' to move backward (with adaptive increment)
      - 'h' to confirm and go to the next motor
    """

    print("--- Calibration started ---")
    if fast:
        base_step = np.pi * D_pulley / 4
        max_multiplier = 10
        accel_threshold = 0.4
    else:
        base_step = np.pi * D_pulley / 12
        max_multiplier = 10                # Original max acceleration  
        accel_threshold = 0.4              # Original acceleration threshold

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
helyx_topics = [
    ('/optitrack/BASE/pose', 'BASE'),
    ('/optitrack/TIP1/pose', 'TIP1'),
    ('/optitrack/TIP2/pose', 'TIP2'),
    ('/optitrack/TIP3/pose', 'TIP3'),
    ('/optitrack/BACKBONE11/pose', 'BACKBONE11'),
    ('/optitrack/BACKBONE12/pose', 'BACKBONE12'),
    ('/optitrack/BACKBONE21/pose', 'BACKBONE21'),
    ('/optitrack/BACKBONE22/pose', 'BACKBONE22'),
    ('/optitrack/BACKBONE31/pose', 'BACKBONE31'),
    ('/optitrack/BACKBONE32/pose', 'BACKBONE32')
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

def get_mocap_data():
    """
    Extract all mocap poses and compute relative coordinates and rotation matrices.
    
    Returns:
        tuple: (mocap_coords, rotation_matrices)
            - mocap_coords: dict with keys 'tip1', 'tip2', 'tip3', 'backbone11', etc.
            - rotation_matrices: dict with keys 'R1', 'R2', 'R3', 'R11', etc.
    """
    
    # Get base pose
    x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")
    
    # Get all other poses
    poses = {}
    for name in ['TIP1', 'TIP2', 'TIP3', 'BACKBONE11', 'BACKBONE12', 
                 'BACKBONE21', 'BACKBONE22', 'BACKBONE31', 'BACKBONE32']:
        poses[name] = get_mocap_pose(name)
    
    # Compute relative coordinates
    mocap_coords = {}
    for name, pose in poses.items():
        x, y, z, yaw, pitch, roll = pose
        mocap_coords[name.lower()] = relative_coordinates(
            x, y, z, yaw, pitch, roll,
            x_base, y_base, z_base, yaw_base, pitch_base, roll_base
        )
    
    # Compute rotation matrices
    rotation_matrices = {}
    name_mapping = {
        'tip1': 'R1', 'tip2': 'R2', 'tip3': 'R3',
        'backbone11': 'R11', 'backbone12': 'R12',
        'backbone21': 'R21', 'backbone22': 'R22',
        'backbone31': 'R31', 'backbone32': 'R32'
    }
    
    for mocap_name, r_name in name_mapping.items():
        rotation_matrices[r_name] = R.from_euler('zyx', mocap_coords[mocap_name][3:]).as_matrix()
    
    return mocap_coords, rotation_matrices

# Load existing data if available
os.makedirs("data", exist_ok=True)
save_path = "data/CurvatureDataCheck.npz"
if os.path.exists(save_path):
    data = np.load(save_path)
    quad_mocap = list(data["quad_mocap"])
    quad_ANN = list(data["quad_ANN"])
    aff_mocap = list(data["aff_mocap"])
    aff_ANN = list(data["aff_ANN"])
    const_mocap = list(data["const_mocap"])
    const_ANN = list(data["const_ANN"])
else:
    quad_mocap = []
    quad_ANN = []
    aff_mocap = []
    aff_ANN = []
    const_mocap = []
    const_ANN = []

print("--- Acquisition started ---")

# Define DELTAL
mocap_coords, rotation_matrices = get_mocap_data()
mocap1 = mocap_coords['tip1']
mocap2 = mocap_coords['tip2']
mocap3 = mocap_coords['tip3']

xyz_initial = np.concatenate([mocap1[:3], mocap2[:3], mocap3[:3]])
DELTAL_initial, _, _ = xyz2tendon(DELTAL0, DxDyDl0, xyz0, xyz_initial - xyz0)
DELTAL_reference = read_DELTAL_from_motors()

time.sleep(1)

# Curvature models
curvature_model_quad = SPCModel(modeAnn='quadratic')
curvature_model_affine = SPCModel(modeAnn='affine')
curvature_model_const = SPCModel(modeAnn='constant')

while len(quad_mocap) < 20:

    # Move motors arbitrarily
    print("Move as needed!")
    calibrate_motors(fast=True)

    # Acquire data if correct
    mocap_coords, rotation_matrices = get_mocap_data()
    mocap1 = mocap_coords['tip1']
    mocap2 = mocap_coords['tip2']
    mocap3 = mocap_coords['tip3']
    
    # Extract rotation matrices
    R1 = rotation_matrices['R1']
    R2 = rotation_matrices['R2']
    R3 = rotation_matrices['R3']
    R11 = rotation_matrices['R11']
    R12 = rotation_matrices['R12']
    R21 = rotation_matrices['R21']
    R22 = rotation_matrices['R22']
    R31 = rotation_matrices['R31']
    R32 = rotation_matrices['R32']

    # Motor positions
    DELTAL = read_DELTAL_from_motors() + DELTAL_initial - DELTAL_reference

    # Curvature parameters
    s1_quad, c01_quad, c11_quad, c21_quad, phi1_quad, \
    s2_quad, c02_quad, c12_quad, c22_quad, phi2_quad, \
    s3_quad, c03_quad, c13_quad, c23_quad, phi3_quad = curvature_model_quad.ComputeANNParameters(
            mocap1=mocap1, mocap2=mocap2, mocap3=mocap3)

    s1_affine, c01_affine, c11_affine, _, phi1_affine, \
    s2_affine, c02_affine, c12_affine, _, phi2_affine, \
    s3_affine, c03_affine, c13_affine, _, phi3_affine = curvature_model_affine.ComputeANNParameters(
            mocap1=mocap1, mocap2=mocap2, mocap3=mocap3)

    s1_constant, c01_constant, _, _, phi1_constant, \
    s2_constant, c02_constant, _, _, phi2_constant, \
    s3_constant, c03_constant, _, _, phi3_constant = curvature_model_const.ComputeANNParameters(
            mocap1=mocap1, mocap2=mocap2, mocap3=mocap3)
    
    quadratic_ANN = np.array([
                s1_quad, c01_quad, c11_quad, c21_quad, phi1_quad,
                s2_quad, c02_quad, c12_quad, c22_quad, phi2_quad,
                s3_quad, c03_quad, c13_quad, c23_quad, phi3_quad
            ])

    affine_ANN = np.array([
                s1_affine, c01_affine, c11_affine, phi1_affine,
                s2_affine, c02_affine, c12_affine, phi2_affine,
                s3_affine, c03_affine, c13_affine, phi3_affine
            ])

    constant_ANN = np.array([
                s1_constant, c01_constant, phi1_constant,
                s2_constant, c02_constant, phi2_constant,
                s3_constant, c03_constant, phi3_constant
            ])
    
    # Convert 9-element DELTAL to 3-element S
    S = np.array([
        L0/3 + np.mean(DELTAL[:3]),   # Mean of first 3 elements
        L0/3 + np.mean(DELTAL[3:6]),  # Mean of second 3 elements  
        L0/3 + np.mean(DELTAL[6:9])   # Mean of last 3 elements
    ])

    c01_quad_mocap, c11_quad_mocap, c21_quad_mocap, phi1_quad_mocap, \
    c02_quad_mocap, c12_quad_mocap, c22_quad_mocap, phi2_quad_mocap, \
    c03_quad_mocap, c13_quad_mocap, c23_quad_mocap, phi3_quad_mocap = curvature_model_quad.ComputeParameters(S, R11, R12, R1, R21, R22, R2, R31, R32, R3, mode='quadratic')

    c01_affine_mocap, c11_affine_mocap, _, phi1_affine_mocap, \
    c02_affine_mocap, c12_affine_mocap, _, phi2_affine_mocap, \
    c03_affine_mocap, c13_affine_mocap, _, phi3_affine_mocap = curvature_model_affine.ComputeParameters(
        S, R11, R12, R1, R21, R22, R2, R31, R32, R3, mode='affine')

    c01_const_mocap, _, _, phi1_const_mocap, \
    c02_const_mocap, _, _, phi2_const_mocap, \
    c03_const_mocap, _, _, phi3_const_mocap = curvature_model_const.ComputeParameters(
        S, R11, R12, R1, R21, R22, R2, R31, R32, R3, mode='constant')

    # Ground truth parameters (mocap-based analytical reconstruction)
    quadratic_mocap = np.array([
        S[0], c01_quad_mocap, c11_quad_mocap, c21_quad_mocap, phi1_quad_mocap,
        S[1], c02_quad_mocap, c12_quad_mocap, c22_quad_mocap, phi2_quad_mocap,
        S[2], c03_quad_mocap, c13_quad_mocap, c23_quad_mocap, phi3_quad_mocap
    ])

    affine_mocap = np.array([
        S[0], c01_affine_mocap, c11_affine_mocap, phi1_affine_mocap,
        S[1], c02_affine_mocap, c12_affine_mocap, phi2_affine_mocap,
        S[2], c03_affine_mocap, c13_affine_mocap, phi3_affine_mocap
    ])

    constant_mocap = np.array([
        S[0], c01_const_mocap, phi1_const_mocap,
        S[1], c02_const_mocap, phi2_const_mocap,
        S[2], c03_const_mocap, phi3_const_mocap
    ])

    # Ask user for confirmation
    print(f"\n--- Save this configuration? (y/n) ---")
    key = get_key()
    
    if key == 'y':
        # Add to all model collections
        quad_mocap.append(quadratic_mocap)
        aff_mocap.append(affine_mocap)
        const_mocap.append(constant_mocap)
        quad_ANN.append(quadratic_ANN)
        aff_ANN.append(affine_ANN)
        const_ANN.append(constant_ANN)

        # Save all models
        np.savez(save_path, 
                quad_mocap=np.array(quad_mocap),
                aff_mocap=np.array(aff_mocap),
                const_mocap=np.array(const_mocap),
                quad_ANN=np.array(quad_ANN),
                aff_ANN=np.array(aff_ANN),
                const_ANN=np.array(const_ANN))
        
        print(f"✓ Configuration saved! Total samples: {len(quad_mocap)}")
        
    else:
        print("Configuration skipped.")
    
    print(f"Progress: {len(quad_mocap)}/20 samples collected")

print("\n--- Data Collection Finished ---")
print(f"Total samples collected: {len(quad_mocap)}")

# Quick analysis for all models
if len(quad_mocap) > 0:
    quad_mocap_array = np.array(quad_mocap)
    quad_ANN_array = np.array(quad_ANN)
    aff_mocap_array = np.array(aff_mocap)
    aff_ANN_array = np.array(aff_ANN)
    const_mocap_array = np.array(const_mocap)
    const_ANN_array = np.array(const_ANN)
    
    # Compute RMSE for each model
    quad_rmse = np.sqrt(np.mean((quad_mocap_array - quad_ANN_array)**2))
    aff_rmse = np.sqrt(np.mean((aff_mocap_array - aff_ANN_array)**2))
    const_rmse = np.sqrt(np.mean((const_mocap_array - const_ANN_array)**2))
    
    print(f"\nQuick Analysis:")
    print(f"  Quadratic RMSE: {quad_rmse:.4f}")
    print(f"  Affine RMSE: {aff_rmse:.4f}")
    print(f"  Constant RMSE: {const_rmse:.4f}")
        