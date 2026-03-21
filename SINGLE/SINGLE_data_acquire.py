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

# User-defined libraries
from SINGLE_jacobians import *
from SINGLE_kinematics import *
from SINGLE_reconstruction_utility import *

# At the beginning, for the optitrack:
# roslaunch optitrack_ros_communication optitrack_nodes.launch

# For the cameras:
# v4l2-ctl --list-devices

# ======================================================================================
#                                    Functions
# ======================================================================================

# Write register
def write_register(motor_id, address, value, size=1):
    """ Writes a single register """
    if size == 1:
        packet_handler.write1ByteTxRx(port_handler, motor_id, address, value)
    elif size == 2:
        packet_handler.write2ByteTxRx(port_handler, motor_id, address, value)
    elif size == 4:
        packet_handler.write4ByteTxRx(port_handler, motor_id, address, value)

def get_key():
    """Read a single character without requiring Enter."""
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
    """Callback function to process messages from each topic"""
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

# Relative coordinates function
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

# Relative point function (to be used without quaternions)
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

# ======================================================================================
#                                   Dynamixel Setting
# ======================================================================================

DEVICENAME      = '/dev/ttyUSB0'
BAUDRATE        = 57600
PROTOCOL_VERSION = 2.0
EXPOSITION_MODE = 4			# position + current (to be changed)

MOTOR_IDs = np.array([1, 2, 3])	# 3 motors (as in the setup)
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

VELOCITY_LIMIT = 60     # velocity limitation

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
# - Base (static): BASE
# - Tip (dynamic): TIP
# - Backbone point 1: BACKBONE1
# - Backbone point 2: BACKBONE2
# Define the TOPICS to subscribe to:
helyx_topics = [
    ('/optitrack/BASE/pose', 'BASE'),
    ('/optitrack/TIP/pose', 'TIP'),
    ('/optitrack/BACKBONE1/pose', 'BACKBONE1'),
    ('/optitrack/BACKBONE2/pose', 'BACKBONE2')
]

# Create subscribers for the topics
subscribers = []
for topic_name, key in helyx_topics:
    sub = rospy.Subscriber(topic_name, PoseStamped, pose_callback, callback_args=key)   # use pose_callback
    subscribers.append(sub)

# Allow messages to start arriving
time.sleep(1)
    
# ======================================================================================
#                           Get to Wanted Configuration (e.g.)
# ======================================================================================

# Initial conditions (rest position)
DELTAL0 = np.array([1e-4, 1e-4, 1e-4])
DxDyDl0 = np.array([1e-4, 1e-4, 1e-4])
coords0 = np.array([1e-4, 1e-4, 145, 1e-4, 1e-4, 1e-4])

# Decide if it is needed to try (default no)
tryFlag = False

if tryFlag:
    # Ask user for coordinate variations (deltax, deltay, deltaz)
    dx = float(input("Enter deltax [mm]: "))
    dy = float(input("Enter deltay [mm]: "))
    dz = float(input("Enter deltaz [mm]: "))
    d_xyz = np.array([dx, dy, dz])

    # Call coordinates2tendon to compute new DELTAL
    DELTAL, DxDyDl, coords = coordinates2tendon(DELTAL0, DxDyDl0, coords0, d_xyz)

    # Print results
    print("\nComputed DELTAL values:")
    print(DELTAL)

    # Application of move_motors()
    move_motors(DELTAL - DELTAL0)
    time.sleep(1)

# ======================================================================================
#                                   Camera Reading
# ======================================================================================

# Use v4l2-ctl --list-devices for the usb camera
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

import threading

latest_frame = None
lock = threading.Lock()

def update_camera():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame

camera_thread = threading.Thread(target=update_camera, daemon=True)
camera_thread.start()

time.sleep(2)

def capture_image():
    with lock:
        return latest_frame.copy() if latest_frame is not None else None

time.sleep(3)

# ======================================================================================
#                               Samples Cycles
# ======================================================================================

# Calibrate motors
print("--- Press 'y' to calibrate the motors, ENTER to go on ---")
key = get_key()
if key == 'y':
  calibrate_motors()
  print("\n")

diameter = 60  # tuned base diameter [mm]
d = diameter / 2  # constant section
L0 = 145  # rest length [mm]

# Print initial position
x_tip, y_tip, z_tip = positions.get('TIP')
yaw_tip, pitch_tip, roll_tip = angles.get('TIP')
x_base, y_base, z_base = positions.get('BASE')
yaw_base, pitch_base, roll_base = angles.get('BASE')
mocap_xyzZYX = relative_coordinates(x_tip,y_tip,z_tip,yaw_tip,pitch_tip,roll_tip,x_base,y_base,z_base,yaw_base,pitch_base,roll_base)

print(f"Initial position = {mocap_xyzZYX}")

# Define initial position
DELTAL_initial, _, _ = coordinates2tendon(DELTAL0, DxDyDl0, coords0, mocap_xyzZYX[:3] - coords0[:3])
DELTAL_reference = read_DELTAL_from_motors()

print(f"DELTAL_initial = {DELTAL_initial}")
print(f"DELTAL_reference = {DELTAL_reference}")

time.sleep(1)

TOTAL_SAMPLES = 10000
BATCH_SIZE = 1000
NUM_BATCHES = TOTAL_SAMPLES // BATCH_SIZE

# Models for curvature
curvature_model = SPCModel()
polynomial_model = SPRegModel()

print("--- Acquisition started ---")

for batch in range(NUM_BATCHES):

    y = []
    backbones = []
    X_images = []
    X_DELTAL = []
    X_quad = []
    X_affine = []
    X_const = []
    X_poly = []

    for i in range(BATCH_SIZE):
        idx = batch * BATCH_SIZE + i + 1

        # Final DELTAL_sample
        DELTAL_sample = np.random.uniform(low=-45, high=10, size=3)

        # Variation with respect to the actual one
        d_DELTAL = DELTAL_sample - (read_DELTAL_from_motors() - DELTAL_reference) 
        d_DELTAL_real = move_motors(d_DELTAL)

        # Evaluate the value after
        DELTAL1 = read_DELTAL_from_motors() + DELTAL_initial - DELTAL_reference

        print(f"Datum {idx}")
        print(f"d_DELTAL = {d_DELTAL_real}")

        time.sleep(0.1)

        # Camera and mocap
        x_tip, y_tip, z_tip = positions.get('TIP')
        yaw_tip, pitch_tip, roll_tip = angles.get('TIP')
        x_base, y_base, z_base = positions.get('BASE')
        yaw_base, pitch_base, roll_base = angles.get('BASE')
        x_bb1, y_bb1, z_bb1 = positions.get('BACKBONE1')
        yaw_bb1, pitch_bb1, roll_bb1 = angles.get('BACKBONE1')
        x_bb2, y_bb2, z_bb2 = positions.get('BACKBONE2')
        yaw_bb2, pitch_bb2, roll_bb2 = angles.get('BACKBONE2')
        image_dataset = capture_image()

        mocap_xyzZYX = relative_coordinates(
            x_tip, y_tip, z_tip, yaw_tip, pitch_tip, roll_tip,
            x_base, y_base, z_base, yaw_base, pitch_base, roll_base
        )
        backbone_1 = relative_coordinates(
            x_bb1, y_bb1, z_bb1, yaw_bb1, pitch_bb1, roll_bb1,
            x_base, y_base, z_base, yaw_base, pitch_base, roll_base            
        )
        backbone_2 = relative_coordinates(
            x_bb2, y_bb2, z_bb2, yaw_bb2, pitch_bb2, roll_bb2,
            x_base, y_base, z_base, yaw_base, pitch_base, roll_base            
        )
        s = L0 + np.mean(DELTAL1)
        backbone = np.concatenate([backbone_1, backbone_2, mocap_xyzZYX, np.array([s])])

        # Points, to be given as (4, 3) array; angles, as (yaw, pitch, roll)
        points = np.vstack([
            np.zeros(3),
            backbone_1[:3],
            backbone_2[:3],
            mocap_xyzZYX[:3]
        ])
        angles_i = np.array([backbone[15], backbone[16], backbone[17]])

        # Building rotation matrix from (yaw, pitch, roll)
        yaw1, pitch1, roll1 = backbone[3], backbone[4], backbone[5]
        yaw2, pitch2, roll2 = backbone[9], backbone[10], backbone[11]
        yaw3, pitch3, roll3 = backbone[15], backbone[16], backbone[17]
        s = backbone[-1]
        R1 = R.from_euler('zyx', [yaw1, pitch1, roll1]).as_matrix()
        R2 = R.from_euler('zyx', [yaw2, pitch2, roll2]).as_matrix()
        R3 = R.from_euler('zyx', [yaw3, pitch3, roll3]).as_matrix()

        # Reconstruction from curvature and polynomial approximations
        c0_quad, c1_quad, c2_quad, phi_quad = curvature_model.curvature_reconstruction(
            S=s,
            Rbase_1=R1,
            Rbase_2=R2,
            Rbase_3=R3,
            mode='quadratic'
        )
        c0_affine, c1_affine, c2_affine, phi_affine = curvature_model.curvature_reconstruction(
            S=s,
            Rbase_1=R1,
            Rbase_2=R2,
            Rbase_3=R3,
            mode='affine'
        )
        c0_constant, c1_constant, c2_constant, phi_constant = curvature_model.curvature_reconstruction(
            S=s,
            Rbase_1=R1,
            Rbase_2=R2,
            Rbase_3=R3,
            mode='constant'
        )
        theta, H, A = polynomial_model.curvature(points, angles_i)
        
        # Function output
        quadratic = np.array([s, c0_quad, c1_quad, c2_quad, phi_quad])
        affine = np.array([s, c0_affine, c1_affine, phi_affine])
        constant = np.array([s, c0_constant, phi_constant])
        poly = np.concatenate((
            np.array([H], dtype=np.float32),
            np.array(A, dtype=np.float32),
            np.array([theta], dtype=np.float32)
        ))

        # Getting BW image
        if image_dataset is not None:
            image_gray = cv2.cvtColor(image_dataset, cv2.COLOR_BGR2GRAY)
            # Show image
            cv2.imshow("Captured Gray Image", image_gray)
            key = cv2.waitKey(1)  # 1 millisecond: does not shut down program
        else:
            print("[ERROR] No image captured.")
            continue

        y.append(mocap_xyzZYX)
        X_images.append(image_gray)
        X_DELTAL.append(DELTAL1)
        backbones.append(backbone)
        
        # Reconstruction
        X_quad.append(quadratic)
        X_affine.append(affine)
        X_const.append(constant)
        X_poly.append(poly)

        time.sleep(0.1) # Small delay to avoid too fast acquisition

    # Save batch
    y_np = np.array(y, dtype=np.float32)
    backbones_np = np.array(backbones, dtype=np.float32)
    X_images_np = np.array(X_images, dtype=np.uint8)
    X_DELTAL_np = np.array(X_DELTAL, dtype=np.float32)
    X_quad_np = np.array(X_quad, dtype = np.float32)
    X_affine_np = np.array(X_affine, dtype = np.float32)
    X_const_np = np.array(X_const, dtype = np.float32)
    X_poly_np = np.array(X_poly, dtype = np.float32)

    X_images_torch = [torch.from_numpy(img).unsqueeze(0).contiguous() for img in X_images_np]
    y_torch = [torch.tensor(v, dtype=torch.float32) for v in y_np]
    backbones_torch = [torch.tensor(v, dtype=torch.float32) for v in backbones_np]
    X_DELTAL_torch = [torch.tensor(v, dtype=torch.float32) for v in X_DELTAL_np]
    X_quad_torch = [torch.tensor(v, dtype=torch.float32) for v in X_quad_np]
    X_affine_torch = [torch.tensor(v, dtype=torch.float32) for v in X_affine_np]
    X_const_torch = [torch.tensor(v, dtype=torch.float32) for v in X_const_np]
    X_poly_torch = [torch.tensor(v, dtype=torch.float32) for v in X_poly_np]

    data_dict = {
        "X_images": X_images_torch,
        "y": y_torch,
        "X_DELTAL": X_DELTAL_torch,
        "backbones": backbones_torch,
        "X_quad": X_quad_torch,
        "X_affine": X_affine_torch,
        "X_const": X_const_torch,
        "X_poly": X_poly_torch
    }

    os.makedirs("data", exist_ok=True)
    torch.save(data_dict, f"data/Dataset{batch}.pt")
    print(f"[OK] Saved: data/Dataset{batch}.pt ({len(y)} samples)")

# =======================================================================================
#                              Close the Camera
# =======================================================================================

# Closing
cap.release()
cv2.destroyAllWindows()