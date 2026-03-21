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
from ctypes import c_uint32
import matplotlib
import matplotlib.pyplot as plt # type: ignore
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D # type: ignore
import termios
import tty
import torch # type: ignore
import threading
from matplotlib.patches import Patch
from sklearn.cluster import KMeans # type: ignore

# User-defined
from MULTI_jacobians import *
from MULTI_kinematics import *
from MULTI_cameras_utility import *

# === Scope of the code ===
JACOBIAN_MOTION = False
RANDOM_MOTION = True
POLYGON_CHECKING = False

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

# === Functions for polygons ===
def generate_polygon_vertices(n_sides, radius, z_height):
    """
    Generate vertices of a regular polygon on plane z=z_height, centered at (0,0).
    """

    angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.full_like(x, z_height)
    return np.vstack([x, y, z]).T  # shape: (n_sides, 3)

def check_vertex_validity(vertices, WS_xyz):
    """
    For each vertex, check if there is at least one workspace point in each of the 8 octants.
    
    Args:
        vertices (np.ndarray): shape (N, 3)
        WS_xyz (np.ndarray): shape (M, 3)
    
    Returns:
        np.ndarray: shape (N,), boolean array indicating if each vertex is valid.
    """

    results = []
    signs = np.array([[sx, sy, sz] for sx in [-1,1] for sy in [-1,1] for sz in [-1,1]])  # 8 combinations

    for v in vertices:
        rel = WS_xyz - v  # shape (M, 3)
        # For each octant, check if at least one point matches its sign combination
        valid = all(np.any((np.sign(rel) == s).all(axis=1)) for s in signs)
        results.append(valid)

    return np.array(results)

def plot_polygon_and_workspace(vertices, WS_xyz, height):
    """
    Plot the workspace and polygon with 3D and top views.
    Simplified version with dark blue vertices numbered.
    """
    
    fig = plt.figure(figsize=(16, 8))
    
    # Left subplot: 3D view
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(WS_xyz[:,0], WS_xyz[:,1], WS_xyz[:,2], s=1, alpha=0.3, c='steelblue')
    
    for i, v in enumerate(vertices):
        ax1.scatter(v[0], v[1], v[2], s=100, color='darkblue', marker='o', 
                   edgecolors='black', linewidth=2)
    
    # Draw polygon edges
    polygon_edges = vertices[[*range(len(vertices)), 0]]
    ax1.plot(*polygon_edges.T, color='black', linewidth=3, alpha=0.8)
    
    ax1.set_title("3D View: Polygon on Workspace", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X [mm]")
    ax1.set_ylabel("Y [mm]")
    ax1.set_zlabel("Z [mm]")
    ax1.view_init(elev=30, azim=45)
    
    # Right subplot: Top view (XY plane)
    ax2 = fig.add_subplot(122)
    ax2.scatter(WS_xyz[:,0], WS_xyz[:,1], s=1, alpha=0.4, c='steelblue')
    
    for i, v in enumerate(vertices):
        ax2.scatter(v[0], v[1], s=150, color='darkblue', marker='o', 
                   edgecolors='black', linewidth=2)
    
    # Draw polygon edges in top view
    ax2.plot(polygon_edges[:,0], polygon_edges[:,1], color='black', linewidth=3, alpha=0.8)
    
    ax2.set_title(f"Top View: Polygon at Z = {height} mm", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X [mm]")
    ax2.set_ylabel("Y [mm]")
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Simplified legend
    legend_elements = [
        Patch(facecolor='darkblue', label='Vertices'),
        Patch(facecolor='steelblue', alpha=0.4, label='Workspace')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)
    
    plt.tight_layout()
    plt.savefig(f"Polygon_{len(vertices)}_{height}.png", dpi=300, bbox_inches='tight')
    plt.close()

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

if JACOBIAN_MOTION:
    # Initial configuration
    DELTAL = np.full(9, 1e-4)
    DxDyDl = np.full(9, 1e-4)
    coords = np.array([0, 0, 145 * 3, 1e-4, 1e-4, 1e-4])

    print("Enter target as: x, y, z, yaw, pitch, roll (in mm and radians). Press 'e' to exit.")

    while True:
        user_input = input("Target >> ")

        if user_input.strip().lower() == 'e':
            print("Terminated.")
            break

        try:
            values = list(map(float, user_input.strip().split(',')))
            if len(values) != 6:
                print("Please enter exactly 6 comma-separated values.")
                continue

            xyzrpy_target = np.array(values)

            # Compute updated state
            DELTAL, DxDyDl, coords = coordinates2tendon(
                DELTAL, DxDyDl, coords, xyzrpy_target - coords, step_size=2.0
            )

            print("Updated configuration:")
            print("  DELTAL =", DELTAL.round(2))
            print("  coords =", coords.round(2))

        except ValueError:
            print("Invalid input. Please enter 6 comma-separated numbers or 'e' to exit.")

def SaveControlWorkspace():
   """
   Saves control workspace using deterministic k-means for maximum coverage.
   Only considers points with z > 200. Uses deterministic initialization from actual workspace points.
   """
   
   WS = np.load("data/Workspace.npz")["pose"]
   
   # Filter points with z > 200
   valid_mask = WS[:, 2] > 200
   valid_WS = WS[valid_mask]
   
   n_points = 20
   
   # Deterministic initialization: select evenly spaced points from workspace
   init_indices = np.linspace(0, len(valid_WS)-1, n_points, dtype=int)
   init_centers = valid_WS[init_indices]
   
   # Use k-means with custom initialization
   kmeans = KMeans(n_clusters=n_points, init=init_centers, n_init=1, random_state=42)
   clusters = kmeans.fit(valid_WS)
   
   ControlWorkspace = []
   for centroid in clusters.cluster_centers_:
       # Find closest actual point to each centroid
       distances = np.linalg.norm(valid_WS - centroid, axis=1)
       closest_idx = np.argmin(distances)
       ControlWorkspace.append(valid_WS[closest_idx])
   
   os.makedirs("results", exist_ok=True)
   np.savez_compressed("results/ControlWorkspace.npz", pose=ControlWorkspace)

if RANDOM_MOTION:
    DEVICENAME      = '/dev/ttyUSB0'
    BAUDRATE        = 57600
    PROTOCOL_VERSION = 2.0
    EXPOSITION_MODE = 4			# position + current (to be changed)

    # Motors according to how they are placed
    MOTOR_IDs = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19])	# 9 motors (as in the setup)
    MOTOR_Num = MOTOR_IDs.shape[0]	# number of motors

    D_pulley = 6				# [mm]
    unit_scale = 4096/(np.pi*D_pulley)	# motor position/DELTAL

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

    VELOCITY_LIMIT = 40     # velocity limitation

    probe_and_assign_cameras()

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

    # Calibrate
    calibrate_motors()

    # Calibrate motors
    print("--- Press 'y' to calibrate the motors, ENTER to go on ---")
    key = get_key()
    if key == 'y':
        calibrate_motors()
        print("\n")

    DELTAL_reference = read_DELTAL_from_motors()

    for i in range(100):
        print(f"...Motion {i+1}")

        # Choose configuration for efficient sampling (between 0 and 2)
        config = np.random.randint(0, 3)

        # Angles
        delta1 = [np.radians(0), np.radians(120), np.radians(-120)]
        delta2 = [np.radians(60), np.radians(120+60), np.radians(-120+60)]
        delta3 = [np.radians(150), np.radians(120+150), np.radians(-120+150)]

        # Sampling (note that it affects how data will be treated in NNs)
        if config==0:
            DELTAL_first = np.random.uniform(low=-50, high=10, size=3)
            DELTAL_second = SamplingSolver(delta1, DELTAL_first, delta2)
            DELTAL_third = SamplingSolver(delta1, DELTAL_first, delta3)
        elif config==1:
            DELTAL_first = np.random.uniform(low=-50, high=10, size=3)
            DELTAL_second = SamplingSolver(delta1, DELTAL_first, delta2)
            DELTAL_third = np.random.uniform(low=-50, high=10, size=3)
        elif config==2:
            DELTAL_first = np.random.uniform(low=-50, high=10, size=3)
            DELTAL_second = np.random.uniform(low=-50, high=10, size=3)
            DELTAL_third = np.random.uniform(low=-50, high=10, size=3)

        # Final DELTAL sample
        DELTAL_sample = np.concatenate([DELTAL_first, DELTAL_second, DELTAL_third])

        # Final d_DELTAL
        d_DELTAL = DELTAL_sample - (read_DELTAL_from_motors() - DELTAL_reference)

        move_motors(d_DELTAL)

        time.sleep(0.1)

if POLYGON_CHECKING:

    available_configs = [
        (30, 80, 380),
        (30, 120, 350),
    ]

    # Load workspace
    WS = np.load("data/Workspace.npz")["tip3"]
    
    for i in range(2):
        # Select configuration to test
        selected_config = available_configs[i]
        n_sides, radius, height = selected_config
    
        # Generate vertices and validate
        vertices = generate_polygon_vertices(n_sides, radius, height)
        valid_mask = check_vertex_validity(vertices, WS)
        plot_polygon_and_workspace(vertices, WS, height)
    
        # Print results
        n_valid = np.sum(valid_mask)
        print(f"Configuration: {n_sides} sides, radius={radius}mm, height={height}mm")
        print(f"Valid vertices: {n_valid}/{len(vertices)}")
        print(f"All vertices valid: {n_valid == len(vertices)}")

# Save control Workspace
if False:
    SaveControlWorkspace()
    print("Saved control workspace")
