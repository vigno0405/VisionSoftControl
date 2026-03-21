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
from matplotlib.patches import Patch

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
from scipy.spatial import cKDTree # type: ignore

warnings.filterwarnings("ignore")
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
import matplotlib
matplotlib.use("Agg")
os.makedirs("results", exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:")
print(f"--- {device} ----")

# Import controller
from SINGLE_controller import SINGLEController, SINGLEOpenLoop

# At the beginning, for the optitrack:
# roslaunch optitrack_ros_communication optitrack_nodes.launch

# For the cameras:
# v4l2-ctl --list-devices

# === Flag variables ===
MOCAP = True           # mocap (always True)
OPEN_LOOP = True        # if open loop
POLYGON = False         # what to do
DATADRIVEN = False      # datadriven or jacobian

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

# Mapping from mocap to robot frame:
# x -> x, y -> -z, z -> y
R_corr = np.array([
    [1, 0,  0],
    [0, 0, -1],
    [0, 1,  0]
])

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

if MOCAP:
    # Initialize ROS node
    rospy.init_node('optitrack_listener')

    # Dictionaries to store positions and angles informations
    positions = {}
    quaternions = {}
    angles = {}

    # Define the TOPICS to subscribe to:
    # - Base (static): BASE
    # - Tip (dynamic): TIP
    # Define the TOPICS to subscribe to:
    helyx_topics = [
        ('/optitrack/BASE/pose', 'BASE'),
        ('/optitrack/TIP/pose', 'TIP')
    ]

    # Create subscribers for the topics
    subscribers = []
    for topic_name, key in helyx_topics:
        sub = rospy.Subscriber(topic_name, PoseStamped, pose_callback, callback_args=key)   # use pose_callback
        subscribers.append(sub)

    # Allow messages to start arriving
    time.sleep(1)

# === WORKSPACE DATA ===

# Sample data for controller testing in the workspace
def SampleWorkspaceData(name="ControlWorkspace", mocap=MOCAP):
    """
    Sample Workspace Data to check the controller.
    Appends data to the .npz file one sample at a time, up to 100.
    """

    max_samples = 100
    if DATADRIVEN:
        save_path = f"results/{name}_datadriven.npz"
    else:
        save_path = f"results/{name}_jacobian.npz"

    # Load existing data if available
    if os.path.exists(save_path):
        data = np.load(save_path)
        target_list = list(data["target"])
        reached_list = list(data["reached"])
        mocap_list = list(data["mocap"]) if mocap and "mocap" in data.files else []
    else:
        target_list = []
        reached_list = []
        mocap_list = [] if mocap else None

    controller = SINGLEController(DATADRIVEN=DATADRIVEN, CAMERA_ID=2)
    WS = np.load("data/Workspace.npz")["xyz"]

    while len(target_list) < max_samples:
        idx = len(target_list)
        print(f"Sample {idx+1}/{max_samples}...")

        target = WS[np.random.randint(0, len(WS))]
        traj, _ = controller.CONTROL(target)
        reached = traj[-1]
        target_list.append(target)
        reached_list.append(reached)

        if mocap:
            x_tip, y_tip, z_tip = positions.get('TIP')
            x_base, y_base, z_base = positions.get('BASE')
            points = np.array([[x_tip, y_tip, z_tip]])
            base = np.array([x_base, y_base, z_base])
            rel = relative_point(points, base, R_corr=R_corr)[0]
            mocap_list.append(rel)

        # Save after each sample
        save_dict = {
            "target": np.stack(target_list),
            "reached": np.stack(reached_list)
        }
        if mocap:
            save_dict["mocap"] = np.stack(mocap_list)

        np.savez_compressed(save_path, **save_dict)

    print(f"Completed: {len(target_list)} samples saved to {save_path}")

# Sample workspace for control
if not POLYGON and False:
    SampleWorkspaceData()

# === POLYGON ===

if POLYGON:
    # Create polygon follower vertices_dict
    WS = np.load("data/Workspace.npz")["xyz"]
    vertices_dict = {"3": {}, "4": {}, "6": {}, "30": {}}

    available_configs = [
        (3, 25, 125),
        (4, 25, 125),
        (6, 25, 125),
        (30, 25, 125),
        (3, 35, 115),
        (4, 35, 115),
        (6, 35, 115),
        (30, 35, 115)
    ]

    # Display available polygon options
    print("\nAvailable polygon configurations:")
    for idx, (n_sides, radius, height) in enumerate(available_configs):
        print(f"[{idx}] {n_sides} sides | radius = {radius} mm | height = {height} mm")

    # User selection
    selected = int(input("\nSelect a configuration index to execute: "))
    assert 0 <= selected < len(available_configs), "Invalid selection index."

    # Extract selected configuration
    n_sides, radius, height = available_configs[selected]
    key = str(n_sides)

    # Generate and visualize selected polygon
    verts = generate_polygon_vertices(n_sides, radius, height).astype(np.float32)
    mask = check_vertex_validity(verts, WS)
    plot_polygon_and_workspace(verts, WS, mask, height)
    vertices_dict[key][str(height)] = verts

    # Print selected polygon vertices
    print(f"\nPolygon with {n_sides} sides at height {height} mm:")
    for i, v in enumerate(verts):
        print(f"  Vertex {i+1}: {v}")

# Polygon tracking
def PolygonTracking(name="PolygonWS", mocap=MOCAP):
    """
    Tracks a selected polygon configuration, collecting full camera trajectory,
    control waypoints, and mocap if enabled, from first vertex to final
    return to the start. One file per polygon is saved.
    """

    controller = SINGLEController(DATADRIVEN=DATADRIVEN, CAMERA_ID=2, discr_wp=1)

    # Define mocap thread function once
    def mocap_thread_fn(container, stop_event):
        while not stop_event.is_set():
            try:
                x_tip, y_tip, z_tip = positions.get('TIP')
                x_base, y_base, z_base = positions.get('BASE')
                point = np.array([x_tip, y_tip, z_tip])
                base = np.array([x_base, y_base, z_base])
                rel = relative_point(point[np.newaxis, :], base, R_corr=R_corr)[0]
                container.append(rel)
                time.sleep(0.1)    # one point every 0.1 seconds
            except Exception as e:
                raise RuntimeError(f"Mocap thread error: {e}")
            
    # Track only the selected polygon
    verts = vertices_dict[str(n_sides)][str(height)]

    reached_list = []
    target_list = []
    mocap_list = []

    # 1. Reach the first vertex without recording
    controller.CONTROL(verts[0])

    # 2. Traverse all edges and return to the first
    for i in range(len(verts)):

        # Vertex
        print(f"\n--- Tracking vertex {i + 1} of {len(verts)} ---")

        # Select target
        target = verts[i + 1] if i < len(verts)-1 else verts[0]

        # Mocap segment
        mocap_segment = []
        if mocap:
            stop_event = threading.Event()
            mocap_thread = threading.Thread(target=mocap_thread_fn, args=(mocap_segment, stop_event))
            mocap_thread.start()

        traj, wps = controller.CONTROL(target)

        if mocap:
            stop_event.set()
            mocap_thread.join()
            mocap_list.extend(mocap_segment)

        reached_list.extend(traj)
        target_list.extend(wps)

    reached = np.stack(reached_list)
    targets = np.stack(target_list)

    # Save data
    if mocap:
        mocap_points = np.stack(mocap_list)
        if DATADRIVEN:
            save_path = f"results/{name}_{n_sides}_{height}_datadriven_mocap.npz"
        else:
            save_path = f"results/{name}_{n_sides}_{height}_jacobian_mocap.npz"
        np.savez_compressed(save_path, target=targets, reached=reached, mocap=mocap_points)
    else:
        if DATADRIVEN:
            save_path = f"results/{name}_{n_sides}_{height}_datadriven.npz"
        else:
            save_path = f"results/{name}_{n_sides}_{height}_jacobian.npz"
        np.savez_compressed(save_path, target=targets, reached=reached)

    print(f"\nSaved polygon with {n_sides} sides at height {height} mm to {save_path}")
    controller.CleanEnv()

# Run if enabled
if POLYGON and False:
    PolygonTracking()

# === OPEN-LOOP ===

# Sample data for controller testing in the workspace
def SampleWorkspaceData_OL(name="ControlWorkspace"):
    """
    Sample Workspace Data to check the controller.
    Appends data to the .npz file one sample at a time, up to 100.
    """

    max_samples = 100
    if DATADRIVEN:
        save_path = f"results/{name}_OL_datadriven.npz"
    else:
        save_path = f"results/{name}_OL_jacobian.npz"

    # Load existing data if available
    if os.path.exists(save_path):
        data = np.load(save_path)
        target_list = list(data["target"])
        mocap_list = list(data["mocap"])
    else:
        target_list = []
        mocap_list = []

    # Get initial position
    x_tip, y_tip, z_tip = positions.get('TIP')
    yaw_tip, pitch_tip, roll_tip = angles.get('TIP')
    x_base, y_base, z_base = positions.get('BASE')
    yaw_base, pitch_base, roll_base = angles.get('BASE')
    x_initial = relative_coordinates(
        x_tip, y_tip, z_tip, yaw_tip, pitch_tip, roll_tip,
        x_base, y_base, z_base, yaw_base, pitch_base, roll_base
    )

    controller = SINGLEOpenLoop(DATADRIVEN=DATADRIVEN, x_initial=x_initial)
    WS = np.load("data/Workspace.npz")["xyz"]
    DELTAL0 = controller.DELTAL_initial
    DxDyDl0 = controller.DxDyDl_initial
    x0 = controller.x_initial
    if DATADRIVEN:
        x0 = x0[:3]

    while len(target_list) < max_samples:
        idx = len(target_list)
        print(f"Sample {idx+1}/{max_samples}...")

        target = WS[np.random.randint(0, len(WS))]
        if DATADRIVEN:
            x0_all, DELTAL0 = controller.CONTROL(x0, target, DELTAL0)
            x0 = x0_all[:3]
        else:
            x0, DELTAL0, DxDyDl0 = controller.CONTROL(x0, target, DELTAL0, DxDyDl0)

        target_list.append(target)
        time.sleep(1)

        x_tip, y_tip, z_tip = positions.get('TIP')
        x_base, y_base, z_base = positions.get('BASE')
        points = np.array([[x_tip, y_tip, z_tip]])
        base = np.array([x_base, y_base, z_base])
        rel = relative_point(points, base, R_corr=R_corr)[0]
        mocap_list.append(rel)

        # Save after each sample
        save_dict = {
            "target": np.stack(target_list),
            "mocap": np.stack(mocap_list)
        }

        np.savez_compressed(save_path, **save_dict)

    print(f"Completed: {len(target_list)} samples saved to {save_path}")

if OPEN_LOOP and not POLYGON:
    SampleWorkspaceData_OL()

# Polygon tracking
def PolygonTracking_OL(name="PolygonWS"):
    """
    Tracks a selected polygon configuration.
    One file per polygon is saved.
    """

    # Get initial position
    x_tip, y_tip, z_tip = positions.get('TIP')
    yaw_tip, pitch_tip, roll_tip = angles.get('TIP')
    x_base, y_base, z_base = positions.get('BASE')
    yaw_base, pitch_base, roll_base = angles.get('BASE')
    x_initial = relative_coordinates(
        x_tip, y_tip, z_tip, yaw_tip, pitch_tip, roll_tip,
        x_base, y_base, z_base, yaw_base, pitch_base, roll_base
    )

    controller = SINGLEOpenLoop(DATADRIVEN=DATADRIVEN, x_initial=x_initial)

    # Define mocap thread function once
    def mocap_thread_fn(container, stop_event):
        while not stop_event.is_set():
            try:
                x_tip, y_tip, z_tip = positions.get('TIP')
                x_base, y_base, z_base = positions.get('BASE')
                point = np.array([x_tip, y_tip, z_tip])
                base = np.array([x_base, y_base, z_base])
                rel = relative_point(point[np.newaxis, :], base, R_corr=R_corr)[0]
                container.append(rel)
                time.sleep(0.1)    # one point every 0.1 seconds
            except Exception as e:
                raise RuntimeError(f"Mocap thread error: {e}")
            
    # Track only the selected polygon
    verts = vertices_dict[str(n_sides)][str(height)]

    # Mocap data
    mocap_list = []

    # Initial values
    DELTAL0 = controller.DELTAL_initial
    DxDyDl0 = controller.DxDyDl_initial
    x0 = controller.x_initial
    if DATADRIVEN:
        x0 = x0[:3]

    # 1. Reach the first vertex without recording
    if DATADRIVEN:
        x0, DELTAL0 = controller.CONTROL(x0, verts[0], DELTAL0)
    else:
        x0, DELTAL0, DxDyDl0 = controller.CONTROL(x0, verts[0], DELTAL0, DxDyDl0)

    # 2. Traverse all edges and return to the first
    for i in range(len(verts)):

        # Vertex
        print(f"\n--- Tracking vertex {i + 1} of {len(verts)} ---")

        # Select target
        target = verts[i + 1] if i < len(verts)-1 else verts[0]

        # Mocap segment
        mocap_segment = []
        stop_event = threading.Event()
        mocap_thread = threading.Thread(target=mocap_thread_fn, args=(mocap_segment, stop_event))
        mocap_thread.start()

        if DATADRIVEN:
            x0, DELTAL0 = controller.CONTROL(x0, target, DELTAL0)
        else:
            x0, DELTAL0, DxDyDl0 = controller.CONTROL(x0, target, DELTAL0, DxDyDl0)

        time.sleep(1)

        stop_event.set()
        mocap_thread.join()
        mocap_list.extend(mocap_segment)

    # Save data
    mocap_points = np.stack(mocap_list)
    if DATADRIVEN:
        save_path = f"results/{name}_{n_sides}_{height}_OL_datadriven.npz"
    else:
        save_path = f"results/{name}_{n_sides}_{height}_OL_jacobian.npz"
    np.savez_compressed(save_path, mocap=mocap_points)

    print(f"\nSaved polygon with {n_sides} sides at height {height} mm to {save_path}")
    controller.CleanEnv()

if OPEN_LOOP and POLYGON:
    PolygonTracking_OL()