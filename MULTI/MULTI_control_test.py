# Libraries
import os
import glob
import random
import warnings
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader, RandomSampler # type: ignore
import torch.nn.functional as F # type: ignore
import shutil
import plotly.graph_objects as go # type: ignore
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau # type: ignore
import torchvision.transforms as T # type: ignore
import torchvision.transforms.functional as TF # type: ignore
from geometry_msgs.msg import PoseStamped

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt # type: ignore
import random
import sys
from torch.amp import GradScaler, autocast # type: ignore
from pytorch_grad_cam import GradCAM # type: ignore
from pytorch_grad_cam.utils.image import show_cam_on_image # type: ignore
from scipy.spatial.transform import Rotation as R # type: ignore
from scipy.linalg import solve # type: ignore
from numpy.linalg import lstsq
from scipy.linalg import null_space # type: ignore
import torchvision.models as models # type: ignore
from scipy.integrate import cumulative_trapezoid # type: ignore
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import rospy
import time
import threading
from matplotlib.patches import Patch

# Ignore warnings
warnings.filterwarnings("ignore")
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# User-defined
from MULTI_jacobians import *
from MULTI_kinematics import *
from MULTI_reconstruction_utility import *
from MULTI_cameras_utility import *
from MULTI_controller import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# At the beginning, for the optitrack:
# roslaunch optitrack_ros_communication optitrack_nodes.launch

# For the cameras:
# v4l2-ctl --list-devices-

# === Define if weighted inference or not ===
WEIGHT = False
if WEIGHT:
    w_string = "W50g"

# === Define if the mocap is being used or not ===
MOCAP = False

# === Define if the null space controller is being checked ===
CHECK = True

# === Polygons redundant configurations ===

available_configs = [
    (30, 80, 380),
    (30, 120, 350),
]

# === Printing while sampling in the WorkSpace ===

def AngError(pred_angles, true_angles):
    """
    Compute angular error considering 360° goniometric circularity.
    """
    diff = pred_angles - true_angles
    # Wrap difference to [-180, 180] range
    diff = (diff + 180) % 360 - 180
    return np.abs(diff)

def PrintError(mocap=None, camera=None, target=None):
    """
    Prints errors with respect to camera (if closed-loop) and ground truth.
    """
    target_rot_deg = np.rad2deg(target[3:])
    mocap_rot_deg = np.rad2deg(mocap[3:])
    
    if camera is not None:
        camera_rot_deg = np.rad2deg(camera[3:])
        error_camera_pos = np.linalg.norm(camera[:3] - target[:3])
        error_camera_rot = np.linalg.norm(AngError(camera_rot_deg, target_rot_deg))
        print(f"Camera: pos. {error_camera_pos}, rot. {error_camera_rot} \n")

    error_mocap_pos = np.linalg.norm(mocap[:3] - target[:3])
    error_mocap_rot = np.linalg.norm(AngError(mocap_rot_deg, target_rot_deg))
    print(f"Mocap: pos. {error_mocap_pos}, rot. {error_mocap_rot}\n")

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

# === Functions for mocap acquisition ===

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

# === Mocap initialization ===

if MOCAP:
    
    # Initialize ROS node
    rospy.init_node('optitrack_listener')

    # Dictionaries to store positions and angles informations
    positions = {}
    quaternions = {}
    angles = {}

    # Define the TOPICS to subscribe to:
    # - Base (static): BASE
    # - Tip3 (dynamic): TIP3
    helyx_topics = [
        ('/optitrack/BASE/pose', 'BASE'),
        ('/optitrack/TIP1/pose', 'TIP1'),
        ('/optitrack/TIP2/pose', 'TIP2'),
        ('/optitrack/TIP3/pose', 'TIP3')
    ]

    # Create subscribers for the topics
    subscribers = []
    for topic_name, key in helyx_topics:
        sub = rospy.Subscriber(topic_name, PoseStamped, pose_callback, callback_args=key)
        subscribers.append(sub)

# Allow messages to start arriving
time.sleep(1)

# Function to get mocap pose
def get_mocap_pose(name):
    """
    Generalized pose acquisition.
    """

    x_tip, y_tip, z_tip = positions.get(name)
    yaw_tip, pitch_tip, roll_tip = angles.get(name)
    return x_tip, y_tip, z_tip, yaw_tip, pitch_tip, roll_tip

# Function for mocap threading
def mocap_thread_fn(container, stop_event):
    while not stop_event.is_set():
        try:
            x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
            x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")
            mocap3 = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                              x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
            container.append(mocap3)
            time.sleep(0.1)     # one point every 0.1 seconds
        except Exception as e:
            raise RuntimeError(f"Mocap thread error: {e}")
           
# === Modalities of motion ===

# What needs to be done
modalities = {
    "Sample datadriven",
    "Sample jacobian",
    "Sample OL datadriven",
    "Sample OL jacobian",
    "Polygon datadriven",
    "Polygon jacobian",
    "Polygon OL datadriven",
    "Polygon OL jacobian",
    "Check Null Space"
}

# === Mean DELTAL for null-space control ===

# Load data-driven DELTAL stats
DELTAL_stats = torch.load('models/kinematics/DELTAL_stats.pt')
DELTAL_mean = DELTAL_stats['mean'].numpy()
os.makedirs("results", exist_ok=True)

# === Testing function for controllers ===

def TestingMultiSectionHelyx(DELTAL_av=DELTAL_mean):
    """
    Interactive testing function for multi-section Helyx robot.
    Inputs:
    - max_samples: maximum number of samples collected
    - DELTA_av: average actuations
    """

    # Convert set to list for indexing
    modality_list = list(modalities)

    print("=== Multi-section helyx testing ===\n")
    for i, modality in enumerate(modality_list, 1):
        print(f"{i}. {modality}")

    # Get user choice
    choice = int(input(f"Choose a modality for testing:\n")) - 1
    selected_modality = modality_list[choice]
    print(f"\nExecuting: {selected_modality}")

    # Maximum samples
    max_samples = 20
    
    # Execute based on selection
    if selected_modality == "Sample datadriven":
        run_sample_datadriven(max_samples, DELTAL_mean=DELTAL_av)
    elif selected_modality == "Sample jacobian":
        run_sample_jacobian(max_samples, DELTAL_mean=DELTAL_av)
    elif selected_modality == "Sample OL datadriven":
        run_sample_OL_datadriven(max_samples, DELTAL_mean=DELTAL_av)
    elif selected_modality == "Sample OL jacobian":
        run_sample_OL_jacobian(max_samples, DELTAL_mean=DELTAL_av)
    elif selected_modality == "Polygon datadriven":
        run_polygon_datadriven(DELTAL_mean=DELTAL_av)
    elif selected_modality == "Polygon jacobian":
        run_polygon_jacobian(DELTAL_mean=DELTAL_av)
    elif selected_modality == "Polygon OL datadriven":
        run_polygon_OL_datadriven(DELTAL_mean=DELTAL_av)
    elif selected_modality == "Polygon OL jacobian":
        run_polygon_OL_jacobian(DELTAL_mean=DELTAL_av)
    elif selected_modality == "Check Null Space":
        run_check_null_space()
    else:
        raise RuntimeError("Not existing modality!\n")

# Define polygon chooser
def ChoosePolygon(available_configs=available_configs):
    """
    Chooses the Polygon to be tracked.
    """

    WS = np.load("data/Workspace.npz")["tip3"]
    vertices_dict = {}
    unique_sides = set(config[0] for config in available_configs)
    vertices_dict = {str(n_sides): [] for n_sides in unique_sides}
    for n_sides, radius, height in available_configs:
        vertices = generate_polygon_vertices(n_sides, radius, height)
        vertices_dict[str(n_sides)].append({
            "vertices": vertices,
            "radius": radius,
            "height": height,
            "config": (n_sides, radius, height)
        })

    # Display options
    print("=== Polygon Selection ===\n")
    all_configs = []
    for i, (n_sides, radius, height) in enumerate(available_configs, 1):
        print(f"{i}. {n_sides} sides, radius={radius}mm, height={height}mm")
        all_configs.append((n_sides, radius, height))

    # Get user choice
    choice = int(input("Choose a polygon configuration:\n")) - 1
    selected_config = all_configs[choice]
    n_sides, radius, height = selected_config

    # Vertices and validation
    vertices = generate_polygon_vertices(n_sides, radius, height)
    _ = check_vertex_validity(vertices, WS)
    plot_polygon_and_workspace(vertices, WS, height)
    print(f"\nSelected: {n_sides} sides, radius={radius}mm, height={height}mm")

    return vertices, selected_config

# Sample, datadriven
def run_sample_datadriven(max_samples, DELTAL_mean=None):
    """
    Sample Workspace Data, in position and orientation, to check the controller.
    Appends data to the .npz file one sample at a time, up to max_samples.
    """

    # Load existing data if available
    if WEIGHT:
        save_path = f"results/ControlWorkspaceDatadriven_{w_string}.npz"
    else:
        save_path = "results/ControlWorkspaceDatadriven.npz"
    if os.path.exists(save_path):
        data = np.load(save_path)
        target_list = list(data["target"])
        mocap_list = list(data["mocap"])
    else:
        target_list = []
        mocap_list = []

    controller = MULTIController(DELTAL_mean, DATADRIVEN=True)
    WS = np.load("results/ControlWorkspace.npz")["pose"]

    while len(target_list) < max_samples:
        idx = len(target_list)
        print(f"Sample {idx+1}/{max_samples}...")

        target_pose = WS[idx]
        pose_traj = controller.CONTROL(target_pose)
        camera_pose = pose_traj[-1]
        target_list.append(target_pose)

        # Mocap acquisition
        x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
        x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")
        mocap3 = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                              x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
        mocap_list.append(mocap3)

        # Printing
        PrintError(mocap=mocap3, camera=camera_pose, target=target_pose)

        # Save after each sample
        save_dict = {
            "target": np.stack(target_list),
            "mocap": np.stack(mocap_list)
        }

        np.savez_compressed(save_path, **save_dict)

    # Clean env
    controller.CleanEnv()

    print(f"Completed: {len(target_list)} samples saved to {save_path}")

# Sample, jacobian
def run_sample_jacobian(max_samples, DELTAL_mean=None):
    """
    Sample Workspace Data, in position and orientation, to check the controller.
    Appends data to the .npz file one sample at a time, up to max_samples.
    """

    # Load existing data if available
    if WEIGHT:
        save_path = f"results/ControlWorkspaceJacobian_{w_string}.npz"
    else:
        save_path = "results/ControlWorkspaceJacobian.npz"
    if os.path.exists(save_path):
        data = np.load(save_path)
        target_list = list(data["target"])
        mocap_list = list(data["mocap"])
    else:
        target_list = []
        mocap_list = []

    controller = MULTIController(DELTAL_mean, DATADRIVEN=False)
    WS = np.load("results/ControlWorkspace.npz")["pose"]

    while len(target_list) < max_samples:
        idx = len(target_list)
        print(f"Sample {idx+1}/{max_samples}...")

        target_pose = WS[idx]
        pose_traj = controller.CONTROL(target_pose)
        camera = pose_traj[-1]
        target_list.append(target_pose)

        # Mocap acquisition
        x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
        x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")
        mocap3 = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                              x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
        mocap_list.append(mocap3)

        # Print errors
        PrintError(mocap=mocap3, camera=camera, target=target_pose)

        # Save after each sample
        save_dict = {
            "target": np.stack(target_list),
            "mocap": np.stack(mocap_list)
        }

        np.savez_compressed(save_path, **save_dict)

    # Clean env
    controller.CleanEnv()

    print(f"Completed: {len(target_list)} samples saved to {save_path}")

# Sample, Open Loop, datadriven
def run_sample_OL_datadriven(max_samples, DELTAL_mean=None):
    """
    Sample Workspace Data, in position and orientation, to check the controller.
    Appends data to the .npz file one sample at a time, up to max_samples.
    """

    # Load existing data if available
    if WEIGHT:
        save_path = f"results/ControlWorkspaceOpenLoopDatadriven_{w_string}.npz"
    else:
        save_path = "results/ControlWorkspaceOpenLoopDatadriven.npz"
    if os.path.exists(save_path):
        data = np.load(save_path)
        target_list = list(data["target"])
        mocap_list = list(data["mocap"])
    else:
        target_list = []
        mocap_list = []

    # Get initial point
    x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
    x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2 = get_mocap_pose("TIP2")
    x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1 = get_mocap_pose("TIP1")
    x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")
    mocap3 = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    mocap2 = relative_coordinates(x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    mocap1 = relative_coordinates(x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    xyz_initial = np.concatenate([mocap1[:3], mocap2[:3], mocap3[:3]])

    # Define controller
    controller = MULTIOpenLoop(DELTAL_mean, DATADRIVEN=True, xyz_initial=xyz_initial)
    WS = np.load("results/ControlWorkspace.npz")["pose"]
    DELTAL0 = controller.DELTAL_initial
    coords0 = controller.coords_initial

    while len(target_list) < max_samples:
        idx = len(target_list)
        print(f"Sample {idx+1}/{max_samples}...")

        target_pose = WS[idx]
        DELTAL0 = controller.CONTROL(coords0, target_pose, DELTAL0)
        coords0 = target_pose
        target_list.append(target_pose)

        # Mocap acquisition
        x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
        x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")
        mocap3 = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                              x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
        mocap_list.append(mocap3)

        # Print error
        PrintError(target=target_pose, mocap=mocap3)

        # Save after each sample
        save_dict = {
            "target": np.stack(target_list),
            "mocap": np.stack(mocap_list)
        }

        np.savez_compressed(save_path, **save_dict)

    # Clean env
    controller.CleanEnv()

    print(f"Completed: {len(target_list)} samples saved to {save_path}")

# Sample, Open Loop, jacobian
def run_sample_OL_jacobian(max_samples, DELTAL_mean=None):
    """
    Sample Workspace Data, in position and orientation, to check the controller.
    Appends data to the .npz file one sample at a time, up to max_samples.
    """

    # Load existing data if available
    if WEIGHT:
        save_path = f"results/ControlWorkspaceOpenLoopJacobian_{w_string}.npz"
    else:
        save_path = "results/ControlWorkspaceOpenLoopJacobian.npz"
    if os.path.exists(save_path):
        data = np.load(save_path)
        target_list = list(data["target"])
        mocap_list = list(data["mocap"])
    else:
        target_list = []
        mocap_list = []

    # Get initial point
    x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
    x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2 = get_mocap_pose("TIP2")
    x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1 = get_mocap_pose("TIP1")
    x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")
    mocap3 = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    mocap2 = relative_coordinates(x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    mocap1 = relative_coordinates(x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    xyz_initial = np.concatenate([mocap1[:3], mocap2[:3], mocap3[:3]])

    # Define controller
    controller = MULTIOpenLoop(DELTAL_mean, DATADRIVEN=False, xyz_initial=xyz_initial)
    WS = np.load("results/ControlWorkspace.npz")["pose"]
    DELTAL0 = controller.DELTAL_initial
    coords0 = controller.coords_initial

    while len(target_list) < max_samples:
        idx = len(target_list)
        print(f"Sample {idx+1}/{max_samples}...")

        target_pose = WS[idx]
        DELTAL0 = controller.CONTROL(coords0, target_pose, DELTAL0)
        coords0 = target_pose
        target_list.append(target_pose)

        # Mocap acquisition
        x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
        x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")
        mocap3 = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                              x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
        mocap_list.append(mocap3)

        # Print error
        PrintError(mocap=mocap3, target=target_pose)

        # Save after each sample
        save_dict = {
            "target": np.stack(target_list),
            "mocap": np.stack(mocap_list)
        }

        np.savez_compressed(save_path, **save_dict)

    # Clean env
    controller.CleanEnv()

    print(f"Completed: {len(target_list)} samples saved to {save_path}")
    
# Polygon, datadriven
def run_polygon_datadriven(DELTAL_mean=None):
    """
    Moves following a pre-defined polygon in space.
    Data-driven approach. One file per polygon is saved.
    """

    # Choose polygon
    vertices, config = ChoosePolygon()

    # Add orientation (ZYX = 0,0,0 for horizontal manipulator)
    vertices_oriented = np.hstack([
        vertices,
        np.zeros((len(vertices), 3))
    ])  # Shape: (N, 6)

    # Base path
    if WEIGHT:
        base_save_path = f"results/PolygonDatadriven_{w_string}_"
    else:
        base_save_path = "results/PolygonDatadriven_"
    n_sides, radius, height = config
    save_path = f"{base_save_path}{n_sides}sides_{radius}mm_{height}mm.npz"

    # Controller
    controller = MULTIController(DELTAL_mean, DATADRIVEN=True, discr_wp=1)

    # Track just mocap (as targets are in vertices)
    mocap_list = []

    # 1. Reach the first vertex without recording
    controller.CONTROL(vertices_oriented[0])

    # 2. Traverse all edges and return to the first
    for i in range(len(vertices_oriented)):

        # Vertex printing
        print(f"\n--- Tracking vertex {i+1} of {len(vertices_oriented)} ---")

        # Select target
        target = vertices_oriented[i+1] if i < len(vertices_oriented)-1 else vertices_oriented[0]

        # Mocap segment
        mocap_segment = []
        stop_event = threading.Event()
        mocap_thread = threading.Thread(target=mocap_thread_fn, args=(mocap_segment, stop_event))
        mocap_thread.start()

        # Control
        controller.CONTROL(target)

        # Extend with mocap segment
        stop_event.set()
        mocap_thread.join()
        mocap_list.extend(mocap_segment)

    # Save data
    mocap_points = np.stack(mocap_list)
    np.savez_compressed(save_path, target=vertices_oriented, mocap=mocap_points)

    # Clean env
    controller.CleanEnv()

    print(f"\nSave polygon with {n_sides} sides at height {height} mm to {save_path}")

# Polygon, jacobian
def run_polygon_jacobian(DELTAL_mean=None):
    """
    Moves following a pre-defined polygon in space.
    Model-based approach. One file per polygon is saved.
    """

    # Choose polygon
    vertices, config = ChoosePolygon()

    # Add orientation (ZYX = 0,0,0 for horizontal manipulator)
    vertices_oriented = np.hstack([
        vertices,
        np.zeros((len(vertices), 3))
    ])  # Shape: (N, 6)

    # Base path
    if WEIGHT:
        base_save_path = f"results/PolygonJacobian_{w_string}_"
    else:
        base_save_path = "results/PolygonJacobian_"
    n_sides, radius, height = config
    save_path = f"{base_save_path}{n_sides}sides_{radius}mm_{height}mm.npz"

    # Controller
    controller = MULTIController(DELTAL_mean, DATADRIVEN=False, discr_wp=1)

    # Track just mocap (as targets are in vertices)
    mocap_list = []

    # 1. Reach the first vertex without recording
    controller.CONTROL(vertices_oriented[0])

    # 2. Traverse all edges and return to the first
    for i in range(len(vertices_oriented)):

        # Vertex printing
        print(f"\n--- Tracking vertex {i+1} of {len(vertices_oriented)} ---")

        # Select target
        target = vertices_oriented[i+1] if i < len(vertices_oriented)-1 else vertices_oriented[0]

        # Mocap segment
        mocap_segment = []
        stop_event = threading.Event()
        mocap_thread = threading.Thread(target=mocap_thread_fn, args=(mocap_segment, stop_event))
        mocap_thread.start()

        # Control
        controller.CONTROL(target)

        # Extend with mocap segment
        stop_event.set()
        mocap_thread.join()
        mocap_list.extend(mocap_segment)

    # Save data
    mocap_points = np.stack(mocap_list)
    np.savez_compressed(save_path, target=vertices_oriented, mocap=mocap_points)

    # Clean env
    controller.CleanEnv()

    print(f"\nSave polygon with {n_sides} sides at height {height} mm to {save_path}")

# Polygon, Open Loop, datadriven
def run_polygon_OL_datadriven(DELTAL_mean=None):
    """
    Moves following a pre-defined polygon in space.
    Data-driven approach, Open-Loop control.
    One file per polygon is saved.
    """

    # Choose polygon
    vertices, config = ChoosePolygon()

    # Add orientation (ZYX = 0,0,0 for horizontal manipulator)
    vertices_oriented = np.hstack([
        vertices,
        np.zeros((len(vertices), 3))
    ])  # Shape: (N, 6)

    # Base path
    if WEIGHT:
        base_save_path = f"results/PolygonOpenLoopDatadriven_{w_string}_"
    else:
        base_save_path = "results/PolygonOpenLoopDatadriven_"
    n_sides, radius, height = config
    save_path = f"{base_save_path}{n_sides}sides_{radius}mm_{height}mm.npz"

    # Get initial point
    x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
    x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2 = get_mocap_pose("TIP2")
    x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1 = get_mocap_pose("TIP1")
    x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")
    mocap3 = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    mocap2 = relative_coordinates(x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    mocap1 = relative_coordinates(x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    xyz_initial = np.concatenate([mocap1[:3], mocap2[:3], mocap3[:3]])

    # Controller
    controller = MULTIOpenLoop(DELTAL_mean, DATADRIVEN=True, xyz_initial=xyz_initial)
    
    # Track just mocap (as targets are in vertices)
    mocap_list = []

    # Initial values
    DELTAL0 = controller.DELTAL_initial
    coords0 = controller.coords_initial

    # 1. Reach the first vertex without recording
    DELTAL0 = controller.CONTROL(coords0, vertices_oriented[0], DELTAL0)
    coords0 = vertices_oriented[0]

    # 2. Traverse all edges and return to the first
    for i in range(len(vertices_oriented)):

        # Vertex printing
        print(f"\n--- Tracking vertex {i+1} of {len(vertices_oriented)} ---")

        # Select target
        target = vertices_oriented[i+1] if i < len(vertices_oriented)-1 else vertices_oriented[0]

        # Mocap segment
        mocap_segment = []
        stop_event = threading.Event()
        mocap_thread = threading.Thread(target=mocap_thread_fn, args=(mocap_segment, stop_event))
        mocap_thread.start()

        # Control
        DELTAL0 = controller.CONTROL(coords0, target, DELTAL0)
        coords0 = target

        # Extend with mocap segment
        stop_event.set()
        mocap_thread.join()
        mocap_list.extend(mocap_segment)

    # Save data
    mocap_points = np.stack(mocap_list)
    np.savez_compressed(save_path, target=vertices_oriented, mocap=mocap_points)

    # Clean env
    controller.CleanEnv()

    print(f"\nSave polygon with {n_sides} sides at height {height} mm to {save_path}")

# Polygon, Open Loop, jacobian
def run_polygon_OL_jacobian(DELTAL_mean=None):
    """
    Moves following a pre-defined polygon in space.
    Model-based approach, Open-Loop control.
    One file per polygon is saved.
    """

    # Choose polygon
    vertices, config = ChoosePolygon()

    # Add orientation (ZYX = 0,0,0 for horizontal manipulator)
    vertices_oriented = np.hstack([
        vertices,
        np.zeros((len(vertices), 3))
    ])  # Shape: (N, 6)

    # Base path
    if WEIGHT:
        base_save_path = f"results/PolygonOpenLoopJacobian_{w_string}_"
    else:
        base_save_path = "results/PolygonOpenLoopJacobian_"
    n_sides, radius, height = config
    save_path = f"{base_save_path}{n_sides}sides_{radius}mm_{height}mm.npz"

    # Get initial point
    x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
    x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2 = get_mocap_pose("TIP2")
    x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1 = get_mocap_pose("TIP1")
    x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")
    mocap3 = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    mocap2 = relative_coordinates(x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    mocap1 = relative_coordinates(x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1,
                    x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    xyz_initial = np.concatenate([mocap1[:3], mocap2[:3], mocap3[:3]])

    # Controller
    controller = MULTIOpenLoop(DELTAL_mean, DATADRIVEN=False, xyz_initial=xyz_initial)

    # Track just mocap (as targets are in vertices)
    mocap_list = []

    # Initial values
    DELTAL0 = controller.DELTAL_initial
    coords0 = controller.coords_initial

    # 1. Reach the first vertex without recording
    DELTAL0 = controller.CONTROL(coords0, vertices_oriented[0], DELTAL0)
    coords0 = vertices_oriented[0]

    # 2. Traverse all edges and return to the first
    for i in range(len(vertices_oriented)):

        # Vertex printing
        print(f"\n--- Tracking vertex {i+1} of {len(vertices_oriented)} ---")

        # Select target
        target = vertices_oriented[i+1] if i < len(vertices_oriented)-1 else vertices_oriented[0]

        # Mocap segment
        mocap_segment = []
        stop_event = threading.Event()
        mocap_thread = threading.Thread(target=mocap_thread_fn, args=(mocap_segment, stop_event))
        mocap_thread.start()

        # Control
        DELTAL0 = controller.CONTROL(coords0, target, DELTAL0)
        coords0 = target

        # Extend with mocap segment
        stop_event.set()
        mocap_thread.join()
        mocap_list.extend(mocap_segment)

    # Save data
    mocap_points = np.stack(mocap_list)
    np.savez_compressed(save_path, target=vertices_oriented, mocap=mocap_points)

    # Clean env
    controller.CleanEnv()

    print(f"\nSave polygon with {n_sides} sides at height {height} mm to {save_path}")

# Function to check if the null-space controller works
def CheckNullSpaceController(num_samples=50, DELTAL_mean=DELTAL_mean):
    """
    Samples Workspace Data, in position and orientation, to check if the
    Null Space controller at least works. Simply shows the error of the
    controller with respect to the camera without saving anything.
    """

    controller = MULTIController(DELTAL_mean, DATADRIVEN=True, CheckNullSpace=True)
    WS = np.load("data/Workspace.npz")["pose"]

    for i in range(num_samples):
        print(f"Sample {i+1}/{num_samples}...")

        # Extract one sample and move there
        target_pose = WS[np.random.randint(0, len(WS))]
        controller.CONTROL(target_pose)
    
    controller.CleanEnv()

# Check null space for redundancy
def run_check_null_space(n_samples=3):
    """
    Checks null space asking confirmation to the user.
    """

    # Ask user for compression quantity
    compression = float(input("Enter compression quantity (mm): "))

    # Use open loop jacobian controller just to move
    xyz_initial = np.array([0, 0, 140, 0, 0, 280, 0, 0, 420])
    controller = MULTIOpenLoop(DELTAL_mean, DATADRIVEN=False, xyz_initial=xyz_initial)
    controller.MotorWriter(np.repeat(np.array(compression), 9))
    controller.CleanEnv()
    
    # Initialize null space checker
    check = NullSpaceCheck(xyz_initial=xyz_initial - np.array([0, 0, compression, 0, 0, compression, 0, 0, compression]))

    i = 0
    while i < n_samples: 

        print(f"Sample {i+1}/{n_samples}...")

        # Ask user for DELTAL_imposed
        print("Enter 9 DELTAL values (space-separated):")
        deltal_input = input().strip().split()
        deltal_imposed = np.array([float(x) for x in deltal_input])

        # Move robot
        check.MOVE(deltal_imposed)

        # Ask user for confirmation
        confirm = input("Confirm the DELTAL movement? (y/n): ").strip().lower()
        if confirm == 'y':
            i += 1
        else:
            print("Movement cancelled.")

    check.CleanEnv()
    print(f"Completed: {i} samples checked.")

if __name__ == "__main__":

    # Run if MOCAP is called
    if MOCAP or CHECK:
        TestingMultiSectionHelyx()
    else:
        CheckNullSpaceController()