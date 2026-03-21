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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:")
print(f"--- {device} ----")

# Import controller
from SINGLE_controller import SINGLEController, SINGLEOpenLoop

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

WS = np.load("data/Workspace.npz")["xyz"]
vertices_dict = {"3": {}, "4": {}, "6": {}, "30": {}}

for i in range(8):
    n_sides, radius, height = available_configs[i]
    key = str(n_sides)

    # Generate and visualize selected polygon
    verts = generate_polygon_vertices(n_sides, radius, height).astype(np.float32)
    mask = check_vertex_validity(verts, WS)
    plot_polygon_and_workspace(verts, WS, height)
    vertices_dict[key][str(height)] = verts

    print(f"Generated {i+1}...")