# VisionSoftControl

Vision-based pose estimation and feedback control of the Helyx soft robotic arm — a cable-tendon-actuated trimmed helicoid continuum manipulator.

---

## Robot

The Helyx is a continuum arm whose shape is determined by cable-tendon actuation applied to a trimmed helicoid structure. Each section is parameterised by curvature coordinates `q = [Dx, Dy, Dl]`, where `Dx` and `Dy` are lateral deflections and `Dl` is the length extension. Tendon-length changes ΔDELTAL [mm] are converted to Dynamixel encoder ticks by `unit_scale = 4096 / (π · D_pulley)`.

Two hardware configurations are implemented:

| | SINGLE | MULTI |
|---|--------|-------|
| Sections | 1 | 3 |
| Motors | 3 (IDs 1–3) | 9 (IDs 11–19) |
| Cameras | 1 USB | 3 USB (threaded) |
| Actuation DoF | 3 | 9 |
| Tip pose DoF | 6 | 6 |

---

## Control

Both configurations support two Jacobian-based control strategies:

**Analytic Jacobian** — symbolic Jacobians (`q2tendon`, `q2coordinates`, `q2xyz`) derived with SymPy from the Piecewise Constant Curvature (PCC) model, evaluated numerically via iterative Jacobian integration.

**Data-driven Jacobian** — Jacobian computed on-the-fly via `torch.autograd.functional.jacobian` applied to learned FK/IK neural networks.

The MULTI configuration is kinematically redundant (9 actuators, 6-DoF tip pose). A null-space controller exploits this redundancy to simultaneously track the tip target and drive the motor configuration toward a nominal posture:

```
dDELTAL = J_pseudo @ delta_pose + N @ (DELTAL_mean - DELTAL_current)
```

---

## Sensing & Reconstruction

**Tip pose** is estimated by a CNN that maps a grayscale camera frame to end-effector position and orientation (`helyx_model.pt`). For the MULTI configuration, three cameras observe the three sections independently.

**Shape reconstruction** fits polynomial curvature models — constant, affine, or quadratic — to rotation matrices derived from OptiTrack backbone markers, via CasADi/IPOPT optimisation. An ANN-based reconstruction path is also available, using section-specific TorchScript models inferred from relative inter-section transforms.

---

## Repository Structure

```
VisionSoftControl/
│
├── SINGLE/
│   ├── SINGLE_jacobians.py              # Symbolic Jacobians: q → tendon, q → pose
│   ├── SINGLE_kinematics.py             # Iterative mappings: tendon ↔ curvature ↔ pose
│   ├── SINGLE_reconstruction_utility.py # Curvature reconstruction (CasADi/IPOPT)
│   ├── SINGLE_controller.py             # SINGLEController (closed-loop), SINGLEOpenLoop
│   ├── SINGLE_data_acquire.py           # Random motor sampling + mocap/camera capture
│   ├── SINGLE_control_test.py           # Polygon tracking and workspace sampling
│   ├── SINGLE_plot_polygon.py           # Polygon waypoint visualisation
│   └── SINGLE_try_motion.py             # Random workspace exploration
│
└── MULTI/
    ├── MULTI_jacobians.py               # Symbolic Jacobians: q → tendon, q → pose, q → xyz
    ├── MULTI_kinematics.py              # Iterative mappings: tendon ↔ xyz ↔ curvature
    ├── MULTI_reconstruction_utility.py  # Curvature reconstruction (analytical + ANN)
    ├── MULTI_cameras_utility.py         # ThreadedCameraSystem for 3 concurrent cameras
    ├── MULTI_controller.py              # MULTIController, MULTIOpenLoop, NullSpaceCheck
    ├── MULTI_data_acquire.py            # Concurrent acquisition: cameras + mocap + curvature
    ├── MULTI_control_test.py            # Evaluation suite (9 test modalities)
    ├── MULTI_curvature_checks.py        # ANN vs. analytical curvature validation
    ├── MULTI_FK_acquire.py              # Forward kinematics dataset collection
    ├── MULTI_try_motion.py              # Random and Jacobian-guided motion exploration
    ├── MULTI_helyx_training.ipynb       # Model training
    ├── MULTI_helyx_testing.ipynb        # Model evaluation
    └── MULTI_control_results.ipynb      # Control result analysis
```

### Pre-trained models

| File | Description |
|------|-------------|
| `helyx_model.pt` | CNN: grayscale image → tip pose |
| `FK_model.pt` | Forward kinematics: ΔDELTAL → tip pose (MULTI) |
| `IK_model.pt` | Inverse kinematics: tip pose → ΔDELTAL (MULTI) |
| `constant_model.pt` / `affine_model.pt` / `quadratic_model.pt` | Polynomial curvature regression |
| `*_for_MULTI_model.pt` | Section-specific curvature models (MULTI reconstruction) |
| `DELTAL_stats.pt` | Normalisation statistics for MULTI motor commands |

Recorded control results and workspace samples are stored as `.npz` archives alongside the source files.

---

## Setup

### Hardware

| Component | Specification |
|-----------|---------------|
| Actuators | Dynamixel XM/XL series, Protocol 2.0, 57600 baud, `/dev/ttyUSB0` |
| Pulley diameter | 6 mm |
| SINGLE motors | IDs 1–3, tendon angles 0°, 120°, −120° |
| MULTI motors | IDs 11–19 (three groups of 3) |
| Cameras | USB cameras (OpenCV V4L2 backend) |
| Motion capture | OptiTrack, via ROS `geometry_msgs/PoseStamped` topics |

The robot base frame and the OptiTrack world frame are related by a fixed correction `R_corr = [[1,0,0],[0,0,-1],[0,1,0]]`. All lengths are in millimetres; angles in radians.

### Dependencies

```
numpy
torch >= 2.0
torchvision
opencv-python
scipy
sympy
casadi
scikit-learn
dynamixel-sdk
rospy        # ROS Noetic
plotly
matplotlib
```

ROS Noetic must be sourced and `rospy` importable. OptiTrack data is consumed from live ROS topics.

---

## Usage

All scripts are run from within the respective `SINGLE/` or `MULTI/` subdirectory.

### Data collection

```bash
# SINGLE: random ΔDELTAL sampling with camera and mocap recording (10,000 samples)
python SINGLE_data_acquire.py

# MULTI: concurrent camera + mocap acquisition with per-batch confirmation
python MULTI_data_acquire.py

# MULTI: forward kinematics dataset (ΔDELTAL → tip pose)
python MULTI_FK_acquire.py
```

### Control evaluation

```bash
# SINGLE: closed-loop and open-loop polygon tracking / workspace sampling
python SINGLE_control_test.py

# MULTI: interactive suite — closed/open-loop × data-driven/Jacobian × null-space check
python MULTI_control_test.py
```

### Utilities

```bash
# Random workspace exploration
python SINGLE_try_motion.py
python MULTI_try_motion.py

# Curvature validation: ANN prediction vs. analytical reconstruction
python MULTI_curvature_checks.py
```

Motor calibration runs interactively at startup (`w`/`s`/`h` keys, adaptive acceleration). Camera-to-position assignment in the MULTI configuration is also interactive.

---

## License

MIT — see [LICENSE](LICENSE).
