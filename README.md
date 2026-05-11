# Internal Camera-based Reconstruction and Closed-loop Control of Soft Robotic Arms

Code for the paper **Internal Camera-based Reconstruction and Closed-loop Control of Soft Robotic Arms** by L. Vignoli, G. Pei, F. Braghin, J. Hughes.

<p align="center">
<img src="asset/setup_intro_a.png" alt="Sensing pipeline and multi-section arm" height="340px"/>
<img src="asset/setup_intro_b.png" alt="Mechanical layout with tendon routing" height="340px"/>
</p>

*(a) Sensorisation and reconstruction pipeline: embedded cameras feed CNNs that predict tip poses and full-body curvature. (b) Tendon-driven multi-section arm, three independent segments with three motors each; sections II and III are routed via Bowden cables.*

Vision-based framework for distributed state estimation and closed-loop control of tendon-driven soft robotic arms. Cameras embedded in each section feed CNNs that regress tip pose and polynomial curvature (constant, affine, quadratic). Two controllers are evaluated: a model-based null-space velocity controller built on the Piecewise Constant Curvature (PCC) model, and a data-driven inverse kinematics network with secondary-objective regularisation. Tested on single- and multi-section manipulators, in open- and closed-loop, with and without external disturbances.

---

## Robot

Continuum manipulator built on a Trimmed Helicoid (TH) structure, actuated by cable tendons. Each section is parameterised by curvature coordinates `q = [Dx, Dy, Dl]` (lateral deflections and length extension). Tendon-length changes ΔDELTAL [mm] map to Dynamixel encoder ticks via `unit_scale = 4096 / (π · D_pulley)`.

| | SINGLE | MULTI |
|---|--------|-------|
| Sections | 1 | 3 |
| Motors | 3 (IDs 1–3) | 9 (IDs 11–19) |
| Cameras | 1 USB | 3 USB (threaded) |
| Actuation DoF | 3 | 9 |
| Tip pose DoF | 6 | 6 |

---

## Control

Two Jacobian-based strategies. The **analytic Jacobian** is symbolic, derived with SymPy from the PCC model and evaluated numerically. The **data-driven Jacobian** is obtained via `torch.autograd.functional.jacobian` on the learned IK network. In MULTI, the model-based controller projects the secondary objective (mean tendon configuration) through the analytic null space; the data-driven controller bakes the same objective into training via a Lagrangian term.

<p align="center">
<img src="asset/model_based_multi.png" alt="Model-based control scheme" height="220px"/>
<img src="asset/data_driven_multi.png" alt="Data-driven control scheme" height="220px"/>
</p>

*Multi-section closed-loop control schemes with camera feedback. (a) Model-based controller with null-space optimisation toward the mean tendon configuration `ℓ̄`. (b) Data-driven controller using the learned inverse Jacobian; no encoder feedback.*

The forward and inverse kinematics networks of the data-driven path share a residual MLP backbone:

<p align="center">
<img src="asset/network_b.png" alt="MLP architecture" width="720px"/>
</p>

*MLP architecture for kinematics approximations. Inputs are projected into a 128-dim latent space and processed through three residual blocks (Linear + BatchNorm1d + LeakyReLU ×2, skip), followed by a linear regression head.*

---

## Sensing & Reconstruction

**Tip pose** is regressed by `helyx_model.pt` from stacked grayscale frames (one per section). **Full-body shape** uses dedicated CNN heads for constant, affine, and quadratic polynomial curvature per section. Ground-truth curvature labels are obtained by fitting these polynomials to OptiTrack backbone-marker rotations via CasADi/IPOPT.

<p align="center">
<img src="asset/network_a.png" alt="CNN architecture" width="780px"/>
</p>

*CNN architecture for full-body reconstruction. Stacked grayscale images go through online augmentations and 1- or 3-channel input convolutions, then a fine-tuned MobileNetV2 backbone with inverted residual bottlenecks. GAP and a linear head regress tip poses or per-section curvature parameters. Transfer learning is shared between pose and curvature networks and across SINGLE and MULTI configurations.*

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
│   ├── SINGLE_try_motion.py             # Random workspace exploration
│   ├── SINGLE_helyx_training.ipynb      # Model training
│   ├── SINGLE_helyx_testing.ipynb       # Model evaluation
│   └── SINGLE_control_results.ipynb     # Control result analysis
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
| `helyx_model.pt` | CNN: grayscale image(s) → tip pose (`f_p^1` or `f_p^3`) |
| `FK_model.pt` | Forward kinematics MLP: ΔDELTAL → tip pose (MULTI) |
| `IK_model.pt` | Inverse kinematics MLP with null-space regularisation: tip pose → ΔDELTAL (MULTI) |
| `constant_model.pt` / `affine_model.pt` / `quadratic_model.pt` | Polynomial curvature regressors |
| `*_for_MULTI_model.pt` | Section-specific curvature models for MULTI reconstruction |
| `DELTAL_stats.pt` | Normalisation statistics for the MULTI null-space objective |

Control results and workspace samples are stored as `.npz` archives. Raw training datasets are available from the corresponding author on reasonable request.

---

## Setup

### Hardware

| Component | Specification |
|-----------|---------------|
| Actuators | DYNAMIXEL XL330-M288-T, Protocol 2.0, 57600 baud, `/dev/ttyUSB0` |
| Pulley diameter | 6 mm |
| SINGLE motors | IDs 1–3, tendon angles 0°, 120°, −120° |
| MULTI motors | IDs 11–19 (three groups of three; sections II/III via Bowden cables) |
| Cameras | USB grayscale, 480 × 640 (OpenCV V4L2 backend) |
| Motion capture | OptiTrack Prime 13 ×6, via ROS `geometry_msgs/PoseStamped` topics |

Robot base and OptiTrack world frames are related by `R_corr = [[1,0,0],[0,0,-1],[0,1,0]]`. Lengths in millimetres, angles in radians.

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

ROS Noetic must be sourced. Before each session:

```bash
roslaunch optitrack_ros_communication optitrack_nodes.launch
v4l2-ctl --list-devices
```

---

## Usage

All scripts run from inside `SINGLE/` or `MULTI/`. Motor calibration is interactive (`w`/`s`/`h` keys, adaptive acceleration); MULTI camera-to-section assignment is also interactive.

### Data collection

```bash
# SINGLE: random ΔDELTAL sampling with camera + mocap (10,000 samples)
python SINGLE_data_acquire.py

# MULTI: concurrent camera + mocap acquisition with per-batch confirmation
python MULTI_data_acquire.py

# MULTI: FK dataset (ΔDELTAL → tip pose) for IK training
python MULTI_FK_acquire.py
```

### Model training and evaluation

```bash
# SINGLE
jupyter notebook SINGLE_helyx_training.ipynb
jupyter notebook SINGLE_helyx_testing.ipynb

# MULTI
jupyter notebook MULTI_helyx_training.ipynb
jupyter notebook MULTI_helyx_testing.ipynb
```

### Control evaluation

```bash
# SINGLE: closed/open-loop polygon tracking and workspace sampling
python SINGLE_control_test.py
jupyter notebook SINGLE_control_results.ipynb

# MULTI: interactive suite (closed/open-loop × data-driven/Jacobian × null-space check)
python MULTI_control_test.py
jupyter notebook MULTI_control_results.ipynb

# MULTI: analytical vs. ANN curvature validation
python MULTI_curvature_checks.py
```

The 50 g payload experiment is run by setting `WEIGHT = True` at the top of [`MULTI_control_test.py`](MULTI/MULTI_control_test.py) and re-running the workspace and polygon modalities.

### Utilities

```bash
# Random workspace exploration
python SINGLE_try_motion.py
python MULTI_try_motion.py
```

---

## Citation

```bibtex
@article{vignoli2026vision,
  author  = {Vignoli, Lorenzo and Pei, Guanran and Braghin, Francesco and Hughes, Josie},
  title   = {Internal Camera-based Reconstruction and Closed-loop Control of Soft Robotic Arms},
  year    = {2026}
}
```

---

## License

MIT, see [LICENSE](LICENSE).
