# Libraries
import numpy as np
import sys
import time
import rospy
import os
from datetime import datetime
from dynamixel_sdk import *
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R # type: ignore
import torch # type: ignore
from dataclasses import dataclass

# User-defined
from MULTI_jacobians import *
from MULTI_kinematics import *

# ======================================================================================
#                                    Functions
# ======================================================================================

def write_register(motor_id, address, value, size=1):
    if size == 1:
        packet_handler.write1ByteTxRx(port_handler, motor_id, address, value)
    elif size == 2:
        packet_handler.write2ByteTxRx(port_handler, motor_id, address, value)
    elif size == 4:
        packet_handler.write4ByteTxRx(port_handler, motor_id, address, value)

def pose_callback(msg, topic_name):
    global positions, quaternions, angles
    position = msg.pose.position
    orientation = msg.pose.orientation
    positions[topic_name] = [position.x, position.y, position.z]
    quaternions[topic_name] = [orientation.x, orientation.y, orientation.z, orientation.w]
    angles[topic_name] = R.from_quat(quaternions[topic_name]).as_euler('zyx')

def move_motors(d_DELTAL, TOLERANCE=20):
    def read_position_float(motor_id):
        raw, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        signed = np.array(raw, dtype=np.uint32).view(np.int32)
        return float(signed)

    start_positions = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)
    goal_positions = start_positions + (unit_scale * d_DELTAL).astype(np.float32)

    groupSyncWrite = GroupSyncWrite(port_handler, packet_handler, ADDR_GOAL_POSITION, 4)
    for motor_id, pos in zip(MOTOR_IDs, goal_positions):
        pos_int = int(round(pos))
        param = [DXL_LOBYTE(DXL_LOWORD(pos_int)), DXL_HIBYTE(DXL_LOWORD(pos_int)),
                 DXL_LOBYTE(DXL_HIWORD(pos_int)), DXL_HIBYTE(DXL_HIWORD(pos_int))]
        groupSyncWrite.addParam(motor_id, param)
    groupSyncWrite.txPacket()
    groupSyncWrite.clearParam()

    tic = time.time()
    while time.time() - tic < 1.5:
        current = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)
        if np.all(np.abs(current - goal_positions) <= TOLERANCE):
            break
        time.sleep(0.05)

    end_positions = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)
    d_tick = end_positions - start_positions
    d_DELTAL_real = d_tick / unit_scale
    return d_DELTAL_real

R_corr = np.array([
    [1, 0,  0],
    [0, 0, -1],
    [0, 1,  0]
])

def relative_coordinates(x_tip, y_tip, z_tip, yaw_tip, pitch_tip, roll_tip,
                         x_base, y_base, z_base, yaw_base, pitch_base, roll_base,
                         R_corr=R_corr):
    p_tip_global = np.array([x_tip, y_tip, z_tip])
    p_base_global = np.array([x_base, y_base, z_base])
    p_rel_global = p_tip_global - p_base_global
    p_rel_robot = R_corr @ p_rel_global

    R_tip_global = R.from_euler('zyx', [yaw_tip, pitch_tip, roll_tip]).as_matrix()
    R_base_global = R.from_euler('zyx', [yaw_base, pitch_base, roll_base]).as_matrix()
    R_rel = R_base_global.T @ R_tip_global
    euler_rel = R.from_matrix(R_rel).as_euler('zyx')

    return np.concatenate((p_rel_robot * 1000, euler_rel))

def SamplingSolver(delta_old, deltaL_old, delta_new):
    s1, s2, s3 = delta_old
    d1, d2, d3 = delta_new
    
    A_old = np.array([
        [np.cos(s1), np.sin(s1), -1],
        [np.cos(s2), np.sin(s2), -1],
        [np.cos(s3), np.sin(s3), -1]
    ])
    
    b = - deltaL_old
    solution = np.linalg.solve(A_old, b)

    A_new = np.array([
        [np.cos(d1), np.sin(d1), -1],
        [np.cos(d2), np.sin(d2), -1],
        [np.cos(d3), np.sin(d3), -1]
    ])

    deltaL_new = - A_new @ solution
    return deltaL_new

def read_DELTAL_from_motors():
    def read_position_float(motor_id):
        raw, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        signed = np.array(raw, dtype=np.uint32).view(np.int32)
        return float(signed)

    positions = np.array([read_position_float(mid) for mid in MOTOR_IDs], dtype=np.float32)
    DELTAL = positions / unit_scale
    return DELTAL

def get_mocap_pose(name):
    x_tip, y_tip, z_tip = positions.get(name)
    yaw_tip, pitch_tip, roll_tip = angles.get(name)
    return x_tip, y_tip, z_tip, yaw_tip, pitch_tip, roll_tip

def get_key():
    import termios
    import tty
    
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def calibrate_motors():
    print("--- Calibration started ---")
    base_step = np.pi * D_pulley / 12
    max_multiplier = 10
    accel_threshold = 0.4

    for idx, motor_id in enumerate(MOTOR_IDs):
        print(f"\n Motor {motor_id} selected. Press 'h' to skip, 'w' to loosen, 's' to tend.")
        last_key = None
        last_time = time.time()
        step_multiplier = 1

        while True:
            key = get_key()
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
                d_DEL = np.zeros(MOTOR_Num, dtype=np.float32)
                direction = 1 if key == 'w' else -1
                step = direction * base_step * step_multiplier
                d_DEL[idx] = step
                move_motors(d_DEL)
            else:
                print("\n Invalid input. Use 'h', 'w', or 's'.")

    print("\n --- Calibration completed ---")

# ======================================================================================
#                                   Dynamixel Setting
# ======================================================================================

DEVICENAME = '/dev/ttyUSB0'
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0
EXPOSITION_MODE = 4

MOTOR_IDs = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19])
MOTOR_Num = MOTOR_IDs.shape[0]

D_pulley = 6
unit_scale = 4096/(np.pi*D_pulley)

# ======================================================================================
#                                   Dynamixel Setup
# ======================================================================================

port_handler = PortHandler(DEVICENAME)
packet_handler = Protocol2PacketHandler()

if not port_handler.openPort():
    print("Failed to open port")
    sys.exit(1)
print("Port opened successfully")

if not port_handler.setBaudRate(BAUDRATE):
    print("Failed to set baudrate")
    sys.exit(1)
print("Baudrate set successfully")

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
ADDR_PROFILE_VELOCITY = 112

for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_OPERATING_MODE, EXPOSITION_MODE, size=1)

for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_TORQUE_ENABLE, 1, size=1)

VELOCITY_LIMIT = 100

for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_PROFILE_VELOCITY, VELOCITY_LIMIT, size=4)

print("--- Motors initialized ---")

# ======================================================================================
#                                Calibrate The Motors
# ======================================================================================

print("--- Press 'y' to calibrate the motors, ENTER to skip ---")
key = get_key()
if key == 'y':
    calibrate_motors()
    print("\n")

# ======================================================================================
#                                   Motion Capture Setup
# ======================================================================================

rospy.init_node('optitrack_listener')

positions = {}
quaternions = {}
angles = {}

helyx_topics = [
    ('/optitrack/BASE/pose', 'BASE'),
    ('/optitrack/TIP1/pose', 'TIP1'),
    ('/optitrack/TIP2/pose', 'TIP2'),
    ('/optitrack/TIP3/pose', 'TIP3')
]

subscribers = []
for topic_name, key in helyx_topics:
    sub = rospy.Subscriber(topic_name, PoseStamped, pose_callback, callback_args=key)
    subscribers.append(sub)

time.sleep(1)

# ======================================================================================
#                               Sequential Data Collection
# ======================================================================================

@dataclass
class DataSample:
    timestamp: float
    DELTAL: np.ndarray
    tip3_pose: np.ndarray

def load_existing_data(filename="data/FKdata.pt"):
    if os.path.exists(filename):
        try:
            print(f"Loading existing data from {filename}...")
            existing_data = torch.load(filename)
            
            existing_samples = []
            for i in range(len(existing_data["DELTAL"])):
                sample = DataSample(
                    timestamp=existing_data["timestamps"][i],
                    DELTAL=existing_data["DELTAL"][i].numpy(),
                    tip3_pose=existing_data["tip3_pose"][i].numpy()
                )
                existing_samples.append(sample)
            
            print(f"Loaded {len(existing_samples)} existing samples")
            return existing_samples
            
        except Exception as e:
            print(f"Failed to load: {e}")
            return []
    return []

def save_data(all_samples, filename="data/FKdata.pt"):
    if not all_samples:
        return

    deltal_torch = [torch.tensor(s.DELTAL, dtype=torch.float32) for s in all_samples]
    tip3_torch = [torch.tensor(s.tip3_pose, dtype=torch.float32) for s in all_samples]
    timestamps = [s.timestamp for s in all_samples]

    data_dict = {
        "DELTAL": deltal_torch,
        "tip3_pose": tip3_torch,
        "timestamps": timestamps,
        "sample_count": len(all_samples)
    }

    os.makedirs("data", exist_ok=True)
    torch.save(data_dict, filename)
    print(f"Saved {len(all_samples)} samples to {filename}")

def collect_sample(DELTAL_initial, DELTAL_reference):
    # Get mocap data
    x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
    x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")
    
    tip3_pose = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                                   x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
    
    # Get motor positions
    DELTAL = read_DELTAL_from_motors() + DELTAL_initial - DELTAL_reference
    
    return DataSample(
        timestamp=time.time(),
        DELTAL=DELTAL.copy(),
        tip3_pose=tip3_pose.copy()
    )

def generate_random_movement():
    delta1 = [np.radians(0), np.radians(120), np.radians(-120)]
    delta2 = [np.radians(60), np.radians(180), np.radians(-60)]
    delta3 = [np.radians(150), np.radians(270), np.radians(30)]
    
    config = np.random.randint(0, 3)
    
    if config == 0:
        DELTAL_first = np.random.uniform(low=-25, high=10, size=3)
        DELTAL_first_2 = SamplingSolver(delta1, DELTAL_first, delta2)
        DELTAL_second = DELTAL_first_2
        DELTAL_first_3 = SamplingSolver(delta1, DELTAL_first, delta3)
        DELTAL_third = DELTAL_first_3
    elif config == 1:
        DELTAL_first = np.random.uniform(low=-25, high=10, size=3)
        DELTAL_first_2 = SamplingSolver(delta1, DELTAL_first, delta2)
        DELTAL_second = DELTAL_first_2
        DELTAL_third = np.random.uniform(low=-35, high=10, size=3)
    elif config == 2:
        DELTAL_first = np.random.uniform(low=-25, high=10, size=3)
        DELTAL_second = np.random.uniform(low=-35, high=10, size=3)
        DELTAL_third = np.random.uniform(low=-40, high=10, size=3)

    return np.concatenate([DELTAL_first, DELTAL_second, DELTAL_third])

def sequential_data_collection(target_samples, DELTAL_initial, DELTAL_reference, settle_time=0.15, save_every=100):
    # Load existing data
    all_samples = load_existing_data()
    existing_count = len(all_samples)
    
    if existing_count >= target_samples:
        print(f"Already have {existing_count} samples (target: {target_samples})")
        return existing_count
    
    needed_samples = target_samples - existing_count
    print(f"Need {needed_samples} more samples")
    
    try:
        print(f"\nStarting sequential collection...")
        print("Press Ctrl+C to stop early")
        
        for i in range(needed_samples):
            # Generate random movement
            DELTAL_target = generate_random_movement()
            current_DELTAL = read_DELTAL_from_motors() - DELTAL_reference
            d_DELTAL = DELTAL_target - current_DELTAL
            
            # Move and settle
            move_motors(d_DELTAL)
            time.sleep(settle_time)
            
            # Collect data sample
            sample = collect_sample(DELTAL_initial, DELTAL_reference)
            all_samples.append(sample)
            
            total_samples = existing_count + i + 1
            print(f"Sample {total_samples}/{target_samples} collected")
            
            # Save periodically
            if (i + 1) % save_every == 0:
                save_data(all_samples)
                print(f"Saved at {total_samples} samples")
        
        # Final save
        save_data(all_samples)
        print(f"\nCollection complete! Total: {len(all_samples)} samples")
        return len(all_samples)
        
    except KeyboardInterrupt:
        print(f"\nUser stopped collection at {len(all_samples)} samples")
        save_data(all_samples)
        return len(all_samples)

# ======================================================================================
#                               Main Execution
# ======================================================================================

# Get initial position
print("Reading initial position...")
x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3 = get_mocap_pose("TIP3")
x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2 = get_mocap_pose("TIP2")
x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1 = get_mocap_pose("TIP1")
x_base, y_base, z_base, yaw_base, pitch_base, roll_base = get_mocap_pose("BASE")

tip3_initial = relative_coordinates(x_tip3, y_tip3, z_tip3, yaw_tip3, pitch_tip3, roll_tip3,
                                   x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
tip2_initial = relative_coordinates(x_tip2, y_tip2, z_tip2, yaw_tip2, pitch_tip2, roll_tip2,
                                   x_base, y_base, z_base, yaw_base, pitch_base, roll_base)
tip1_initial = relative_coordinates(x_tip1, y_tip1, z_tip1, yaw_tip1, pitch_tip1, roll_tip1,
                                   x_base, y_base, z_base, yaw_base, pitch_base, roll_base)

# Initial conditions
DELTAL0 = np.full(9, 1e-4)
DxDyDl0 = np.full(9, 1e-4)
xyz0 = np.array([1e-4, 1e-4, 145, 1e-4, 1e-4, 145 * 2, 1e-4, 1e-4, 145 * 3])

# Calculate references
xyz_initial = np.concatenate([tip1_initial[:3], tip2_initial[:3], tip3_initial[:3]])
DELTAL_initial, _, _ = xyz2tendon(DELTAL0, DxDyDl0, xyz0, xyz_initial - xyz0)
DELTAL_reference = read_DELTAL_from_motors()

print(f"DELTAL_initial = {DELTAL_initial}")
print(f"Initial xyz = {xyz_initial}")

# Get target samples
while True:
    try:
        target_samples = int(input("How many total samples? "))
        if target_samples > 0:
            break
        else:
            print("Please enter positive number")
    except ValueError:
        print("Please enter valid number")

# Confirm start
while True:
    ready = input(f"\nReady to collect {target_samples} samples sequentially? (y/n): ").lower().strip()
    if ready in ['y', 'yes']:
        break
    elif ready in ['n', 'no']:
        print("Exiting...")
        for motor_id in MOTOR_IDs:
            write_register(motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, size=1)
        port_handler.closePort()
        sys.exit(0)

print("\nStarting in 3 seconds...")
for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)

# Run sequential collection
final_count = sequential_data_collection(target_samples, DELTAL_initial, DELTAL_reference)

print(f"\n=== Collection Complete ===")
print(f"Total samples: {final_count}")
print(f"Data saved to: FKdata.pt")

# Close motors
for motor_id in MOTOR_IDs:
    write_register(motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, size=1)
port_handler.closePort()
print("Motors disabled and port closed")
