import cv2
import threading
import time
import numpy as np

class ThreadedCameraSystem:
    """
    Multi-camera system using threading for concurrent frame capture.
    Handles sequential initialization to avoid USB/driver conflicts.
    """
    
    def __init__(self, camera_ids):
        """
        Initialize the camera system.
        
        Args:
            camera_ids (list): List of camera device IDs [cam1_id, cam2_id, cam3_id]
        """
        self.camera_ids = camera_ids
        self.caps = []
        self.latest_frames = [None] * len(camera_ids)
        self.locks = [threading.Lock() for _ in range(len(camera_ids))]
        self.threads = []
        self.running = False
        
    def initialize_cameras(self):
        """
        Initialize cameras sequentially to avoid conflicts.
        Each camera is opened with a delay to prevent simultaneous access.
        """
        print("Initializing cameras sequentially...")
        
        for i, cam_id in enumerate(self.camera_ids):
            print(f"Opening camera {i+1} (device {cam_id})...")
            
            # Open camera with V4L2 backend
            cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
            
            if not cap.isOpened():
                print(f"ERROR: Could not open camera {i+1} (device {cam_id})")
                self.caps.append(None)
                continue
            
            # Configure camera settings for optimal threading performance
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Minimal buffer for fresh frames
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduce resolution for speed
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 3)            # Lower FPS to reduce bandwidth
            
            self.caps.append(cap)
            print(f"Camera {i+1} initialized successfully")
            
            # Wait between camera initializations to avoid conflicts
            if i < len(self.camera_ids) - 1:
                time.sleep(1)
        
        print("Camera initialization complete\n")
    
    def camera_capture_thread(self, camera_index):
        """
        Continuous frame capture thread for a single camera.
        
        Args:
            camera_index (int): Index of the camera to capture from
        """
        cap = self.caps[camera_index]
        
        if cap is None:
            print(f"Thread {camera_index+1}: Camera not available")
            return
        
        print(f"Thread {camera_index+1}: Starting capture...")
        
        while self.running:
            try:
                # Capture frame
                ret, frame = cap.read()
                
                if ret:
                    # Thread-safe frame storage
                    with self.locks[camera_index]:
                        self.latest_frames[camera_index] = frame.copy()
                else:
                    print(f"Thread {camera_index+1}: Failed to read frame")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.27)
                
            except Exception as e:
                print(f"Thread {camera_index+1}: Error - {e}")
                break
        
        print(f"Thread {camera_index+1}: Stopped")
    
    def start_capture_threads(self):
        """
        Start all camera capture threads with sequential delays.
        """
        print("Starting capture threads...")
        self.running = True
        
        for i in range(len(self.caps)):
            if self.caps[i] is not None:
                thread = threading.Thread(
                    target=self.camera_capture_thread, 
                    args=(i,), 
                    daemon=True,
                    name=f"Camera_{i+1}_Thread"
                )
                thread.start()
                self.threads.append(thread)
                
                # Delay between thread starts to reduce initial conflicts
                time.sleep(2)
        
        print("All threads started")
        
        # Wait for frames to populate
        print("Waiting for initial frames...")
        time.sleep(3)
        print("System ready!\n")
    
    def get_all_frames(self):
        """
        Retrieve the latest frames from all cameras.
        Thread-safe access to current frame data.
        
        Returns:
            list: List of latest frames [frame1, frame2, frame3]
        """
        frames = []
        
        for i in range(len(self.caps)):
            with self.locks[i]:
                if self.latest_frames[i] is not None:
                    frames.append(self.latest_frames[i].copy())
                else:
                    frames.append(None)
        
        return frames
    
    def get_frame_status(self):
        """
        Get status of all cameras (whether they have valid frames).
        
        Returns:
            list: Boolean list indicating frame availability
        """
        status = []
        for i in range(len(self.caps)):
            with self.locks[i]:
                status.append(self.latest_frames[i] is not None)
        return status
    
    def stop_capture(self):
        """
        Stop all capture threads and release camera resources.
        """
        print("Stopping capture system...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        # Release all cameras
        for i, cap in enumerate(self.caps):
            if cap is not None:
                cap.release()
                print(f"Camera {i+1} released")
        
        cv2.destroyAllWindows()
        print("System stopped")

def measure_capture_performance(camera_system, num_measurements=100, silent=False):
    """
    Measure the time required to read frames from all cameras.
    
    Args:
        camera_system: ThreadedCameraSystem instance
        num_measurements (int): Number of measurements to take
        silent (bool): If True, don't print during measurement
    
    Returns:
        dict: Performance statistics
    """
    if not silent:
        print(f"Measuring capture performance over {num_measurements} readings...\n")
    
    times = []
    successful_reads = 0
    
    for i in range(num_measurements):
        start_time = time.time()
        
        # Get frames from all cameras
        frames = camera_system.get_all_frames()
        
        end_time = time.time()
        capture_time = end_time - start_time
        times.append(capture_time)
        
        # Check frame status
        frame_status = [f is not None for f in frames]
        valid_frames = sum(frame_status)
        
        if valid_frames == len(frames):
            successful_reads += 1
        
        if not silent and i % 10 == 0:  # Print status every 10 readings
            print(f"Reading {i+1:3d}: {capture_time*1000:.2f}ms, "
                  f"Valid frames: {valid_frames}/{len(frames)}")
    
    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    std_time = np.std(times) * 1000
    success_rate = (successful_reads / num_measurements) * 100
    
    results = {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'std_time': std_time,
        'success_rate': success_rate,
        'effective_fps': 1000/avg_time
    }
    
    return results

def display_cameras(camera_system, duration=10):
    """
    Display camera feeds for visual verification.
    
    Args:
        camera_system: ThreadedCameraSystem instance
        duration (int): Display duration in seconds
    """
    print(f"Displaying camera feeds for {duration} seconds...")
    print("Press 'q' to stop early")
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        frames = camera_system.get_all_frames()
        
        # Display each valid frame
        for i, frame in enumerate(frames):
            if frame is not None:
                cv2.imshow(f"Camera {i+1}", frame)
            else:
                # Create placeholder for missing camera
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Camera {i+1}", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(placeholder, "NO SIGNAL", (220, 280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(f"Camera {i+1}", placeholder)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.033)  # ~30 FPS display rate
    
    cv2.destroyAllWindows()

# Probe and assign cameras
def probe_and_assign_cameras(max_index=15):
    assignments = {}
    assigned_ids = set()

    print("Scanning for cameras and displaying them one by one...")
    print("Press 1, 2, or 3 to assign the current camera to that position.")
    print("Press any other key to skip to the next camera.\n")

    for i in range(max_index):
        print(f"Testing camera {i}...")
        
        # Try to open camera
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            print(f"Camera {i}: Could not open")
            cap.release()
            continue

        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {i}: Could not read frame")
            cap.release()
            continue

        print(f"Camera {i}: Working! Displaying...")
        
        # Show the camera feed continuously until key press
        window_name = f"Camera {i} - Press 1/2/3 to assign or any key to skip"
        cv2.namedWindow(window_name)
        
        while True:
            ret, frame = cap.read()
            if ret:
                # Resize for easier viewing
                resized = cv2.resize(frame, (640, 480))
                cv2.imshow(window_name, resized)
                
                # Check for key press every 30ms
                key = cv2.waitKey(30) & 0xFF
                if key != 255:  # A key was pressed
                    break
            else:
                print(f"Camera {i}: Lost connection")
                break
        
        # Close the window
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # Allow window to close
        
        # Process the key press
        if key in [ord('1'), ord('2'), ord('3')]:
            cam_num = int(chr(key))
            if cam_num in assigned_ids:
                print(f"Camera position {cam_num} is already assigned. Skipping camera {i}.")
            else:
                assignments[f"camera_{cam_num}"] = i
                assigned_ids.add(cam_num)
                print(f"✓ Assigned Camera {i} as camera_{cam_num}")
        else:
            print(f"Skipped camera {i}")
        
        cap.release()
        
        # Stop if we have all 3 cameras assigned
        if len(assignments) == 3:
            print("All 3 cameras assigned!")
            break

    return assignments

def color2gray(frame_image):
    """
    Converts color image to gray one.
    """

    if frame_image is not None:
        image_gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = None
    return image_gray
