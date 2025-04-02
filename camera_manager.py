"""
Camera Manager
-------------
Handles camera discovery, initialization, and frame capture for Pupil Core cameras.
"""
import os
import time
import logging
import platform
import re
from fractions import Fraction
import multiprocessing as mp
import threading

import numpy as np
import uvc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exceptions
class NonMonotonicTimestampError(ValueError):
    """Raised when timestamps are not monotonically increasing."""
    pass

class CameraMissingError(Exception):
    """Raised when a camera cannot be found."""
    pass

class UVCSource:
    """
    Camera source implementation using the UVC driver.
    Manages camera connection, frame capture, and camera controls.
    """
    
    def __init__(self, name=None, uid=None, frame_size=(1280, 720), frame_rate=30, exposure_mode="auto"):
        """
        Initialize a UVC camera source.
        
        Args:
            name: Camera name or pattern to match
            uid: Specific camera UID
            frame_size: Desired frame size (width, height)
            frame_rate: Desired frame rate in fps
            exposure_mode: "auto" or "manual" exposure control
        """
        self.uvc_capture = None
        self._recent_frame = None
        self._last_ts = None
        self._restart_in = 3
        
        # Store initial settings for reconnection
        self.name_pattern = name
        self.uid = uid
        self.frame_size_backup = frame_size
        self.frame_rate_backup = frame_rate
        self.exposure_mode = exposure_mode
        
        # Add backup variables for camera controls
        self.exposure_time_backup = None
        self.gamma_backup = None
        self.saturation_backup = None
        
        # Find and initialize camera
        self.devices = uvc.Device_List()
        
        if uid:
            self._init_with_uid(uid)
        elif name:
            self._init_with_name(name)
        else:
            raise ValueError("Either name or uid must be provided")
        
        # Configure camera if initialized successfully
        if self.uvc_capture:
            self.configure_capture(frame_size, frame_rate)
    
    def _init_with_uid(self, uid):
        """Initialize camera with specific UID."""
        try:
            self.uvc_capture = uvc.Capture(uid)
            logger.info(f"Initialized camera with UID: {uid}")
        except uvc.OpenError:
            logger.warning(f"Camera with UID {uid} found but not available")
        except uvc.InitError:
            logger.error(f"Camera with UID {uid} failed to initialize")
        except uvc.DeviceNotFoundError:
            logger.warning(f"No camera found with UID {uid}")
    
    def _init_with_name(self, name_pattern):
        """Initialize first camera matching the name pattern."""
        found = False
        for device in self.devices:
            if name_pattern in device['name']:
                try:
                    self.uvc_capture = uvc.Capture(device['uid'])
                    logger.info(f"Initialized camera: {device['name']}")
                    found = True
                    break
                except (uvc.OpenError, uvc.InitError):
                    logger.warning(f"Camera {device['name']} found but not available")
        
        if not found:
            logger.warning(f"No available camera found matching: {name_pattern}")
    
    def configure_capture(self, frame_size, frame_rate):
        """Configure camera settings."""
        # Set timestamp handling based on camera type
        if "Pupil Cam" in self.uvc_capture.name:
            if platform.system() == "Windows":
                # Hardware timestamps problematic on Windows
                self.ts_offset = -0.01
            else:
                # Use hardware timestamps on other platforms
                self.ts_offset = None
            
            # Set bandwidth factor for Pupil cameras
            if "ID0" in self.uvc_capture.name or "ID1" in self.uvc_capture.name:
                self.uvc_capture.bandwidth_factor = 1.3
            else:
                self.uvc_capture.bandwidth_factor = 2.0
        else:
            # Non-Pupil cameras use software timestamps
            self.ts_offset = -0.1
        
        # Set frame size to closest available
        sizes = [
            abs(r[0] - frame_size[0]) + abs(r[1] - frame_size[1])
            for r in self.uvc_capture.frame_sizes
        ]
        best_size_idx = sizes.index(min(sizes))
        size = self.uvc_capture.frame_sizes[best_size_idx]
        self.uvc_capture.frame_size = size
        
        # Set frame rate to closest available
        rates = [abs(r - frame_rate) for r in self.uvc_capture.frame_rates]
        best_rate_idx = rates.index(min(rates))
        rate = self.uvc_capture.frame_rates[best_rate_idx]
        self.uvc_capture.frame_rate = rate
        
        # Log actual settings
        logger.info(f"Camera configured: {self.name} at {self.frame_size}@{self.frame_rate}fps")
        
        # Store settings for reconnection
        self.frame_size_backup = size
        self.frame_rate_backup = rate
        
        # Configure camera controls based on camera type
        controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
        
        # First set auto exposure controls for all cameras
        try:
            controls_dict["Auto Exposure Priority"].value = 0
        except KeyError:
            pass
        
        # Set auto exposure mode based on setting
        try:
            if self.exposure_mode == "manual":
                controls_dict["Auto Exposure Mode"].value = 1  # Manual mode
        except KeyError:
            pass
            
        if "Pupil Cam1" in self.uvc_capture.name:
            if "ID0" in self.uvc_capture.name or "ID1" in self.uvc_capture.name:
                # Eye camera settings
                # Only set exposure time if in manual mode
                if self.exposure_mode == "manual":
                    try:
                        if self.exposure_time_backup is None:
                            controls_dict["Absolute Exposure Time"].value = 63
                            self.exposure_time_backup = 63
                    except KeyError:
                        pass
                    
                try:
                    if self.gamma_backup is None:
                        controls_dict["Gamma"].value = 100
                        self.gamma_backup = 100
                except KeyError:
                    pass
                    
                try:
                    if self.saturation_backup is None:
                        controls_dict["Saturation"].value = 0
                        self.saturation_backup = 0
                except KeyError:
                    pass
            else:
                # World camera settings for ID2
                try:
                    if self.gamma_backup is None:
                        controls_dict["Gamma"].value = 100
                        self.gamma_backup = 100
                except KeyError:
                    pass
                    
        elif "Pupil Cam2" in self.uvc_capture.name:
            # Pupil Cam2 settings
            max_exposure = 32
            if self.frame_rate == 200:
                max_exposure = 28
            elif self.frame_rate == 180:
                max_exposure = 31
            
            # Only set exposure time if in manual mode
            if self.exposure_mode == "manual":
                try:
                    if self.exposure_time_backup is None:
                        controls_dict["Absolute Exposure Time"].value = max_exposure
                        self.exposure_time_backup = max_exposure
                except KeyError:
                    pass
                
            try:
                if self.gamma_backup is None:
                    controls_dict["Gamma"].value = 200
                    self.gamma_backup = 200
            except KeyError:
                pass
                
            try:
                if self.saturation_backup is None:
                    controls_dict["Saturation"].value = 0
                    self.saturation_backup = 0
            except KeyError:
                pass
    
    @property
    def exposure_time(self):
        """Get current absolute exposure time."""
        if not self.uvc_capture:
            return self.exposure_time_backup
        
        try:
            controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
            return controls_dict["Absolute Exposure Time"].value
        except KeyError:
            return None

    @exposure_time.setter
    def exposure_time(self, value):
        """Set the exposure time."""
        if not self.uvc_capture:
            self.exposure_time_backup = value
            return
        
        try:
            controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
            
            # Only apply if in manual mode
            if self.exposure_mode == "manual":
                # Ensure auto exposure is in manual mode
                try:
                    controls_dict["Auto Exposure Mode"].value = 1  # Manual mode
                except KeyError:
                    pass
                
                # Determine the appropriate maximum exposure based on frame rate
                max_exposure = 32
                if self.frame_rate >= 200:
                    max_exposure = 28
                elif self.frame_rate >= 180:
                    max_exposure = 31
                
                # Constrain value to valid range
                value = min(max_exposure, max(1, value))
                
                # Only set if significantly different from current
                current_value = controls_dict["Absolute Exposure Time"].value
                if abs(value - current_value) >= 1:
                    controls_dict["Absolute Exposure Time"].value = value
                    self.exposure_time_backup = value
            else:
                logger.debug("Camera in auto exposure mode. Manual exposure values ignored.")
        except KeyError:
            logger.warning("Could not set exposure time: control not found")

    @property
    def gamma(self):
        """Get current gamma value."""
        if not self.uvc_capture:
            return self.gamma_backup
        
        try:
            controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
            return controls_dict["Gamma"].value
        except KeyError:
            return None

    @gamma.setter
    def gamma(self, value):
        """Set the gamma value."""
        if not self.uvc_capture:
            self.gamma_backup = value
            return
        
        try:
            controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
            
            # Only set if significantly different from current
            current_value = controls_dict["Gamma"].value
            if abs(value - current_value) >= 1:
                controls_dict["Gamma"].value = value
                self.gamma_backup = value
        except KeyError:
            logger.warning("Could not set gamma: control not found")

    @property
    def saturation(self):
        """Get current saturation value."""
        if not self.uvc_capture:
            return self.saturation_backup
        
        try:
            controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
            return controls_dict["Saturation"].value
        except KeyError:
            return None

    @saturation.setter
    def saturation(self, value):
        """Set the saturation value."""
        if not self.uvc_capture:
            self.saturation_backup = value
            return
        
        try:
            controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
            
            # Only set if significantly different from current
            current_value = controls_dict["Saturation"].value
            if abs(value - current_value) >= 1:
                controls_dict["Saturation"].value = value
                self.saturation_backup = value
        except KeyError:
            logger.warning("Could not set saturation: control not found")
    
    def get_frame(self):
        """
        Get the latest frame from the camera.
        
        Returns:
            Frame object or None if no frame is available
        """
        if not self.uvc_capture:
            return None
        
        try:
            # Get frame with timeout
            frame = self.uvc_capture.get_frame(0.05)
            
            # Handle invalid timestamps (can happen after reconnection)
            if np.isclose(frame.timestamp, 0):
                logger.debug("Frame has invalid timestamp (0). Dropping frame.")
                return None
            
            # Apply timestamp offset for software timestamps
            if self.ts_offset is not None:
                frame.timestamp = uvc.get_time_monotonic() + self.ts_offset
            
            # Ensure monotonic timestamps
            if self._last_ts is not None and frame.timestamp <= self._last_ts:
                logger.debug(f"Non-monotonic timestamps: {self._last_ts} -> {frame.timestamp}. Dropping frame.")
                return None
            
            # Store and return valid frame
            self._last_ts = frame.timestamp
            self._recent_frame = frame
            return frame
            
        except (uvc.StreamError, TimeoutError):
            # Handle streaming errors
            self._restart_logic()
            return None
            
        except (AttributeError, uvc.InitError):
            # Handle initialization errors
            time.sleep(0.02)
            self._restart_logic()
            return None
    
    def _restart_logic(self):
        """Handle camera disconnection and attempt reconnection."""
        if self._restart_in <= 0:
            if self.uvc_capture:
                logger.warning("Camera disconnected. Reconnecting...")
                
                if self.uvc_capture:
                    self.uvc_capture.close()
                    self.uvc_capture = None
            
            # Update device list
            self.devices.update()
            
            # Try to reconnect
            if self.uid:
                self._init_with_uid(self.uid)
            elif self.name_pattern:
                self._init_with_name(self.name_pattern)
                
            # Reconfigure if reconnected
            if self.uvc_capture:
                self.configure_capture(self.frame_size_backup, self.frame_rate_backup)
                
            self._restart_in = int(5 / 0.02)  # Reset retry counter
        else:
            self._restart_in -= 1
    
    @property
    def name(self):
        """Get camera name."""
        if self.uvc_capture:
            return self.uvc_capture.name
        return "(disconnected)"
    
    @property
    def frame_size(self):
        """Get current frame size."""
        if self.uvc_capture:
            return self.uvc_capture.frame_size
        return self.frame_size_backup
    
    @property
    def frame_rate(self):
        """Get current frame rate."""
        if self.uvc_capture:
            return self.uvc_capture.frame_rate
        return self.frame_rate_backup
    
    @property
    def online(self):
        """Check if camera is online."""
        return self.uvc_capture is not None
    
    @property
    def jpeg_support(self):
        """Check if camera supports JPEG compression."""
        return self.uvc_capture is not None
    
    def get_all_controls(self):
        """Get all camera controls as a dictionary."""
        controls = {}
        if self.uvc_capture:
            for c in self.uvc_capture.controls:
                controls[c.display_name] = {
                    "value": c.value,
                    "min": c.min_val,
                    "max": c.max_val,
                    "step": c.step,
                    "default": c.def_val
                }
        return controls
    
    def set_control_value(self, control_name, value):
        """Set a camera control to a specific value."""
        if not self.uvc_capture:
            return False
        
        try:
            controls_dict = {c.display_name: c for c in self.uvc_capture.controls}
            control = controls_dict[control_name]
            control.value = value
            return True
        except (KeyError, Exception) as e:
            logger.warning(f"Could not set {control_name} to {value}: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.devices:
            self.devices.cleanup()
            self.devices = None
        
        if self.uvc_capture:
            self.uvc_capture.close()
            self.uvc_capture = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def list_available_cameras():
    """
    List all available UVC cameras.
    
    Returns:
        List of dicts with camera info
    """
    devices = uvc.Device_List()
    return [
        {
            "id": i,
            "name": d["name"],
            "uid": d["uid"]
        }
        for i, d in enumerate(devices)
    ]


def find_camera_by_name(name_pattern):
    """
    Find a camera by name pattern.
    
    Args:
        name_pattern: String pattern to match in camera name
        
    Returns:
        Camera info dict or None if not found
    """
    devices = uvc.Device_List()
    for device in devices:
        if name_pattern in device['name']:
            return device
    return None


def find_camera_by_uid(uid):
    """
    Find a camera by UID.
    
    Args:
        uid: Camera UID to find
        
    Returns:
        Camera info dict or None if not found
    """
    devices = uvc.Device_List()
    for device in devices:
        if device['uid'] == uid:
            return device
    return None


def create_camera_manager(world_name="ID2", eye0_name="ID0", eye1_name="ID1", 
                         world_exposure_mode="auto",  # Separate exposure mode for world camera
                         eye_exposure_mode="manual",  # Separate exposure mode for eye cameras
                         world_size=(1280, 720), world_fps=30, 
                         eye_size=(192, 192), eye_fps=120):
    """
    Create a complete camera manager with world and eye cameras.
    
    Args:
        world_name: Name pattern for world camera
        eye0_name: Name pattern for eye0 camera
        eye1_name: Name pattern for eye1 camera
        world_exposure_mode: "auto" or "manual" exposure control for world camera
        eye_exposure_mode: "auto" or "manual" exposure control for eye cameras
        world_size: Frame size for world camera
        world_fps: Frame rate for world camera
        eye_size: Frame size for eye cameras
        eye_fps: Frame rate for eye cameras
        
    Returns:
        Dict with cameras
    """
    # Initialize cameras with different exposure modes
    world_cam = UVCSource(name=world_name, frame_size=world_size, frame_rate=world_fps, 
                          exposure_mode=world_exposure_mode)
    
    eye0_cam = UVCSource(name=eye0_name, frame_size=eye_size, frame_rate=eye_fps, 
                         exposure_mode=eye_exposure_mode)
    
    eye1_cam = UVCSource(name=eye1_name, frame_size=eye_size, frame_rate=eye_fps, 
                         exposure_mode=eye_exposure_mode)
    
    # Collect a few frames to stabilize cameras
    for _ in range(10):
        world_cam.get_frame()
        eye0_cam.get_frame()
        eye1_cam.get_frame()
        time.sleep(0.01)
    
    return {
        "world_cam": world_cam,
        "eye0_cam": eye0_cam,
        "eye1_cam": eye1_cam
    }


class CameraCapture:
    """
    High-level camera capture class with threading support.
    Provides continuous frame capture in a separate thread.
    """
    
    def __init__(self, camera):
        """
        Initialize camera capture.
        
        Args:
            camera: UVCSource instance
        """
        self.camera = camera
        self.running = False
        self.thread = None
        self.current_frame = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.frame_callback = None
    
    def start_capture(self, frame_callback=None):
        """
        Start frame capture in a separate thread.
        
        Args:
            frame_callback: Optional callback function to call for each new frame
        """
        if self.running:
            return
        
        self.running = True
        self.frame_callback = frame_callback
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_capture(self):
        """Stop frame capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
    
    def get_latest_frame(self):
        """Get the most recent captured frame."""
        with self.lock:
            return self.current_frame
    
    def wait_for_frame(self, timeout=1.0):
        """
        Wait for a new frame to be captured.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if a new frame is available, False on timeout
        """
        if self.new_frame_event.wait(timeout):
            self.new_frame_event.clear()
            return True
        return False
    
    def _capture_loop(self):
        """Internal frame capture loop."""
        while self.running and self.camera.online:
            frame = self.camera.get_frame()
            if frame:
                with self.lock:
                    self.current_frame = frame
                    self.frame_count += 1
                
                self.new_frame_event.set()
                
                if self.frame_callback:
                    self.frame_callback(frame)
            
            # Small sleep to prevent busy waiting
            time.sleep(0.001)


# Simple usage example
if __name__ == "__main__":
    # List all cameras
    print("Available cameras:")
    for cam in list_available_cameras():
        print(f"  {cam['id']}: {cam['name']} (UID: {cam['uid']})")
    
    # Create a camera manager
    print("\nInitializing cameras...")
    manager = create_camera_manager()
    
    # Show camera info
    for name, cam in manager.items():
        if cam.online:
            print(f"{name}: {cam.name} at {cam.frame_size}@{cam.frame_rate}fps")
    
    # # Capture a few frames from the world camera
    # if manager["world_cam"].online:
    #     print("\nCapturing 10 frames from world camera...")
    #     for i in range(10):
    #         frame = manager["world_cam"].get_frame()
    #         if frame:
    #             print(f"Frame {i+1}: {frame.width}x{frame.height} at time {frame.timestamp:.3f}")
    #         time.sleep(0.1)
    
    # Clean up
    for cam in manager.values():
        cam.cleanup()
    
    print("Done.")