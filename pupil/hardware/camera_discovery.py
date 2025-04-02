"""
Camera Discovery
--------------
Functions for listing, finding, and initializing cameras for Pupil Core.
"""
import logging
import uvc
from .camera import UVCCamera

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class CameraManager:
    """
    Camera manager class that handles multiple cameras used in Pupil Core.
    Manages initialization, configuration, and provides access to cameras.
    """
    def __init__(self, world_name="ID2", eye0_name="ID0", eye1_name="ID1", 
                world_exposure_mode="auto", eye_exposure_mode="manual",
                world_size=(1280, 720), world_fps=30, 
                eye_size=(192, 192), eye_fps=120):
        """
        Create a camera manager with world and eye cameras.
        
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
        """
        # Initialize cameras with different exposure modes
        self.world_cam = UVCCamera(name=world_name, frame_size=world_size, 
                                  frame_rate=world_fps, exposure_mode=world_exposure_mode)
        
        self.eye0_cam = UVCCamera(name=eye0_name, frame_size=eye_size, 
                                 frame_rate=eye_fps, exposure_mode=eye_exposure_mode)
        
        self.eye1_cam = UVCCamera(name=eye1_name, frame_size=eye_size, 
                                 frame_rate=eye_fps, exposure_mode=eye_exposure_mode)
        
        # Dictionary access for cameras
        self.cameras = {
            "world_cam": self.world_cam,
            "eye0_cam": self.eye0_cam,
            "eye1_cam": self.eye1_cam
        }
        
        # Collect a few frames to stabilize cameras
        self._initialize_cameras()

    def _initialize_cameras(self):
        """Collect a few frames to stabilize cameras."""
        import time
        for _ in range(10):
            self.world_cam.get_frame()
            self.eye0_cam.get_frame()
            self.eye1_cam.get_frame()
            time.sleep(0.01)
    
    def get_camera(self, name):
        """
        Get a camera by name.
        
        Args:
            name: Camera name ("world_cam", "eye0_cam", or "eye1_cam")
            
        Returns:
            Camera object or None if not found
        """
        return self.cameras.get(name)
    
    def get_all_cameras(self):
        """
        Get all cameras as a dictionary.
        
        Returns:
            Dict with camera objects
        """
        return self.cameras
    
    def get_online_cameras(self):
        """
        Get all online cameras.
        
        Returns:
            Dict with online camera objects
        """
        return {name: cam for name, cam in self.cameras.items() if cam.online}
    
    def cleanup(self):
        """Clean up all cameras."""
        for cam in self.cameras.values():
            cam.cleanup()


# Factory function for backward compatibility
def create_camera_manager(world_name="ID2", eye0_name="ID0", eye1_name="ID1", 
                         world_exposure_mode="auto", eye_exposure_mode="manual",
                         world_size=(1280, 720), world_fps=30, 
                         eye_size=(192, 192), eye_fps=120):
    """
    Create a camera manager with world and eye cameras.
    
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
        CameraManager object
    """
    manager = CameraManager(
        world_name=world_name, 
        eye0_name=eye0_name,
        eye1_name=eye1_name,
        world_exposure_mode=world_exposure_mode,
        eye_exposure_mode=eye_exposure_mode,
        world_size=world_size,
        world_fps=world_fps,
        eye_size=eye_size,
        eye_fps=eye_fps
    )
    
    return manager.get_all_cameras()