"""
Recorder
--------
Video recorder for multiple cameras with session management.
"""
import os
import time
import logging
import uuid
import json
import threading
from pathlib import Path

import numpy as np

from .video_writer import MPEGWriter, JPEGWriter, H264Writer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Recorder:
    """Video recorder for multiple cameras."""
    
    def __init__(self, rec_dir=None, session_name=None, use_jpeg=True):
        """
        Initialize the recorder.
        
        Args:
            rec_dir: Root directory for recordings
            session_name: Name for the recording session
            use_jpeg: Whether to use JPEG encoding when available
        """
        # Set recording directory
        if not rec_dir:
            rec_dir = os.path.join(str(Path.home()), "pupil_recordings")
        self.rec_dir = rec_dir
        os.makedirs(rec_dir, exist_ok=True)
        
        # Set session name
        if not session_name:
            session_name = time.strftime("%Y_%m_%d", time.localtime())
        self.session_name = session_name
        
        # Initialize state
        self.use_jpeg = use_jpeg
        self.running = False
        self.world_writer = None
        self.eye_writers = {}
        self.current_rec_path = None
        self.frame_count = 0  # Keep for backward compatibility
        # Track frames for each camera separately
        self.frame_counts = {"world": 0, "eye0": 0, "eye1": 0}
        self.start_time = None
        
        # Add a lock for thread safety
        self.lock = threading.Lock()
    
    def start(self, world_cam=None, eye0_cam=None, eye1_cam=None):
        """
        Start recording from cameras.
        
        Args:
            world_cam: World camera
            eye0_cam: First eye camera
            eye1_cam: Second eye camera
            
        Returns:
            Path to the recording directory
        """
        with self.lock:  # Thread safety
            if self.running:
                logger.warning("Recording already running")
                return self.current_rec_path
            
            # Get synchronized start time - use time.monotonic() if uvc not available
            try:
                import uvc
                self.start_time = uvc.get_time_monotonic()
            except ImportError:
                self.start_time = time.monotonic()
                
            logger.info(f"Recording start time: {self.start_time}")
            
            # Create session directory
            session_dir = os.path.join(self.rec_dir, self.session_name)
            os.makedirs(session_dir, exist_ok=True)
            
            # Create new numbered recording directory
            counter = 0
            while True:
                rec_path = os.path.join(session_dir, f"{counter:03d}")
                if not os.path.exists(rec_path):
                    os.makedirs(rec_path)
                    break
                counter += 1
            
            self.current_rec_path = rec_path
            logger.info(f"Created recording directory: {rec_path}")
            
            # Create metadata
            self._save_recording_info()
            
            # Initialize video writers
            try:
                if world_cam and world_cam.online:
                    self._init_world_writer(world_cam)
                
                if eye0_cam and eye0_cam.online:
                    self._init_eye_writer(eye0_cam, "eye0")
                
                if eye1_cam and eye1_cam.online:
                    self._init_eye_writer(eye1_cam, "eye1")
                
                self.running = True
                self.frame_count = 0
                # Reset frame counters
                self.frame_counts = {"world": 0, "eye0": 0, "eye1": 0}
                
            except Exception as e:
                logger.error(f"Error initializing writers: {e}")
                # Clean up on error
                self._cleanup_writers()
                raise
                
            return self.current_rec_path
    
    def update(self, world_frame=None, eye0_frame=None, eye1_frame=None):
        """
        Update recording with new frames with thread safety.
        
        Args:
            world_frame: New world camera frame
            eye0_frame: New eye0 camera frame
            eye1_frame: New eye1 camera frame
        """
        with self.lock:  # Thread safety
            if not self.running:
                return
            
            # Write world frame
            if world_frame and self.world_writer:
                if self.world_writer.write_frame(world_frame):
                    self.frame_count += 1  # Keep for backward compatibility
                    self.frame_counts["world"] += 1
            
            # Write eye frames
            if eye0_frame and "eye0" in self.eye_writers:
                if self.eye_writers["eye0"].write_frame(eye0_frame):
                    self.frame_counts["eye0"] += 1
            
            if eye1_frame and "eye1" in self.eye_writers:
                if self.eye_writers["eye1"].write_frame(eye1_frame):
                    self.frame_counts["eye1"] += 1
    
    def stop(self):
        """
        Stop recording and finalize files.
        
        Returns:
            Tuple of (recording path, statistics)
        """
        with self.lock:  # Thread safety
            if not self.running:
                # Return empty stats if not running
                return self.current_rec_path, {
                    "world": {"frames": 0, "fps": 0},
                    "eye0": {"frames": 0, "fps": 0},
                    "eye1": {"frames": 0, "fps": 0}
                }
            
            # Calculate recording duration
            try:
                import uvc
                duration = uvc.get_time_monotonic() - self.start_time
            except ImportError:
                duration = time.monotonic() - self.start_time
            
            # Calculate statistics
            stats = {}
            for camera_id in ["world", "eye0", "eye1"]:
                frames = self.frame_counts[camera_id]
                fps = frames / duration if duration > 0 else 0
                stats[camera_id] = {"frames": frames, "fps": fps}
            
            # Close all writers
            self._cleanup_writers()
            
            # Update metadata with duration and stats
            self._update_recording_info(duration, stats)
            
            # Set state to not running before returning
            self.running = False
            logger.info(f"Recording stopped. Duration: {duration:.2f}s, Total frames: {self.frame_count}")
            
            # Store path before possibly changing in a new recording
            current_path = self.current_rec_path
            
            return current_path, stats
    
    def _cleanup_writers(self):
        """Clean up all writers properly with error handling."""
        try:
            if self.world_writer:
                self.world_writer.close()
                self.world_writer = None
        except Exception as e:
            logger.error(f"Error closing world writer: {e}")
            self.world_writer = None
        
        for eye_id, writer in list(self.eye_writers.items()):
            try:
                writer.close()
            except Exception as e:
                logger.error(f"Error closing {eye_id} writer: {e}")
            finally:
                self.eye_writers.pop(eye_id, None)
    
    def _init_world_writer(self, camera):
        """Initialize world camera writer."""
        path = os.path.join(self.current_rec_path, "world.mp4")
        
        # Safely check if camera has a recent frame
        if not hasattr(camera, '_recent_frame') or camera._recent_frame is None:
            logger.warning("World camera has no recent frame, cannot initialize writer")
            return
        
        # Check for H264 buffer first
        if hasattr(camera._recent_frame, "h264_buffer"):
            self.world_writer = H264Writer(path, camera.frame_size[0], camera.frame_size[1], camera.frame_rate)
            logger.info("Using H264 encoding for world camera")
        # Use JPEG writer if supported and enabled
        elif self.use_jpeg and hasattr(camera._recent_frame, "jpeg_buffer"):
            self.world_writer = JPEGWriter(path, self.start_time)
            logger.info("Using JPEG encoding for world camera")
        else:
            self.world_writer = MPEGWriter(path, self.start_time)
            logger.info("Using MPEG encoding for world camera")
    
    def _init_eye_writer(self, camera, eye_id):
        """Initialize eye camera writer."""
        path = os.path.join(self.current_rec_path, f"{eye_id}.mp4")
        
        # Safely check if camera has a recent frame
        if not hasattr(camera, '_recent_frame') or camera._recent_frame is None:
            logger.warning(f"{eye_id} camera has no recent frame, cannot initialize writer")
            return
        
        # Check for H264 buffer first
        if hasattr(camera._recent_frame, "h264_buffer"):
            self.eye_writers[eye_id] = H264Writer(path, camera.frame_size[0], camera.frame_size[1], camera.frame_rate)
            logger.info(f"Using H264 encoding for {eye_id} camera")
        # Use JPEG writer if supported and enabled
        elif self.use_jpeg and hasattr(camera._recent_frame, "jpeg_buffer"):
            self.eye_writers[eye_id] = JPEGWriter(path, self.start_time)
            logger.info(f"Using JPEG encoding for {eye_id} camera")
        else:
            self.eye_writers[eye_id] = MPEGWriter(path, self.start_time)
            logger.info(f"Using MPEG encoding for {eye_id} camera")
    
    def _save_recording_info(self):
        """Save recording metadata."""
        try:
            info = {
                "recording_software": "Pupil Core Minimal",
                "recording_uuid": str(uuid.uuid4()),
                "start_time_system": time.time(),
                "start_time_synced": self.start_time,
                "session_name": self.session_name
            }
            
            with open(os.path.join(self.current_rec_path, "info.json"), "w") as f:
                json.dump(info, f, indent=4)
                
            logger.info(f"Saved recording info")
            
        except Exception as e:
            logger.error(f"Failed to save recording info: {e}")
    
    def _update_recording_info(self, duration, stats=None):
        """Update recording metadata with duration and stats."""
        try:
            info_path = os.path.join(self.current_rec_path, "info.json")
            
            if not os.path.exists(info_path):
                logger.warning(f"Recording info file not found: {info_path}")
                return
                
            with open(info_path, "r") as f:
                info = json.load(f)
            
            info["duration"] = duration
            info["frame_count"] = self.frame_count
            
            # Add per-camera statistics if available
            if stats:
                info["frame_stats"] = stats
            
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4)
                
            logger.info(f"Updated recording info with duration: {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to update recording info: {e}")
    
    @property
    def recording_time(self):
        """Get current recording duration in seconds."""
        if not self.running or not self.start_time:
            return 0
        
        try:
            import uvc
            return uvc.get_time_monotonic() - self.start_time
        except ImportError:
            return time.monotonic() - self.start_time
    
    def get_recording_stats(self):
        """Get current recording statistics with thread safety."""
        with self.lock:
            stats = {}
            duration = self.recording_time
            
            for camera_id in ["world", "eye0", "eye1"]:
                frames = self.frame_counts[camera_id]
                fps = frames / duration if duration > 0 else 0
                stats[camera_id] = {"frames": frames, "fps": fps}
            
            return stats