"""
Minimal Pupil Core Implementation
--------------------------------
A focused implementation of just the camera initialization and recording functionality.
"""
import os
import time
import logging
import platform
import uuid
from pathlib import Path
from fractions import Fraction
import multiprocessing as mp
import threading

import numpy as np
import uvc
import av
from av.packet import Packet

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

###############################################################################
# Camera Source
###############################################################################

class UVCSource:
    """Simple UVC camera source implementation."""
    
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
            logger.warning(f"Camera {self.name} disconnected. Attempting reconnection...")
            
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
    
    def cleanup(self):
        """Clean up resources."""
        if self.devices:
            self.devices.cleanup()
            self.devices = None
        
        if self.uvc_capture:
            self.uvc_capture.close()
            self.uvc_capture = None


###############################################################################
# Video Writer
###############################################################################

class VideoWriter:
    """Base class for video writers."""
    
    def __init__(self, output_file_path, start_time):
        """
        Initialize a video writer.
        
        Args:
            output_file_path: Path to output video file
            start_time: Recording start time (monotonic)
        """
        self.output_file_path = output_file_path
        self.temp_path = output_file_path + ".writing"
        self.start_time = start_time
        self.timestamps = []
        self.last_pts = -1
        self.time_base = Fraction(1, 65535)  # High precision timebase
        self.configured = False
        self.closed = False
        
        # Extract file extension
        _, ext = os.path.splitext(output_file_path)
        if not ext:
            raise ValueError("Output file must have an extension")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Initialize container
        try:
            self.container = av.open(self.temp_path, 'w', format=ext[1:])
            logger.info(f"Created video container: {self.temp_path}")
        except Exception as e:
            logger.error(f"Failed to create video container: {e}")
            raise
        
        # Create video stream with codec
        self.stream = self.container.add_stream(self.codec, rate=1/self.time_base)
        
        # Set bitrate and threading for better performance
        bitrate = 15000 * 10000
        self.stream.bit_rate = bitrate
        self.stream.bit_rate_tolerance = bitrate // 20
        self.stream.thread_count = max(1, mp.cpu_count() - 1)
    
    def write_frame(self, frame):
        """
        Write a frame to the video file.
        
        Args:
            frame: Frame to write
        """
        if self.closed:
            logger.warning("Cannot write to closed writer")
            return False
        
        # Configure on first frame
        if not self.configured:
            self.stream.width = frame.width
            self.stream.height = frame.height
            self.configured = True
            self._init_encoder(frame)
            logger.info(f"Configured video stream: {frame.width}x{frame.height}")
        
        # Get frame timestamp and calculate pts
        timestamp = frame.timestamp
        
        # ALWAYS write the first frame, otherwise do timestamp check
        if self.timestamps and timestamp < self.start_time:
            # Log and skip frames that arrive before recording start time
            logger.debug(f"Skipping early frame: ts={timestamp}, start={self.start_time}")
            return False
        
        # Check for monotonic timestamps
        if self.timestamps and timestamp <= self.timestamps[-1]:
            logger.warning(f"Non-monotonic timestamp: {self.timestamps[-1]} -> {timestamp}")
            return False
        
        # Calculate presentation timestamp (PTS)
        pts = int((timestamp - self.start_time) / self.time_base)
        
        # Ensure PTS is always increasing
        pts = max(pts, self.last_pts + 1)
        
        # Encode and write frame
        try:
            packets = list(self._encode_frame(frame, pts))
            if not packets:
                logger.warning("Frame encoding produced no packets")
                return False
            
            for packet in packets:
                self.container.mux(packet)
            
            # Update state
            self.last_pts = pts
            self.timestamps.append(timestamp)
            
            # Log occasional success
            if len(self.timestamps) in [1, 10, 50, 100]:
                logger.info(f"Successfully wrote frame {len(self.timestamps)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing frame: {e}")
            return False
    
    def close(self):
        """Close the writer and finalize the video file."""
        if self.closed:
            return
        
        try:
            # Flush any remaining frames
            if self.configured:
                for packet in self.stream.encode(None):
                    self.container.mux(packet)
            
            # Close container
            self.container.close()
            
            # Rename from temp file to final file
            if os.path.exists(self.temp_path):
                os.rename(self.temp_path, self.output_file_path)
                logger.info(f"Finalized video file: {self.output_file_path}")
                
                # Save timestamps
                self._save_timestamps()
            else:
                logger.warning(f"Temp file not found: {self.temp_path}")
                
        except Exception as e:
            logger.error(f"Error closing video writer: {e}")
        finally:
            self.closed = True
    
    def _save_timestamps(self):
        """Save timestamps to a numpy file."""
        try:
            ts_file = os.path.splitext(self.output_file_path)[0] + "_timestamps.npy"
            np.save(ts_file, np.array(self.timestamps))
            logger.info(f"Saved timestamps to {ts_file}")
        except Exception as e:
            logger.error(f"Failed to save timestamps: {e}")
    
    def _init_encoder(self, frame):
        """Initialize the frame encoder (implemented by subclasses)."""
        raise NotImplementedError()
    
    def _encode_frame(self, frame, pts):
        """Encode a frame (implemented by subclasses)."""
        raise NotImplementedError()
    
    @property
    def codec(self):
        """Codec to use (implemented by subclasses)."""
        raise NotImplementedError()


class MPEGWriter(VideoWriter):
    """MPEG4 video writer."""
    
    @property
    def codec(self):
        return "mpeg4"
    
    def _init_encoder(self, frame):
        """Initialize MPEG encoder with frame."""
        # Determine pixel format based on frame
        if hasattr(frame, 'yuv_buffer') and frame.yuv_buffer is not None:
            pix_format = "yuv422p"
        else:
            pix_format = "bgr24"
        
        # Create AV frame for encoding
        self.av_frame = av.VideoFrame(frame.width, frame.height, pix_format)
        self.av_frame.time_base = self.time_base
    
    def _encode_frame(self, frame, pts):
        """Encode frame to MPEG."""
        # Fill pixel data based on format
        if hasattr(frame, 'yuv_buffer') and frame.yuv_buffer is not None:
            y, u, v = frame.yuv422
            self.av_frame.planes[0].update(y)
            self.av_frame.planes[1].update(u)
            self.av_frame.planes[2].update(v)
        else:
            self.av_frame.planes[0].update(frame.img)
        
        # Set presentation timestamp
        self.av_frame.pts = pts
        
        # Encode and return packets
        return self.stream.encode(self.av_frame)


class JPEGWriter(VideoWriter):
    """Motion JPEG video writer."""
    
    @property
    def codec(self):
        return "mjpeg"
    
    def _init_encoder(self, frame):
        """Initialize JPEG encoder."""
        # Set pixel format for MJPEG
        self.stream.pix_fmt = "yuvj422p"
    
    def _encode_frame(self, frame, pts):
        """Encode frame to JPEG."""
        # Check for JPEG buffer
        if not hasattr(frame, 'jpeg_buffer') or frame.jpeg_buffer is None:
            logger.warning("Frame has no JPEG buffer")
            yield from []
            return
        
        # Create packet from JPEG buffer
        try:
            packet = Packet()
            packet.payload = frame.jpeg_buffer
        except AttributeError:
            packet = Packet(frame.jpeg_buffer)
        
        # Set packet properties
        packet.stream = self.stream
        packet.time_base = self.time_base
        packet.pts = pts
        packet.dts = pts
        
        yield packet


###############################################################################
# Recorder
###############################################################################

class SimpleRecorder:
    """Simple camera recorder."""
    
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
        if self.running:
            logger.warning("Recording already running")
            return self.current_rec_path
        
        # Get synchronized start time
        self.start_time = uvc.get_time_monotonic()
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
        return self.current_rec_path
    
    def update(self, world_frame=None, eye0_frame=None, eye1_frame=None):
        """
        Update recording with new frames.
        
        Args:
            world_frame: New world camera frame
            eye0_frame: New eye0 camera frame
            eye1_frame: New eye1 camera frame
        """
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
        """Stop recording and finalize files."""
        if not self.running:
            return
        
        # Calculate recording duration
        duration = uvc.get_time_monotonic() - self.start_time
        
        # Calculate statistics
        stats = {}
        for camera_id in ["world", "eye0", "eye1"]:
            frames = self.frame_counts[camera_id]
            fps = frames / duration if duration > 0 else 0
            stats[camera_id] = {"frames": frames, "fps": fps}
        
        # Close all writers
        if self.world_writer:
            self.world_writer.close()
            self.world_writer = None
        
        for writer in self.eye_writers.values():
            writer.close()
        self.eye_writers = {}
        
        # Update metadata with duration and stats
        self._update_recording_info(duration, stats)
        
        self.running = False
        logger.info(f"Recording stopped. Duration: {duration:.2f}s, Total frames: {self.frame_count}")
        
        return self.current_rec_path, stats
    
    def _init_world_writer(self, camera):
        """Initialize world camera writer."""
        path = os.path.join(self.current_rec_path, "world.mp4")
        
        # Use JPEG writer if supported and enabled
        if self.use_jpeg and hasattr(camera._recent_frame, "jpeg_buffer"):
            self.world_writer = JPEGWriter(path, self.start_time)
            logger.info("Using JPEG encoding for world camera")
        else:
            self.world_writer = MPEGWriter(path, self.start_time)
            logger.info("Using MPEG encoding for world camera")
    
    def _init_eye_writer(self, camera, eye_id):
        """Initialize eye camera writer."""
        path = os.path.join(self.current_rec_path, f"{eye_id}.mp4")
        
        # Use JPEG writer if supported and enabled
        if self.use_jpeg and hasattr(camera._recent_frame, "jpeg_buffer"):
            self.eye_writers[eye_id] = JPEGWriter(path, self.start_time)
            logger.info(f"Using JPEG encoding for {eye_id} camera")
        else:
            self.eye_writers[eye_id] = MPEGWriter(path, self.start_time)
            logger.info(f"Using MPEG encoding for {eye_id} camera")
    
    def _save_recording_info(self):
        """Save recording metadata."""
        try:
            import json
            
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
            import json
            
            info_path = os.path.join(self.current_rec_path, "info.json")
            
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


###############################################################################
# Helper Functions
###############################################################################

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


def create_eye_tracker(world_name="ID2", eye0_name="ID0", eye1_name="ID1", exposure_mode="auto"):
    """
    Create a complete eye tracker setup.
    
    Args:
        world_name: Name pattern for world camera
        eye0_name: Name pattern for eye0 camera
        eye1_name: Name pattern for eye1 camera
        exposure_mode: "auto" or "manual" exposure control
        
    Returns:
        Dict with cameras and recorder
    """
    # Initialize cameras
    world_cam = UVCSource(name=world_name, frame_size=(1280, 720), frame_rate=30, exposure_mode=exposure_mode)
    eye0_cam = UVCSource(name=eye0_name, frame_size=(192, 192), frame_rate=120, exposure_mode=exposure_mode)
    eye1_cam = UVCSource(name=eye1_name, frame_size=(192, 192), frame_rate=120, exposure_mode=exposure_mode)
    
    # Initialize recorder
    recorder = SimpleRecorder()
    
    # Collect a few frames to stabilize cameras
    for _ in range(10):
        world_cam.get_frame()
        eye0_cam.get_frame()
        eye1_cam.get_frame()
        time.sleep(0.01)
    
    return {
        "world_cam": world_cam,
        "eye0_cam": eye0_cam,
        "eye1_cam": eye1_cam,
        "recorder": recorder
    }


def simple_recording_example():
    """Run a simple recording example."""
    # List available cameras
    cameras = list_available_cameras()
    print("Available cameras:")
    for cam in cameras:
        print(f"  {cam['id']}: {cam['name']} (UID: {cam['uid']})")
    
    # Create tracker
    tracker = create_eye_tracker()
    
    # Show camera info
    for name, cam in tracker.items():
        if name != "recorder" and cam.online:
            print(f"{name}: {cam.name} at {cam.frame_size}@{cam.frame_rate}fps")
    
    # Start recording
    if any(cam.online for name, cam in tracker.items() if name != "recorder"):
        print("Starting recording for 5 seconds...")
        
        # Start recording
        rec_path = tracker["recorder"].start(
            world_cam=tracker["world_cam"],
            eye0_cam=tracker["eye0_cam"],
            eye1_cam=tracker["eye1_cam"]
        )
        
        # Record for 5 seconds
        start_time = time.time()
        frames_processed = 0
        
        # Since we can't set timeouts on get_frame, we'll use a tight loop
        # to poll each camera as quickly as possible
        while time.time() - start_time < 5:
            any_frames = False
            
            # Get and process world frame
            world_frame = tracker["world_cam"].get_frame() if tracker["world_cam"].online else None
            if world_frame:
                any_frames = True
                tracker["recorder"].update(
                    world_frame=world_frame,
                    eye0_frame=None,
                    eye1_frame=None
                )
                frames_processed += 1
                if frames_processed % 10 == 0:
                    print(".", end="", flush=True)
            
            # Get and process eye0 frame
            eye0_frame = tracker["eye0_cam"].get_frame() if tracker["eye0_cam"].online else None
            if eye0_frame:
                any_frames = True
                tracker["recorder"].update(
                    world_frame=None,
                    eye0_frame=eye0_frame,
                    eye1_frame=None
                )
            
            # Get and process eye1 frame
            eye1_frame = tracker["eye1_cam"].get_frame() if tracker["eye1_cam"].online else None
            if eye1_frame:
                any_frames = True
                tracker["recorder"].update(
                    world_frame=None,
                    eye0_frame=None,
                    eye1_frame=eye1_frame
                )
            
            # Only sleep if we didn't get any frames
            if not any_frames:
                time.sleep(0.001)  # Minimal sleep to prevent CPU hogging
        
        # Stop recording and get stats
        rec_path, stats = tracker["recorder"].stop()
        print(f"\nRecording saved to: {rec_path}")
        
        # Display recording statistics
        print("\nRecording statistics:")
        for camera_id, data in stats.items():
            if data["frames"] > 0:
                print(f"  {camera_id.capitalize()}: {data['frames']} frames, {data['fps']:.2f} fps")
        
        # Check file sizes
        world_path = os.path.join(rec_path, "world.mp4")
        eye0_path = os.path.join(rec_path, "eye0.mp4")
        eye1_path = os.path.join(rec_path, "eye1.mp4")
        
        print("\nFile sizes:")
        if os.path.exists(world_path):
            print(f"  World: {os.path.getsize(world_path)} bytes")
        if os.path.exists(eye0_path):
            print(f"  Eye0: {os.path.getsize(eye0_path)} bytes")
        if os.path.exists(eye1_path):
            print(f"  Eye1: {os.path.getsize(eye1_path)} bytes")
    else:
        print("No cameras found!")
    
    # Clean up
    for name, component in tracker.items():
        if hasattr(component, "cleanup"):
            component.cleanup()   


def optimized_recording_example():
    """Run an optimized multi-threaded recording example."""
    # Create tracker
    tracker = create_eye_tracker()
    
    # Show camera info
    for name, cam in tracker.items():
        if name != "recorder" and cam.online:
            print(f"{name}: {cam.name} at {cam.frame_size}@{cam.frame_rate}fps")
    
    # Create thread control flag and locks
    running = True
    import threading
    recorder_lock = threading.Lock()  # Add this lock to prevent recorder contention
    
    # Define thread functions for each camera with optimizations
    def capture_world():
        frames = 0
        cam = tracker["world_cam"]
        # Pre-capture a frame to warm up
        cam.get_frame()
        while running and cam.online:
            frame = cam.get_frame()
            if frame:
                with recorder_lock:  # Use lock to prevent contention
                    tracker["recorder"].update(world_frame=frame, eye0_frame=None, eye1_frame=None)
                frames += 1
                if frames % 30 == 0:
                    print("W", end="", flush=True)
    
    def capture_eye0():
        frames = 0
        cam = tracker["eye0_cam"]
        # Pre-capture a frame to warm up
        cam.get_frame()
        while running and cam.online:
            frame = cam.get_frame()
            if frame:
                with recorder_lock:  # Use lock to prevent contention
                    tracker["recorder"].update(world_frame=None, eye0_frame=frame, eye1_frame=None)
                frames += 1
                if frames % 120 == 0:
                    print("0", end="", flush=True)
    
    def capture_eye1():
        frames = 0
        cam = tracker["eye1_cam"]
        # Pre-capture a frame to warm up
        cam.get_frame()
        while running and cam.online:
            frame = cam.get_frame()
            if frame:
                with recorder_lock:  # Use lock to prevent contention
                    tracker["recorder"].update(world_frame=None, eye0_frame=None, eye1_frame=frame)
                frames += 1
                if frames % 120 == 0:
                    print("1", end="", flush=True)
    
    # Start recording
    if any(cam.online for name, cam in tracker.items() if name != "recorder"):
        print("Starting recording for 5 seconds...")
        
        rec_path = tracker["recorder"].start(
            world_cam=tracker["world_cam"],
            eye0_cam=tracker["eye0_cam"],
            eye1_cam=tracker["eye1_cam"]
        )
        
        # Start capture threads
        threads = []
        if tracker["world_cam"].online:
            threads.append(threading.Thread(target=capture_world, name="WorldCam"))
        if tracker["eye0_cam"].online:
            threads.append(threading.Thread(target=capture_eye0, name="Eye0Cam"))
        if tracker["eye1_cam"].online:
            threads.append(threading.Thread(target=capture_eye1, name="Eye1Cam"))
        
        for t in threads:
            t.daemon = True
            t.start()
        
        # Record for 5 seconds
        time.sleep(5)
        
        # Stop threads and recording
        running = False
        for t in threads:
            t.join(timeout=1.0)
        
        rec_path, stats = tracker["recorder"].stop()
        print(f"\nRecording saved to: {rec_path}")
        
        # Display recording statistics
        print("\nRecording statistics:")
        for camera_id, data in stats.items():
            if data["frames"] > 0:
                print(f"  {camera_id.capitalize()}: {data['frames']} frames, {data['fps']:.2f} fps")
    else:
        print("No cameras found!")
    
    # Clean up
    for name, component in tracker.items():
        if hasattr(component, "cleanup"):
            component.cleanup()
if __name__ == "__main__":
    optimized_recording_example()