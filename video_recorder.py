"""
Video Recorder
-------------
Handles video file recording, session management, and metadata for Pupil Core cameras.
"""
import os
import time
import logging
import uuid
import threading
from pathlib import Path
from fractions import Fraction
import json

import numpy as np
import av
from av.packet import Packet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom exceptions
class NonMonotonicTimestampError(ValueError):
    """Raised when timestamps are not monotonically increasing."""
    pass


###############################################################################
# Video Writer Base Class
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
        self.lock = threading.Lock()  # Add lock for thread safety
        
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
        self.stream.thread_count = max(1, os.cpu_count() - 1)
    
    def write_frame(self, frame):
        """
        Write a frame to the video file.
        
        Args:
            frame: Frame to write
        """
        with self.lock:  # Thread safety
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
        with self.lock:  # Thread safety
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
    
    def release(self):
        """Alias for close() to match OpenCV interface."""
        self.close()


###############################################################################
# Specific Video Writer Implementations
###############################################################################

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
# H264 Writer Implementation
###############################################################################

class H264Writer:
    """H264 video writer for cameras that provide H264 encoded frames."""
    
    def __init__(self, output_file_path, width, height, fps):
        """
        Initialize an H264 writer.
        
        Args:
            output_file_path: Path to output video file
            width: Frame width
            height: Frame height
            fps: Frame rate
        """
        self.output_file_path = output_file_path
        self.temp_path = output_file_path + ".writing"
        self.width = width
        self.height = height
        self.fps = fps
        self.configured = False
        self.closed = False
        self.timestamps = []
        self.lock = threading.Lock()  # Add lock for thread safety
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Initialize container
        try:
            self.container = av.open(self.temp_path, 'w', format='mp4')
            logger.info(f"Created H264 container: {self.temp_path}")
            
            # Create stream
            self.stream = self.container.add_stream('h264', rate=fps)
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = 'yuv420p'
            
            # Set bitrate for quality
            bitrate = 15000 * 10000
            self.stream.bit_rate = bitrate
            self.stream.bit_rate_tolerance = bitrate // 20
            
            # Set as configured
            self.configured = True
        except Exception as e:
            logger.error(f"Failed to create H264 container: {e}")
            raise
    
    def write_frame(self, frame):
        """
        Write an H264 encoded frame to the video file.
        
        Args:
            frame: Frame with h264_buffer to write
        """
        with self.lock:  # Thread safety
            if self.closed:
                logger.warning("Cannot write to closed writer")
                return False
            
            # Check for H264 buffer
            if not hasattr(frame, 'h264_buffer') or frame.h264_buffer is None:
                logger.warning("Frame has no H264 buffer")
                return False
            
            # Store timestamp
            self.timestamps.append(frame.timestamp)
            
            # Create packet from H264 buffer
            try:
                packet = Packet(frame.h264_buffer)
                packet.stream = self.stream
                self.container.mux(packet)
                return True
            except Exception as e:
                logger.error(f"Error writing H264 frame: {e}")
                return False
    
    def close(self):
        """Close the writer and finalize the video file."""
        with self.lock:  # Thread safety
            if self.closed:
                return
            
            try:
                # Close container
                self.container.close()
                
                # Rename from temp file to final file
                if os.path.exists(self.temp_path):
                    os.rename(self.temp_path, self.output_file_path)
                    logger.info(f"Finalized H264 video file: {self.output_file_path}")
                    
                    # Save timestamps
                    self._save_timestamps()
                else:
                    logger.warning(f"Temp file not found: {self.temp_path}")
            except Exception as e:
                logger.error(f"Error closing H264 writer: {e}")
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
    
    def release(self):
        """Alias for close() to match other interfaces."""
        self.close()


###############################################################################
# Recorder
###############################################################################

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


###############################################################################
# Simple Example Usage
###############################################################################

def record_from_cameras(cameras, duration=5, output_dir=None, session_name=None):
    """
    Record from cameras for a specified duration.
    
    Args:
        cameras: Dict with world_cam, eye0_cam, eye1_cam
        duration: Recording duration in seconds
        output_dir: Output directory for recordings
        session_name: Session name for the recording
    
    Returns:
        Path to the recording
    """
    # Initialize recorder
    recorder = Recorder(rec_dir=output_dir, session_name=session_name)
    
    # Start recording
    rec_path = recorder.start(
        world_cam=cameras.get("world_cam"),
        eye0_cam=cameras.get("eye0_cam"),
        eye1_cam=cameras.get("eye1_cam")
    )
    
    # Record for the specified duration
    start_time = time.time()
    frames_processed = 0
    
    logger.info(f"Recording for {duration} seconds...")
    
    try:
        while time.time() - start_time < duration:
            any_frames = False
            
            # Get and process world frame
            world_cam = cameras.get("world_cam")
            if world_cam and world_cam.online:
                world_frame = world_cam.get_frame()
                if world_frame:
                    any_frames = True
                    recorder.update(world_frame=world_frame)
                    frames_processed += 1
                    if frames_processed % 10 == 0:
                        print(".", end="", flush=True)
            
            # Get and process eye0 frame
            eye0_cam = cameras.get("eye0_cam")
            if eye0_cam and eye0_cam.online:
                eye0_frame = eye0_cam.get_frame()
                if eye0_frame:
                    any_frames = True
                    recorder.update(eye0_frame=eye0_frame)
            
            # Get and process eye1 frame
            eye1_cam = cameras.get("eye1_cam")
            if eye1_cam and eye1_cam.online:
                eye1_frame = eye1_cam.get_frame()
                if eye1_frame:
                    any_frames = True
                    recorder.update(eye1_frame=eye1_frame)
            
            # Only sleep if we didn't get any frames
            if not any_frames:
                time.sleep(0.001)  # Minimal sleep to prevent CPU hogging
    
    except KeyboardInterrupt:
        logger.info("Recording interrupted by user")
    except Exception as e:
        logger.error(f"Error during recording: {e}")
    
    # Stop recording and get stats
    try:
        rec_path, stats = recorder.stop()
        
        # Display recording statistics
        print("\nRecording statistics:")
        for camera_id, data in stats.items():
            if data["frames"] > 0:
                print(f"  {camera_id.capitalize()}: {data['frames']} frames, {data['fps']:.2f} fps")
        
        print(f"\nRecording saved to: {rec_path}")
        return rec_path
    except Exception as e:
        logger.error(f"Error stopping recording: {e}")
        return None


# Simple usage example
if __name__ == "__main__":
    # This requires camera_manager.py to run
    try:
        from camera_manager import create_camera_manager
        
        # Create cameras
        cameras = create_camera_manager()
        
        # Record for 5 seconds
        record_from_cameras(cameras, duration=5)
        
        # Clean up
        for cam in cameras.values():
            cam.cleanup()
            
    except ImportError:
        logger.error("camera_manager.py required for this example")