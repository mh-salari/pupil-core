"""
Video Writer
-----------
Base classes and implementations for video encoding and writing.
"""
import os
import time
import logging
import threading
from fractions import Fraction
from abc import ABC, abstractmethod

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


class BaseVideoWriter(ABC):
    """Base abstract class for video writers."""
    
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
        
        # Initialize file and container
        self._initialize_container()
    
    def _initialize_container(self):
        """Initialize the AV container for video writing."""
        # Extract file extension
        _, ext = os.path.splitext(self.output_file_path)
        if not ext:
            raise ValueError("Output file must have an extension")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
        
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
    
    @abstractmethod
    def _init_encoder(self, frame):
        """Initialize the frame encoder (implemented by subclasses)."""
        pass
    
    @abstractmethod
    def _encode_frame(self, frame, pts):
        """Encode a frame (implemented by subclasses)."""
        pass
    
    @property
    @abstractmethod
    def codec(self):
        """Codec to use (implemented by subclasses)."""
        pass
    
    def release(self):
        """Alias for close() to match OpenCV interface."""
        self.close()


class MPEGWriter(BaseVideoWriter):
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


class JPEGWriter(BaseVideoWriter):
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