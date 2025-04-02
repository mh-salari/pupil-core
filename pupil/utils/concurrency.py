"""
Concurrency Utilities
-------------------
Helper classes and functions for thread safety and concurrency.
"""
import threading
from queue import Queue, Empty
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreadSafeObject:
    """Base class for thread-safe objects using a lock."""
    
    def __init__(self):
        """Initialize with a lock."""
        self._lock = threading.Lock()
    
    def synchronized(self, func):
        """Decorator to make a method synchronized with the object lock."""
        def wrapper(*args, **kwargs):
            with self._lock:
                return func(*args, **kwargs)
        return wrapper


class FrameBuffer:
    """Thread-safe buffer for camera frames with capacity limit."""
    
    def __init__(self, max_size=30):
        """
        Initialize frame buffer.
        
        Args:
            max_size: Maximum number of frames to store
        """
        self.queue = Queue(maxsize=max_size)
        self.lock = threading.Lock()
        self.frame_count = 0
    
    def add_frame(self, frame):
        """
        Add a frame to the buffer, removing oldest if full.
        
        Args:
            frame: Frame to add
        """
        with self.lock:
            # Remove oldest frame if queue is full
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
                    
            # Add new frame
            self.queue.put(frame)
            self.frame_count += 1
    
    def get_latest_frame(self):
        """
        Get the most recent frame without removing it.
        
        Returns:
            Most recent frame or None if empty
        """
        with self.lock:
            if self.queue.empty():
                return None
                
            # Get all frames
            frames = []
            while not self.queue.empty():
                try:
                    frames.append(self.queue.get_nowait())
                except Empty:
                    break
                    
            # Put all frames back except the last one
            for frame in frames[:-1]:
                self.queue.put(frame)
                
            # Put the last frame back and return it
            latest_frame = frames[-1]
            self.queue.put(latest_frame)
            return latest_frame
    
    def drain(self):
        """
        Remove all frames from the buffer.
        
        Returns:
            List of all frames in order
        """
        with self.lock:
            frames = []
            while not self.queue.empty():
                try:
                    frames.append(self.queue.get_nowait())
                except Empty:
                    break
            return frames
            
    def clear(self):
        """Clear all frames from the buffer."""
        with self.lock:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Empty:
                    break
                    
            self.frame_count = 0
    
    def __len__(self):
        """Get the current number of frames in the buffer."""
        return self.queue.qsize()