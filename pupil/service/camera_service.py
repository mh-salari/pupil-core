"""
Camera Service
------------
ZeroMQ-based service for remote camera access and recording control.
"""
import os
import time
import json
import logging
import threading
from queue import Queue, Empty
from typing import Dict, List, Any, Optional, Tuple

import zmq
import numpy as np
import cv2

# Import from our package
from ..hardware.camera_discovery import CameraManager, list_available_cameras
from ..io.recorder import Recorder
from .message_types import MessageType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CameraService:
    """
    ZeroMQ-based service that manages cameras and provides remote access.
    Acts as a server for client applications to connect to.
    """
    
    def __init__(self, host="127.0.0.1", port=5555, recording_dir=None):
        """
        Initialize the camera service.
        
        Args:
            host: Host address to bind to
            port: Port to bind to
            recording_dir: Directory to store recordings
        """
        self.host = host
        self.port = port
        self.recording_dir = recording_dir
        
        # Initialize ZeroMQ context
        self.context = zmq.Context()
        
        # Create sockets
        self.command_socket = self.context.socket(zmq.REP)  # Request-Reply for commands
        self.command_socket.bind(f"tcp://{host}:{port}")
        
        self.pub_socket = self.context.socket(zmq.PUB)  # Publisher for frames/status
        self.pub_socket.bind(f"tcp://{host}:{port+1}")
        
        # Initialize camera manager and recorder
        self.camera_manager = None
        self.recorder = None
        self.init_cameras()
        self.init_recorder()
        
        # Status tracking
        self.is_recording = False
        self.recording_path = None
        self.frame_counts = {"world": 0, "eye0": 0, "eye1": 0}
        self.clients = set()
        self.active_streams = set()  # Track which cameras are being streamed
        
        # Frame queue for each camera
        self.frame_queues = {
            "world": Queue(maxsize=10),
            "eye0": Queue(maxsize=30),
            "eye1": Queue(maxsize=30)
        }
        
        # Control variables
        self.running = False
        self.threads = []
        
        # Add locks for thread safety
        self.recorder_lock = threading.Lock()
        self.camera_lock = threading.Lock()
        self.client_lock = threading.Lock()
        self.stream_lock = threading.Lock()
    
    def init_cameras(self):
        """Initialize camera manager."""
        try:
            logger.info("Initializing cameras...")
            self.camera_manager = CameraManager()
            self.cameras = self.camera_manager.get_all_cameras()
            
            # Log camera status
            for name, cam in self.cameras.items():
                if cam.online:
                    logger.info(f"Camera {name}: {cam.name} online at {cam.frame_size}@{cam.frame_rate}fps")
                else:
                    logger.warning(f"Camera {name}: offline")
        except Exception as e:
            logger.error(f"Failed to initialize cameras: {e}")
            self.cameras = {}
    
    def init_recorder(self):
        """Initialize video recorder."""
        try:
            logger.info("Initializing recorder...")
            self.recorder = Recorder(rec_dir=self.recording_dir)
        except Exception as e:
            logger.error(f"Failed to initialize recorder: {e}")
            self.recorder = None
    
    def start(self):
        """Start the camera service and all threads."""
        if self.running:
            logger.warning("Service already running")
            return
        
        logger.info(f"Starting camera service on {self.host}:{self.port}")
        self.running = True
        
        # Start command handling thread
        command_thread = threading.Thread(target=self.handle_commands, daemon=True)
        command_thread.start()
        self.threads.append(command_thread)
        
        # Start frame capture threads
        for cam_id in ["world", "eye0", "eye1"]:
            capture_thread = threading.Thread(
                target=self.capture_frames,
                args=(cam_id,),
                daemon=True
            )
            capture_thread.start()
            self.threads.append(capture_thread)
        
        # Start status update thread
        status_thread = threading.Thread(target=self.publish_status, daemon=True)
        status_thread.start()
        self.threads.append(status_thread)
        
        logger.info("Camera service started")
    
    def stop(self):
        """Stop the camera service and clean up resources."""
        if not self.running:
            return
        
        logger.info("Stopping camera service...")
        self.running = False
        
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Clean up cameras
        if self.camera_manager:
            self.camera_manager.cleanup()
        
        # Close ZeroMQ sockets
        self.command_socket.close()
        self.pub_socket.close()
        self.context.term()
        
        logger.info("Camera service stopped")
    
    def handle_commands(self):
        """Handle incoming commands from clients."""
        logger.info("Command handler started")
        
        while self.running:
            try:
                # Wait for next command with timeout
                if not self.command_socket.poll(1000):
                    continue
                
                # Receive command
                message = self.command_socket.recv_json()
                logger.debug(f"Received command: {message}")
                
                # Extract command and parameters
                command = message.get("command", "")
                params = message.get("params", {})
                client_id = message.get("client_id", "unknown")
                
                # Add client to tracking
                with self.client_lock:
                    self.clients.add(client_id)
                
                # Process command
                response = self.process_command(command, params, client_id)
                
                # Send response
                self.command_socket.send_json(response)
                
            except zmq.ZMQError as e:
                logger.error(f"ZMQ error while handling commands: {e}")
            except Exception as e:
                logger.error(f"Error handling command: {e}")
                # Send error response
                try:
                    self.command_socket.send_json({
                        "status": "error",
                        "error": str(e)
                    })
                except:
                    pass
    
    def process_command(self, command, params, client_id):
        """
        Process a command from a client.
        
        Args:
            command: Command string
            params: Command parameters
            client_id: Client identifier
            
        Returns:
            Response dict
        """
        # Handle different commands
        if command == "get_status":
            return self.get_status()
            
        elif command == "list_cameras":
            return {
                "status": "ok",
                "cameras": self.get_camera_info()
            }
            
        elif command == "start_recording":
            session_name = params.get("session_name")
            return self.start_recording(session_name)
            
        elif command == "stop_recording":
            return self.stop_recording()
            
        elif command == "start_streaming":
            cam_id = params.get("camera_id", "world")
            with self.stream_lock:
                self.active_streams.add(cam_id)
            return {
                "status": "ok",
                "message": f"Streaming started for {cam_id}"
            }
            
        elif command == "stop_streaming":
            cam_id = params.get("camera_id", "world")
            with self.stream_lock:
                if cam_id in self.active_streams:
                    self.active_streams.remove(cam_id)
            return {
                "status": "ok",
                "message": f"Streaming stopped for {cam_id}"
            }
            
        elif command == "set_camera_property":
            cam_id = params.get("camera_id", "world")
            property_name = params.get("property")
            value = params.get("value")
            return self.set_camera_property(cam_id, property_name, value)
            
        elif command == "disconnect":
            with self.client_lock:
                if client_id in self.clients:
                    self.clients.remove(client_id)
            return {
                "status": "ok",
                "message": "Disconnected"
            }
            
        else:
            return {
                "status": "error",
                "message": f"Unknown command: {command}"
            }
    
    def get_status(self):
        """Get current system status."""
        status = {
            "status": "ok",
            "recording": self.is_recording,
            "cameras": {},
            "frame_counts": self.frame_counts,
            "clients": len(self.clients),
            "active_streams": list(self.active_streams)
        }
        
        # Add camera status
        with self.camera_lock:
            for cam_id, cam in self.cameras.items():
                if cam:
                    status["cameras"][cam_id] = {
                        "online": cam.online,
                        "name": cam.name,
                        "frame_size": cam.frame_size,
                        "frame_rate": cam.frame_rate
                    }
        
        # Add recording info if recording
        if self.is_recording and self.recorder:
            try:
                with self.recorder_lock:
                    status["recording_info"] = {
                        "path": self.recording_path,
                        "duration": self.recorder.recording_time,
                        "frame_stats": self.recorder.get_recording_stats()
                    }
            except Exception as e:
                logger.error(f"Error getting recording info: {e}")
                status["recording_info"] = {
                    "path": self.recording_path,
                    "error": str(e)
                }
        
        return status
    
    def get_camera_info(self):
        """Get information about all cameras."""
        camera_info = {}
        
        # Add connected cameras
        with self.camera_lock:
            for cam_id, cam in self.cameras.items():
                if cam:
                    camera_info[cam_id] = {
                        "online": cam.online,
                        "name": cam.name,
                        "frame_size": cam.frame_size,
                        "frame_rate": cam.frame_rate
                    }
                    
                    # Add camera controls if online
                    if cam.online:
                        try:
                            camera_info[cam_id]["controls"] = cam.controls
                        except Exception as e:
                            logger.error(f"Error getting controls for {cam_id}: {e}")
                            camera_info[cam_id]["controls"] = {}
        
        # Add available cameras
        try:
            camera_info["available"] = list_available_cameras()
        except Exception as e:
            logger.error(f"Error listing available cameras: {e}")
            camera_info["available"] = []
        
        return camera_info
    
    def start_recording(self, session_name=None):
        """Start recording from all cameras."""
        with self.recorder_lock:
            if self.is_recording:
                return {
                    "status": "error",
                    "message": "Recording already in progress"
                }
            
            if not self.recorder:
                return {
                    "status": "error",
                    "message": "Recorder not initialized"
                }
            
            try:
                # Clear frame queues before starting a new recording
                for queue in self.frame_queues.values():
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except Empty:
                            break
                
                # Start recording
                self.recording_path = self.recorder.start(
                    world_cam=self.cameras.get("world_cam"),
                    eye0_cam=self.cameras.get("eye0_cam"),
                    eye1_cam=self.cameras.get("eye1_cam")
                )
                
                self.is_recording = True
                logger.info(f"Recording started: {self.recording_path}")
                
                return {
                    "status": "ok",
                    "message": "Recording started",
                    "path": self.recording_path
                }
                
            except Exception as e:
                logger.error(f"Failed to start recording: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to start recording: {e}"
                }
    
    def stop_recording(self):
        """Stop recording."""
        with self.recorder_lock:
            if not self.is_recording:
                return {
                    "status": "error",
                    "message": "No recording in progress"
                }
            
            try:
                # Stop recording
                path, stats = self.recorder.stop()
                
                self.is_recording = False
                logger.info(f"Recording stopped: {path}")
                
                return {
                    "status": "ok",
                    "message": "Recording stopped",
                    "path": path,
                    "stats": stats
                }
                
            except Exception as e:
                logger.error(f"Failed to stop recording: {e}")
                # Make sure to reset recording state even if there's an error
                self.is_recording = False
                return {
                    "status": "error",
                    "message": f"Failed to stop recording: {e}"
                }
    
    def set_camera_property(self, cam_id, property_name, value):
        """Set a camera property."""
        with self.camera_lock:
            if cam_id not in self.cameras:
                return {
                    "status": "error",
                    "message": f"Unknown camera: {cam_id}"
                }
            
            cam = self.cameras.get(f"{cam_id}_cam")
            if not cam or not cam.online:
                return {
                    "status": "error",
                    "message": f"Camera {cam_id} is offline"
                }
            
            try:
                # Handle different properties
                if property_name == "exposure_time":
                    cam.exposure_time = value
                    return {
                        "status": "ok",
                        "message": f"Set {cam_id} exposure time to {value}",
                        "new_value": cam.exposure_time
                    }
                
                elif property_name == "gamma":
                    cam.gamma = value
                    return {
                        "status": "ok",
                        "message": f"Set {cam_id} gamma to {value}",
                        "new_value": cam.gamma
                    }
                
                elif property_name == "saturation":
                    cam.saturation = value
                    return {
                        "status": "ok",
                        "message": f"Set {cam_id} saturation to {value}",
                        "new_value": cam.saturation
                    }
                
                elif property_name in cam.controls:
                    result = cam.set_control_value(property_name, value)
                    if result:
                        return {
                            "status": "ok",
                            "message": f"Set {cam_id} {property_name} to {value}"
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"Failed to set {property_name}"
                        }
                
                else:
                    return {
                        "status": "error",
                        "message": f"Unknown property: {property_name}"
                    }
                    
            except Exception as e:
                logger.error(f"Error setting camera property: {e}")
                return {
                    "status": "error",
                    "message": f"Error: {e}"
                }
    
    def capture_frames(self, cam_id):
        """Capture frames from a camera and add to queue."""
        logger.info(f"Starting frame capture for {cam_id}")
        
        camera = self.cameras.get(f"{cam_id}_cam")
        if not camera:
            logger.warning(f"No camera found for {cam_id}")
            return
        
        queue = self.frame_queues[cam_id]
        
        while self.running:
            try:
                # Skip if camera is offline
                if not camera.online:
                    time.sleep(0.1)
                    continue
                
                # Get frame from camera
                frame = camera.get_frame()
                if not frame:
                    time.sleep(0.001)
                    continue
                
                # Update frame count
                self.frame_counts[cam_id] += 1
                
                # Add frame to queue, replacing oldest if full
                if queue.full():
                    try:
                        queue.get_nowait()
                    except Empty:
                        pass
                queue.put(frame)
                
                # If recording, update recorder with proper locking
                if self.is_recording and self.recorder:
                    with self.recorder_lock:
                        if not self.is_recording:
                            # Double-check that recording is still active
                            continue
                            
                        try:
                            if cam_id == "world":
                                self.recorder.update(world_frame=frame, eye0_frame=None, eye1_frame=None)
                            elif cam_id == "eye0":
                                self.recorder.update(world_frame=None, eye0_frame=frame, eye1_frame=None)
                            elif cam_id == "eye1":
                                self.recorder.update(world_frame=None, eye0_frame=None, eye1_frame=frame)
                        except Exception as e:
                            logger.error(f"Error updating recorder with {cam_id} frame: {e}")
                
                # If camera is being streamed, publish frame
                with self.stream_lock:
                    if cam_id in self.active_streams:
                        self.publish_frame(cam_id, frame)
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error capturing frame from {cam_id}: {e}")
                time.sleep(0.1)
    
    def publish_frame(self, cam_id, frame):
        """Publish a frame to clients."""
        try:
            # Get image from frame
            img = frame.img.copy()
            
            # Rotate eye0 camera image (it's upside down)
            if cam_id == "eye0":
                img = cv2.rotate(img, cv2.ROTATE_180)
            
            # Convert frame to JPEG-encoded bytes with highest quality for detection
            # Eye cameras need higher quality for accurate pupil detection
            jpeg_quality = 95 if cam_id.startswith("eye") else 90
            _, img_encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            img_bytes = img_encoded.tobytes()
            
            # Create frame metadata
            metadata = {
                "type": MessageType.FRAME_RESPONSE,
                "camera_id": cam_id,
                "timestamp": frame.timestamp,
                "width": frame.width,
                "height": frame.height
            }
            
            # Send frame metadata followed by binary data
            self.pub_socket.send_json(metadata, zmq.SNDMORE)
            self.pub_socket.send(img_bytes)
            
        except Exception as e:
            logger.error(f"Error publishing frame: {e}")
    
    def publish_status(self):
        """Periodically publish status updates."""
        logger.info("Status publisher started")
        
        while self.running:
            try:
                # Get current status
                status = self.get_status()
                status["type"] = MessageType.STATUS_UPDATE
                
                # Publish status
                self.pub_socket.send_json(status)
                
                # Wait before next update
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error publishing status: {e}")
                time.sleep(1.0)
    
    def cleanup(self):
        """Clean up all resources."""
        logger.info("Cleaning up resources...")
        
        # Stop recording if active
        if self.is_recording:
            try:
                self.stop_recording()
            except Exception as e:
                logger.error(f"Error stopping recording during cleanup: {e}")
        
        # Clean up cameras
        if self.camera_manager:
            try:
                self.camera_manager.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up cameras: {e}")
        
        # Close ZeroMQ sockets
        try:
            self.command_socket.close()
            self.pub_socket.close()
            self.context.term()
        except Exception as e:
            logger.error(f"Error closing ZeroMQ sockets: {e}")
        
        logger.info("Cleanup completed")