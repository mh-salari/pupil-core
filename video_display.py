"""
Multi-Camera Video Display
------------------------
Client that connects to the camera service and displays all three cameras:
world, eye0, and eye1 simultaneously.
"""
import os
import time
import json
import logging
import argparse
import uuid
import sys

import zmq
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultiCameraDisplay")

class MessageType:
    """Message types for ZeroMQ communication."""
    COMMAND = 0
    FRAME_REQUEST = 1
    FRAME_RESPONSE = 2
    STATUS_UPDATE = 3
    ERROR = 4

class MultiCameraDisplay:
    """
    Video display for multiple cameras.
    Displays world, eye0, and eye1 cameras simultaneously.
    Single-threaded implementation to avoid threading issues on macOS.
    """
    
    def __init__(self, server_host="127.0.0.1", server_port=5555):
        """
        Initialize the multi-camera video display.
        
        Args:
            server_host: Camera service host address
            server_port: Camera service command port
        """
        self.server_host = server_host
        self.server_port = server_port
        self.client_id = f"multi_display_{uuid.uuid4().hex[:8]}"
        
        # List of cameras to display
        self.cameras = ["world", "eye0", "eye1"]
        
        # Store latest frames for each camera
        self.current_frames = {
            "world": None,
            "eye0": None,
            "eye1": None
        }
        
        # Initialize ZeroMQ context
        self.context = zmq.Context()
        
        # Create command socket (REQ)
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.connect(f"tcp://{server_host}:{server_port}")
        
        # Create subscriber socket for frames
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{server_host}:{server_port+1}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        
        # Set socket timeout for polling
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        
        # Status variables
        self.is_connected = False
        self.is_recording = False
        self.recording_path = None
        self.recording_duration = 0
        self.camera_statuses = {}
        
        # Window names
        self.window_names = {
            "world": "Camera: World",
            "eye0": "Camera: Eye 0",
            "eye1": "Camera: Eye 1"
        }
        
        # Control variables
        self.running = False
    
    def connect(self):
        """Connect to the camera service and start streaming for all cameras."""
        try:
            # Send initial status request
            response = self.send_command("get_status")
            if response.get("status") == "ok":
                self.is_connected = True
                logger.info(f"Connected to camera service at {self.server_host}:{self.server_port}")
                
                # Update camera statuses
                if "cameras" in response:
                    self.camera_statuses = response["cameras"]
                
                # Start streaming for all cameras
                success = True
                for camera_id in self.cameras:
                    response = self.send_command("start_streaming", {"camera_id": camera_id})
                    if response.get("status") == "ok":
                        logger.info(f"Started streaming for {camera_id} camera")
                    else:
                        logger.error(f"Failed to start streaming for {camera_id}: {response.get('message')}")
                        success = False
                
                return success
            else:
                logger.error(f"Failed to connect: {response.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to camera service: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the camera service."""
        try:
            # Stop streaming for all cameras
            for camera_id in self.cameras:
                self.send_command("stop_streaming", {"camera_id": camera_id})
            
            # Send disconnect command
            self.send_command("disconnect")
            
            # Close ZeroMQ sockets
            self.command_socket.close()
            self.sub_socket.close()
            self.context.term()
            
            logger.info("Disconnected from camera service")
            
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    def send_command(self, command, params=None):
        """
        Send a command to the camera service.
        
        Args:
            command: Command string
            params: Command parameters (optional)
            
        Returns:
            Response dict
        """
        if params is None:
            params = {}
        
        try:
            # Create command message
            message = {
                "command": command,
                "params": params,
                "client_id": self.client_id,
                "timestamp": time.time()
            }
            
            # Send command
            self.command_socket.send_json(message)
            
            # Wait for response
            response = self.command_socket.recv_json()
            
            return response
        except zmq.ZMQError as e:
            logger.error(f"ZMQ error sending command: {e}")
            return {"status": "error", "message": f"ZMQ error: {e}"}
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return {"status": "error", "message": f"Error: {e}"}
    
    def create_windows(self):
        """Create OpenCV windows for all cameras."""
        try:
            # Create window for each camera
            for camera_id, window_name in self.window_names.items():
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                
                # Set different sizes for world vs eye cameras
                if camera_id == "world":
                    cv2.resizeWindow(window_name, 800, 450)
                else:
                    cv2.resizeWindow(window_name, 400, 300)
            
            # Create status window
            cv2.namedWindow("Status", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Status", 800, 100)
            
            logger.info("Created OpenCV windows")
            return True
        except Exception as e:
            logger.error(f"Failed to create OpenCV windows: {e}")
            return False
    
    def run(self):
        """Run the main display loop."""
        # Connect to service
        if not self.connect():
            logger.error("Failed to connect to camera service")
            return
        
        # Create windows
        if not self.create_windows():
            logger.error("Failed to create windows, cannot continue")
            self.disconnect()
            return
        
        logger.info("Starting display loop")
        self.running = True
        
        try:
            # Main loop
            while self.running:
                # Check for status updates and frames
                try:
                    # Receive message from sub socket
                    message = self.sub_socket.recv_multipart(zmq.NOBLOCK)
                    
                    # Process message
                    if len(message) == 1:
                        # Status update (JSON)
                        status = json.loads(message[0])
                        if status.get("type") == MessageType.STATUS_UPDATE:
                            self.update_status(status)
                    
                    elif len(message) == 2:
                        # Frame data (metadata + binary)
                        metadata = json.loads(message[0])
                        frame_data = message[1]
                        
                        if metadata.get("type") == MessageType.FRAME_RESPONSE:
                            cam_id = metadata.get("camera_id")
                            
                            # Process frames for any of our cameras
                            if cam_id in self.cameras:
                                # Decode frame
                                img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                                
                                # Add timestamp
                                timestamp = metadata.get("timestamp")
                                self.add_timestamp(img, timestamp)
                                
                                # Store current frame
                                self.current_frames[cam_id] = img
                                
                                # Display frame
                                cv2.imshow(self.window_names[cam_id], img)
                                
                except zmq.Again:
                    # No message available, continue
                    pass
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                
                # Update status display
                self.display_status()
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    self.running = False
                elif key == ord('r'):
                    self.toggle_recording()
                
                # Small sleep to prevent high CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            # Clean up
            self.running = False
            self.disconnect()
            cv2.destroyAllWindows()
    
    def update_status(self, status):
        """Update internal status from server status."""
        if status.get("recording") is not None:
            self.is_recording = status["recording"]
        
        # Update recording info if available
        if "recording_info" in status:
            self.recording_path = status["recording_info"].get("path")
            self.recording_duration = status["recording_info"].get("duration", 0)
        else:
            self.recording_duration = 0
            
        # Update camera statuses
        if "cameras" in status:
            self.camera_statuses = status["cameras"]
    
    def add_timestamp(self, img, timestamp):
        """Add timestamp to image."""
        # Format timestamp as HH:MM:SS.mmm
        time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
        time_str += f".{int((timestamp % 1) * 1000):03d}"
        
        # Add text to image
        cv2.putText(
            img, time_str, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    
    def display_status(self):
        """Display status overlay with information for all cameras."""
        try:
            # Create status image
            height, width = 100, 800
            status_img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add recording status
            if self.is_recording:
                # Red recording indicator
                cv2.circle(status_img, (20, 20), 10, (0, 0, 255), -1)
                
                # Recording time
                time_str = self.format_duration(self.recording_duration)
                cv2.putText(
                    status_img, f"REC {time_str}", (40, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
                
                # Recording path
                if self.recording_path:
                    path_str = f"Path: {os.path.basename(self.recording_path)}"
                    cv2.putText(
                        status_img, path_str, (40, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )
            else:
                cv2.putText(
                    status_img, "Press 'R' to start recording", (20, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                )
            
            # Add camera statuses
            status_text = ""
            for i, cam_id in enumerate(self.cameras):
                # Get status for this camera
                cam_status = self.camera_statuses.get(cam_id, {})
                online = cam_status.get("online", False)
                status_text += f" | {cam_id}: {'ONLINE' if online else 'OFFLINE'}"
            
            cv2.putText(
                status_img, status_text, (20, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            
            # Add help text
            cv2.putText(
                status_img, "Q: Quit | R: Record", (600, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )
            
            # Display status overlay
            cv2.imshow("Status", status_img)
        except Exception as e:
            logger.error(f"Error displaying status: {e}")
    
    def toggle_recording(self):
        """Toggle recording state."""
        if self.is_recording:
            logger.info("Stopping recording")
            response = self.send_command("stop_recording")
            if response.get("status") == "ok":
                self.is_recording = False
                path = response.get("path")
                logger.info(f"Recording stopped: {path}")
        else:
            logger.info("Starting recording")
            response = self.send_command("start_recording")
            if response.get("status") == "ok":
                self.is_recording = True
                self.recording_path = response.get("path")
                logger.info(f"Recording started: {self.recording_path}")
    
    def format_duration(self, seconds):
        """Format duration as MM:SS.ms."""
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:05.2f}"


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Camera Video Display Client")
    parser.add_argument("--host", default="127.0.0.1", help="Camera service host address")
    parser.add_argument("--port", type=int, default=5555, help="Camera service port")
    args = parser.parse_args()
    
    # Create and run display
    display = MultiCameraDisplay(
        server_host=args.host,
        server_port=args.port
    )
    
    # Run the display
    display.run()


if __name__ == "__main__":
    main()