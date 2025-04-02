"""
Multi-Camera Video Display
------------------------
Simplified client that connects to the camera service and displays camera feeds:
world, eye0, and eye1 simultaneously. Display only, no control functionality.
Optimized for smooth display performance at ~30 FPS.
"""
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
logger = logging.getLogger("SimpleCameraDisplay")

class MessageType:
    """Message types for ZeroMQ communication."""
    COMMAND = 0
    FRAME_REQUEST = 1
    FRAME_RESPONSE = 2
    STATUS_UPDATE = 3
    ERROR = 4

class SimpleCameraDisplay:
    """
    Video display for multiple cameras.
    Displays world, eye0, and eye1 cameras simultaneously.
    Display only, with no recording or control functionality.
    Optimized for smooth display performance at ~30 FPS.
    """
    
    def __init__(self, server_host="127.0.0.1", server_port=5555, target_fps=30):
        """
        Initialize the simple camera display.
        
        Args:
            server_host: Camera service host address
            server_port: Camera service command port
            target_fps: Target frames per second for display
        """
        self.server_host = server_host
        self.server_port = server_port
        self.client_id = f"simple_display_{uuid.uuid4().hex[:8]}"
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        
        # List of cameras to display
        self.cameras = ["world", "eye0", "eye1"]
        
        # Store latest frames for each camera
        self.current_frames = {
            "world": None,
            "eye0": None,
            "eye1": None
        }
        
        # Performance tracking
        self.frame_count = 0
        self.fps_stats = {cam: {"count": 0, "last_time": 0, "fps": 0} for cam in self.cameras}
        self.last_display_time = time.monotonic()
        
        # Initialize ZeroMQ context
        self.context = zmq.Context()
        
        # Create command socket (REQ)
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.connect(f"tcp://{server_host}:{server_port}")
        
        # Create subscriber socket for frames
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{server_host}:{server_port+1}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        
        # Set socket options for better performance
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout
        self.sub_socket.setsockopt(zmq.LINGER, 0)
        
        # Window names
        self.window_names = {
            "world": "World Camera",
            "eye0": "Eye Camera 0",
            "eye1": "Eye Camera 1"
        }
        
        # Control variables
        self.running = False
    
    def connect(self):
        """Connect to the camera service and start streaming for all cameras."""
        try:
            # Send initial status request
            response = self.send_command("get_status")
            if response.get("status") == "ok":
                logger.info(f"Connected to camera service at {self.server_host}:{self.server_port}")
                
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
                    cv2.resizeWindow(window_name, 300, 300)
            
            logger.info("Created OpenCV windows")
            return True
        except Exception as e:
            logger.error(f"Failed to create OpenCV windows: {e}")
            return False
    
    def process_frame(self, metadata, frame_data):
        """Process a single frame from the camera service."""
        cam_id = metadata.get("camera_id")
        
        # Skip if not one of our cameras
        if cam_id not in self.cameras:
            return False
        
        # For eye cameras, we might want to skip some frames 
        # when the rate is higher than our display needs
        if cam_id.startswith("eye"):
            # Assuming eye cameras run at 120fps and we want 30fps display
            # We'll just take every 4th frame
            self.fps_stats[cam_id]["count"] += 1
            if self.fps_stats[cam_id]["count"] % 4 != 0:
                return False
        
        try:
            # Decode frame - lower quality for improved performance
            img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Resize for eye cameras for better performance
            if cam_id.startswith("eye"):
                img = cv2.resize(img, (192, 192))  # Smaller size for eye cameras
            
            # Add timestamp text to image
            timestamp = metadata.get("timestamp")
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            time_str += f".{int((timestamp % 1) * 1000):03d}"
            cv2.putText(img, time_str, (10, 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Store current frame
            self.current_frames[cam_id] = img
            
            # Display frame
            cv2.imshow(self.window_names[cam_id], img)
            
            # Update FPS statistics for this camera
            now = time.monotonic()
            if self.fps_stats[cam_id]["last_time"] > 0:
                elapsed = now - self.fps_stats[cam_id]["last_time"]
                if elapsed > 0:
                    self.fps_stats[cam_id]["fps"] = 1.0 / elapsed
            self.fps_stats[cam_id]["last_time"] = now
            
            return True
        except Exception as e:
            logger.error(f"Error processing frame for {cam_id}: {e}")
            return False
    
    def run(self):
        """Run the main display loop with optimized timing."""
        # Connect to service
        if not self.connect():
            logger.error("Failed to connect to camera service")
            return
        
        # Create windows
        if not self.create_windows():
            logger.error("Failed to create windows, cannot continue")
            self.disconnect()
            return
        
        logger.info(f"Starting display loop with target {self.target_fps} FPS")
        self.running = True
        self.last_display_time = time.monotonic()
        self.frame_count = 0
        fps_report_time = time.monotonic()
        
        try:
            # Main loop
            while self.running:
                loop_start_time = time.monotonic()
                frames_processed = 0
                
                # Process up to 10 messages per frame to avoid getting stuck
                # processing a backlog of messages
                for _ in range(10):
                    try:
                        # Use NOBLOCK to avoid blocking in the loop
                        message = self.sub_socket.recv_multipart(zmq.NOBLOCK)
                        
                        # Process message
                        if len(message) == 2:
                            # Frame data (metadata + binary)
                            metadata = json.loads(message[0])
                            frame_data = message[1]
                            
                            if metadata.get("type") == MessageType.FRAME_RESPONSE:
                                # Process frame and increment counter if successful
                                if self.process_frame(metadata, frame_data):
                                    frames_processed += 1
                        
                    except zmq.Again:
                        # No more messages available right now
                        break
                    except Exception as e:
                        logger.error(f"Error receiving or processing message: {e}")
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("Quit requested")
                    self.running = False
                
                # Calculate how long to sleep to maintain target frame rate
                current_time = time.monotonic()
                elapsed = current_time - loop_start_time
                sleep_time = max(0, self.target_frame_time - elapsed)
                
                # Log FPS every 5 seconds
                if current_time - fps_report_time > 5.0:
                    fps_report_time = current_time
                    fps_msg = f"Display FPS: "
                    for cam_id in self.cameras:
                        fps_msg += f"{cam_id}={self.fps_stats[cam_id]['fps']:.1f} "
                    logger.info(fps_msg)
                
                # Increment frame count
                self.frame_count += 1
                
                if frames_processed > 0:
                    # If we processed frames, sleep to maintain timing
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    # If no frames were processed, sleep a small amount to avoid CPU overuse
                    time.sleep(0.005)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.exception(e)
        finally:
            # Clean up
            self.running = False
            self.disconnect()
            cv2.destroyAllWindows()
            logger.info("Display stopped")

def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simple Camera Display Client")
    parser.add_argument("--host", default="127.0.0.1", help="Camera service host address")
    parser.add_argument("--port", type=int, default=5555, help="Camera service port")
    parser.add_argument("--fps", type=int, default=30, help="Target display frames per second")
    args = parser.parse_args()
    
    # Print instructions
    print("Simple Camera Display")
    print("--------------------")
    print("Press 'q' or ESC to quit")
    
    # Create and run display
    display = SimpleCameraDisplay(
        server_host=args.host,
        server_port=args.port,
        target_fps=args.fps
    )
    
    # Run the display
    display.run()

if __name__ == "__main__":
    main()