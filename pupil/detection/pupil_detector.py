"""
Pupil Detector
------------
Client to receive eye camera frames and detect pupils.
"""
import time
import json
import logging
import uuid
import threading
import cv2
import numpy as np
import zmq

from ..service.message_types import MessageType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PupilDetector:
    """
    Pupil detection client that connects to the camera service,
    receives eye camera frames, and performs pupil detection.
    """
    
    def __init__(self, server_host="127.0.0.1", server_port=5555):
        """
        Initialize the pupil detector.
        
        Args:
            server_host: Camera service host address
            server_port: Camera service command port
        """
        self.server_host = server_host
        self.server_port = server_port
        self.client_id = f"pupil_detector_{uuid.uuid4().hex[:8]}"
        
        # Detection parameters
        self.min_pupil_radius = 10
        self.max_pupil_radius = 100
        self.detection_threshold = 80  # Threshold for binary image
        
        # Result storage
        self.eye0_result = None
        self.eye1_result = None
        self.last_eye0_time = 0
        self.last_eye1_time = 0
        
        # Processing rates
        self.eye0_fps = 0
        self.eye1_fps = 0
        
        # Visualization windows
        self.show_visualization = False
        
        # Threading and control
        self.running = False
        self.threads = []
        self.result_lock = threading.Lock()
        
        # Connect to ZeroMQ
        self._init_zeromq()
    
    def _init_zeromq(self):
        """Initialize ZeroMQ sockets."""
        # Initialize ZeroMQ context
        self.context = zmq.Context()
        
        # Create command socket (REQ)
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.connect(f"tcp://{self.server_host}:{self.server_port}")
        
        # Create subscriber socket for frames
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{self.server_host}:{self.server_port+1}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        
        # Set socket options for better performance
        self.sub_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        self.sub_socket.setsockopt(zmq.LINGER, 0)
    
    def start(self, show_visualization=False):
        """
        Start pupil detection.
        
        Args:
            show_visualization: Whether to show detection visualization
        """
        if self.running:
            logger.warning("Pupil detector already running")
            return
        
        self.show_visualization = show_visualization
        if show_visualization:
            cv2.namedWindow("Eye0 Detection", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Eye1 Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Eye0 Detection", 400, 400)
            cv2.resizeWindow("Eye1 Detection", 400, 400)
        
        # Connect and start streaming
        logger.info("Starting pupil detector...")
        self._send_command("start_streaming", {"camera_id": "eye0"})
        self._send_command("start_streaming", {"camera_id": "eye1"})
        
        # Start processing thread
        self.running = True
        process_thread = threading.Thread(target=self._process_frames, daemon=True)
        process_thread.start()
        self.threads.append(process_thread)
        
        logger.info("Pupil detector started")
    
    def stop(self):
        """Stop pupil detection and clean up resources."""
        if not self.running:
            return
        
        logger.info("Stopping pupil detector...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Stop streaming
        self._send_command("stop_streaming", {"camera_id": "eye0"})
        self._send_command("stop_streaming", {"camera_id": "eye1"})
        
        # Close ZeroMQ sockets
        self.command_socket.close()
        self.sub_socket.close()
        self.context.term()
        
        # Close visualization windows
        if self.show_visualization:
            cv2.destroyAllWindows()
        
        logger.info("Pupil detector stopped")
    
    def _send_command(self, command, params=None):
        """
        Send a command to the camera service.
        
        Args:
            command: Command string
            params: Command parameters
        
        Returns:
            Response dictionary
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
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return {"status": "error", "message": str(e)}
    
    def _process_frames(self):
        """Process incoming frames and detect pupils."""
        last_fps_time = time.time()
        eye0_count = 0
        eye1_count = 0
        
        try:
            while self.running:
                try:
                    # Receive frame message
                    message = self.sub_socket.recv_multipart()
                    
                    # Process frame if it's from an eye camera
                    if len(message) == 2:
                        # Parse metadata
                        metadata = json.loads(message[0])
                        
                        # Check if it's a frame response
                        if metadata.get("type") == MessageType.FRAME_RESPONSE:
                            camera_id = metadata.get("camera_id")
                            
                            # Only process eye cameras
                            if camera_id in ["eye0", "eye1"]:
                                # Decode JPEG image
                                frame_data = message[1]
                                img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_GRAYSCALE)
                                
                                # Detect pupil
                                result = self._detect_pupil(img)
                                
                                # Visualize if enabled
                                if self.show_visualization and img is not None and result is not None:
                                    self._visualize_detection(camera_id, img, result)
                                
                                # Store result with thread safety
                                with self.result_lock:
                                    if camera_id == "eye0":
                                        self.eye0_result = result
                                        self.last_eye0_time = time.time()
                                        eye0_count += 1
                                    elif camera_id == "eye1":
                                        self.eye1_result = result
                                        self.last_eye1_time = time.time()
                                        eye1_count += 1
                
                except zmq.Again:
                    # No message available, continue
                    pass
                
                # Calculate FPS every second
                now = time.time()
                if now - last_fps_time >= 1.0:
                    # Calculate FPS
                    elapsed = now - last_fps_time
                    self.eye0_fps = eye0_count / elapsed
                    self.eye1_fps = eye1_count / elapsed
                    
                    # Log FPS
                    logger.debug(f"Pupil detection FPS - Eye0: {self.eye0_fps:.1f}, Eye1: {self.eye1_fps:.1f}")
                    
                    # Reset counters
                    last_fps_time = now
                    eye0_count = 0
                    eye1_count = 0
                
                # Handle keyboard input if visualization is enabled
                if self.show_visualization and cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    logger.info("ESC key pressed, stopping detection")
                    self.running = False
                    break
                
                # Sleep a small amount to avoid busy waiting
                time.sleep(0.001)
                
        except Exception as e:
            logger.error(f"Error in frame processing loop: {e}")
            logger.exception(e)
            self.running = False
    
    def _detect_pupil(self, image):
        """
        Detect pupil in grayscale eye image.
        
        Args:
            image: Grayscale eye image
            
        Returns:
            Dictionary with pupil properties or None if not detected
        """
        try:
            if image is None or image.size == 0:
                return None
            
            # Apply blur to reduce noise
            blurred = cv2.GaussianBlur(image, (7, 7), 0)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(blurred, self.detection_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest contour
            if not contours:
                return None
                
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fit a circle to the contour
            area = cv2.contourArea(largest_contour)
            if area < 50:  # Minimum area to consider
                return None
                
            # Find center and radius using minEnclosingCircle
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Validate radius
            if radius < self.min_pupil_radius or radius > self.max_pupil_radius:
                return None
            
            # Calculate confidence based on circularity
            circularity = 4 * np.pi * area / (cv2.arcLength(largest_contour, True) ** 2)
            confidence = circularity * 100  # Scale to percentage
            
            # Return detection result
            return {
                "center": center,
                "radius": radius,
                "confidence": confidence,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in pupil detection: {e}")
            return None
    
    def _visualize_detection(self, camera_id, image, detection):
        """
        Visualize pupil detection.
        
        Args:
            camera_id: Camera identifier ('eye0' or 'eye1')
            image: Grayscale eye image
            detection: Detection result
        """
        try:
            # Convert grayscale to BGR for colored annotations
            display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Draw pupil circle
            center = detection.get("center")
            radius = detection.get("radius")
            confidence = detection.get("confidence", 0)
            
            if center and radius:
                # Draw circle
                cv2.circle(display_img, center, radius, (0, 255, 0), 2)
                cv2.circle(display_img, center, 2, (0, 0, 255), -1)  # Center point
                
                # Add text with confidence
                conf_text = f"Conf: {confidence:.1f}%"
                cv2.putText(display_img, conf_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display window
            window_name = f"{camera_id} Detection"
            cv2.imshow(window_name, display_img)
            
        except Exception as e:
            logger.error(f"Error in detection visualization: {e}")
    
    def get_pupil_positions(self):
        """
        Get current pupil positions for both eyes.
        
        Returns:
            Dictionary with eye0 and eye1 pupil positions and properties
        """
        with self.result_lock:
            return {
                "eye0": self.eye0_result,
                "eye1": self.eye1_result,
                "eye0_fps": self.eye0_fps,
                "eye1_fps": self.eye1_fps
            }
    
    def is_detecting(self):
        """Check if pupil detection is currently active."""
        return self.running