"""
Pupil Detector Client
-------------------
Client to connect to a running pupil detector service to get pupil detection results.
"""
import time
import json
import logging
import uuid
import zmq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PupilDetectorClient:
    """
    Client to connect to a running pupil detector service and get pupil positions.
    This is a lightweight client that does not perform detection itself.
    """
    
    def __init__(self, server_host="127.0.0.1", server_port=5555):
        """
        Initialize the client.
        
        Args:
            server_host: Camera service host address
            server_port: Camera service port
        """
        self.server_host = server_host
        self.server_port = server_port
        self.client_id = f"pupil_client_{uuid.uuid4().hex[:8]}"
        
        # Store latest results
        self.latest_results = {
            "eye0": None,
            "eye1": None,
            "eye0_fps": 0,
            "eye1_fps": 0
        }
        
        # Connection state
        self.connected = False
        self.detector_running = False
        
        # Connect to ZeroMQ
        self._init_connection()
    
    def _init_connection(self):
        """Initialize ZeroMQ connection."""
        try:
            # Initialize ZeroMQ context
            self.context = zmq.Context()
            
            # Create REQ socket for client-server communication
            # Pupil detector service runs on camera service port + 2
            detector_port = self.server_port + 2
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.server_host}:{detector_port}")
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
            
            # Check if detector is running
            self._check_detector_status()
            
            self.connected = True
            logger.info(f"Connected to pupil detector service at {self.server_host}:{detector_port}")
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to pupil detector service: {e}")
            return False
    
    def _check_detector_status(self):
        """Check if the pupil detector is running."""
        try:
            response = self._send_command("get_pupil_status")
            self.detector_running = response.get("is_detecting", False)
            return self.detector_running
        except Exception as e:
            logger.warning(f"Failed to check detector status: {e}")
            self.detector_running = False
            return False
    
    def _send_command(self, command, params=None):
        """
        Send a command to the server.
        
        Args:
            command: Command string
            params: Command parameters (optional)
            
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
            self.socket.send_json(message)
            
            # Wait for response
            response = self.socket.recv_json()
            
            return response
        except zmq.ZMQError as e:
            logger.error(f"ZMQ error sending command: {e}")
            return {"status": "error", "message": f"ZMQ error: {e}"}
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return {"status": "error", "message": f"Error: {e}"}
    
    def get_pupil_positions(self):
        """
        Get the current pupil positions from the detector service.
        
        Returns:
            Dictionary with eye0 and eye1 pupil positions
        """
        try:
            # Check if detector is running
            if not self.detector_running:
                if not self._check_detector_status():
                    return self.latest_results
            
            # Get pupil positions
            response = self._send_command("get_pupil_positions")
            
            if response.get("status") == "ok":
                # Update latest results
                self.latest_results = response.get("data", self.latest_results)
            
            return self.latest_results
        except Exception as e:
            logger.error(f"Error getting pupil positions: {e}")
            return self.latest_results
    
    def disconnect(self):
        """Disconnect from the pupil detector service."""
        if self.connected:
            try:
                # Send disconnect command
                self._send_command("disconnect")
                
                # Close socket
                self.socket.close()
                self.context.term()
                
                self.connected = False
                logger.info("Disconnected from pupil detector service")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
    
    def is_detector_running(self):
        """
        Check if the detector is running.
        
        Returns:
            True if the detector is running, False otherwise
        """
        return self._check_detector_status()