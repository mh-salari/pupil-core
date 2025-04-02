"""
Remote Controller
---------------
Terminal-based client for controlling the camera service remotely.
"""
import os
import time
import json
import logging
import threading
import uuid
import sys
from queue import Queue, Empty

import zmq

from ..service.message_types import MessageType
from ..utils.timestamp import format_duration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RemoteController:
    """
    Terminal-based remote controller for camera service.
    Provides commands to start/stop recording and query status.
    """
    
    def __init__(self, server_host="127.0.0.1", server_port=5555):
        """
        Initialize the remote controller.
        
        Args:
            server_host: Camera service host address
            server_port: Camera service command port
        """
        self.server_host = server_host
        self.server_port = server_port
        self.client_id = f"remote_controller_{uuid.uuid4().hex[:8]}"
        
        # Initialize ZeroMQ context
        self.context = zmq.Context()
        
        # Create command socket (REQ)
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.connect(f"tcp://{server_host}:{server_port}")
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.command_socket.RCVTIMEO = 5000  # 5 second timeout
        
        # Create subscriber socket for status
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{server_host}:{server_port+1}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        
        # Status variables
        self.is_connected = False
        self.is_recording = False
        self.recording_path = None
        self.recording_duration = 0
        self.status_data = {}
        
        # Control variables
        self.running = False
        self.threads = []
    
    def connect(self):
        """Connect to the camera service."""
        try:
            # Send initial status request
            response = self.send_command("get_status")
            if response.get("status") == "ok":
                self.is_connected = True
                logger.info(f"Connected to camera service at {self.server_host}:{self.server_port}")
                
                # Update status from response
                self.update_status(response)
                
                return True
            else:
                logger.error(f"Failed to connect: {response.get('message', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to camera service: {e}")
            return False
    
    def start(self):
        """Start the remote controller."""
        if self.running:
            logger.warning("Controller already running")
            return
        
        # Connect to camera service
        if not self.is_connected and not self.connect():
            logger.error("Failed to connect to camera service")
            return
        
        logger.info("Starting remote controller")
        self.running = True
        
        # Start status receiver thread
        receiver_thread = threading.Thread(target=self.receive_status, daemon=True)
        receiver_thread.start()
        self.threads.append(receiver_thread)
        
        # Start command processor
        self.process_commands()
    
    def stop(self):
        """Stop the remote controller."""
        if not self.running:
            return
        
        logger.info("Stopping remote controller...")
        self.running = False
        
        # Send disconnect command
        self.send_command("disconnect")
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Close ZeroMQ sockets
        self.command_socket.close()
        self.sub_socket.close()
        self.context.term()
        
        logger.info("Remote controller stopped")
    
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
    
    def receive_status(self):
        """Receive status updates from the camera service."""
        logger.info("Status receiver started")
        
        while self.running:
            try:
                # Wait for next message with timeout
                if not self.sub_socket.poll(1000):
                    continue
                
                # Receive message - handling both single and multipart messages
                try:
                    # Try to receive a multipart message
                    message = self.sub_socket.recv_multipart(zmq.NOBLOCK)
                    
                    # If it's a multipart message, it may be frame data
                    # We're only interested in the first part which is JSON metadata
                    if message and len(message) > 0:
                        # Parse the first part as JSON, ignore other parts (binary data)
                        metadata = json.loads(message[0])
                        
                        # Check for status update
                        if metadata.get("type") == MessageType.STATUS_UPDATE:
                            self.update_status(metadata)
                            
                except zmq.Again:
                    # If no multipart message, try simple JSON message
                    try:
                        message = self.sub_socket.recv_json()
                        
                        # Check for status update
                        if message.get("type") == MessageType.STATUS_UPDATE:
                            self.update_status(message)
                    except zmq.Again:
                        # No messages available
                        continue
                
            except zmq.ZMQError as e:
                logger.error(f"ZMQ error receiving status: {e}")
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error receiving status: {e}")
                time.sleep(0.1)
    
    def update_status(self, status):
        """Update internal status from server status."""
        # Store full status data
        self.status_data = status
        
        # Update recording status
        if status.get("recording") is not None:
            self.is_recording = status["recording"]
        
        # Update recording info if available
        if "recording_info" in status:
            self.recording_path = status["recording_info"].get("path")
            self.recording_duration = status["recording_info"].get("duration", 0)
        else:
            self.recording_duration = 0
    
    def display_help(self):
        """Display help information."""
        print("\nAvailable commands:")
        print("  status        - Show camera service status")
        print("  cameras       - List available cameras")
        print("  record start  - Start recording")
        print("  record stop   - Stop recording")
        print("  property list [camera_id] - List camera properties")
        print("  property set <camera_id> <property> <value> - Set camera property")
        print("  help          - Show this help message")
        print("  exit          - Exit the remote controller")
        print()
    
    def display_status(self):
        """Display current status."""
        print("\n=== Camera Service Status ===")
        
        # Recording status
        if self.is_recording:
            duration_str = format_duration(self.recording_duration)
            print(f"Recording: ACTIVE ({duration_str})")
            print(f"Recording path: {self.recording_path}")
        else:
            print("Recording: INACTIVE")
        
        # Camera status
        print("\nCameras:")
        cameras = self.status_data.get("cameras", {})
        for cam_id, cam_info in cameras.items():
            online = cam_info.get("online", False)
            status = "ONLINE" if online else "OFFLINE"
            name = cam_info.get("name", "Unknown")
            print(f"  {cam_id}: {status} - {name}")
            
            if online:
                size = cam_info.get("frame_size", (0, 0))
                rate = cam_info.get("frame_rate", 0)
                print(f"    {size[0]}x{size[1]} @ {rate}fps")
        
        # Frame counts
        print("\nFrame counts:")
        frame_counts = self.status_data.get("frame_counts", {})
        for cam_id, count in frame_counts.items():
            print(f"  {cam_id}: {count}")
        
        # Client info
        client_count = self.status_data.get("clients", 0)
        print(f"\nConnected clients: {client_count}")
        
        print()
    
    def display_cameras(self):
        """Display detailed camera information."""
        # Get camera information
        response = self.send_command("list_cameras")
        if response.get("status") != "ok":
            print(f"Error: {response.get('message', 'Failed to get camera list')}")
            return
        
        cameras = response.get("cameras", {})
        
        print("\n=== Camera Information ===")
        
        # Show connected cameras
        print("\nConnected cameras:")
        for cam_id, cam_info in cameras.items():
            if cam_id not in ["world", "eye0", "eye1", "available"]:
                continue
                
            if cam_id == "available":
                continue
                
            online = cam_info.get("online", False)
            status = "ONLINE" if online else "OFFLINE"
            name = cam_info.get("name", "Unknown")
            print(f"  {cam_id}: {status} - {name}")
            
            if online:
                size = cam_info.get("frame_size", (0, 0))
                rate = cam_info.get("frame_rate", 0)
                print(f"    Resolution: {size[0]}x{size[1]}")
                print(f"    Frame rate: {rate}fps")
        
        # Show available cameras
        print("\nAvailable cameras:")
        available = cameras.get("available", [])
        if not available:
            print("  No cameras detected")
        else:
            for i, cam in enumerate(available):
                print(f"  {i}: {cam.get('name')} (UID: {cam.get('uid')})")
        
        print()
    
    def display_camera_properties(self, cam_id):
        """Display camera properties."""
        # Get camera information
        response = self.send_command("list_cameras")
        if response.get("status") != "ok":
            print(f"Error: {response.get('message', 'Failed to get camera list')}")
            return
        
        cameras = response.get("cameras", {})
        
        # Check if camera exists
        if cam_id not in cameras:
            print(f"Error: Camera '{cam_id}' not found")
            return
        
        # Check if camera is online
        if not cameras[cam_id].get("online", False):
            print(f"Error: Camera '{cam_id}' is offline")
            return
        
        print(f"\n=== Properties for {cam_id} camera ===")
        
        # Display camera controls
        controls = cameras[cam_id].get("controls", {})
        if not controls:
            print("  No properties available")
        else:
            print(f"{'Property':<30} {'Value':<10} {'Range':<20} {'Default':<10}")
            print("-" * 70)
            
            for name, info in controls.items():
                value = info.get("value", "N/A")
                min_val = info.get("min", "N/A")
                max_val = info.get("max", "N/A")
                default = info.get("default", "N/A")
                
                range_str = f"{min_val} - {max_val}"
                print(f"{name:<30} {value:<10} {range_str:<20} {default:<10}")
        
        print()
    
    def set_camera_property(self, cam_id, property_name, value):
        """Set a camera property."""
        try:
            # Convert value to number if possible
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Keep as string
            
            # Send command
            response = self.send_command("set_camera_property", {
                "camera_id": cam_id,
                "property": property_name,
                "value": value
            })
            
            if response.get("status") == "ok":
                print(f"Property set: {response.get('message', 'Success')}")
                if "new_value" in response:
                    print(f"New value: {response['new_value']}")
            else:
                print(f"Error: {response.get('message', 'Failed to set property')}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    def start_recording(self):
        """Start recording."""
        if self.is_recording:
            print("Recording already in progress")
            return
        
        # Send command
        response = self.send_command("start_recording")
        
        if response.get("status") == "ok":
            self.is_recording = True
            self.recording_path = response.get("path")
            print(f"Recording started: {self.recording_path}")
        else:
            print(f"Error: {response.get('message', 'Failed to start recording')}")
    
    def stop_recording(self):
        """Stop recording."""
        if not self.is_recording:
            print("No recording in progress")
            return
        
        # Send command
        response = self.send_command("stop_recording")
        
        if response.get("status") == "ok":
            self.is_recording = False
            path = response.get("path")
            stats = response.get("stats", {})
            
            print(f"Recording stopped: {path}")
            
            # Display statistics
            if stats:
                print("\nRecording statistics:")
                for cam_id, data in stats.items():
                    frames = data.get("frames", 0)
                    fps = data.get("fps", 0)
                    if frames > 0:
                        print(f"  {cam_id}: {frames} frames, {fps:.2f} fps")
        else:
            print(f"Error: {response.get('message', 'Failed to stop recording')}")
    
    def process_commands(self):
        """Process commands from the user."""
        self.display_help()
        
        while self.running:
            try:
                # Get command from user
                cmd_line = input("remote> ").strip()
                
                # Skip empty commands
                if not cmd_line:
                    continue
                
                # Split command and arguments
                parts = cmd_line.split()
                cmd = parts[0].lower()
                args = parts[1:]
                
                # Process command
                if cmd == "help":
                    self.display_help()
                    
                elif cmd == "exit":
                    self.running = False
                    break
                    
                elif cmd == "status":
                    self.display_status()
                    
                elif cmd == "cameras":
                    self.display_cameras()
                    
                elif cmd == "record":
                    if len(args) < 1:
                        print("Error: Missing subcommand (start/stop)")
                        continue
                        
                    if args[0].lower() == "start":
                        self.start_recording()
                    elif args[0].lower() == "stop":
                        self.stop_recording()
                    else:
                        print(f"Error: Unknown subcommand: {args[0]}")
                
                elif cmd == "property":
                    if len(args) < 1:
                        print("Error: Missing subcommand (list/set)")
                        continue
                        
                    if args[0].lower() == "list":
                        cam_id = args[1] if len(args) > 1 else "world"
                        self.display_camera_properties(cam_id)
                    elif args[0].lower() == "set":
                        if len(args) < 4:
                            print("Error: Usage - property set <camera_id> <property> <value>")
                            continue
                        
                        cam_id = args[1]
                        property_name = args[2]
                        value = args[3]
                        self.set_camera_property(cam_id, property_name, value)
                    else:
                        print(f"Error: Unknown subcommand: {args[0]}")
                
                else:
                    print(f"Error: Unknown command: {cmd}")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                print(f"Error: {e}")