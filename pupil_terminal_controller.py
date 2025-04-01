#!/usr/bin/env python3
"""
Pupil Camera Terminal Controller
A simple terminal-based event manager for Pupil Core cameras.
"""

import sys
import os
import time
import threading
import signal
import argparse
from datetime import timedelta

# Import from pupil_camera_recorder.py
from pupil_camera_recorder import (
    list_available_cameras,
    UVCSource,
    SimpleRecorder
)

# Custom tracker initializer to handle frame size and rate at creation time
def create_custom_tracker(
    world_name="ID2", 
    eye0_name="ID0", 
    eye1_name="ID1", 
    exposure_mode="auto",
    world_size=(1280, 720),
    world_fps=30,
    eye_size=(192, 192),
    eye_fps=120
):
    """Create a custom eye tracker with specified frame sizes and rates"""
    # Initialize cameras with custom settings
    world_cam = UVCSource(name=world_name, frame_size=world_size, frame_rate=world_fps, exposure_mode=exposure_mode)
    eye0_cam = UVCSource(name=eye0_name, frame_size=eye_size, frame_rate=eye_fps, exposure_mode=exposure_mode)
    eye1_cam = UVCSource(name=eye1_name, frame_size=eye_size, frame_rate=eye_fps, exposure_mode=exposure_mode)
    
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

# Global variables for thread control
running = True
recording = False
tracker = None
status_thread = None
camera_threads = []
recording_start_time = None

def format_time(seconds):
    """Format seconds into a readable time string"""
    return str(timedelta(seconds=seconds)).split('.')[0]

def show_status():
    """Display current status information"""
    global tracker, recording, recording_start_time
    
    if not recording:
        print("Status: Not recording")
        return
    
    elapsed = time.time() - recording_start_time
    world_frames = tracker["recorder"].frame_counts["world"]
    eye0_frames = tracker["recorder"].frame_counts["eye0"]
    eye1_frames = tracker["recorder"].frame_counts["eye1"]
    
    print("\nRecording Status:")
    print(f"  Duration: {format_time(elapsed)}")
    print(f"  World camera frames: {world_frames} ({world_frames/elapsed:.1f} fps)")
    print(f"  Eye0 camera frames: {eye0_frames} ({eye0_frames/elapsed:.1f} fps)")
    print(f"  Eye1 camera frames: {eye1_frames} ({eye1_frames/elapsed:.1f} fps)")
    print(f"  Recording path: {tracker['recorder'].current_rec_path}")

def status_loop():
    """Thread function to periodically display status"""
    global running, recording
    while running:
        if recording:
            show_status()
        time.sleep(5)  # Update every 5 seconds

def capture_world():
    """Thread function to capture frames from world camera"""
    global tracker, running, recording
    frames = 0
    cam = tracker["world_cam"]
    while running and cam.online:
        frame = cam.get_frame()
        if frame and recording:
            tracker["recorder"].update(world_frame=frame, eye0_frame=None, eye1_frame=None)
            frames += 1

def capture_eye0():
    """Thread function to capture frames from eye0 camera"""
    global tracker, running, recording
    frames = 0
    cam = tracker["eye0_cam"]
    while running and cam.online:
        frame = cam.get_frame()
        if frame and recording:
            tracker["recorder"].update(world_frame=None, eye0_frame=frame, eye1_frame=None)
            frames += 1

def capture_eye1():
    """Thread function to capture frames from eye1 camera"""
    global tracker, running, recording
    frames = 0
    cam = tracker["eye1_cam"]
    while running and cam.online:
        frame = cam.get_frame()
        if frame and recording:
            tracker["recorder"].update(world_frame=None, eye0_frame=None, eye1_frame=frame)
            frames += 1

def start_recording():
    """Start recording from all cameras"""
    global tracker, recording, recording_start_time, camera_threads
    
    if recording:
        print("Recording already in progress")
        return
    
    try:
        print("Starting recording...")
        rec_path = tracker["recorder"].start(
            world_cam=tracker["world_cam"],
            eye0_cam=tracker["eye0_cam"],
            eye1_cam=tracker["eye1_cam"]
        )
        
        # Start capture threads if not already running
        if not camera_threads:
            camera_threads = []
            if tracker["world_cam"].online:
                camera_threads.append(threading.Thread(target=capture_world, name="WorldCam"))
            if tracker["eye0_cam"].online:
                camera_threads.append(threading.Thread(target=capture_eye0, name="Eye0Cam"))
            if tracker["eye1_cam"].online:
                camera_threads.append(threading.Thread(target=capture_eye1, name="Eye1Cam"))
            
            for t in camera_threads:
                t.daemon = True
                t.start()
        
        recording = True
        recording_start_time = time.time()
        print(f"Recording started. Output: {rec_path}")
    except Exception as e:
        print(f"Error starting recording: {e}")

def stop_recording():
    """Stop current recording"""
    global tracker, recording
    
    if not recording:
        print("No recording in progress")
        return
    
    try:
        print("Stopping recording...")
        rec_path, stats = tracker["recorder"].stop()
        recording = False
        
        print("\nRecording Statistics:")
        for camera_id, data in stats.items():
            if data["frames"] > 0:
                print(f"  {camera_id.capitalize()}: {data['frames']} frames, {data['fps']:.2f} fps")
        
        print(f"\nRecording saved to: {rec_path}")
    except Exception as e:
        print(f"Error stopping recording: {e}")

def list_cameras():
    """List all available cameras"""
    cameras = list_available_cameras()
    if not cameras:
        print("No cameras detected")
        return
    
    print("\nAvailable cameras:")
    for cam in cameras:
        print(f"  {cam['id']}: {cam['name']} (UID: {cam['uid']})")
    
    if tracker:
        print("\nActive cameras:")
        for name, cam in tracker.items():
            if name != "recorder" and hasattr(cam, "online"):
                status = "online" if cam.online else "offline"
                if cam.online:
                    print(f"  {name}: {cam.name} at {cam.frame_size}@{cam.frame_rate}fps ({status})")
                else:
                    print(f"  {name}: {cam.name} ({status})")

def cleanup():
    """Clean up resources and exit"""
    global tracker, running
    
    print("Cleaning up resources...")
    running = False
    
    if recording:
        stop_recording()
    
    # Wait for threads to finish
    for t in camera_threads:
        t.join(timeout=1.0)
    
    if tracker:
        for name, component in tracker.items():
            if hasattr(component, "cleanup"):
                component.cleanup()
    
    print("Exited.")

def signal_handler(sig, frame):
    """Handle Ctrl+C and other signals"""
    print("\nReceived interrupt, shutting down...")
    cleanup()
    sys.exit(0)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Terminal controller for Pupil Core cameras")
    parser.add_argument("--session", type=str, default=None, 
                        help="Session name for recordings (default: auto-generated date)")
    parser.add_argument("--dir", type=str, default=None,
                        help="Root directory for recordings (default: ~/pupil_recordings)")
    parser.add_argument("--exposure", type=str, default="auto", choices=["auto", "manual"],
                        help="Camera exposure mode (default: auto)")
    parser.add_argument("--world-size", type=str, default="1280,720",
                        help="World camera resolution (default: 1280,720)")
    parser.add_argument("--eye-size", type=str, default="192,192",
                        help="Eye camera resolution (default: 192,192)")
    parser.add_argument("--world-fps", type=int, default=30,
                        help="World camera frame rate (default: 30)")
    parser.add_argument("--eye-fps", type=int, default=120,
                        help="Eye camera frame rate (default: 120)")
    return parser.parse_args()

def main():
    """Main function"""
    global tracker, running, status_thread
    
    args = parse_arguments()
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Pupil Camera Terminal Controller")
    print("--------------------------------")
    
    # Parse camera settings
    world_size = tuple(map(int, args.world_size.split(',')))
    eye_size = tuple(map(int, args.eye_size.split(',')))
    
    # List available cameras
    list_cameras()
    
    # Create eye tracker with custom settings
    print("\nInitializing cameras...")
    tracker = create_custom_tracker(
        exposure_mode=args.exposure,
        world_size=world_size,
        world_fps=args.world_fps,
        eye_size=eye_size,
        eye_fps=args.eye_fps
    )
    
    # Configure recorder if custom settings provided
    if args.dir or args.session:
        recorder = SimpleRecorder(rec_dir=args.dir, session_name=args.session)
        tracker["recorder"] = recorder
    
    # Initialize status thread
    status_thread = threading.Thread(target=status_loop, daemon=True)
    status_thread.start()
    
    # Show active cameras
    print("\nInitialized cameras:")
    for name, cam in tracker.items():
        if name != "recorder" and hasattr(cam, "online"):
            status = "online" if cam.online else "offline"
            if cam.online:
                print(f"  {name}: {cam.name} at {cam.frame_size}@{cam.frame_rate}fps ({status})")
            else:
                print(f"  {name}: {cam.name} ({status})")
    
    # Command loop
    print("\nCommands: start, stop, status, list, quit")
    while running:
        try:
            cmd = input("\n> ").strip().lower()
            
            if cmd == "start":
                start_recording()
            elif cmd == "stop":
                stop_recording()
            elif cmd == "status":
                show_status()
            elif cmd == "list":
                list_cameras()
            elif cmd in ["quit", "exit", "q"]:
                cleanup()
                break
            elif cmd == "help":
                print("Available commands:")
                print("  start  - Start recording from all cameras")
                print("  stop   - Stop current recording")
                print("  status - Show recording status")
                print("  list   - List available cameras")
                print("  quit   - Exit the application")
            elif cmd:
                print(f"Unknown command: {cmd}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("Exiting...")

if __name__ == "__main__":
    main()