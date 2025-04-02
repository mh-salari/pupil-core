#!/usr/bin/env python3
"""
Pupil Detection Example
---------------------
Demonstration of pupil detection using eye camera feeds.
"""
import argparse
import logging
import time
import signal
import sys

from pupil.detection.pupil_detector import PupilDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PupilDetection")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pupil Detection")
    parser.add_argument("--host", default="127.0.0.1", help="Camera service host address")
    parser.add_argument("--port", type=int, default=5555, help="Camera service port")
    parser.add_argument("--visualize", action="store_true", help="Show visualization windows")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create detector
    detector = PupilDetector(
        server_host=args.host,
        server_port=args.port
    )
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        detector.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start detection
    detector.start(show_visualization=args.visualize)
    
    # Print instructions
    print("\nPupil Detection Running")
    print("----------------------")
    print("Press Ctrl+C to exit")
    if args.visualize:
        print("Press ESC in visualization window to exit")
    print("\nDetection results (updated every second):")
    
    # Display detection results every second
    try:
        while detector.is_detecting():
            # Get current pupil positions
            results = detector.get_pupil_positions()
            
            # Display information
            print("\033[H\033[J", end="")  # Clear terminal
            print("Pupil Detection Results:")
            print("-----------------------")
            
            # Eye0 info
            eye0 = results.get("eye0")
            if eye0:
                print(f"Eye0: Center={eye0['center']}, Radius={eye0['radius']:.1f}, "
                      f"Confidence={eye0['confidence']:.1f}%")
            else:
                print("Eye0: No pupil detected")
                
            # Eye1 info
            eye1 = results.get("eye1")
            if eye1:
                print(f"Eye1: Center={eye1['center']}, Radius={eye1['radius']:.1f}, "
                      f"Confidence={eye1['confidence']:.1f}%")
            else:
                print("Eye1: No pupil detected")
                
            # FPS info
            print(f"\nProcessing: Eye0 {results['eye0_fps']:.1f} FPS, "
                  f"Eye1 {results['eye1_fps']:.1f} FPS")
            
            # Wait before next update
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        detector.stop()


if __name__ == "__main__":
    main()