#!/usr/bin/env python3
"""
Pupil Core Camera Service
------------------------
Main entry point to start the camera service.
"""
import argparse
import logging
import signal
import sys
import time

from pupil.service.camera_service import CameraService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PupilService")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pupil Core Camera Service")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=5555, help="Port to bind to")
    parser.add_argument("--recording-dir", help="Directory to store recordings")
    args = parser.parse_args()
    
    # Create and start service
    service = CameraService(
        host=args.host,
        port=args.port,
        recording_dir=args.recording_dir
    )
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start service
    try:
        service.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        service.stop()


if __name__ == "__main__":
    main()