#!/usr/bin/env python3
"""
Pupil Core Video Display
---------------------
Client application to display video streams from the camera service.
"""
import argparse
import logging

from pupil.ui.video_display import VideoDisplay

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DisplayVideo")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pupil Core Video Display")
    parser.add_argument("--host", default="127.0.0.1", help="Camera service host address")
    parser.add_argument("--port", type=int, default=5555, help="Camera service port")
    parser.add_argument("--fps", type=int, default=30, help="Target display frames per second")
    args = parser.parse_args()
    
    # Print instructions
    print("Pupil Core Video Display")
    print("---------------------")
    print("Press 'q' or ESC to quit")
    
    # Create and run display
    display = VideoDisplay(
        server_host=args.host,
        server_port=args.port,
        target_fps=args.fps
    )
    
    # Run the display
    display.run()


if __name__ == "__main__":
    main()