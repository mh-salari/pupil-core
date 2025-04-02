#!/usr/bin/env python3
"""
Pupil Core Remote Controller
--------------------------
Terminal-based client to control the Pupil Core camera service remotely.
"""
import argparse
import logging

from pupil.ui.remote_controller import RemoteController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RemoteControl")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Pupil Core Remote Controller")
    parser.add_argument("--host", default="127.0.0.1", help="Camera service host address")
    parser.add_argument("--port", type=int, default=5555, help="Camera service port")
    args = parser.parse_args()
    
    # Create and start controller
    controller = RemoteController(
        server_host=args.host,
        server_port=args.port
    )
    
    try:
        # Start controller
        controller.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()