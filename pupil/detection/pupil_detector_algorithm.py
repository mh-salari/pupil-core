"""
Pupil Detector Algorithm
---------------
Standalone pupil detection algorithm that uses Pupil Labs detector.
"""
import logging
import time
import cv2
import numpy as np
from pupil_detectors import Detector2D

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a global detector instance for reuse
_detector = None
def get_detector():
    """Get or initialize the detector"""
    global _detector
    if _detector is None:
        _detector = Detector2D()
    return _detector

def detect_pupil(image, min_pupil_radius=10, max_pupil_radius=100):
    """
    Pupil detection using Pupil Labs detector library.
    
    Args:
        image: Grayscale eye image
        min_pupil_radius: Minimum valid pupil radius (optional, used for validation)
        max_pupil_radius: Maximum valid pupil radius (optional, used for validation)
        
    Returns:
        Dictionary with pupil properties or None if not detected
    """
    try:
        if image is None or image.size == 0:
            return None
        
        # Ensure the image is grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Get the detector instance
        detector = get_detector()
        
        # Run detection
        result = detector.detect(gray)
        
        # Check if a pupil was detected
        if result is None or "ellipse" not in result or not result["confidence"]:
            return None
            
        # Extract ellipse parameters
        ellipse = result["ellipse"]
        center = tuple(int(v) for v in ellipse["center"])
        axes = tuple(int(v / 2) for v in ellipse["axes"])
        angle = ellipse["angle"]
        
        # Calculate equivalent radius (average of semi-axes)
        radius = int((axes[0] + axes[1]) / 2)
        
        # Validate radius if needed
        if not (min_pupil_radius <= radius <= max_pupil_radius):
            return None
            
        # Extract confidence
        confidence = float(result["confidence"]) * 100  # Scale to percentage
        
        # Return detection result in compatible format
        return {
            "center": center,
            "radius": radius,  # Approximate radius
            "ellipse": {       # Additional ellipse data
                "center": center,
                "axes": axes,
                "angle": angle
            },
            "confidence": confidence,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error in pupil detection: {e}")
        return None