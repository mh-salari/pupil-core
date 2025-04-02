"""
Timestamp Utilities
-----------------
Functions for handling monotonic timestamps and time conversions.
"""
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_monotonic_time():
    """
    Get a monotonic timestamp. Uses uvc.get_time_monotonic() if available,
    otherwise falls back to time.monotonic().
    
    Returns:
        Monotonic timestamp in seconds
    """
    try:
        import uvc
        return uvc.get_time_monotonic()
    except ImportError:
        return time.monotonic()


def ensure_monotonic(new_ts, last_ts):
    """
    Ensure timestamp is monotonically increasing.
    
    Args:
        new_ts: New timestamp
        last_ts: Last timestamp
        
    Returns:
        Tuple of (valid, adjusted_timestamp)
        valid: True if timestamp is valid, False otherwise
        adjusted_timestamp: Original or adjusted timestamp
    """
    if last_ts is None:
        return True, new_ts
        
    if new_ts <= last_ts:
        # If the difference is very small, just increment by a tiny amount
        if abs(new_ts - last_ts) < 0.001:
            return True, last_ts + 0.001
        else:
            logger.warning(f"Non-monotonic timestamp: {last_ts} -> {new_ts}")
            return False, last_ts
            
    return True, new_ts


def format_time(timestamp, include_ms=True):
    """
    Format timestamp as human-readable string.
    
    Args:
        timestamp: Unix timestamp
        include_ms: Whether to include milliseconds
        
    Returns:
        Formatted time string
    """
    timestr = time.strftime("%H:%M:%S", time.localtime(timestamp))
    if include_ms:
        timestr += f".{int((timestamp % 1) * 1000):03d}"
    return timestr


def format_duration(seconds):
    """
    Format duration as MM:SS.ms.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"