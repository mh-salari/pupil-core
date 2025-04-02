"""
Message Types
------------
Shared message type definitions for ZeroMQ communication between
camera service and clients.
"""


class MessageType:
    """Message types for ZeroMQ communication."""
    COMMAND = 0        # Command message (request/response)
    FRAME_REQUEST = 1  # Request for a frame
    FRAME_RESPONSE = 2 # Frame data response
    STATUS_UPDATE = 3  # Status update broadcast
    ERROR = 4          # Error message