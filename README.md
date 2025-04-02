# Pupil Core Minimal

A simplified, modular implementation of Pupil Core camera recording, streaming, display, and pupil detection functionality.

## Architecture Overview

The codebase follows a modular, layered architecture:

```
pupil/
├── hardware/      # Camera hardware interfaces
├── io/            # File reading/writing utilities
├── service/       # Network services and messaging
├── ui/            # User interfaces (terminal, display)
├── detection/     # Pupil and eye feature detection
└── utils/         # Shared utilities
```

## Components

### Core Components

- **Camera System**: Abstract base camera interface with UVC implementation
- **Recorder**: Video recording with multiple encoder options (JPEG, MPEG4, H264)
- **Camera Service**: ZeroMQ-based server for remote camera control
- **Remote Controller**: Terminal client for controlling recording
- **Video Display**: Optimized multi-camera video display
- **Pupil Detector**: Client for real-time pupil detection from eye cameras

## Running the Applications

### Camera Service

Start the camera service to detect cameras and provide access to them:

```bash
python start_service.py [--host HOST] [--port PORT] [--recording-dir DIR]
```

### Remote Controller

Control recording and camera settings from the terminal:

```bash
python remote_control.py [--host HOST] [--port PORT]
```

### Video Display

Display video streams from the cameras:

```bash
python display_video.py [--host HOST] [--port PORT] [--fps FPS]
```

### Pupil Detection

Run real-time pupil detection using the eye camera streams:

```bash
python detect_pupils.py [--host HOST] [--port PORT] [--visualize] [--debug]
```

Options:
- `--visualize`: Show visualization windows with pupil detection
- `--debug`: Enable debug logging for more detailed information

## Design Choices

1. **Abstraction**: Base classes for cameras and video writers
2. **Separation of Concerns**: Clean separation between hardware, I/O, and UI
3. **Thread Safety**: Proper locking and thread management
4. **Client-Server Architecture**: ZeroMQ for network communication
5. **Extensibility**: Easy to add new camera types, encoders, or detection algorithms

## Pupil Detection

The pupil detection system:
- Connects to the camera service as a client
- Subscribes to eye camera streams
- Processes frames in real-time using OpenCV
- Detects pupils using contour detection and circular shape analysis
- Provides visualization of detection results
- Returns pupil location, size, and confidence