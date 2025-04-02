# Pupil Core Minimal

A simplified, modular implementation of Pupil Core camera recording, streaming, display, and pupil detection functionality.

## Requirements

- **Python 3.11** (required for pupil-detectors compatibility)
- OpenCV 4.x
- ZeroMQ
- Other dependencies listed in requirements.txt

## Installation

1. Create a Python 3.11 environment:
   ```bash
   conda create -n pupil-env python=3.11
   conda activate pupil-env
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

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
- Processes frames in real-time using the Pupil Labs detector (requires Python 3.11)
- Detects pupils using advanced ellipse fitting algorithms
- Provides visualization of detection results with ellipse or circle overlay
- Returns pupil location, size, orientation, and confidence
- Other display applications can connect to the pupil detector service

## Notes

- The pupil detection uses the Pupil Labs' detector library which requires Python 3.11
- Python 3.12 is currently not supported by the pupil-detectors library
- If you encounter library loading errors, ensure you're using Python 3.11