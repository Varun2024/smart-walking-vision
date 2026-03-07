# Smart Walking Stick Script

Real-time assistive vision pipeline for a smart walking stick, built around YOLO detection, voice alerts, obstacle awareness, and optional fall-risk detection.

## What This Project Does

- Detects objects from webcam, stream, video file, or ESP32-CAM snapshot URL.
- Supports multiple detector backends:
  - `yolov3` (OpenCV DNN)
  - `yolov8onnx` (ONNX Runtime)
  - `yoloworld` (open-vocabulary detection)
  - `hybrid` (fast YOLOv8 ONNX + periodic YOLO-World scans)
- Uses different bounding-box colors per class.
- Announces detections with natural `gTTS` speech.
- Avoids repeated announcements unless new objects enter the scene.
- Shows per-class counts and smooths count flicker over time.
- Estimates proximity from bounding-box size (`FAR` to `VERY CLOSE`) and displays urgency.
- Optional pose-based fall-risk detection.

## Project Structure

```text
.
|-- improved_obj_detect.py          # Entry point
|-- requirements.txt
|-- coco.names
|-- yolov3.cfg
|-- yolov3.weights
|-- yolov8n.pt                     # Export source model (generated)
|-- yolov8n.onnx                   # ONNX model (generated)
|-- detector/
|   |-- __init__.py
|   |-- runtime/                   # Orchestration + config
|   |   |-- runner.py
|   |   |-- config.py
|   |   |-- runtime_settings.py
|   |-- backends/                  # Detection backends + merge helpers
|   |   |-- yolo.py
|   |   |-- yolo_onnx.py
|   |   |-- yolo_world.py
|   |   |-- merge_utils.py
|   |-- io/                        # Input/output interfaces
|   |   |-- sources.py
|   |   |-- speech.py
|   |-- safety/                    # Temporal safety and fall logic
|   |   |-- temporal.py
|   |   |-- pose.py
|-- tools/
|   |-- quantize_onnx.py           # INT8 ONNX quantization helper
|-- espcam_code/                   # Arduino ESP32-CAM firmware
|-- dist_node_server/              # Arduino distance node firmware
|-- ip_nodemcu/                    # Arduino NodeMCU network firmware
```

## Installation

1. Create and activate a Python virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Runtime Config File

Hardcoded runtime defaults are now moved to JSON config:

- `config/runtime_defaults.json`

You can edit this file to tune defaults such as:

- model backend and model paths
- quality presets
- world classes
- speech, persistence, and fall-detection defaults

Use a custom config file:

```powershell
python .\improved_obj_detect.py --config .\config\runtime_defaults.json --source 0
```

## Quick Start

Run with default backend (`yolov8onnx`) on webcam:

```powershell
python .\improved_obj_detect.py --source 0
```

Run hybrid mode (recommended for broader obstacle coverage):

```powershell
python .\improved_obj_detect.py --source 0 --model-type hybrid
```

Run YOLO-World only:

```powershell
python .\improved_obj_detect.py --source 0 --model-type yoloworld
```

Press `q` to close the window.

## Input Sources

- Webcam: `--source 0` (or `1`, `2`, ...)
- Video/stream path: `--source path_or_url`
- ESP32-CAM snapshot mode: `--source espcam`
- Direct snapshot URL: `--source http://<ip>/cam-hi.jpg`

## Detection Modes

- `yolov3`: legacy baseline.
- `yolov8onnx`: better speed/accuracy on CPU.
- `yoloworld`: open-vocabulary classes (detects broader concepts such as watch and road obstacles).
- `hybrid`: fast ONNX every inference cycle + periodic YOLO-World scans for rare/long-tail objects.

## Performance Features

- `infer_every`: run detector every N frames and predict positions between frames.
- Async capture thread: grabs frames while inference runs.
- Quality presets: `fast`, `balanced`, `accurate`.
- Optional INT8 model preference for ONNX backend.

Example low-latency hybrid command:

```powershell
python .\improved_obj_detect.py --source 0 --model-type hybrid --quality fast --infer-every 2 --hybrid-world-every 8
```

## Reliability Features

- Detection persistence smoothing to reduce count flicker.
- New-object-only voice announcements.
- Proximity estimation and urgency overlay from box size.
- Optional fall-risk detection with pose model.

Example with fall detection enabled:

```powershell
python .\improved_obj_detect.py --source 0 --model-type hybrid --enable-fall-detection --pose-every 10
```

## ONNX Export and INT8 Quantization

Export YOLOv8 to ONNX:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)"
```

Quantize ONNX model to INT8:

```powershell
python .\tools\quantize_onnx.py --input .\yolov8n.onnx --output .\yolov8n.int8.onnx
```

Use INT8 model at runtime:

```powershell
python .\improved_obj_detect.py --source 0 --model-type yolov8onnx --prefer-int8 --onnx-int8-path .\yolov8n.int8.onnx
```

## Useful CLI Options

- `--model-type yolov3|yolov8onnx|yoloworld|hybrid`
- `--quality fast|balanced|accurate`
- `--infer-every N`
- `--hybrid-world-every N`
- `--world-classes "person,watch,cell phone,pothole,traffic cone"`
- `--persistence-window N`
- `--no-async-capture`
- `--disable-distance-estimation`
- `--enable-fall-detection`
- `--pose-model yolov8n-pose.pt`
- `--no-speech`
- `--speech-cooldown`, `--speech-gap`

## Recommended Presets

| Scenario | Goal | Command |
| --- | --- | --- |
| Indoor navigation | Balanced accuracy + smooth speech | `python .\improved_obj_detect.py --source 0 --model-type hybrid --quality balanced --infer-every 2 --hybrid-world-every 8 --persistence-window 5` |
| Outdoor road obstacles | Broad obstacle coverage | `python .\improved_obj_detect.py --source 0 --model-type hybrid --quality balanced --infer-every 2 --hybrid-world-every 6 --world-classes "person,car,bicycle,motorcycle,bus,truck,curb,pothole,speed bump,traffic cone,barrier,construction sign"` |
| Low-light scene | More sensitive detections | `python .\improved_obj_detect.py --source 0 --model-type hybrid --quality accurate --confidence 0.35 --nms-threshold 0.45 --infer-every 2 --hybrid-world-every 8` |
| Max FPS on CPU | Lowest latency | `python .\improved_obj_detect.py --source 0 --model-type yolov8onnx --quality fast --infer-every 3 --prefer-int8 --persistence-window 4` |
| Fall monitoring mode | Add fall-risk alerts | `python .\improved_obj_detect.py --source 0 --model-type hybrid --enable-fall-detection --pose-every 10 --infer-every 2` |

Tip: if your camera feed flickers between detected and not-detected objects, increase `--persistence-window` to `7` or `9`.

## Arduino Components

This repository also contains microcontroller code for hardware-side integration:

- `espcam_code/`: ESP32-CAM image endpoint firmware.
- `dist_node_server/`: distance node firmware.
- `ip_nodemcu/`: NodeMCU network configuration firmware.

The Python vision pipeline can be tested independently from Arduino code.

## Troubleshooting

- Webcam not opening on Windows:
  - Close apps using camera (Teams, Zoom, browser).
  - Try another index: `--source 1`.
- ONNX input size mismatch:
  - Re-export model and use current `yolov8n.onnx`.
- Low FPS:
  - Use `--quality fast --infer-every 2`.
  - Use `--prefer-int8` with quantized model.
- Too many repeated alerts:
  - Increase `--speech-gap` and `--speech-cooldown`.

## Current Stage

This is an actively evolving prototype with modularized detection backends and assistive safety features intended for iterative experimentation.
