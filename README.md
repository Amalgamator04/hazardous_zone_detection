# AI Hazardous Zone Intrusion Detection

FastAPI + YOLO + ByteTrack + OpenCV + WebSockets + SQLite.

## Requirements
Python 3.11 recommended. Webcam for live camera mode. Internet is needed on first launch to download YOLOv8n weights.

## Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
Open http://127.0.0.1:8000

## Linux/macOS
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

## Use
1. Start server webcam or upload MP4/AVI/MOV/MKV.
2. Click Draw zone and click at least 3 points on the video.
3. Finish zone, name it and choose severity.
4. Intrusion is detected when a person's bottom-centre (foot) point enters a polygon.
5. Review event history and screenshot evidence.

Webcam mode opens camera index 0 on the machine running FastAPI, not necessarily the browser device.

## GPU
The app automatically uses CUDA if your PyTorch installation supports it; otherwise it uses CPU. For NVIDIA GPU-specific PyTorch installation, use https://pytorch.org/get-started/locally/

## API
GET /api/health, GET /api/status, POST /api/source/webcam, POST /api/source/upload, POST /api/stop, GET/POST /api/zones, GET /api/events, WS /ws/stream.

Zone points are normalized 0..1 relative to the video frame.

## Safety
Prototype only. Do not replace certified safety interlocks, emergency stops, physical barriers, trained personnel or required industrial safety controls.


## Multi-class detection and improved tracking (YOLO26)

This version uses **YOLO26n** for object detection and **TrackTrack with appearance ReID** for multi-object tracking. The nano model is selected as a practical starting point for laptops with 4 GB VRAM. The first run downloads `yolo26n.pt` automatically.

Detected COCO classes:
- Person
- Bicycle
- Car
- Motorcycle
- Bus
- Train
- Truck

Every tracked object receives a class label and track ID. Three lower-edge points are evaluated for both people and vehicles: left/right point entry gives a blinking yellow CAUTION; centre point entry gives a red ALERT and records an event. The event database migrates older tables automatically to add object type and class name.

### Install/update dependencies

Use Python 3.13 in a virtual environment, then run:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`ultralytics` must be version 8.4.63 or newer for the configured TrackTrack tracker and ReID support. If you already have an older environment, run `pip install --upgrade "ultralytics>=8.4.63,<9.0.0"`.

### Tracking configuration

The tracker settings are in `backend/trackers/tracktrack_reid.yaml`. `track_buffer`, thresholds, and ReID settings can be tuned for the camera and scene. ReID improves identity association but adds compute cost. Performance and ID stability depend on camera angle, resolution, lighting, object size, and occlusion; test with representative CCTV footage before operational use.

The system is an AI monitoring prototype, not a certified safety interlock.
