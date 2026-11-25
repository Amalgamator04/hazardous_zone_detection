from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
EVIDENCE_DIR = BASE_DIR / "evidence"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "events.db"
TRACKER_CONFIG = BASE_DIR / "backend" / "trackers" / "tracktrack_reid.yaml"

for folder in (UPLOAD_DIR, EVIDENCE_DIR, DATA_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# YOLO26 nano is a practical starting point for a laptop GPU with 4 GB VRAM.
# Ultralytics downloads the official weights automatically on first startup.
MODEL_NAME = "yolo26n.pt"
IMAGE_SIZE = 640
CONFIDENCE = 0.20
MAX_DETECTIONS = 150
EVENT_COOLDOWN_SECONDS = 8
FRAME_JPEG_QUALITY = 80

# COCO class IDs: person, bicycle, car, motorcycle, bus, train, truck.
TARGET_CLASSES = [0, 1, 2, 3, 5, 6, 7]
CLASS_NAMES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    6: "Train",
    7: "Truck",
}
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 6, 7}
