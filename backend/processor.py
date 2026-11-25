import cv2
import time
import threading
from datetime import datetime

from ultralytics import YOLO

from backend.config import (
    MODEL_NAME, IMAGE_SIZE, CONFIDENCE, MAX_DETECTIONS,
    EVENT_COOLDOWN_SECONDS, FRAME_JPEG_QUALITY, EVIDENCE_DIR,
    TRACKER_CONFIG, TARGET_CLASSES, CLASS_NAMES, VEHICLE_CLASS_IDS,
)
from backend.database import add_event
from backend.zone_manager import point_inside_zone, draw_zones


class VideoProcessor:
    """YOLO26 multi-class detection + TrackTrack/ReID + three-point zone checks."""
    GREEN = (0, 220, 0)
    YELLOW = (0, 255, 255)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)

    def __init__(self):
        self.lock = threading.RLock()
        self.capture = None
        self.thread = None
        self.running = False
        self.source_name = None
        self.latest_jpeg = None
        self.zones = []
        self.model = None
        self.last_event_time = {}
        self.latest_events = []
        self.event_sequence = 0
        self.last_error = None
        self.frame_size = None

    def _load_model(self):
        if self.model is None:
            self.model = YOLO(MODEL_NAME)
        return self.model

    def start(self, source, source_name):
        self.stop()
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Could not open selected video source.")
        with self.lock:
            self.capture, self.source_name, self.running = cap, source_name, True
            self.last_error, self.latest_jpeg = None, None
            self.last_event_time.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False
            cap, self.capture = self.capture, None
        if cap is not None:
            cap.release()
        thread = self.thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        with self.lock:
            self.thread, self.source_name = None, None

    def set_zones(self, zones):
        with self.lock:
            self.zones = zones

    def get_zones(self):
        with self.lock:
            return list(self.zones)

    def status(self):
        with self.lock:
            return {
                "running": self.running,
                "source": self.source_name,
                "frame_ready": self.latest_jpeg is not None,
                "error": self.last_error,
                "frame_size": self.frame_size,
                "model": MODEL_NAME,
                "tracker": "TrackTrack + ReID",
                "classes": [CLASS_NAMES[c] for c in TARGET_CLASSES],
            }

    def snapshot(self):
        with self.lock:
            return self.latest_jpeg, self.event_sequence, list(self.latest_events)

    @staticmethod
    def _zone_for_point(point, zones, width, height):
        for zone in zones:
            if point_inside_zone(point, zone, width, height):
                return zone
        return None

    def _record_intrusion(self, track_id, zone, frame, object_type, class_name):
        now = time.time()
        key = (int(track_id), zone["id"], object_type)
        if now - self.last_event_time.get(key, 0) < EVENT_COOLDOWN_SECONDS:
            return
        self.last_event_time[key] = now
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        filename = f"intrusion_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(str(EVIDENCE_DIR / filename), frame)
        event_id = add_event(timestamp, int(track_id), zone, filename, object_type, class_name)
        event = {
            "id": event_id, "timestamp": timestamp, "track_id": int(track_id),
            "zone_id": zone["id"], "zone_name": zone["name"],
            "severity": zone["severity"], "screenshot": filename,
            "object_type": object_type, "class_name": class_name,
        }
        with self.lock:
            self.latest_events = (self.latest_events + [event])[-20:]
            self.event_sequence += 1

    def _run(self):
        try:
            model = self._load_model()
            while True:
                with self.lock:
                    if not self.running or self.capture is None:
                        break
                    cap, zones, source_name = self.capture, list(self.zones), self.source_name

                ok, frame = cap.read()
                if not ok:
                    if source_name == "uploaded video":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.03)
                        continue
                    break

                height, width = frame.shape[:2]
                with self.lock:
                    self.frame_size = {"width": width, "height": height}

                # All selected COCO categories are detected and tracked together.
                results = model.track(
                    frame,
                    persist=True,
                    tracker=str(TRACKER_CONFIG),
                    classes=TARGET_CLASSES,
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    max_det=MAX_DETECTIONS,
                    iou=0.50,
                    verbose=False,
                )

                annotated = frame.copy()
                draw_zones(annotated, zones)
                confirmed_intrusions = []
                counts = {"person": 0, "vehicle": 0}
                alert_counts = {"person": 0, "vehicle": 0}
                caution_counts = {"person": 0, "vehicle": 0}
                blink_on = (int(time.time() * 3) % 2) == 0

                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    class_ids = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [0] * len(xyxy)
                    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)

                    for box, class_id, track_id in zip(xyxy, class_ids, track_ids):
                        class_name = CLASS_NAMES.get(int(class_id))
                        if class_name is None:
                            continue
                        object_type = "person" if int(class_id) == 0 else "vehicle"
                        counts[object_type] += 1

                        x1, y1, x2, y2 = map(int, box)
                        box_width = max(1, x2 - x1)
                        # Ground-contact samples along the lower edge of each bbox.
                        left_point = (int(x1 + box_width * 0.15), y2)
                        centre_point = (int((x1 + x2) / 2), y2)
                        right_point = (int(x2 - box_width * 0.15), y2)

                        left_zone = self._zone_for_point(left_point, zones, width, height)
                        centre_zone = self._zone_for_point(centre_point, zones, width, height)
                        right_zone = self._zone_for_point(right_point, zones, width, height)

                        if centre_zone is not None:
                            state, state_zone, box_color = "ALERT", centre_zone, self.RED
                            alert_counts[object_type] += 1
                            if track_id is not None:
                                confirmed_intrusions.append((track_id, centre_zone, object_type, class_name))
                        elif left_zone is not None or right_zone is not None:
                            state, state_zone, box_color = "CAUTION", left_zone or right_zone, self.YELLOW
                            caution_counts[object_type] += 1
                        else:
                            state, state_zone, box_color = "SAFE", None, self.GREEN

                        display_id = track_id if track_id is not None else "pending"
                        label = f"{class_name} ID:{display_id} | {state}"
                        if state_zone is not None:
                            label += f" - {state_zone['name']}"
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(annotated, label, (x1, max(22, y1 - 9)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color, 2)

                        for point, point_color, short_label in (
                            (left_point, self.YELLOW if left_zone else self.GREEN, "L"),
                            (centre_point, self.RED if centre_zone else self.GREEN, "C"),
                            (right_point, self.YELLOW if right_zone else self.GREEN, "R"),
                        ):
                            cv2.circle(annotated, point, 6, point_color, -1)
                            cv2.circle(annotated, point, 8, self.WHITE, 1)
                            cv2.putText(annotated, short_label, (point[0] - 4, point[1] - 11),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, point_color, 1)

                for track_id, zone, object_type, class_name in confirmed_intrusions:
                    self._record_intrusion(track_id, zone, annotated, object_type, class_name)

                total_alerts = sum(alert_counts.values())
                total_cautions = sum(caution_counts.values())
                if total_alerts:
                    cv2.rectangle(annotated, (12, 12), (width - 12, 58), (0, 0, 150), -1)
                    msg = f"HAZARD ALERT | People: {alert_counts['person']} | Vehicles: {alert_counts['vehicle']}"
                    cv2.putText(annotated, msg, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, self.WHITE, 2)
                elif total_cautions and blink_on:
                    cv2.rectangle(annotated, (12, 12), (width - 12, 58), (0, 170, 220), -1)
                    msg = f"CAUTION | People: {caution_counts['person']} | Vehicles: {caution_counts['vehicle']}"
                    cv2.putText(annotated, msg, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)

                cv2.putText(annotated,
                            f"People: {counts['person']} | Vehicles: {counts['vehicle']} | YOLO26n + TrackTrack/ReID",
                            (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.WHITE, 2)

                ok, encoded = cv2.imencode(".jpg", annotated,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_JPEG_QUALITY])
                if ok:
                    with self.lock:
                        self.latest_jpeg = encoded.tobytes()
                time.sleep(0.001)

        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
        finally:
            with self.lock:
                cap, self.capture = self.capture, None
                self.running, self.source_name = False, None
            if cap is not None:
                cap.release()


processor = VideoProcessor()
