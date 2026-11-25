import asyncio, shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from backend.config import UPLOAD_DIR, EVIDENCE_DIR, BASE_DIR
from backend.database import initialize_database, get_events
from backend.processor import processor
from backend.schemas import WebcamRequest, ZonesRequest

app = FastAPI(title="Hazardous Zone Intrusion Detection",version="1.0.0")
app.mount("/static",StaticFiles(directory=BASE_DIR/"frontend"),name="static")
app.mount("/evidence",StaticFiles(directory=EVIDENCE_DIR),name="evidence")

@app.on_event("startup")
def startup(): initialize_database()

@app.get("/")
def home(): return FileResponse(BASE_DIR/"frontend"/"index.html")

@app.get("/api/health")
def health(): return {"ok":True,"service":"hazardous-zone-detection"}

@app.get("/api/status")
def status(): return processor.status()

@app.post("/api/source/webcam")
def start_webcam(request:WebcamRequest):
    try:
        processor.start(request.camera_index,"server webcam")
        return {"ok":True,"message":"Webcam started"}
    except Exception as exc: raise HTTPException(status_code=400,detail=str(exc))

@app.post("/api/source/upload")
async def upload_video(file:UploadFile=File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4",".avi",".mov",".mkv",".mpeg",".mpg"}:
        raise HTTPException(status_code=400,detail="Upload MP4, AVI, MOV, MKV or MPEG video.")
    destination = UPLOAD_DIR/f"video_input{suffix}"
    with destination.open("wb") as buffer: shutil.copyfileobj(file.file,buffer)
    try:
        processor.start(str(destination),"uploaded video")
        return {"ok":True,"message":"Uploaded video started"}
    except Exception as exc: raise HTTPException(status_code=400,detail=str(exc))

@app.post("/api/stop")
def stop_video():
    processor.stop()
    return {"ok":True,"message":"Video stopped"}

@app.get("/api/zones")
def zones_get(): return {"zones":processor.get_zones()}

@app.post("/api/zones")
def zones_save(request:ZonesRequest):
    for zone in request.zones:
        if len(zone.points)<3: raise HTTPException(status_code=400,detail=f"Zone {zone.name} needs at least 3 points.")
        for point in zone.points:
            if len(point)!=2 or not all(0<=v<=1 for v in point):
                raise HTTPException(status_code=400,detail="Zone coordinates must be between 0 and 1.")
    processor.set_zones([z.model_dump() for z in request.zones])
    return {"ok":True,"zones":processor.get_zones()}

@app.get("/api/events")
def events(limit:int=100): return {"events":get_events(max(1,min(limit,500)))}

@app.websocket("/ws/stream")
async def stream(websocket:WebSocket):
    await websocket.accept()
    last_sequence=-1
    try:
        while True:
            jpeg,sequence,recent_events=processor.snapshot()
            if sequence!=last_sequence:
                await websocket.send_json({"type":"events","events":recent_events})
                last_sequence=sequence
            if jpeg: await websocket.send_bytes(jpeg)
            await asyncio.sleep(0.06)
    except (WebSocketDisconnect,RuntimeError): pass
