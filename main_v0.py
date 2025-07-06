import os
import uuid
import cv2
import base64
import threading
import logging
from typing import Dict, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import json
import asyncio

# Configure logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Create necessary folder
Path("camera_snapshots").mkdir(exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8x.pt")

# FastAPI instance
app = FastAPI()

# Allow CORS for localhost and others
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# In-memory database for camera configs
camera_db: Dict[str, Dict] = {}
camera_threads: Dict[str, threading.Thread] = {}
camera_stop_flags: Dict[str, threading.Event] = {}
person_counts: Dict[str, int] = {}
active_websockets: List[WebSocket] = []

# Load users from local JSON file
with open("users.json", "r") as f:
    users_list = json.load(f)
VALID_USERS = {user["username"]: user["password"] for user in users_list}
TOKENS = {}

class AuthRequest(BaseModel):
    username: str
    password: str

def verify_token_header(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid or missing token format")
    token = authorization.replace("Bearer ", "")
    if token not in TOKENS.values():
        raise HTTPException(status_code=403, detail="Invalid or missing token")

@app.post("/login")
def login(auth: AuthRequest):
    if VALID_USERS.get(auth.username) == auth.password:
        token = str(uuid.uuid4())
        TOKENS[auth.username] = token
        return {"token": token}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/")
def root():
    return {"status": "Camera Processing API running"}

# Model for adding streams
class StreamInput(BaseModel):
    url: str

def detect_people(frame, conf_threshold=0.5):
    results = model(frame)
    count = 0
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0 and box.conf[0] >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                count += 1
    return frame, count

async def broadcast_to_websockets(payload):
    for ws in active_websockets[:]:
        try:
            await ws.send_json(payload)
        except Exception:
            active_websockets.remove(ws)

def process_camera(camera_id: str, url: str, stop_event: threading.Event):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Failed to open camera: {url}")
        return

    while not stop_event.is_set():
        for _ in range(3):  # Retry reading the frame up to 3 times
            ret, frame = cap.read()
            if ret:
                break
            time.sleep(0.5)  # small delay before retrying

        if not ret:
            print(f"Camera {camera_id}: Stream unresponsive, attempting to reopen.")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                print(f"Camera {camera_id}: Failed to reopen stream.")
                continue
            print(f"Camera {camera_id}: Stream reopened.")
            continue

        frame, count = detect_people(frame)
        snapshot_path = f"camera_snapshots/{camera_id}.jpg"
        cv2.imwrite(snapshot_path, frame)
        person_counts[camera_id] = count

        # Broadcast to all connected WebSocket clients
        _, buffer = cv2.imencode(".jpg", frame)
        b64_result = base64.b64encode(buffer).decode("utf-8")
        asyncio.run(broadcast_to_websockets({"camera_id": camera_id, "person_count": count, "image_base64": b64_result}))

    cap.release()

@app.post("/add_stream", dependencies=[Depends(verify_token_header)])
def add_stream(stream: StreamInput):
    cam_id = str(uuid.uuid4())
    if cam_id in camera_db:
        raise HTTPException(status_code=400, detail="Camera already exists")

    camera_db[cam_id] = {"url": stream.url}
    stop_event = threading.Event()
    camera_stop_flags[cam_id] = stop_event
    thread = threading.Thread(target=process_camera, args=(cam_id, stream.url, stop_event), daemon=True)
    camera_threads[cam_id] = thread
    thread.start()
    return {"id": cam_id, "url": stream.url}

@app.delete("/delete_stream/{cam_id}", dependencies=[Depends(verify_token_header)])
def delete_stream(cam_id: str):
    if cam_id not in camera_db:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_stop_flags[cam_id].set()
    del camera_db[cam_id]
    del camera_threads[cam_id]
    del camera_stop_flags[cam_id]
    person_counts.pop(cam_id, None)
    return {"message": "Camera deleted"}

@app.get("/list_streams", dependencies=[Depends(verify_token_header)])
def list_streams():
    return camera_db

@app.get("/count_people/{cam_id}", dependencies=[Depends(verify_token_header)])
def count_people(cam_id: str):
    if cam_id not in camera_db:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"camera_id": cam_id, "person_count": person_counts.get(cam_id, 0)}

@app.get("/snapshot/{cam_id}", dependencies=[Depends(verify_token_header)])
def get_snapshot(cam_id: str, return_type: str = "base64"):
    path = f"camera_snapshots/{cam_id}.jpg"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Snapshot not found")

    if return_type == "image":
        return FileResponse(path, media_type="image/jpeg")

    with open(path, "rb") as img_file:
        img_bytes = img_file.read()
        return {"image_base64": base64.b64encode(img_bytes).decode("utf-8")}

@app.post("/detect_base64", dependencies=[Depends(verify_token_header)])
def detect_base64_image(file: UploadFile = File(None), base64_str: str = Form(None), compressed_base64: str = Form(None), return_type: str = "base64"):
    if file:
        image_data = file.file.read()
    elif base64_str:
        image_data = base64.b64decode(base64_str)
    elif compressed_base64:
        image_data = base64.b64decode(compressed_base64)
    else:
        raise HTTPException(status_code=400, detail="No image file or base64 string provided")

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    frame, count = detect_people(img)
    _, buffer = cv2.imencode(".jpg", frame)

    if return_type == "image":
        temp_path = "camera_snapshots/temp_detect.jpg"
        with open(temp_path, "wb") as f:
            f.write(buffer)
        return FileResponse(temp_path, media_type="image/jpeg")

    if return_type == "count":
        return {"person_count": count}

    b64_result = base64.b64encode(buffer).decode("utf-8")
    return {"person_count": count, "image_base64": b64_result}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        auth = await websocket.receive_json()
        token = auth.get("token")
        if token not in TOKENS.values():
            await websocket.send_json({"error": "Unauthorized"})
            await websocket.close()
            return

        active_websockets.append(websocket)
        await websocket.send_json({"status": "connected"})

        while True:
            data = await websocket.receive_json()
            image_data = None
            if 'base64_str' in data:
                image_data = base64.b64decode(data['base64_str'])
            elif 'compressed_base64' in data:
                image_data = base64.b64decode(data['compressed_base64'])
            else:
                await websocket.send_json({"error": "No valid image data provided"})
                continue

            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame, count = detect_people(img)
            _, buffer = cv2.imencode(".jpg", frame)
            b64_result = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_json({"person_count": count, "image_base64": b64_result})

    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        print("WebSocket disconnected")
