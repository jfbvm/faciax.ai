import os
import time
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
from datetime import datetime
import pytz

import subprocess
import re

conf_threshold = 0.4  # Default confidence threshold for YOLO detection

# Configure logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Create necessary folder
Path("camera_snapshots").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8x.pt")
model_seg = YOLO("yolo11x_segment.pt")

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

# Load persistent data from JSON files
def load_json(file_path, default):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return default

def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

camera_db: Dict[str, Dict] = load_json("data/camera_db.json", {})
camera_threads: Dict[str, threading.Thread] = {}
camera_stop_flags: Dict[str, threading.Event] = {}
person_counts: Dict[str, int] = load_json("data/person_counts.json", {})

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

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

"""
detect_people_segmented(frame, enable_nms=True, conf_threshold=0.5)

This function performs person detection using the YOLOv8 segmentation model. It detects human figures in the given frame, draws bounding boxes and segmentation masks for each detected person, and returns the annotated frame along with the total count of people detected.

Parameters:
- frame:
    The input image (as a NumPy array) to process.
- enable_nms (bool):
    To avoid detection of same person multiple times.
    Whether to apply Non-Maximum Suppression (NMS) manually to reduce overlapping detections.
- conf_threshold (float):
    Confidence threshold for YOLO detection.

Returns:
- frame: The output image with annotations.
- count: The number of people detected in the frame.
"""
def detect_people_segmented(frame, enable_nms=True, conf_threshold=conf_threshold):
    results = model_seg.predict(frame, conf=conf_threshold, classes=[0], task="segment")
    count = 0

    for r in results:
        boxes = r.boxes
        masks = r.masks

        if boxes is not None and masks is not None:
            selected = list(range(len(boxes)))

            if enable_nms:
                selected = []
                used = [False] * len(boxes)
                for i in range(len(boxes)):
                    if used[i]:
                        continue
                    selected.append(i)
                    for j in range(i + 1, len(boxes)):
                        if iou(boxes[i].xyxy[0].cpu().numpy(), boxes[j].xyxy[0].cpu().numpy()) > 0.5:
                            used[j] = True

            count = len(selected)

            for i in selected:
                # Caixa delimitadora
                b = boxes[i].xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = b
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # # Máscara da segmentação
                # mask = masks.data[i].cpu().numpy()
                # colored_mask = (mask * 255).astype('uint8')
                # contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Adicionar data e hora no canto inferior direito
    tz = pytz.timezone("America/Sao_Paulo")
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    height, width = frame.shape[:2]
    cv2.putText(frame, now, (width - 300, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame, count

def detect_people(frame, conf_threshold=conf_threshold):
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

def get_video_resolution(url):
    try:
        command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0', url
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            width, height = map(int, result.stdout.strip().split(','))
            return width, height
    except Exception as e:
        print(f"FFprobe error: {e}")
    return 1920, 1080  # fallback default

def read_frame_ffmpeg(url):
    width, height = get_video_resolution(url)
    command = [
        'ffmpeg',
        '-i', url,
        '-loglevel', 'quiet',
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-frames:v', '1',
        'pipe:1'
    ]
    try:
        pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_image = pipe.stdout.read(width * height * 3) # type: ignore
        pipe.stdout.close() # type: ignore
        pipe.wait()
        if raw_image:
            frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 3)).copy()
            return True, frame
    except Exception as e:
        print(f"FFmpeg error: {e}")
    return False, None

def process_camera_ffmpeg(camera_id: str, url: str, stop_event: threading.Event):
    url = url.strip()
    ret, frame = False, None

    while not stop_event.is_set():
        for _ in range(3):  # Retry reading the frame up to 3 times
            ret, frame = read_frame_ffmpeg(url)
            if ret:
                break
            time.sleep(0.5)  # small delay before retrying

        if not ret:
            print(f"Camera {camera_id}: Stream unresponsive, retrying.")
            time.sleep(2)
            continue

        frame, count = detect_people_segmented(frame)
        snapshot_path = f"camera_snapshots/{camera_id}.jpg"
        if frame is not None:
            cv2.imwrite(snapshot_path, frame)
        person_counts[camera_id] = count
        save_json("data/person_counts.json", person_counts)

        # Broadcast to all connected WebSocket clients
        # OLD CODE
        # _, buffer = cv2.imencode(".jpg", frame)
        # b64_result = base64.b64encode(buffer).decode("utf-8")
        # asyncio.run(broadcast_to_websockets({"camera_id": camera_id, "person_count": count, "image_base64": b64_result}))
        if frame is not None:
            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                b64_result = base64.b64encode(buffer).decode("utf-8")
                asyncio.run(broadcast_to_websockets({"camera_id": camera_id, "person_count": count, "image_base64": b64_result}))


        time.sleep(5)  # Add a fixed interval of 5 seconds between processed frames

def process_camera_cv2(camera_id: str, url: str, stop_event: threading.Event):
    url = url.strip()
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"Failed to open camera: {url}")
        return

    while not stop_event.is_set():
        for _ in range(3):  # Retry reading the frame up to 3 times
            cap = cv2.VideoCapture(url)
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

        frame, count = detect_people_segmented(frame)
        snapshot_path = f"camera_snapshots/{camera_id}.jpg"
        cv2.imwrite(snapshot_path, frame)
        person_counts[camera_id] = count
        save_json("data/person_counts.json", person_counts)

        # Broadcast to all connected WebSocket clients
        _, buffer = cv2.imencode(".jpg", frame)
        b64_result = base64.b64encode(buffer).decode("utf-8")
        asyncio.run(broadcast_to_websockets({"camera_id": camera_id, "person_count": count, "image_base64": b64_result}))

        cap.release()
        time.sleep(5)  # Add a fixed interval of 5 seconds between processed frames

    cap.release()

def register_camera_stream(url: str) -> Dict[str, str]:
    cam_id = str(uuid.uuid4())
    if cam_id in camera_db:
        raise HTTPException(status_code=400, detail="Camera already exists")

    camera_db[cam_id] = {"url": url}
    save_json("data/camera_db.json", camera_db)
    stop_event = threading.Event()
    camera_stop_flags[cam_id] = stop_event
    thread = threading.Thread(target=process_camera_ffmpeg, args=(cam_id, url, stop_event), daemon=True)
    camera_threads[cam_id] = thread
    thread.start()
    return {"id": cam_id, "url": url}

@app.post("/add_stream", dependencies=[Depends(verify_token_header)])
def add_stream(stream: StreamInput):
    return register_camera_stream(stream.url)

@app.delete("/delete_stream/{cam_id}", dependencies=[Depends(verify_token_header)])
def delete_stream(cam_id: str):
    if cam_id not in camera_db:
        raise HTTPException(status_code=404, detail="Camera not found")

    camera_stop_flags[cam_id].set()
    del camera_db[cam_id]
    del camera_threads[cam_id]
    del camera_stop_flags[cam_id]
    person_counts.pop(cam_id, None)
    save_json("data/camera_db.json", camera_db)
    save_json("data/person_counts.json", person_counts)
    return {"message": "Camera deleted"}

@app.get("/list_streams", dependencies=[Depends(verify_token_header)])
def list_streams():
    return camera_db

@app.get("/count_people/{cam_id}", dependencies=[Depends(verify_token_header)])
def count_people(cam_id: str):
    if cam_id not in camera_db:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"camera_id": cam_id, "person_count": person_counts.get(cam_id, 0)}

@app.get("/recognition/{cam_id}", dependencies=[Depends(verify_token_header)])
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
    frame, count = detect_people_segmented(img)
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
            frame, count = detect_people_segmented(img)
            _, buffer = cv2.imencode(".jpg", frame)
            b64_result = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_json({"person_count": count, "image_base64": b64_result})

    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        print("WebSocket disconnected")


# Reinitialize existing streams from saved camera_db
for cam_id, cam_data in camera_db.items():
    stop_event = threading.Event()
    camera_stop_flags[cam_id] = stop_event
    thread = threading.Thread(target=process_camera_ffmpeg, args=(cam_id, cam_data["url"], stop_event), daemon=True)
    camera_threads[cam_id] = thread
    thread.start()
