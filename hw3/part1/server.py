import time
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

app = FastAPI()


model = YOLO("yolov8n.pt")
PERSON_CLASS_ID = 0 


latest_people_count = 0
latest_infer_ms = 0.0
latest_server_time = 0.0


def count_persons(frame_bgr: np.ndarray) -> int:
    
    results = model(frame_bgr, verbose=False)[0]

    if results.boxes is None:
        return 0

    cls = results.boxes.cls.cpu().numpy().astype(int)
    return int(np.sum(cls == PERSON_CLASS_ID))


@app.post("/upload_frame")
async def upload_frame(frame: UploadFile = File(...)):
   
    global latest_people_count, latest_infer_ms, latest_server_time

    
    if frame.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(status_code=400, detail="frame must be JPEG or PNG image")

    file_bytes = await frame.read()
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="could not decode image")

    t0 = time.time()
    people_count = count_persons(img)
    t1 = time.time()

    latest_people_count = people_count
    latest_infer_ms = (t1 - t0) * 1000.0
    latest_server_time = t1

    return {
        "people": people_count,
        "server_infer_ms": latest_infer_ms,
        "server_time": latest_server_time,
    }


@app.get("/people_count")
def people_count():
   
    return {
        "people": latest_people_count,
        "server_infer_ms": latest_infer_ms,
        "server_time": latest_server_time,
    }
