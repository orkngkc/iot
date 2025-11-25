# server_pi_http.py
import time
import cv2
import requests
from ultralytics import YOLO
import numpy as np

SERVER_IP = "192.168.1.112"   # <-- server makinenin IP'si
SERVER_URL = f"http://{SERVER_IP}:80/update_count"

# Tiny / nano model 
model = YOLO("yolov8n.pt") 

PERSON_CLASS_ID = 0  # COCO person

def detect_people_local(frame_bgr: np.ndarray) -> int:
    """
    Run Tiny/Nano YOLO on the local device and return number of persons.
    """
    results = model(frame_bgr, verbose=False)[0]

    if results.boxes is None:
        return 0

    cls = results.boxes.cls.cpu().numpy().astype(int)
    return int((cls == PERSON_CLASS_ID).sum())

def main():
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera (or video).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Sending local YOLO people counts to {SERVER_URL}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            time.sleep(0.1)
            continue

        t0 = time.time()
        people_count = detect_people_local(frame)
        t1 = time.time()

        payload = {
            "count": int(people_count),
            "device_time": t1
        }

        try:
            resp = requests.post(SERVER_URL, json=payload, timeout=5)
        except Exception as e:
            print("Request failed:", e)
            time.sleep(0.5)
            continue

        if not resp.ok:
            print("Server error:", resp.status_code, resp.text)
        else:
            data = resp.json()
            print(
                f"Local count: {people_count} | "
                f"local infer: {(t1 - t0)*1000:.1f} ms | "
                f"server ack: {data.get('status')}"
            )

        
        time.sleep(0.2)

    cap.release()

if __name__ == "__main__":
    main()