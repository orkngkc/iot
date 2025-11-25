import time
import cv2
import requests

SERVER_IP = "192.168.1.112"        
SERVER_URL = f"http://{SERVER_IP}:80/upload_frame" 

def main():
   
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        raise RuntimeError("Camera not found. Check /dev/video0 or VideoCapture source.")

    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Streaming frames to {SERVER_URL}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            time.sleep(0.1)
            continue

        
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            print("JPEG encode failed, skipping frame.")
            continue

        files = {
            "frame": ("frame.jpg", jpg.tobytes(), "image/jpeg")
        }

        t0 = time.time()
        try:
            resp = requests.post(SERVER_URL, files=files, timeout=5)
        except Exception as e:
            print("Request failed:", e)
            time.sleep(0.5)
            continue

        if not resp.ok:
            print("Server error:", resp.status_code, resp.text)
            time.sleep(0.5)
            continue

        data = resp.json()
        people = data.get("people")
        infer_ms = data.get("server_infer_ms")

        t1 = time.time()
        rtt_ms = (t1 - t0) * 1000.0

        
        if infer_ms is not None:
            print(f"People: {people} | server infer: {infer_ms:.1f} ms | RTT: {rtt_ms:.1f} ms")
        else:
            print(f"People: {people} | RTT: {rtt_ms:.1f} ms")

       
        time.sleep(0.05)

    cap.release()

if __name__ == "__main__":
    main()
