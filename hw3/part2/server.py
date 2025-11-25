# server_part2.py
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class CountUpdate(BaseModel):
    count: int
    device_time: float | None = None  # optional, istersen cihaz saatini de yollarsın

# global state
latest_people_count: int = 0
latest_update_ts: float = 0.0
latest_device_time: float | None = None

@app.post("/update_count")
async def update_count(payload: CountUpdate):
    
    
    global latest_people_count, latest_update_ts, latest_device_time

    if payload.count < 0:
        raise HTTPException(status_code=400, detail="count must be >= 0")

    latest_people_count = int(payload.count)
    latest_update_ts = time.time()
    latest_device_time = payload.device_time

    return {
        "status": "ok",
        "stored_count": latest_people_count,
        "server_update_ts": latest_update_ts,
    }

@app.get("/people_count")
def people_count():
    
    return {
        "people": latest_people_count,
        "last_update_ts": latest_update_ts,
        "last_device_time": latest_device_time,
    }
