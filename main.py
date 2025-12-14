import asyncio
import os
import threading
import time
from typing import Dict

import uvicorn
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel

from src.detect_scripyts.sub_server import run_detect_script, run_detect_script_new
from src.domain.ActionState import action_state
from src.domain.StreamState import StreamState
from src.mq.MQTTPublisher import create_default_publisher, send_move_from_quest2Tony
from src.voice_process.voice_process import transcribe_audio_bytes, transcribe_audio_bytes_new


# Local modules


# ==============================
# Models
# ==============================
class Cmd(BaseModel):
    """Command data model for Unity → Python control commands."""
    cmd: str


app = FastAPI(title="Video Push/Stream Relay")

# ==============================
# MQTT publisher init
# ==============================
try:
    # Connect to the MQTT broker that controls the robot
    publisher = create_default_publisher(brokers=["192.168.0.101"], topic="tony_one/cmd")
except Exception as e:
    print(e)


BOUNDARY = 'frameboundary'  # Boundary token for MJPEG streaming


# ==============================
# CORS settings
# (Allow Quest3 / LAN devices to call the API)
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use a whitelist in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# Manage multiple video streams
# ==============================
streams: Dict[str, StreamState] = {}


def get_stream(stream_id: str) -> StreamState:
    """Retrieve or create a StreamState object for every camera stream."""
    if stream_id not in streams:
        streams[stream_id] = StreamState()
    return streams[stream_id]


# ==============================
# Background detection thread
# ==============================
def sub_server():
    """
    Background thread that receives frames from `usb_cam`
    and runs your object-detection script.
    """

    cam_state = get_stream('tony_cam')
    # run_detect_script(cam_state,is_yolo=True)
    run_detect_script_new(cam_state,is_yolo=True)


# ==============================
# Root endpoint
# ==============================
@app.get("/", response_class=PlainTextResponse)
async def root():
    """Basic health check for the server."""
    return (
        "Video Relay is running.\n"
        "POST /push/{stream_id}  (body = JPEG bytes)\n"
        "GET  /stream/{stream_id} (MJPEG)\n"
        "WS   /ws/{stream_id} (binary JPEG frames)\n"
    )


# ==============================
# PUSH endpoint
# Robot → PC
# ==============================
@app.post("/push/{stream_id}")
async def push_frame(stream_id: str, request: Request):
    """
    Robots or Raspberry Pi send raw JPEG bytes to the server.
    The server stores the latest frame for subscribers.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    st = get_stream(stream_id)
    st.latest_frame = body
    st.latest_ts = time.time()

    # Notify subscribers waiting for new frames
    st.frame_event.set()
    st.frame_event = asyncio.Event()

    return {"ok": True, "stream_id": stream_id, "ts": st.latest_ts}




# ==============================
# Command endpoint
# Unity → Python → MQTT → Robot
# ==============================
@app.post("/cmd")
async def receive_cmd(command: Cmd):
    start_time = time.perf_counter()
    """
    Receive button commands from Unity.
    Forward them to the robot through MQTT.
    """
    print(f"Received Unity command: {command}")
    str_cmd = command.cmd

    # MQTT publish command to Tony robot
    send_move_from_quest2Tony(publisher, str_cmd)
    exec_time = (time.perf_counter() - start_time) * 1000  #  ms
    print(f"[receive_cmd] Execution time: {exec_time:.2f} ms")
    return {"status": "ok", "received": str_cmd}


# ==============================
# Audio ASR endpoint
# Quest3/Unity → PC → Whisper
# ==============================
# audio_path = r'D:\programs\python_projects\quest_robots\audio'
# SAVE_PATH = os.path.join(audio_path, "record.wav")


@app.post("/asr")
async def asr_audio(file: UploadFile = File(...)):
    """
    Receive uploaded WAV audio file from Unity and run Whisper ASR.
    Returns the recognized text.
    """
    print(f"Received audio file: {file.filename}, type: {file.content_type}")
    start_time = time.perf_counter()
    data = await file.read()

    # Save audio to disk
    # with open(SAVE_PATH, "wb") as f:
    #     f.write(data)
    #
    # print(f"Saved audio to {SAVE_PATH}, size = {len(data)} bytes")

    # Run Whisper / ASR model
    # result_text = transcribe_audio_bytes(data) # dont use vad
    result_text = transcribe_audio_bytes_new(data) #use vad
    print("ASR detected text: " + result_text)
    exec_time = (time.perf_counter() - start_time) * 1000  # ms
    print(f"[voice process] Execution time: {exec_time:.2f} ms")
    return {
        "status": "ok",
        "msg": "receive success",
        "text": result_text,
    }
@app.get("/detect/{name}")
async def set_detect(name: str):
    d_name = name

    if d_name =='start':
        action_state.set_start_detect(True)
    elif d_name =='stop':
        action_state.set_start_detect(False)
    else:
        action_state.set_detect_class([d_name])
    return {
        "status": "ok",
        "msg": "receive success",

    }

# ==============================
# MJPEG streaming endpoint
# PC → Unity / Web browser
# ==============================
@app.get("/stream/{stream_id}")
async def mjpeg_stream(stream_id: str):
    """
    Clients (Unity/Web) request this URL to receive MJPEG streaming.
    Each frame is sent as a multipart/x-mixed-replace response.
    """
    st = get_stream(stream_id)

    # ----------------------
    # FPS 统计变量
    # ----------------------
    fps_counter = 0
    fps_last_time = time.time()

    async def frame_generator():
        nonlocal fps_counter, fps_last_time

        heartbeat_interval = 10.0
        last_sent = time.time()

        while True:
            try:
                # Wait for new frame or timeout
                try:
                    await asyncio.wait_for(st.frame_event.wait(), timeout=heartbeat_interval)
                except asyncio.TimeoutError:
                    pass


                if st.latest_frame:

                    yield (
                        f"--{BOUNDARY}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(st.latest_frame)}\r\n\r\n"
                    ).encode("ascii") + st.latest_frame + b"\r\n"

                    last_sent = time.time()

                    # -------------  FPS -------------
                    fps_counter += 1
                    now = time.time()
                    if now - fps_last_time >= 1.0:
                        # print(f"[STREAM:{stream_id}] Push FPS = {fps_counter:.2f}")
                        fps_counter = 0
                        fps_last_time = now
                    # -----------------------------------


                else:
                    if time.time() - last_sent >= heartbeat_interval:
                        yield (
                            f"--{BOUNDARY}\r\n"
                            "Content-Type: text/plain\r\n\r\n"
                            "heartbeat\r\n"
                        ).encode("utf-8")
                        last_sent = time.time()

            except asyncio.CancelledError:
                break

    return StreamingResponse(
        frame_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
    )


# ==============================
# App entry point
# ==============================
if __name__ == "__main__":
    # Start background detection thread   can comment out this line if it's not needed.”
    t = threading.Thread(target=sub_server, daemon=True)
    t.start()

    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, access_log=False)
