import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO


from src.domain.ActionState import action_state
from src.domain.StreamCamState import StreamCamState
from src.domain.StreamState import StreamState
from src.mq.MQTTPublisher import create_default_publisher

try:
    publisher = create_default_publisher(brokers=["192.168.0.101"], topic="jetauto/cmd")

except Exception as e:
    print(e)

MODEL_PATH = r"src/yolo_model/yolov8s-worldv2.pt"


if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"can not find file: {MODEL_PATH}")
print(f"loading model: {MODEL_PATH}")


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO(MODEL_PATH)


yolo_model.to(DEVICE)   #



# 图像尺寸（宽, 高）
img_size = (640, 480)



FPS = 20
JPEG_QUALITY = 80

# MJPEG
Push_URL = "http://192.168.0.100:8000/push/detect_cam"


# ===============================pull  and push streaming ==================================
def get_opencv_frame(stream: StreamState):

    with stream._lock:
        if stream.latest_frame is None:
            return None

        jpg_bytes = stream.latest_frame

    # JPEG → numpy array → BGR
    nparr = np.frombuffer(jpg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return frame




# convert opencv frame to jpeg
def frame_to_jpeg(frame: Optional[np.ndarray], quality: int = 80) -> Optional[bytes]:

    if frame is None:
        return None

    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None

    return buf.tobytes()

def push_loop(state, url, name):

    session = requests.Session()
    frame_interval = 1.0 / FPS
    last_time = 0

    print(f"📡 video push thread start [{name}] → {url}")

    while True:
        now = time.time()
        if now - last_time < frame_interval:
            time.sleep(0.001)
            continue
        last_time = now

        frame = state.get_frame_copy()
        if frame is None:
            time.sleep(0.05)
            continue

        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            print(f"⚠️ {name} JPEG encode fail")
            continue

        try:
            session.post(
                url,
                data=buf.tobytes(),
                headers={"Content-Type": "image/jpeg"},
                timeout=0.5
            )
        except Exception as e:
            print(f"⚠️ {name} 推流异常: {e}")
            time.sleep(0.2)




IMG_SIZE = 640
CONF_THRES = 0.3
DEVICE = "0"

# ====================================detect object==========================================
def run_detect_script(pull_state:StreamState,is_yolo:bool):

    push_state = StreamCamState()
    t2 = threading.Thread(target=push_loop, args=(push_state, Push_URL, 'detect_cam'), daemon=True)
    t2.start()
    def detect_by_yolo(frame):
        detect_class = action_state.get_detect_class()
        if detect_class:
            yolo_model.set_classes(detect_class)
        results = yolo_model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            device=DEVICE,
            save=False,
            save_txt=False,
            verbose=False,
            show=False,

            max_det=20,
            agnostic_nms=False,
            vid_stride=1,
        )

        return  results

    try:
        while True:
            in_frame = get_opencv_frame(pull_state)
            if is_yolo:
                is_detect = action_state.get_start_detect()
                if is_detect:
                    yolo_start = time.perf_counter()
                    res = detect_by_yolo(in_frame)
                    yolo_time = (time.perf_counter() - yolo_start) * 1000
                    print(f"[YOLO] Inference time: {yolo_time:.2f} ms")

                    annotated = res[0].plot(conf=False)
                    push_state.latest_frame = annotated

                    cv2.imshow("YOLO Stream", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else: push_state.latest_frame = in_frame



    except KeyboardInterrupt:
        print("\n[Main] Exit by user.")
    finally:

        cv2.destroyAllWindows()





