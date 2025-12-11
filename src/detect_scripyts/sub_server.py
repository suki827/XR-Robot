import json
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
    publisher = create_default_publisher(brokers=["192.168.0.101"], topic="tony_one/cmd")

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
        # 统一走安全设置逻辑


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
                detect_class = action_state.get_detect_class()
                cls_len  = len(detect_class)
                # action_state.set_start_detect(True)
                if is_detect:
                    yolo_start = time.perf_counter()
                    res = detect_by_yolo(in_frame)
                    yolo_time = (time.perf_counter() - yolo_start) * 1000
                    # print(f"[YOLO] Inference time: {yolo_time:.2f} ms")

                    annotated = res[0].plot(conf=False)
                    push_state.latest_frame = annotated

                    # cv2.imshow("YOLO Stream", annotated)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
                else: push_state.latest_frame = in_frame

    except KeyboardInterrupt:
        print("\n[Main] Exit by user.")
    finally:

        cv2.destroyAllWindows()


def run_detect_script_new(pull_state: StreamState, is_yolo: bool):
    push_state = StreamCamState()
    t2 = threading.Thread(target=push_loop, args=(push_state, Push_URL, 'detect_cam'), daemon=True)
    t2.start()

    # ============ IoU 计算 ============
    def bbox_iou(box1, box2):
        """
        box: [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h

        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

        union = area1 + area2 - inter + 1e-6
        return inter / union

    # ============ YOLO 调用 ============
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
        return results

    # ============ 稳定判断相关状态 ============
    tracked_box = None      # 当前追踪的框
    hit_count = 0           # 命中次数
    gap_count = 0           # 连续 miss 帧数

    IOU_THRESHOLD = 0.7
    MIN_HITS = 10            # 命中足够多次认为稳定
    MAX_GAP = 5             # 允许中间断几帧

    last_class_key = None   # 上一次的 detect_class（tuple 形式）
    stable_done = False     # 这一段“单类”是否已经稳定触发过一次

    try:
        while True:
            in_frame = get_opencv_frame(pull_state)

            if is_yolo:
                is_detect = action_state.get_start_detect()
                detect_class = action_state.get_detect_class() or []  # list
                cls_len = len(detect_class)

                # 当前 detect_class 的“签名”，用来判断是否变化
                class_key = tuple(detect_class) if detect_class else None

                # ========= 只要数组内容变了，就重置稳定逻辑 =========
                if class_key != last_class_key:
                    last_class_key = class_key
                    tracked_box = None
                    hit_count = 0
                    gap_count = 0
                    stable_done = False
                    try:
                        action_state.set_stable_detect(False)
                    except AttributeError:
                        pass

                if is_detect:
                    res = detect_by_yolo(in_frame)
                    boxes = res[0].boxes

                    # 默认当前帧不触发稳定
                    stable = False

                    # ========= 只在“检测类别数量 == 1” 时做稳定判断 =========
                    if cls_len == 1 and not stable_done:
                        if boxes is not None and len(boxes) > 0:
                            # 取置信度最高的框
                            confs = boxes.conf.cpu().numpy()
                            idx = int(confs.argmax())
                            curr_xyxy = boxes.xyxy[idx].cpu().numpy()  # [x1,y1,x2,y2]

                            if tracked_box is None:
                                tracked_box = curr_xyxy
                                hit_count = 1
                                gap_count = 0
                            else:
                                iou = bbox_iou(tracked_box, curr_xyxy)
                                if iou >= IOU_THRESHOLD:
                                    hit_count += 1
                                    gap_count = 0
                                else:
                                    gap_count += 1
                                    if gap_count > MAX_GAP:
                                        # 认为原来的 box 不再稳定，切到新 box
                                        tracked_box = curr_xyxy
                                        hit_count = 1
                                        gap_count = 0
                        else:
                            # 这一帧没有检测到目标
                            if tracked_box is not None:
                                gap_count += 1
                                if gap_count > MAX_GAP:
                                    tracked_box = None
                                    hit_count = 0
                                    gap_count = 0

                        # ========= 是否达到稳定条件 =========
                        if tracked_box is not None and hit_count >= MIN_HITS:
                            # 这一帧触发“稳定事件”
                            stable = True          # 👉 这帧 stable=True（脉冲）
                            stable_done = True     # 标记这个单类已经触发过

                    # cls_len == 1 且 stable_done == True 的情况下：
                    # 这一段“单类”已经触发过一次了，之后 stable 一律 False，
                    # 直到 detect_class 数组变了（上面 class_key != last_class_key 会重置）。
                    # cls_len != 1：YOLO 照跑，但不做稳定判断，stable=False



                    # ========= 显示画面 & 调试文字 =========
                    annotated = res[0].plot(conf=False)
                    push_state.latest_frame = annotated
                    if stable:
                        payload = {
                            "type": "cmd",
                            "data": 'raise',
                            "voice_text": str(detect_class[0])
                        }
                        publisher.publish(json.dumps(payload))
                    cv2.putText(
                        annotated,
                        f"cls_len={cls_len}, hits={hit_count}, gaps={gap_count}, "
                        f"stable={int(stable)}, done={int(stable_done)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("YOLO Stream", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                else:
                    # 未开启检测，直接透传
                    if in_frame is not None:
                        # cv2.imshow("YOLO Stream", in_frame)
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break
                        push_state.latest_frame = in_frame

    except KeyboardInterrupt:
        print("\n[Main] Exit by user.")
    finally:
        cv2.destroyAllWindows()


