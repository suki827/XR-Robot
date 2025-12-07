import json
import math
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests
import torch
import yaml


from ultralytics import YOLO

from src.domain.ActionState import action_state
from src.domain.StreamCamState import StreamCamState
from src.domain.StreamState import StreamState
from src.mq.MQTTPublisher import create_default_publisher

try:
    publisher = create_default_publisher(brokers=["192.168.0.102"], topic="jetauto/cmd")

except Exception as e:
    print(e)

    # human_publish = create_default_publisher(brokers=["192.168.0.101"], topic="jetauto/cmd")


# MODEL_PATH = r"D:\programs\python_projects\quest_robots\yolo_model\ball_120.pt"
MODEL_PATH = r"D:\programs\python_projects\quest_robots\yolo_model\yolov8s-worldv2.pt"

# 加载模型
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"找不到模型文件: {MODEL_PATH}")
print(f"✅ 加载模型: {MODEL_PATH}")

# 1. 选 device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO(MODEL_PATH)


yolo_model.to(DEVICE)   # 🌟 保证模型和后面的 device 一致





# ---------- 相机内参 K ----------
K = np.array([
    [473.4506985179141, 0.0, 323.5512181265506],
    [0.0, 474.2169451085363, 238.6016133237558],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# ---------- 畸变系数 D ----------
dist = np.array([
    -0.02749631166333748,
    -0.0005351606400560652,
    -0.005899822232353089,
    -0.003423120278803321,
    -0.03559287109655516
], dtype=np.float32)

# 图像尺寸（宽, 高）
img_size = (640, 480)

# ---------- 使用 projection_matrix 里的新内参 ----------
K_new = np.array([
    [465.8162841796875, 0.0, 321.2985662909559],
    [0.0, 470.4940490722656, 236.298464380614],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# 生成去畸变映射（只做一次）
map1, map2 = cv2.initUndistortRectifyMap(
    K, dist,
    R=None,  # rectification_matrix 是单位阵，用 None 即可
    newCameraMatrix=K_new,
    size=img_size,
    m1type=cv2.CV_16SC2
)

FPS = 20
JPEG_QUALITY = 80

# MJPEG 拉流地址
Push_URL = "http://192.168.0.100:8000/push/detect_cam"





# HSV 颜色范围（示例，根据实际环境再调）
COLOR_RANGES = {
    "red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255])),
    ],
    "blue": [
        (np.array([100, 120, 70]), np.array([130, 255, 255])),
    ],
    "green": [
        (np.array([40, 70, 70]), np.array([80, 255, 255])),
    ],
    "yellow": [
        (np.array([20, 120, 120]), np.array([35, 255, 255])),
    ],
    "white": [
        # 白色: 低饱和度 + 高亮度
        (np.array([0, 0, 200]), np.array([180, 40, 255])),
    ],
}

def load_bbox_from_yaml(path, margin=None):
    """
    从包含以下结构的 YAML 文件读取 bbox：
      roi:
        x_min: ...
        x_max: ...
        y_min: ...
        y_max: ...

    若文件里的 bbox 是“扩大的”（例如 expand_bbox(margin=10) 后保存的），
    可以传入 margin 进行反推还原：
        margin=10  → 宽高各减 20
    """
    if not os.path.exists(path):
        print(f"[ERROR] ROI YAML not found: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "roi" not in data:
        print("[ERROR] YAML missing 'roi' field")
        return None

    r = data["roi"]
    x_min = int(r["x_min"])
    x_max = int(r["x_max"])
    y_min = int(r["y_min"])
    y_max = int(r["y_max"])

    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min

    # 如果 YAML 里保存的是“扩大的 bbox”，这里可反推回正常 bbox
    if margin is not None and margin > 0:
        x += margin
        y += margin
        w -= margin * 2
        h -= margin * 2

    return (x, y, w, h)

def draw_bbox(frame, bbox, color=(0, 255, 0), label=None, thickness=2):
    """
    在 frame 上绘制一个矩形框（bbox）。

    参数:
        frame : np.ndarray    - BGR 图像
        bbox  : (x, y, w, h)  - 外接框
        color : (B, G, R)     - 颜色
        label : str 或 None   - 在框上方写字
        thickness: int        - 线宽
    """
    if bbox is None:
        return frame

    x, y, w, h = bbox

    # 画矩形
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    # 画文字
    if label is not None:
        cv2.putText(
            frame, label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )

    return frame




# ===============================pull  and push streaming ==================================



"""
    从 FastAPI 的 /stream/{stream_id} (MJPEG) 拉流，
    解码每一帧，并写入 StreamState。
    """

def get_opencv_frame(stream: StreamState):
    """
    从 FastAPI 的 stream.latest_frame (JPEG bytes) 解码成 BGR 图像
    """
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
    """
    将 OpenCV BGR 图像转成 FastAPI StreamState.latest_frame 可直接使用的 JPEG bytes。

    输入:
        frame: np.ndarray 或 None
    输出:
        Optional[bytes] —— 与 FastAPI 推流结构保持一致
    """
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

    print(f"📡 推流线程启动 [{name}] → {url}")

    while True:
        now = time.time()
        if now - last_time < frame_interval:
            time.sleep(0.001)
            continue
        last_time = now

        frame = state.get_frame_copy()
        # ret, frame = cap.read()
        if frame is None:
            # print(f"⚠️ {name} 读取失败")
            time.sleep(0.05)
            continue

        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            print(f"⚠️ {name} JPEG 编码失败")
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


# stable check =======================================================================
def check_stable(dx, dy, tolerance=5, threshold=100):
    """
    判断 dx、dy 是否连续稳定 threshold 次。
    返回值：
        - None ：未稳定
        - (dx, dy) ：稳定后的最终值
    """

    if not hasattr(check_stable, "prev_dx"):
        check_stable.prev_dx = None
        check_stable.prev_dy = None
        check_stable.counter = 0

    if check_stable.prev_dx is not None:
        dx_stable = abs(dx - check_stable.prev_dx) <= tolerance
        dy_stable = abs(dy - check_stable.prev_dy) <= tolerance

        if dx_stable and dy_stable:
            check_stable.counter += 1
        else:
            check_stable.counter = 0
    else:
        check_stable.counter = 0

    check_stable.prev_dx = dx
    check_stable.prev_dy = dy

    if check_stable.counter >= threshold:
        return (dx, dy)

    return None

# ==================================images process area=======================================================
def detect_color(state = None,frame_in=None):
    # frame_in = None, state = None
    """
    从 StreamState 中取最新的一帧图像，做去畸变 + 颜色检测 + 圆形形态学过滤。
    返回: (best_color_name, best_area, best_bbox, frame_undistorted)
      - best_color_name: 最优颜色名称(str) 或 None
      - best_area: 最大轮廓面积(float)（已经通过圆形度过滤）
      - best_bbox: (x, y, w, h) 或 None
      - frame_undistorted: 去畸变后的图像
    """
    frame = frame_in
    if state is not None:
        frame = state.get_frame_copy()

    if frame is None:
        return None, 0.0, None, None

    # 1. 去畸变
    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    # 2. BGR -> HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    best_color = None
    best_score = 0.0  # 用 area * circularity 作为综合评分
    best_area = 0.0
    best_bbox = None  # 最大目标的外接矩形

    # 形态学内核（可按需求调大/调小）
    kernel = np.ones((5, 5), np.uint8)

    # 面积和圆形度阈值，后面可以根据实际画面慢慢调
    MIN_AREA = 330.0  # 像素面积，比这个小的直接忽略
    MIN_CIRCULARITY = 0.6  # 0~1，越接近1越圆，0.6~0.8 比较常用

    for color_name, ranges in COLOR_RANGES.items():
        # 2.1 合并该颜色的所有 HSV 区间
        mask_total = None
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            if mask_total is None:
                mask_total = mask
            else:
                mask_total = cv2.bitwise_or(mask_total, mask)

        if mask_total is None:
            continue

        # 2.2 形态学操作：去小噪点 + 填小洞
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

        # 2.3 找轮廓
        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)

            if area < MIN_AREA:
                continue  # 面积太小当作噪声
            # print(f"=========ares:{area}")
            # 计算圆形度 circularity = 4πA / P^2
            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)

            if circularity < MIN_CIRCULARITY:
                # 过滤掉细长、扁长、不规则的区域
                continue

            # 通过了面积 + 圆形度检查，认为是候选“球”
            x, y, w, h = cv2.boundingRect(c)

            # 综合评分：面积 * 圆形度（大且圆）
            score = area * circularity

            if score > best_score:
                best_score = score
                best_area = area
                best_color = color_name
                best_bbox = (x, y, w, h)

    return best_color, best_area, best_bbox, frame




def bbox_center_xywh(bbox):
    x, y, w, h = bbox
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy

# 算出距离偏差
def draw_right_triangle_and_offsets(frame, bbox1, bbox2,
                                    color_line=(0, 255, 255),
                                    color_point1=(0, 0, 255),
                                    color_point2=(0, 255, 0)):
    """
    在 frame 上：
      - 画出 bbox1、bbox2 的中心点
      - 用这两个点画一个直角三角形（水平 + 垂直 + 斜边）
      - 分别显示 x、y 方向的像素距离（含正负）

    返回: frame, dx, dy, dist
      dx, dy: c2 相对 c1 的水平/垂直偏移（像素，可为负）
      dist:   两点欧氏距离（像素）
    """
    if bbox1 is None or bbox2 is None:
        return frame, None, None, None

    c1 = bbox_center_xywh(bbox1)
    c2 = bbox_center_xywh(bbox2)

    x1, y1 = c1
    x2, y2 = c2

    dx = x2 - x1  # >0 说明 c2 在 c1 右侧
    dy = y2 - y1  # >0 说明 c2 在 c1 下方
    dist = math.hypot(dx, dy)

    return frame, dx, dy, dist


"""
first process dx   then process dy
"""


def action_proc_new(dx, dy):
    # if -25 < dy < 25 and -55 < dy < 55:
    def dx_proc(dx):
        x_linear = 0.1
        x_direction = 1
        if dx < 0:
            x_direction = -1

        x_p_distance = abs(dx)

        #  如果像素长度大于100 那么就最多走2s 控制
        if x_p_distance > 100:
            x_action_duration = x_p_distance // 100
            x_action_duration = 2 if x_action_duration >= 2 else 1
        else:
            if x_p_distance >= 50:
                x_linear = 0.03
            else:
                x_linear = 0.02
            x_action_duration = 1

        data = [0, x_linear * x_direction, x_action_duration]

        payload = {"type": 'move', "data": data}
        publisher.publish(json.dumps(payload))

    def dy_proc(dy):
        y_linear = 0.1
        y_direction = 1
        if dy < 0:
            y_direction = -1

        y_p_distance = abs(dy)

        #  如果像素长度大于100 那么就最多走2s 控制
        if y_p_distance > 100:
            y_action_duration = y_p_distance // 100
            y_action_duration = 2 if y_action_duration >= 2 else 1
        else:
            if y_p_distance >= 50:
                y_linear = 0.03
            else:
                y_linear = 0.02

            y_action_duration = 1

        data = [y_linear * y_direction, 0, y_action_duration]

        payload = {"type": 'move', "data": data}
        publisher.publish(json.dumps(payload))

    # 如果在范围内就直接捡球

    if -40 <= dx <=40 and -25 <= dy <= 30:
        cmd_type = 'cmd'
        payload = {"type": cmd_type, "data": "pick_place_ball_big_craw"}
        publisher.publish(json.dumps(payload))
        time.sleep(1.5)
        action_state.set_picking(False)
        return

    else:
        # if pick_flag: return
        if abs(dx)>40:
            dx_proc(dx)
            time.sleep(1.5)
        else:
            if dy < -25 or dy > 30:
                dy_proc(dy)
                time.sleep(1.5)


IMG_SIZE = 640
CONF_THRES = 0.3        # 置信度阈值(0~1)
DEVICE = "0"

# ====================================检测球体==========================================
def run_detect_script(pull_state:StreamState,is_yolo:bool):
    roi_path = r"src/cfg/pick_roi.yaml"
    roi_bbox = load_bbox_from_yaml(roi_path)

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
            save=False,  # ❗ 不保存
            save_txt=False,  # ❗ 不保存 txt
            verbose=False,
            show=False,

            max_det=20,  # ✅ 限制每帧最多检测多少个目标（默认 300），减少 NMS 开销
            agnostic_nms=False,  # ✅ 类别无关 NMS 一般不需要，关掉略快一点
            vid_stride=1,
        )

        return  results

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    def detect_by_color():
        color, area, bbox, frame = detect_color(frame_in=in_frame)
        if frame is None:
            time.sleep(0.05)
            return

        # 在图像上画检测到的最大颜色块
        if bbox is not None and color is not None:



            # 颜色框（黄色）
            draw_bbox(frame, bbox, color=(0, 255, 255), label="BALL")

            frame, dx, dy, dist = draw_right_triangle_and_offsets(frame, bbox, roi_bbox)

            # 准备要显示的文字
            text1 = f"dx: {dx:.1f} px"
            text2 = f"dy:   {dy:.1f} px"
            text_tips = f"The ball has been detected"
            # 字体 & 颜色设置
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 255, 0)  # 绿色
            thickness = 2
            cv2.putText(frame, text_tips, (20, 40), font, 0.8, color, thickness)
            # 在画面左上角依次显示三行
            # cv2.putText(frame, text1, (20, 40), font, 0.8, color, thickness)
            # cv2.putText(frame, text2, (20, 70), font, 0.8, color, thickness)

            # 4. 调用稳定判断
            stable = check_stable(dx, dy)

            if stable is not None:
                final_dx, final_dy = stable
                # print("📌 dx dy 已连续稳定 100 次：", final_dx, final_dy)
                if action_state.is_picking():
                    action_proc_new(final_dx, final_dy)

            return  frame

    try:
        while True:
            # 颜色检测（含去畸变）
            in_frame = get_opencv_frame(pull_state)
            if is_yolo:
                # action_state.set_start_detect(True)
                is_detect = action_state.get_start_detect()
                if is_detect:
                    yolo_start = time.perf_counter()
                    res = detect_by_yolo(in_frame)
                    yolo_time = (time.perf_counter() - yolo_start) * 1000  # 毫秒
                    print(f"[YOLO] Inference time: {yolo_time:.2f} ms")
                    # YOLO 返回一个 list，所以取第一项
                    annotated = res[0].plot(conf=False)  # 带框的 BGR 图像
                    push_state.latest_frame = annotated
                    # if class_len == 1:
                    #     count = len(res[0].boxes)
                    #     print("检测到目标数量:", count)
                    #     voice_text = f"I detect {count} {detectClasses[0]}"


                    cv2.imshow("YOLO Stream", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else: push_state.latest_frame = in_frame

            else:
                out_frame = detect_by_color()

                # push_frame = frame_to_jpeg(frame)
                push_state.latest_frame = out_frame
                # 显示画面
                # cv2.imshow("Object Detection", out_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

    except KeyboardInterrupt:
        print("\n[Main] Exit by user.")
    finally:

        cv2.destroyAllWindows()





