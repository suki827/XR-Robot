import json
import time
import whisper
import io
import numpy as np
import soundfile as sf
import librosa

from src.domain.ActionState import action_state
from src.mq.MQTTPublisher import create_default_publisher

# loading whisper model when run this script (small medium)

model = whisper.load_model("medium")

try:
    publisher_human = create_default_publisher(brokers=["192.168.0.101"], topic="tony_one/cmd")
except Exception as e:
    print(e)
WHISPER_SR = 16000  # Whisper Expected sampling rate

def bytes_to_whisper_audio(audio_bytes: bytes) -> np.ndarray:


    data, sr = sf.read(io.BytesIO(audio_bytes))   # data: np.ndarray


    if data.ndim > 1:
        data = np.mean(data, axis=1)


    if sr != WHISPER_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=WHISPER_SR)


    data = data.astype(np.float32)

    return data

def transcribe(audio_path: str) -> str:

    result = model.transcribe(
        audio_path,
        language='en',  # en
        task='transcribe',  # no translate
        temperature=0,
        beam_size=5,
        best_of=5,
        fp16=True,
        patience=1.0,
        condition_on_previous_text=True,
    )

    return result["text"]

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    total_start = time.perf_counter()

    data, sr = sf.read(io.BytesIO(audio_bytes))  # data: np.ndarray


    if data.ndim > 1:
        data = np.mean(data, axis=1)


    if sr != WHISPER_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=WHISPER_SR)


    audio_np = data.astype(np.float32)

    # 5
    whisper_start = time.perf_counter()
    result = model.transcribe(
        audio_np,
        language='en',
        task='transcribe',
        temperature=0,
        beam_size=5,
        best_of=5,
        fp16=True,
        patience=1.0,
        condition_on_previous_text=True,
    )
    whisper_time = (time.perf_counter() - whisper_start) * 1000  # 毫秒
    print(f"[Whisper] Inference time: {whisper_time:.2f} ms")
    text = result.get("text", "")
    print("recognize text print=======:", text)
    text = text.lower()
    if text is not None:
        select_cmd_object(text)
    total_time = (time.perf_counter() - total_start) * 1000
    print(f"[transcribe_audio_bytes] Total function time: {total_time:.2f} ms")
    return text

         #     }
forward_commands = ['forward','move forward','go forward','walk forward','advance','move ahead','go ahead','walk ahead']
backward_commands = ['backward','move backward','go backward','walk backward','retreat','move back','go back','walk back']
move_left_commands = ['move left','go left','walk left']
move_right_commands = ['move right','go right','walk right']
turn_left_commands = ['turn left']
turn_right_commands = ['turn right']
rotate_commands = ['rotate left','rotate right']
pick_commands = ['pick up','pick ball','picking ball','pick the ball']
wave_commands = ['wave','play wave']
dance_commands = ['dance']
start_commands = ['start detect','activate detect']
stop_commands = ['stop detect','deactivate detect']
detect_commands = ['detect','find','search']





def parse_detect_classes(text: str) -> list[str]:
    text = text.strip()

    # 1. 去掉 detect（忽略大小写）
    if text.lower().startswith("detect "):
        text = text[7:].strip()
    if text.lower().startswith("search "):
        text = text[7:].strip()
    if text.lower().startswith("find "):
        text = text[5:].strip()
    parts = [x.strip() for x in text.split("and")]

    parts = [p for p in parts if p]
    print(parts)

    return parts


def human_action(text):
    text = text.lower()
    print(text)
    action_name = ''
    payload = {}
    if any(cmd in text for cmd in forward_commands):
        action_name = 'forward'
    elif any(cmd in text for cmd in backward_commands):
        action_name = 'backward'
    elif any(cmd in text for cmd in move_left_commands):
        action_name = 'left'
    elif any(cmd in text for cmd in move_right_commands):
        action_name = 'right'
    elif any(cmd in text for cmd in turn_left_commands):
        action_name = 'turn_left'
    elif any(cmd in text for cmd in turn_right_commands):
        action_name = 'turn_right'
    elif any(cmd in text for cmd in wave_commands):
        action_name = 'wave'
    elif any(cmd in text for cmd in dance_commands):
        action_name = 'dance'
    elif any(cmd in text for cmd in start_commands):
        print('start detect')
        action_state.set_start_detect(True)
    elif any(cmd in text for cmd in stop_commands):
        print('stop detect')

        detect_class = ['banana', 'apple', 'knife', 'teddy bear', 'bottle', 'chair', 'bottle', 'cup', 'spoon', 'book', 'fork', 'ball', 'hand bag','scissors']

        action_state.set_detect_class(detect_class)
        action_state.set_start_detect(False)

    elif any(cmd in text for cmd in detect_commands):
        detect_detect = parse_detect_classes(text)
        action_state.set_detect_class(detect_detect)


    if action_name !='':

        payload = {
            "type": "cmd",
            "data": action_name
        }
        if payload['data'] is not None:
            human_payload = json.dumps(payload)
            publisher_human.publish(human_payload)





def select_cmd_object(text: str) -> str:
    if text is  None:
        return ""
    else:
        human_action(text)


def main():
    audio_path = r'D:\programs\python_projects\quest_robots\audio\record.wav'
    res  = transcribe(audio_path)
    print("recognize result：" + res)
if __name__ == "__main__":
    main()
