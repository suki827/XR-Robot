import io
import json
import time

import librosa
import numpy as np
import soundfile as sf
import whisper
import webrtcvad

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


import numpy as np
import webrtcvad


def apply_vad_to_audio(
        audio: np.ndarray,
        sr: int = 16000,
        vad_mode: int = 2,
        frame_ms: int = 30,
) -> np.ndarray:
    """
    对一段已录制音频做 VAD 处理，去掉静音部分，返回拼接后的语音片段。

    参数：
        audio: np.ndarray，一维或二维，float32 [-1, 1] 或 int16
        sr:    采样率，必须是 8000 / 16000 / 32000 / 48000 之一（webrtcvad 限制）
        vad_mode: 0~3，数值越大越“宽松”，更容易被判定为语音
        frame_ms: VAD 帧长（10, 20, 30 ms 之一）

    返回：
        去掉静音后的一维 float32 音频（同采样率 sr）
        若整段都被认为是静音，则返回长度为 0 的数组
    """
    assert frame_ms in (10, 20, 30), "frame_ms 必须是 10 / 20 / 30"
    assert sr in (8000, 16000, 32000, 48000), "sr 必须是 8k / 16k / 32k / 48k"

    # 1) 转为单声道
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # 2) 统一转为 float32 [-1, 1]
    if audio.dtype == np.int16:
        audio_float = audio.astype(np.float32) / 32768.0
    else:
        audio_float = audio.astype(np.float32)

    # 3) 转为 int16 PCM bytes（VAD 要求）
    audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
    pcm_bytes = audio_int16.tobytes()

    vad = webrtcvad.Vad(vad_mode)
    frame_len = int(sr * frame_ms / 1000)  # 每帧样本数
    byte_len = frame_len * 2  # int16 = 2 字节

    voiced_frames = []  # 保存保留的“有语音”帧（用 float32 存）
    sample_index = 0  # 在原始 audio_float 里的索引

    for start in range(0, len(pcm_bytes), byte_len):
        chunk = pcm_bytes[start:start + byte_len]
        if len(chunk) < byte_len:
            break

        is_speech = vad.is_speech(chunk, sr)
        if is_speech:
            # 对应的样本区间
            end_index = sample_index + frame_len
            voiced_frames.append(audio_float[sample_index:end_index])

        sample_index += frame_len

    if not voiced_frames:
        # 全是静音
        return np.zeros(0, dtype=np.float32)

    # 4) 拼接所有“有语音”的帧
    return np.concatenate(voiced_frames).astype(np.float32)


def transcribe_audio_bytes_new(audio_bytes: bytes) -> str:
    total_start = time.perf_counter()

    # Step 1. 解码音频（wav bytes → numpy）
    data, sr = sf.read(io.BytesIO(audio_bytes))

    # Step 2. 转单声道
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Step 3. 重采样到 Whisper 需要的 16k
    if sr != WHISPER_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=WHISPER_SR)
        sr = WHISPER_SR

    # Step 4. VAD 去掉静音（
    vad_start = time.perf_counter()
    data_vad = apply_vad_to_audio(data, sr=sr, vad_mode=2, frame_ms=30)
    vad_time = (time.perf_counter() - vad_start) * 1000

    print(f"[VAD] audio length: {len(data)/sr:.2f}s → {len(data_vad)/sr:.2f}s  |  {vad_time:.2f} ms")

    # Whisper 输入必须是 float32
    audio_np = data_vad.astype(np.float32)

    # Step 5. Whisper 转写
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
        condition_on_previous_text=False,   # 更安全
    )
    whisper_time = (time.perf_counter() - whisper_start) * 1000
    print(f"[Whisper] Inference time: {whisper_time:.2f} ms")

    text = result.get("text", "")
    print("Recognized text:", text)

    # Step 6. 执行后续动作（如果不为空）
    text_lower = text.lower()
    if text_lower:
        select_cmd_object(text_lower)

    total_time = (time.perf_counter() - total_start) * 1000
    print(f"[transcribe_audio_bytes] Total time: {total_time:.2f} ms")

    return text_lower



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
turn_left_commands = ['turn left','rotate left']
turn_right_commands = ['turn right','rotate right']
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
