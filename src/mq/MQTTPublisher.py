#!/usr/bin/env python3
# encoding: utf-8

import json
import threading
import time

import paho.mqtt.client as mqtt

from src.domain.ActionState import action_state


class MQTTPublisher:
    def __init__(self, brokers, port=1883, topic="flag/topic", keepalive=60):
        """
        brokers: List[str]
        """
        self.brokers = brokers
        self.port = port
        self.topic = topic
        self.keepalive = keepalive
        self.client = mqtt.Client()
        self.lock = threading.Lock()
        self.connected = False
        self.current_broker_index = 0

        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

        self.connect()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT broker: {self.brokers[self.current_broker_index]}")
            self.connected = True
        else:
            print(f"❌ Failed to connect with code {rc}")

    def on_disconnect(self, client, userdata, rc):
        print("Disconnected from broker.")
        self.connected = False

    def connect(self):
        """尝试连接主地址，如果失败则切备用地址"""
        for attempt in range(len(self.brokers)):
            broker = self.brokers[self.current_broker_index]
            try:
                self.client.connect(broker, self.port, self.keepalive)
                self.client.loop_start()
                print(f"Trying to connect to {broker} ...")
                time.sleep(1.5)
                if self.connected:
                    return
            except Exception as e:
                print(f"Connection to {broker} failed: {e}")
                self.current_broker_index = (self.current_broker_index + 1) % len(self.brokers)
        raise ConnectionError("Unable to connect to any MQTT broker.")

    def publish(self, message: str, topic: str = None):
        """线程安全的发布函数"""
        with self.lock:
            if not self.connected:
                print("Reconnecting before publish...")
                self.connect()
            try:
                topic = topic or self.topic
                result = self.client.publish(topic, message)
                # paho-mqtt 返回的是 MQTTMessageInfo 对象，这里用 rc 判断是否成功
                status = result.rc
                if status == 0:
                    print(f"📤 Sent '{message}' to topic '{topic}'")
                else:
                    print(f"Failed to send message to topic {topic}, rc={status}")
            except Exception as e:
                print(f"Publish error: {e}")
                self.connected = False


# ================= 对外封装的“可调用函数” =================

def create_default_publisher(
    brokers=None,
    topic="jetauto/cmd",
    port=1883,
    keepalive=60,
) -> MQTTPublisher:
    """
    创建一个默认的发布器，方便在其他脚本中直接调用。
    """
    if brokers is None:
        # 默认 broker 列表，你可以按需要改
        brokers = ["192.168.0.101"]
    return MQTTPublisher(brokers=brokers, port=port, topic=topic, keepalive=keepalive)


def send_move_from_quest2Tony(default_publisher: MQTTPublisher = None,cmd: str = None) :

    if cmd is not None:

        action_name = ''
        if cmd =='forward':
            action_name = 'forward'
        elif cmd =='backward':
            action_name = 'backward'
        elif cmd =='left':
            action_name = 'left'
        elif cmd =='right':
            action_name = 'right'
        elif cmd == 'turn_left':
            action_name = 'turn_left'
        elif cmd == 'turn_right':
            action_name = 'turn_right'
        elif cmd =='wave':
            action_name = 'wave'
        elif cmd =='dance':
            action_name = 'dance'
        elif cmd =='activate':
            action_state.set_start_detect(True)


        elif cmd =='deactivate':

            l = ['banana', 'apple', 'knife', 'teddy bear', 'bottle', 'chair', 'bottle', 'cup', 'spoon', 'book', 'fork', 'ball', 'hand bag','scissors']
            action_state.set_detect_class(l)
            action_state.set_start_detect(False)


        if action_name != '':
            payload = {
                'type':  'cmd',
                'data': action_name
            }
            default_publisher.publish(json.dumps(payload))






