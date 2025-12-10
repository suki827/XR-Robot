# laptop network configure and HiWonder Robot Remote Connection Guide

This document describes how to configure Laptop IP address and connect to the HiWonder robot, configure the network, and run the project scripts on the Raspberry Pi. .

---

## 📡 1. Overview

The HiWonder robot runs on a Raspberry Pi.  
This project allows you to remotely log in to the Raspberry Pi, configure the network, and start the robot control program.

This guide covers:

- Network setup  
- SSH remote access  
- Robot connection  
- Running the main control program  

---

## 🌐 2. Network Setup

### The goal is to keep all devices on the same local area network. Currently, there is one router with SSID robots_wifi and password robots123.
### **Option 1: Configure Laptop

*  Connect your laptop to robots_wifi and ensure, or set, the IP address to 192.168.0.100.

### **Option 2: Connect Robot 

#### AP
* AP-SSID: HW-9E00CA0
* PSW：hiwonder

#### ethernet
* your laptop can share network to raspberry pi and The Ethernet gateway will become 192.168.137.1. 

* Then use an SSH client to connect to the Raspberry Pi via 192.168.137.165.  
* username: pi  password:raspberrypi


### **Option 3: Device networking

* Connect to the external router using wlan1 and configure a static IP address.

```bash

sudo nmcli dev wifi connect "robots_wifi" password "robots123" ifname wlan1 

sudo nmcli connection modify "robots_wifi" ipv4.addresses "192.168.0.101/24" ipv4.gateway "192.168.0.1" ipv4.dns "8.8.8.8"

sudo nmcli connection up "robots_wifi"

```

## The program that runs the robot

```bash

cd /home/pi/TonyPi/Functions/mqtt

python3  mq_subcribe.py
```