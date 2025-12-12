# TP-Link Archer A6 Router Management Document
# TP-Link Archer A6 路由器管理文档

## 1. Management Interface / 管理接口
- **Address / 地址**: 192.168.0.1
- **Password / 密码**: aut666

## 2. WiFi Connection Information / WiFi 连接信息
- **SSID**: robots_wifi
- **Password / 密码**: robots123

## 3. Static IP Address Assignment / 设备固定 IP 地址分配
To prevent device IP address changes, it is recommended to assign a static IP address to each device:
为避免设备 IP 地址变动，建议为每个设备分配固定 IP 地址：

1. Log in to the router management interface (192.168.0.1)
   登录路由器管理界面 (192.168.0.1)
2. Go to **Advanced** → **Network** → **DHCP Server**
   进入 **高级** → **网络** → **DHCP服务器**
3. Select **Address Reservation**
   选择 **地址保留**
4. Click **Add** to assign a static IP address for the specified device
   点击 **添加**，为指定设备分配固定 IP 地址

## 4. Automatic WiFi Connection Script / 自动连接 WiFi 脚本
An automatic WiFi connection script is prepared in tonypi:
tonypi 中已准备自动连接 WiFi 脚本：

**Connect to Raspberry Pi AP First / 先连接树莓派AP**
   - **SSID**: HW-9E00C3A0
   - **Password / 密码**: hiwonder
**SSH Login / SSH 登录**
   ```bash
   ssh pi@192.168.149.1
   
   pwd raspberrypi
   ```


- **Script Name / 脚本名称**: connect_wifi_static_ip.sh
- **Usage Scenario / 使用场景**:
  - Normally, the device will automatically connect to WiFi via networkmanager
    正常情况下，设备会通过 networkmanager 自动连接 WiFi
  - When automatic connection fails, you can manually run this script
    当自动连接失败时，可手动运行此脚本


