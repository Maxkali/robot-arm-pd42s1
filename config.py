"""
AI 机械臂系统配置
"""

# ===================== 串口配置 =====================
SERIAL_PORT = 'COM3'          # USB-RS485 串口号 (Linux 下为 /dev/ttyUSB0)
SERIAL_BAUDRATE = 115200
SIMULATE = True               # True=软件仿真模式 (无硬件时调试), False=连接真实硬件

# ===================== 电机地址配置 (RS485 总线) =====================
# 5 自由度机械臂: 底座旋转 / 大臂 / 小臂 / 腕部俯仰 / 腕部翻滚
MOTOR_ADDR = {
    'base':       1,   # 底座旋转 (J1)
    'shoulder':   2,   # 大臂 (J2)
    'elbow':      3,   # 小臂 (J3)
    'wrist':      4,   # 腕部俯仰 (J4)
    'wrist_roll': 5,   # 腕部翻滚 (J5) — 控制夹爪朝向, 与位置解耦
}

# ===================== 机械臂几何参数 (mm) =====================
# 4-DOF 桌面机械臂尺寸 (根据实际结构修改)
ARM = {
    'L1': 50.0,    # 底座到大臂关节的高度
    'L2': 150.0,   # 大臂长度 (肩->肘)
    'L3': 150.0,   # 小臂长度 (肘->腕)
    'L4': 50.0,    # 腕部到夹爪末端长度
}

# ===================== 关节角度限位 (度) =====================
JOINT_LIMITS = {
    'base':       (-180.0, 180.0),
    'shoulder':   (-45.0, 150.0),
    'elbow':      (-90.0, 180.0),
    'wrist':      (-180.0, 180.0),
    'wrist_roll': (-180.0, 180.0),  # 夹爪可整圈旋转
}

# ===================== 腕部翻滚轴模式 =====================
# True  = 有第5轴 (wrist_roll 电机), 直接驱动夹爪旋转 (推荐, 全向抓取)
# False = 无第5轴, 用底座旋转补偿 (仅径向对齐物体有效, 定位精度下降)
WRIST_ROLL_AXIS = True

# ===================== 运动参数 =====================
DEFAULT_SPEED = 600      # 默认速度 RPM
DEFAULT_ACC = 50         # 默认加速度
MOVE_TIMEOUT = 30.0      # 单步超时 (秒)

# ===================== 工作流点位 (度) =====================
# 抓料投炉的关键位置
POSES = {
    'home':       {'base': 0,    'shoulder': 90,  'elbow': 90,  'wrist': 0,   'wrist_roll': 0},
    'pick_ready': {'base': 0,    'shoulder': 60,  'elbow': 120, 'wrist': 0,   'wrist_roll': 0},
    'pick':       {'base': 0,    'shoulder': 45,  'elbow': 135, 'wrist': 0,   'wrist_roll': 0},
    'lift':       {'base': 0,    'shoulder': 80,  'elbow': 100, 'wrist': 0,   'wrist_roll': 0},
    'furnace':    {'base': 90,   'shoulder': 60,  'elbow': 120, 'wrist': 0,   'wrist_roll': 0},
    'place':      {'base': 90,   'shoulder': 45,  'elbow': 135, 'wrist': 0,   'wrist_roll': 0},
}

# ===================== AI 视觉配置 =====================
VISION = {
    'camera_id': 0,                       # 摄像头编号
    'target_color_hsv': ((35, 80, 80), (85, 255, 255)),  # 绿色物料 HSV 范围
    'min_area': 500,                      # 最小识别面积 (像素)
    'workspace_x': (100, 500),            # 工作区 X (像素)
    'workspace_y': (100, 400),            # 工作区 Y (像素)
}

# ===================== 安全配置 =====================
SAFETY = {
    'max_payload_g': 100,       # 最大负载 (克)
    'emergency_stop_key': 'q',  # 急停按键
    'clog_retry_max': 3,        # 堵转最大重试次数
}
