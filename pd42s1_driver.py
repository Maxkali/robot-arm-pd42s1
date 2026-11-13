"""
PD42S1 步进电机闭环驱动器 - Python 协议驱动
移植自正点原子官方 STM32 例程 (smd.c / smd.h / process_frame.c)

协议帧格式:
    发送: 0xC5 + 从机地址 + 功能码 + [数据...] + 校验和 + 0x5C
    应答: 0xC5 + 从机地址 + 功能码 + 错误码 + [数据...] + 校验和 + 0x5C

校验和 = 从帧头到数据末尾所有字节累加 (低8位)
"""

import struct
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List

try:
    import serial
except ImportError:
    serial = None  # 允许在无 pyserial 时仅做仿真

from logger import get_logger, hex_dump

log = get_logger('pd42s1')


# ============================================================
# 帧定义
# ============================================================
FRAME_HEAD = 0xC5
FRAME_TAIL = 0x5C

PULSES_PER_REV = 51200  # 一圈脉冲数 (细分后)


# ============================================================
# 功能码 (与官方 smd.h 一致)
# ============================================================
class FCT:
    # 系统命令 0x00~0x0F
    IDLE = 0x00
    CAL_ENCODER = 0x01
    RESTART = 0x02
    RESET_FACTORY = 0x03
    PARAM_SAVE = 0x04

    # 读取命令 0x20~0x3F
    READ_SOFT_HARD_VER = 0x20
    READ_PSI = 0x21
    READ_PHASE_RES_IND = 0x22
    READ_PHASE_MA = 0x23
    READ_VOL = 0x24
    READ_MA_PID = 0x25
    READ_SPEED_PID = 0x26
    READ_POS_PID = 0x27
    READ_TOTAL_PULSE = 0x28
    READ_ROTATE_SPEED = 0x29
    READ_POS = 0x2A
    READ_POS_ERROR = 0x2B
    READ_MOTOR_STA = 0x2C
    READ_CLOG_FLAG = 0x2D
    READ_CLOG_CUR = 0x2E
    READ_ENABLE_STA = 0x2F
    READ_ARRIVED_STA = 0x30
    READ_SYS_PARAM = 0x31
    READ_DRIVE_PARAMS = 0x32

    # 设置命令 0x60~0x7F
    SET_SLAVE_ADD = 0x60
    SET_GROUP_ADD = 0x61
    SET_MODE = 0x62
    SET_POS_PID = 0x63
    SET_POS_TORQUE = 0x64
    SET_STEP = 0x65
    SET_MA = 0x66
    SET_UART_BAUD = 0x67
    SET_CAN_BAUD = 0x68
    SET_MODBUS = 0x69
    SET_CLOG_PRO = 0x6A
    SET_CLOG_CUR = 0x6B
    SET_CAN_ID = 0x6C
    SET_DIR_LEVEL = 0x6D
    SET_EN_LEVEL = 0x6E
    SET_CMD_ECHO = 0x6F
    SET_KEY_LOCK = 0x70
    SET_AUTO_NOT_DISPLAY = 0x71
    SET_IO_START_LEVEL = 0x72
    SET_SPEED_PID = 0x73

    # 回零 0x90~0x99
    ORIGIN_SET_LEFT_POS = 0x90
    ORIGIN_LIMIT_HOME = 0x91
    ORIGIN_TRIG = 0x92
    ORIGIN_BREAK = 0x93
    ORIGIN_READ_PARAMS = 0x94
    ORIGIN_SET_PARAMS = 0x95
    ORIGIN_READ_STA = 0x96
    ORIGIN_AOTO_ZERO = 0x97
    ORIGIN_SET_RIGHT_POS = 0x98
    ORIGIN_SWITCH = 0x99

    # 运动控制 0xE0~0xFF
    OL_SPEED_MODE = 0xE0
    OL_POS_MODE = 0xE1
    OL_POS_REL_MODE = 0xE2
    OL_PULSES_MODE = 0xE3
    IO_RUN_MODE = 0xE4
    TORQUE_MODE = 0xF0
    SPEED_MODE = 0xF1
    POS_MODE = 0xF2          # 绝对位置模式
    POS_REL_MODE = 0xF3      # 相对位置模式
    PULSES_MODE = 0xF4
    PULSE_WIDTH_POS_MODE = 0xF5
    PULSE_WIDTH_MA_MODE = 0xF6
    PULSE_WIDTH_SPEED_MODE = 0xF7
    ANGLE_ZERO = 0xF8        # 当前位置清零
    CLEAR_CLOG_PRO = 0xF9
    MOTOR_ENABLE = 0xFA
    CLEAR_STATE = 0xFB
    STOP_NOW = 0xFC          # 立即停止


# ============================================================
# 应答帧结构
# ============================================================
@dataclass
class SerialFrame:
    slave_addr: int = 0
    function_code: int = 0
    error_code: int = 0
    data: bytes = b''
    checksum: int = 0
    valid: bool = False

    def ok(self) -> bool:
        return self.valid and self.error_code == 0x01


# ============================================================
# 工具函数
# ============================================================
def checksum(data: bytes) -> int:
    """计算校验和 (累加取低8位)"""
    return sum(data) & 0xFF


def build_frame(addr: int, func: int, payload: bytes = b'') -> bytes:
    """
    构建发送帧
    :param addr: 从机地址 (1~255)
    :param func: 功能码
    :param payload: 数据载荷
    :return: 完整帧字节
    """
    body = bytes([FRAME_HEAD, addr, func]) + payload
    cs = checksum(body)
    return body + bytes([cs, FRAME_TAIL])


def parse_frame(buffer: bytes) -> Optional[SerialFrame]:
    """
    解析应答帧
    应答格式: 0xC5 + addr + func + error_code + [data...] + checksum + 0x5C
    """
    if len(buffer) < 6:
        return None
    if buffer[0] != FRAME_HEAD or buffer[-1] != FRAME_TAIL:
        return None

    frame = SerialFrame()
    frame.slave_addr = buffer[1]
    frame.function_code = buffer[2]
    frame.error_code = buffer[3]
    frame.checksum = buffer[-2]

    # 数据部分 = 总长 - 6 (头+地址+功能码+错误码+校验+尾)
    data_len = len(buffer) - 6
    if data_len > 0:
        frame.data = bytes(buffer[4:4 + data_len])

    # 校验: body = 头+地址+功能码+错误码+数据
    body = buffer[:-2]
    if checksum(body) == frame.checksum:
        frame.valid = True
    return frame


def bytes_to_int32(b: bytes) -> int:
    """4字节大端转 int32"""
    return struct.unpack('>i', b[:4])[0]


def bytes_to_uint32(b: bytes) -> int:
    return struct.unpack('>I', b[:4])[0]


def bytes_to_int16(b: bytes) -> int:
    return struct.unpack('>h', b[:2])[0]


def bytes_to_float(b: bytes) -> float:
    """4字节转 float (协议中浮点数为大端传输)"""
    return struct.unpack('>f', b[:4])[0]


# ============================================================
# 串口传输层
# ============================================================
class SerialTransport:
    """RS485 串口传输层"""

    def __init__(self, port: str = 'COM3', baudrate: int = 115200,
                 timeout: float = 0.1, simulate: bool = False):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.simulate = simulate
        self._ser = None
        self._lock = threading.Lock()

        if not simulate and serial is None:
            raise RuntimeError("未安装 pyserial，请执行: pip install pyserial")

        if not simulate:
            self._ser = serial.Serial(port, baudrate, timeout=timeout)

    def send_and_recv(self, frame: bytes, timeout_ms: int = 100) -> Optional[bytes]:
        """发送一帧并等待应答"""
        if self.simulate:
            resp = self._simulate_response(frame)
            log.debug('TX[sim] %s -> RX[sim] %s', hex_dump(frame), hex_dump(resp))
            return resp

        with self._lock:
            self._ser.reset_input_buffer()
            log.debug('TX %s', hex_dump(frame))
            self._ser.write(frame)
            self._ser.flush()
            resp = self._read_response(timeout_ms)
            if resp:
                log.debug('RX %s', hex_dump(resp))
            else:
                log.warning('超时无应答 (addr=%d, func=0x%02X)', frame[1], frame[2])
            return resp

    def _read_response(self, timeout_ms: int) -> Optional[bytes]:
        """读取应答帧 (按 0xC5...0x5C 帧界定)"""
        deadline = time.time() + timeout_ms / 1000.0
        buf = bytearray()
        in_frame = False
        while time.time() < deadline:
            ch = self._ser.read(1)
            if not ch:
                if in_frame:
                    break
                continue
            buf.append(ch[0])
            if ch[0] == FRAME_HEAD:
                in_frame = True
                buf = bytearray([FRAME_HEAD])
            elif ch[0] == FRAME_TAIL and in_frame:
                return bytes(buf)
        return bytes(buf) if buf else None

    def _simulate_response(self, frame: bytes) -> bytes:
        """
        软件仿真: 根据功能码生成合理应答
        维护每个从机的模拟状态 (位置/到位/堵转等)
        """
        if not hasattr(self, '_sim_state'):
            self._sim_state = {}  # addr -> {'pos': int, 'arrived': bool, ...}

        addr = frame[1]
        func = frame[2]
        if addr not in self._sim_state:
            self._sim_state[addr] = {'pos': 0, 'arrived': True, 'clog': False, 'enable': True}
        st = self._sim_state[addr]

        data = b''
        # 根据功能码生成应答数据
        if func in (FCT.POS_MODE, FCT.POS_REL_MODE, FCT.SPEED_MODE, FCT.TORQUE_MODE):
            # 运动命令: 标记为运行中, 到位=False
            st['arrived'] = False
            # 模拟位置更新 (绝对/相对)
            if func in (FCT.POS_MODE, FCT.OL_POS_MODE):
                pulses = struct.unpack('>I', frame[7:11])[0]
                st['pos'] = pulses if frame[3] == 0 else -pulses
            elif func in (FCT.POS_REL_MODE, FCT.OL_POS_REL_MODE):
                pulses = struct.unpack('>I', frame[7:11])[0]
                st['pos'] += pulses if frame[3] == 0 else -pulses
        elif func == FCT.READ_POS:
            data = struct.pack('>i', st['pos'])
        elif func == FCT.READ_ARRIVED_STA:
            # 运动命令后下一次查询时设为到位 (模拟运动完成)
            st['arrived'] = True
            data = bytes([1 if st['arrived'] else 0])
        elif func == FCT.READ_CLOG_FLAG:
            data = bytes([1 if st['clog'] else 0])
        elif func == FCT.READ_ENABLE_STA:
            data = bytes([0 if st['enable'] else 1])
        elif func == FCT.READ_MOTOR_STA:
            data = bytes([2 if not st['arrived'] else 1])  # 2=运行中, 1=已完成
        elif func == FCT.READ_ROTATE_SPEED:
            data = struct.pack('>h', 0)
        elif func == FCT.READ_POS_ERROR:
            data = struct.pack('>i', 0)
        elif func == FCT.MOTOR_ENABLE:
            st['enable'] = (frame[3] == 0)
        elif func == FCT.STOP_NOW:
            st['arrived'] = True
        elif func == FCT.ANGLE_ZERO:
            st['pos'] = 0

        body = bytes([FRAME_HEAD, addr, func, 0x01]) + data
        return body + bytes([checksum(body), FRAME_TAIL])

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


# ============================================================
# PD42S1 单个电机驱动器
# ============================================================
class PD42S1:
    """PD42S1 电机驱动器封装"""

    def __init__(self, transport: SerialTransport, addr: int = 1, name: str = ''):
        self.transport = transport
        self.addr = addr
        self.name = name or f'motor_{addr}'

    # -------------------- 底层收发 --------------------
    def _send(self, func: int, payload: bytes = b'',
              timeout_ms: int = 100) -> SerialFrame:
        frame = build_frame(self.addr, func, payload)
        raw = self.transport.send_and_recv(frame, timeout_ms)
        if raw is None:
            return SerialFrame(error_code=0xFF, valid=False)
        parsed = parse_frame(raw)
        return parsed if parsed else SerialFrame(error_code=0xFE, valid=False)

    # -------------------- 系统命令 --------------------
    def cal_encoder(self):
        """校准编码器"""
        return self._send(FCT.CAL_ENCODER)

    def restart(self):
        """复位重启"""
        return self._send(FCT.RESTART)

    def reset_factory(self):
        """恢复出厂设置"""
        return self._send(FCT.RESET_FACTORY)

    def param_save(self):
        """参数保存"""
        return self._send(FCT.PARAM_SAVE)

    # -------------------- 读取命令 --------------------
    def read_version(self):
        """读取软硬件版本"""
        f = self._send(FCT.READ_SOFT_HARD_VER)
        if f.ok() and len(f.data) >= 2:
            return f'software V{f.data[0] // 10}.{f.data[0] % 10}, ' \
                   f'hardware V{f.data[1] // 10}.{f.data[1] % 10}'
        return None

    def read_pos(self) -> Optional[int]:
        """读取实时位置 (脉冲数, 51200为一圈)"""
        f = self._send(FCT.READ_POS)
        if f.ok() and len(f.data) >= 4:
            return bytes_to_int32(f.data)
        return None

    def read_speed(self) -> Optional[int]:
        """读取实时转速 (RPM)"""
        f = self._send(FCT.READ_ROTATE_SPEED)
        if f.ok() and len(f.data) >= 2:
            return bytes_to_int16(f.data)
        return None

    def read_pos_error(self) -> Optional[int]:
        """读取位置误差"""
        f = self._send(FCT.READ_POS_ERROR)
        if f.ok() and len(f.data) >= 4:
            return bytes_to_int32(f.data)
        return None

    def read_motor_sta(self) -> Optional[str]:
        """读取电机状态"""
        states = {0: '空闲', 1: '已完成', 2: '运行中', 3: '过载', 4: '堵转', 5: '欠压'}
        f = self._send(FCT.READ_MOTOR_STA)
        if f.ok() and len(f.data) >= 1:
            return states.get(f.data[0], f'未知({f.data[0]})')
        return None

    def read_arrived(self) -> Optional[bool]:
        """读取到位状态"""
        f = self._send(FCT.READ_ARRIVED_STA)
        if f.ok() and len(f.data) >= 1:
            return f.data[0] == 1
        return None

    def read_clog(self) -> Optional[bool]:
        """读取堵转标志"""
        f = self._send(FCT.READ_CLOG_FLAG)
        if f.ok() and len(f.data) >= 1:
            return f.data[0] == 1
        return None

    def read_enable(self) -> Optional[bool]:
        """读取使能状态 (True=使能)"""
        f = self._send(FCT.READ_ENABLE_STA)
        if f.ok() and len(f.data) >= 1:
            return f.data[0] == 0
        return None

    # -------------------- 设置命令 --------------------
    def set_mode(self, mode: int):
        """
        设置工作模式
        0:通信位置 1:通信速度 2:通信力矩 3:脉冲 4:脉宽位置
        5:脉宽速度 6:脉宽力矩 7:回零 8:开环速度 9:开环位置 10:开环脉冲
        """
        return self._send(FCT.SET_MODE, bytes([mode]))

    def set_step(self, step: int):
        """设置细分 (1~256)"""
        return self._send(FCT.SET_STEP, struct.pack('>H', step))

    def set_pos_torque(self, torque_ma: int):
        """设置位置环力矩限制 (100~3000mA)"""
        return self._send(FCT.SET_POS_TORQUE, struct.pack('>h', torque_ma))

    def set_uart_baud(self, baud: int):
        """设置串口波特率"""
        return self._send(FCT.SET_UART_BAUD, struct.pack('>I', baud))

    def set_clog_protect(self, enable: bool):
        """设置堵转保护"""
        return self._send(FCT.SET_CLOG_PRO, bytes([1 if enable else 0]))

    def set_clog_current(self, ma: int):
        """设置堵转电流 (0~3000mA)"""
        return self._send(FCT.SET_CLOG_CUR, struct.pack('>h', ma))

    # -------------------- 回零 --------------------
    def origin_homing(self, limit_enable: int = 0, direction: int = 0,
                      speed_rpm: int = 100, curr_limit: int = 500):
        """
        回零
        :param limit_enable: 0无限位 1有限位
        :param direction: 0正转 1反转
        :param speed_rpm: 转速 (建议100RPM以内)
        :param curr_limit: 无限位到位电流 (200~1000mA)
        """
        payload = bytes([limit_enable, direction]) + \
                  struct.pack('>i', speed_rpm) + struct.pack('>h', curr_limit)
        return self._send(FCT.ORIGIN_LIMIT_HOME, payload, timeout_ms=500)

    def origin_trig(self, mode: int = 2):
        """
        触发回零
        :param mode: 0单圈 1就近 2多圈
        """
        return self._send(FCT.ORIGIN_TRIG, bytes([mode]), timeout_ms=500)

    def origin_read_sta(self) -> Optional[str]:
        """读取回零状态"""
        states = {0: '空闲', 1: '找零中', 2: '找到零点', 3: '错误'}
        f = self._send(FCT.ORIGIN_READ_STA)
        if f.ok() and len(f.data) >= 1:
            return states.get(f.data[0], f'未知({f.data[0]})')
        return None

    # -------------------- 运动控制 --------------------
    def pos_mode(self, direction: int, acc: int, speed_rpm: int, pulses: int):
        """
        绝对位置模式
        :param direction: 0正转(CW) 1反转(CCW)
        :param acc: 加速度 0~200 (0=直接启动)
        :param speed_rpm: 最大速度 0~3000 RPM
        :param pulses: 目标脉冲数 (51200=1圈)
        """
        payload = bytes([direction, acc]) + \
                  struct.pack('>H', speed_rpm) + struct.pack('>I', pulses)
        return self._send(FCT.POS_MODE, payload)

    def pos_rel_mode(self, direction: int, acc: int, speed_rpm: int, pulses: int):
        """相对位置模式"""
        payload = bytes([direction, acc]) + \
                  struct.pack('>H', speed_rpm) + struct.pack('>I', pulses)
        return self._send(FCT.POS_REL_MODE, payload)

    def speed_mode(self, direction: int, acc: int, speed_rpm: float):
        """
        速度模式
        :param speed_rpm: 0.1~3000 RPM
        """
        payload = bytes([direction, acc]) + struct.pack('>f', speed_rpm)
        return self._send(FCT.SPEED_MODE, payload)

    def torque_mode(self, direction: int, current_ma: int):
        """力矩模式 (0~3000mA)"""
        payload = bytes([direction]) + struct.pack('>H', current_ma)
        return self._send(FCT.TORQUE_MODE, payload)

    def angle_to_zero(self):
        """将当前位置角度清零"""
        return self._send(FCT.ANGLE_ZERO)

    def motor_enable(self, enable: bool):
        """电机使能控制 (True=使能, False=失能)"""
        return self._send(FCT.MOTOR_ENABLE, bytes([0 if enable else 1]))

    def stop_now(self):
        """立即停止 (急停)"""
        return self._send(FCT.STOP_NOW)

    def clear_state(self):
        """清除电机状态 (堵转/失能/刹车)"""
        return self._send(FCT.CLEAR_STATE)

    def clear_clog(self):
        """解除堵转保护"""
        return self._send(FCT.CLEAR_CLOG_PRO)

    # -------------------- 高级封装 --------------------
    def move_to_angle(self, angle_deg: float, speed_rpm: int = 600,
                      acc: int = 50, block: bool = True,
                      timeout_s: float = 30.0) -> bool:
        """
        移动到绝对角度 (度)
        :param angle_deg: 目标角度
        :param speed_rpm: 速度
        :param acc: 加速度
        :param block: 是否阻塞等待到位
        :param timeout_s: 超时
        :return: 是否成功到位
        """
        pulses = int(angle_deg / 360.0 * PULSES_PER_REV)
        direction = 0 if pulses >= 0 else 1
        self.pos_mode(direction, acc, speed_rpm, abs(pulses))

        if not block:
            return True

        # 轮询到位状态
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            arrived = self.read_arrived()
            if arrived is True:
                return True
            # 检查堵转
            if self.read_clog():
                self.stop_now()
                return False
            time.sleep(0.05)
        return False

    def move_relative(self, angle_deg: float, speed_rpm: int = 600,
                      acc: int = 50, block: bool = True,
                      timeout_s: float = 30.0) -> bool:
        """相对角度移动"""
        pulses = int(abs(angle_deg) / 360.0 * PULSES_PER_REV)
        direction = 0 if angle_deg >= 0 else 1
        self.pos_rel_mode(direction, acc, speed_rpm, pulses)

        if not block:
            return True
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.read_arrived() is True:
                return True
            if self.read_clog():
                self.stop_now()
                return False
            time.sleep(0.05)
        return False

    def get_angle(self) -> Optional[float]:
        """获取当前角度 (度)"""
        pos = self.read_pos()
        if pos is None:
            return None
        return pos / PULSES_PER_REV * 360.0

    def wait_arrived(self, timeout_s: float = 30.0) -> bool:
        """等待到位"""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.read_arrived() is True:
                return True
            if self.read_clog():
                return False
            time.sleep(0.05)
        return False
