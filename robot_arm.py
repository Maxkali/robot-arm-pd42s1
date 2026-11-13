"""
4-DOF 机械臂控制器
- 基于 PD42S1 驱动的四轴桌面机械臂
- 含正运动学 (FK) 与逆运动学 (IK)
- 支持关节空间与笛卡尔空间运动
"""

import math
import time
import threading
from typing import Dict, List, Optional, Tuple

from pd42s1_driver import PD42S1, SerialTransport
from trajectory_planner import TrajectoryPlanner
from logger import get_logger
import config

log = get_logger('arm')


class Kinematics:
    """4-DOF 机械臂运动学 (平面臂 + 底座旋转)"""

    def __init__(self, L1: float, L2: float, L3: float, L4: float):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4

    def forward(self, joints: Dict[str, float]) -> Tuple[float, float, float]:
        """
        正运动学: 关节角 -> 末端 (x, y, z)
        坐标系: 底座中心为原点, z 向上, x 向前
        :param joints: {'base':θ1, 'shoulder':θ2, 'elbow':θ3, 'wrist':θ4} 度
        :return: (x, y, z) mm
        """
        t1 = math.radians(joints['base'])
        t2 = math.radians(joints['shoulder'])
        t3 = math.radians(joints['elbow'])
        t4 = math.radians(joints['wrist'])

        # 大臂末端 (肘) 相对肩的位置
        x2 = self.L2 * math.cos(t2)
        z2 = self.L2 * math.sin(t2)

        # 小臂末端 (腕) 相对肘的位置 (肘角相对大臂)
        x3 = x2 + self.L3 * math.cos(t2 + t3)
        z3 = z2 + self.L3 * math.sin(t2 + t3)

        # 末端 (夹爪) 相对腕
        x4 = x3 + self.L4 * math.cos(t2 + t3 + t4)
        z4 = z3 + self.L4 * math.sin(t2 + t3 + t4)

        # 底座旋转
        x = x4 * math.cos(t1)
        y = x4 * math.sin(t1)
        z = self.L1 + z4
        return (x, y, z)

    def inverse(self, x: float, y: float, z: float,
                wrist_angle: float = 0.0) -> Optional[Dict[str, float]]:
        """
        逆运动学: 末端 (x,y,z) -> 关节角
        :param x, y, z: 末端位置 mm
        :param wrist_angle: 腕部俯仰角 (度), 末端相对水平方向
        :return: {'base', 'shoulder', 'elbow', 'wrist'} 或 None (无解)
        """
        # 底座角
        t1 = math.atan2(y, x)

        # 在 x-z 平面 (投影距离 r)
        r = math.hypot(x, y)
        z_eff = z - self.L1

        # 腕部位置 (扣掉末端长度 L4)
        wt = math.radians(wrist_angle)
        r_wrist = r - self.L4 * math.cos(wt)
        z_wrist = z_eff - self.L4 * math.sin(wt)

        # 肘角 (余弦定理)
        d2 = r_wrist ** 2 + z_wrist ** 2
        d = math.sqrt(d2)
        # 检查可达
        if d > self.L2 + self.L3 or d < abs(self.L2 - self.L3):
            return None

        cos_t3 = (d2 - self.L2 ** 2 - self.L3 ** 2) / (2 * self.L2 * self.L3)
        cos_t3 = max(-1.0, min(1.0, cos_t3))
        # 肘角 (0=伸直, π=折叠)
        t3 = math.acos(cos_t3)

        # 肩角: 肘朝上解 (t3>0) -> t2 = alpha - beta
        alpha = math.atan2(z_wrist, r_wrist)
        beta = math.acos(
            max(-1.0, min(1.0, (d2 + self.L2 ** 2 - self.L3 ** 2) / (2 * d * self.L2)))
        )
        t2 = alpha - beta

        # 腕角 (使末端水平为 wrist_angle)
        t4 = wt - (t2 + t3)

        return {
            'base': math.degrees(t1),
            'shoulder': math.degrees(t2),
            'elbow': math.degrees(t3),
            'wrist': math.degrees(t4),
        }

    def check_limits(self, joints: Dict[str, float]) -> bool:
        """检查关节角是否在限位内 (只检查 joints 中存在的关节)"""
        for name, ang in joints.items():
            if name not in config.JOINT_LIMITS:
                continue
            lo, hi = config.JOINT_LIMITS[name]
            if not (lo <= ang <= hi):
                return False
        return True


class RobotArm:
    """4-DOF 机械臂控制器"""

    def __init__(self, transport: SerialTransport):
        self.motors: Dict[str, PD42S1] = {
            name: PD42S1(transport, addr, name)
            for name, addr in config.MOTOR_ADDR.items()
        }
        self.kin = Kinematics(config.ARM['L1'], config.ARM['L2'],
                              config.ARM['L3'], config.ARM['L4'])
        self._lock = threading.Lock()
        self._emergency = False

    # -------------------- 基础控制 --------------------
    def get_joint_angles(self) -> Dict[str, Optional[float]]:
        """读取所有关节当前角度"""
        return {name: m.get_angle() for name, m in self.motors.items()}

    def move_joint(self, name: str, angle: float, speed: int = None,
                   acc: int = None, block: bool = True) -> bool:
        """移动单关节到绝对角度"""
        if name not in self.motors:
            log.warning('关节 %s 不存在 (可能未配置电机), 跳过', name)
            return False
        lo, hi = config.JOINT_LIMITS[name]
        if not (lo <= angle <= hi):
            log.warning('关节 %s 角度 %.1f 超出限位 [%.1f, %.1f]', name, angle, lo, hi)
            return False
        log.info('关节 %s -> %.1f° (speed=%d)', name, angle, speed or config.DEFAULT_SPEED)
        speed = speed or config.DEFAULT_SPEED
        acc = acc or config.DEFAULT_ACC
        ok = self.motors[name].move_to_angle(angle, speed, acc, block,
                                               config.MOVE_TIMEOUT)
        if not ok:
            log.error('关节 %s 移动失败 (可能堵转/超时)', name)
        return ok

    def move_joints(self, joints: Dict[str, float], speed: int = None,
                    block: bool = True, sequential: bool = True) -> bool:
        """
        移动多关节到目标角度
        :param sequential: True=依次移动(更安全), False=同时发送
        """
        # 限位检查 (跳过未配置的关节)
        for name, ang in joints.items():
            if name not in self.motors:
                continue
            lo, hi = config.JOINT_LIMITS[name]
            if not (lo <= ang <= hi):
                log.warning('关节 %s 角度 %.1f 超出限位', name, ang)
                return False

        speed = speed or config.DEFAULT_SPEED
        ok = True
        active = {n: a for n, a in joints.items() if n in self.motors}

        with self._lock:
            if sequential:
                for name, ang in active.items():
                    if self._emergency:
                        return False
                    if not self.move_joint(name, ang, speed, block=True):
                        ok = False
                        break
            else:
                # 同时发送 (不阻塞)
                for name, ang in active.items():
                    self.move_joint(name, ang, speed, block=False)
                # 等待全部到位
                if block:
                    for name in active:
                        if not self.motors[name].wait_arrived(config.MOVE_TIMEOUT):
                            ok = False
        return ok

    def move_joints_synced(self, target: Dict[str, float],
                           max_speed_deg_s: float = 360.0,
                           accel_deg_s2: float = 1000.0,
                           update_rate_hz: float = 50.0) -> bool:
        """
        多关节同步运动 (T 型速度曲线插值)
        - 所有关节同时启动、同时到位, 运动平滑无卡顿
        - 用于笛卡尔直线运动的平滑跟随
        :param target: 目标关节角 {joint: deg}
        :param max_speed_deg_s: 最大关节速度 (度/秒)
        :param accel_deg_s2: 加/减速度 (度/秒²)
        :param update_rate_hz: 插值更新频率 (Hz)
        """
        # 限位检查 (跳过未配置关节)
        target = {n: a for n, a in target.items() if n in self.motors}
        for name, ang in target.items():
            lo, hi = config.JOINT_LIMITS[name]
            if not (lo <= ang <= hi):
                log.warning('关节 %s 角度 %.1f 超出限位', name, ang)
                return False

        # 读取当前位置作为起点 (只取目标中包含的关节, 保证键一致)
        start_all = self.get_joint_angles()
        start = {n: start_all[n] for n in target if n in start_all}
        if any(a is None for a in start.values()):
            log.warning('无法读取当前关节角, 退化为顺序运动')
            return self.move_joints(target, block=True)

        # 规划同步轨迹
        tp = TrajectoryPlanner(max_speed_deg_s, accel_deg_s2)
        traj = tp.plan(start, target)
        log.info('同步运动: %s', traj)

        dt = 1.0 / update_rate_hz
        steps = int(traj.duration * update_rate_hz) + 1

        with self._lock:
            for i in range(steps):
                if self._emergency:
                    return False
                t = min(i * dt, traj.duration)
                angles = traj.at(t)
                # 非阻塞发送, 各关节并行执行
                for name, ang in angles.items():
                    self.motors[name].move_to_angle(
                        ang, speed_rpm=config.DEFAULT_SPEED,
                        acc=config.DEFAULT_ACC, block=False)
                time.sleep(dt)

            # 最终到位确认
            for name in target:
                if not self.motors[name].wait_arrived(config.MOVE_TIMEOUT):
                    return False
        return True

    def move_to_pose(self, pose_name: str, speed: int = None) -> bool:
        """移动到预设点位"""
        if pose_name not in config.POSES:
            log.error('未知点位: %s', pose_name)
            return False
        log.info('移动到点位: %s', pose_name)
        return self.move_joints(config.POSES[pose_name], speed)

    # -------------------- 笛卡尔空间 --------------------
    def set_wrist_roll(self, angle_deg: float, speed: int = None) -> bool:
        """
        设置腕部翻滚角 (夹爪朝向)
        - 5-DOF 模式 (WRIST_ROLL_AXIS=True): 直接驱动 wrist_roll 电机, 与位置解耦
        - 4-DOF 降级模式: 用底座旋转补偿 (仅对径向对齐物体有效)
        :param angle_deg: 目标翻滚角 (度)
        """
        speed = speed or config.DEFAULT_SPEED
        if config.WRIST_ROLL_AXIS and 'wrist_roll' in self.motors:
            # 第5轴直驱: 独立控制夹爪旋转, 不影响位置
            return self.move_joint('wrist_roll', angle_deg, speed)
        else:
            # 降级: 底座旋转补偿
            # 原理: 底座旋转会带动夹爪在XY平面旋转, 近似实现翻滚
            # 局限: 仅当物体径向对齐时定位准确, 切向物体会有偏移
            log.warning('4-DOF 降级模式: 用底座旋转补偿翻滚 %.1f° (定位精度下降)', angle_deg)
            cur = self.get_joint_angles()
            base_ang = cur.get('base', 0.0)
            if base_ang is None:
                base_ang = 0.0
            new_base = base_ang + angle_deg
            return self.move_joint('base', new_base, speed)

    def move_to_xyz(self, x: float, y: float, z: float,
                    wrist_angle: float = 0.0, wrist_roll: float = None,
                    speed: int = None) -> bool:
        """
        笛卡尔空间移动到 (x,y,z) (单次 IK, 关节空间插值)
        :param wrist_angle: 腕部俯仰角 (度), 末端相对水平方向
        :param wrist_roll: 腕部翻滚角 (度), 夹爪朝向 (None=不设置)
        """
        joints = self.kin.inverse(x, y, z, wrist_angle)
        if joints is None:
            log.warning('IK 目标 (%.1f,%.1f,%.1f) 不可达', x, y, z)
            return False
        if not self.kin.check_limits(joints):
            log.warning('IK 解超出关节限位: %s', joints)
            return False
        log.info('IK (%.1f,%.1f,%.1f) -> %s', x, y, z,
                 {k: round(v, 1) for k, v in joints.items()})
        if not self.move_joints(joints, speed):
            return False
        # 位置到位后设置腕部翻滚
        if wrist_roll is not None:
            self.set_wrist_roll(wrist_roll, speed)
        return True

    def move_to_xyz_linear(self, x: float, y: float, z: float,
                           wrist_angle: float = 0.0, speed: int = None,
                           steps: int = 20) -> bool:
        """
        笛卡尔空间直线运动 (保证末端走直线)
        通过在起点→终点间做笛卡尔线性插值, 每步重新 IK 求解并发送关节角
        适用于: 垂直下降抓取 / 垂直提升 / 水平移动等需要直线轨迹的场景
        """
        # 获取当前末端位置作为起点
        cur = self.get_joint_angles()
        if any(a is None for a in cur.values()):
            # 无法读取当前位置, 退化为单次 IK
            return self.move_to_xyz(x, y, z, wrist_angle, speed)
        sx, sy, sz = self.kin.forward(cur)

        ok = True
        for i in range(1, steps + 1):
            if self._emergency:
                return False
            t = i / steps
            ix = sx + (x - sx) * t
            iy = sy + (y - sy) * t
            iz = sz + (z - sz) * t
            joints = self.kin.inverse(ix, iy, iz, wrist_angle)
            if joints is None or not self.kin.check_limits(joints):
                log.warning('直线运动 第%d步 IK 失败 (%.1f,%.1f,%.1f)', i, ix, iy, iz)
                ok = False
                break
            # 每步用同步运动保证多关节协调 (比顺序移动更平滑)
            if not self.move_joints_synced(joints, max_speed_deg_s=speed * 6):
                ok = False
                break
        return ok

    def get_xyz(self) -> Optional[Tuple[float, float, float]]:
        """获取末端笛卡尔坐标"""
        angles = self.get_joint_angles()
        if any(a is None for a in angles.values()):
            return None
        return self.kin.forward(angles)

    # -------------------- 安全控制 --------------------
    def emergency_stop(self):
        """急停: 所有电机立即停止"""
        self._emergency = True
        for m in self.motors.values():
            m.stop_now()
        log.critical('急停触发: 所有电机已停止')

    def reset_emergency(self):
        """解除急停状态"""
        self._emergency = False
        for m in self.motors.values():
            m.clear_state()

    def enable_all(self, enable: bool = True):
        """使能/失能所有电机"""
        for m in self.motors.values():
            m.motor_enable(enable)

    def zero_all(self):
        """所有关节当前位置清零 (用于标定零点)"""
        for m in self.motors.values():
            m.angle_to_zero()

    def home_all(self):
        """回到 home 点位"""
        return self.move_to_pose('home')

    # -------------------- 状态监控 --------------------
    def read_status(self) -> Dict[str, str]:
        """读取所有电机状态"""
        return {name: m.read_motor_sta() for name, m in self.motors.items()}

    def check_clog(self) -> List[str]:
        """检查堵转, 返回堵转的电机名列表"""
        clogged = []
        for name, m in self.motors.items():
            if m.read_clog():
                clogged.append(name)
        return clogged
