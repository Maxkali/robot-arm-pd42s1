"""
抓料投炉工作流引擎 (增强版: 两阶段视觉抓取)
流程: 全局视觉定位 -> 移动到上方 -> 腕部近距识别最佳角度 -> 旋转腕部 -> 下降抓取
"""

import time
import math
from typing import Optional

from robot_arm import RobotArm
from ai_vision import AIVision, DetectedObject
from wrist_vision import WristVision
from grasp_planner import GraspPlanner, GraspPose
from logger import get_logger
import config

log = get_logger('workflow')


class Gripper:
    """
    夹爪控制 (平动夹爪)
    支持三种驱动后端:
      1. simulate=True: 软件仿真
      2. servo_serial: 串口舵机 (Arduino/STM32 输出 PWM)
      3. pd42s1: 额外一个 PD42S1 电机驱动丝杆 (需传入 PD42S1 实例)
    """

    def __init__(self, simulate: bool = True, servo_serial=None,
                 pd42s1_motor=None, open_angle: int = 0, close_angle: int = 90):
        self.simulate = simulate
        self.servo_serial = servo_serial    # 串口舵机后端
        self.pd42s1 = pd42s1_motor          # PD42S1 丝杆后端
        self.open_angle = open_angle        # 舵机打开角度
        self.close_angle = close_angle      # 舵机闭合角度
        self._open = True
        self._grip_force = 0

    def open(self):
        self._open = True
        if self.simulate:
            log.info('[夹爪] 打开')
        elif self.servo_serial is not None:
            self._set_servo_angle(self.open_angle)
        elif self.pd42s1 is not None:
            self.pd42s1.move_to_angle(0, speed_rpm=300)

    def close(self, force: int = 70):
        self._open = False
        self._grip_force = force
        if self.simulate:
            log.info('[夹爪] 闭合 (夹持力 %d%%)', force)
        elif self.servo_serial is not None:
            angle = int(self.open_angle + (self.close_angle - self.open_angle) * force / 100)
            self._set_servo_angle(angle)
        elif self.pd42s1 is not None:
            self.pd42s1.set_pos_torque(int(3000 * force / 100))
            self.pd42s1.move_relative(-30, speed_rpm=300)

    def _set_servo_angle(self, angle: int):
        """通过串口发送舵机角度指令 (协议: S<angle>\\n)"""
        try:
            self.servo_serial.write(f'S{angle}\n'.encode())
        except Exception as e:
            print(f'[夹爪] 舵机控制失败: {e}')

    @property
    def is_open(self) -> bool:
        return self._open


class FurnaceFeedingWorkflow:
    """
    抓料投炉工作流 (两阶段视觉抓取)
    IDLE -> DETECTING -> APPROACHING -> REFINING -> GRASPING
          -> LIFTING -> TRANSFERRING -> PLACING -> RETURNING -> IDLE
    """

    STATE_IDLE = 'IDLE'
    STATE_DETECTING = 'DETECTING'
    STATE_APPROACHING = 'APPROACHING'
    STATE_REFINING = 'REFINING'
    STATE_GRASPING = 'GRASPING'
    STATE_LIFTING = 'LIFTING'
    STATE_TRANSFERRING = 'TRANSFERRING'
    STATE_PLACING = 'PLACING'
    STATE_RETURNING = 'RETURNING'
    STATE_ERROR = 'ERROR'

    def __init__(self, arm: RobotArm, vision: AIVision, gripper: Gripper,
                 wrist_vision: WristVision = None):
        self.arm = arm
        self.vision = vision
        self.gripper = gripper
        self.wrist_vis = wrist_vision
        self.planner = GraspPlanner(vision, wrist_vision) if wrist_vision else None
        self.state = self.STATE_IDLE
        self.cycle_count = 0
        self._running = False
        self._pose: Optional[GraspPose] = None

    # -------------------- 阶段1: 全局检测 --------------------
    def _detect(self) -> bool:
        """全局视觉检测物料位置"""
        log.info('[%s] AI 视觉识别物料...', self.STATE_DETECTING)
        if self.planner:
            self._pose = self.planner.plan_grasp()
        else:
            obj = self.vision.pick_target()
            if obj:
                x, y = self.vision.pixel_to_mm(obj)
                self._pose = GraspPose(x, y, 80, obj.angle_deg, 0.5, obj)
            else:
                self._pose = None

        if self._pose is None:
            log.warning('未检测到物料')
            return False
        log.info('检测到物料: 位置(%.1f,%.1f)mm, 粗角度=%.1f°',
                 self._pose.x, self._pose.y, self._pose.wrist_roll)
        return True

    # -------------------- 阶段2: 接近 (预抓取点 + 开爪) --------------------
    def _approach(self) -> bool:
        """移动到物料上方 (预抓取高度), 并提前打开夹爪, 预设粗翻滚角"""
        log.info('[%s] 移动到物料上方 (z=%.1fmm, 粗roll=%.1f°)...',
                 self.STATE_APPROACHING, self._pose.z, self._pose.wrist_roll)
        if not self.arm.move_to_xyz(
            self._pose.x, self._pose.y, self._pose.z,
            wrist_angle=0, wrist_roll=self._pose.wrist_roll, speed=400,
        ):
            return False
        log.info('打开夹爪 (预备抓取)...')
        self.gripper.open()
        time.sleep(0.3)
        return True

    def _refine_angle(self) -> bool:
        """腕部摄像头近距识别最佳抓取角度, 旋转腕部对齐"""
        if self.wrist_vis is None or self.planner is None:
            log.info('(无腕部视觉, 跳过精修)')
            return True

        log.info('[%s] 腕部近距识别最佳抓取角度...', self.STATE_REFINING)
        refined = self.planner.refine_with_wrist(self._pose)
        self._pose = refined
        log.info('最佳抓取角度: %.1f° (置信=%.2f)', refined.wrist_roll, refined.confidence)
        # 精修后立即旋转腕部翻滚轴对齐夹爪 (下降前完成朝向调整)
        if not self.arm.set_wrist_roll(refined.wrist_roll, speed=400):
            log.warning('腕部翻滚对齐失败, 继续抓取 (可能角度不对)')
        return True

    def _grasp(self) -> bool:
        """垂直直线下降到物料表面, 闭合夹爪"""
        grasp_z = self._pose.z
        log.info('[%s] 垂直直线下降到 z=%.1fmm...', self.STATE_GRASPING, grasp_z)
        if not self.arm.move_to_xyz_linear(
            self._pose.x, self._pose.y, grasp_z,
            wrist_angle=0, speed=200, steps=15,
        ):
            return False
        log.info('闭合夹爪抓取...')
        self.gripper.close(force=80)
        time.sleep(0.5)
        return True

    def _lift(self) -> bool:
        """垂直直线提升物料到安全高度"""
        lift_z = self._pose.z + 80
        log.info('[%s] 垂直直线提升到 z=%.1fmm...', self.STATE_LIFTING, lift_z)
        return self.arm.move_to_xyz_linear(
            self._pose.x, self._pose.y, lift_z,
            wrist_angle=0, speed=400, steps=15,
        )

    def _transfer(self) -> bool:
        log.info('[%s] 移送到炉口...', self.STATE_TRANSFERRING)
        return self.arm.move_to_pose('furnace')

    def _place(self) -> bool:
        log.info('[%s] 投料入炉...', self.STATE_PLACING)
        if not self.arm.move_to_pose('place'):
            return False
        time.sleep(0.3)
        self.gripper.open()
        time.sleep(0.5)
        return True

    def _return_home(self) -> bool:
        log.info('[%s] 复位...', self.STATE_RETURNING)
        return self.arm.home_all()

    # -------------------- 完整循环 --------------------
    def run_cycle(self) -> bool:
        self._running = True
        try:
            self.state = self.STATE_DETECTING
            if not self._detect():
                self.state = self.STATE_IDLE
                return False

            self.state = self.STATE_APPROACHING
            if not self._approach():
                return self._error('接近失败')

            self.state = self.STATE_REFINING
            if not self._refine_angle():
                return self._error('角度精修失败')

            self.state = self.STATE_GRASPING
            if not self._grasp():
                return self._error('抓取失败')

            self.state = self.STATE_LIFTING
            if not self._lift():
                return self._error('提升失败')

            self.state = self.STATE_TRANSFERRING
            if not self._transfer():
                return self._error('移送失败')

            self.state = self.STATE_PLACING
            if not self._place():
                return self._error('投料失败')

            self.state = self.STATE_RETURNING
            if not self._return_home():
                return self._error('复位失败')

            self.cycle_count += 1
            self.state = self.STATE_IDLE
            log.info('第 %d 次抓料投炉循环成功!', self.cycle_count)
            return True

        except Exception as e:
            return self._error(f'异常: {e}')
        finally:
            self._running = False

    def run_continuous(self, max_cycles: int = 0, interval_s: float = 2.0):
        log.info('启动连续抓料投炉 (%s 次)', '无限' if max_cycles == 0 else max_cycles)
        count = 0
        while max_cycles == 0 or count < max_cycles:
            if not self._running:
                self.run_cycle()
                count += 1
                time.sleep(interval_s)

    def _error(self, msg: str) -> bool:
        self.state = self.STATE_ERROR
        log.error('工作流错误: %s', msg)
        self.arm.emergency_stop()
        self.gripper.open()
        return False

    def stop(self):
        self._running = False
        self.arm.emergency_stop()
