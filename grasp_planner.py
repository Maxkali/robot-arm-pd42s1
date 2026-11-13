"""
抓取规划器 - 整合全局视觉定位 + 腕部近距角度决策
输出完整抓取位姿: (x, y, z, wrist_roll_angle)
"""

import math
from typing import Optional, Tuple

from ai_vision import AIVision, DetectedObject
from wrist_vision import WristVision, GraspResult
import config


class GraspPose:
    """完整抓取位姿"""

    def __init__(self, x_mm: float, y_mm: float, z_mm: float,
                 wrist_roll_deg: float, confidence: float,
                 obj: DetectedObject = None):
        self.x = x_mm                   # 抓取点 x (mm, 机械臂基坐标系)
        self.y = y_mm                   # 抓取点 y
        self.z = z_mm                   # 抓取点 z (桌面高度)
        self.wrist_roll = wrist_roll_deg  # 腕部旋转角 (度)
        self.confidence = confidence
        self.obj = obj

    def __repr__(self):
        return (f'GraspPose(x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f}, '
                f'roll={self.wrist_roll:.1f}°, conf={self.confidence:.2f})')


class GraspPlanner:
    """抓取规划器"""

    def __init__(self, global_vision: AIVision, wrist_vision: WristVision,
                 table_z_mm: float = 0.0):
        self.global_vis = global_vision
        self.wrist_vis = wrist_vision
        self.table_z = table_z_mm          # 桌面高度 (mm)
        self.approach_z = table_z_mm + 120  # 接近高度 (物体上方 120mm)
        self.grasp_z = table_z_mm + 5       # 抓取高度 (物体上方 5mm)

    def plan_from_global(self) -> Optional[GraspPose]:
        """
        第一阶段: 全局视觉粗定位
        用固定摄像头找到物体位置, 生成初步抓取位姿
        """
        obj = self.global_vis.pick_target()
        if obj is None:
            return None

        x_mm, y_mm = self.global_vis.pixel_to_mm(obj)
        # 初步抓取角度用全局视觉的方向 (精度有限)
        roll = obj.angle_deg
        conf = 0.5  # 粗定位置信度较低

        return GraspPose(x_mm, y_mm, self.approach_z, roll, conf, obj)

    def refine_with_wrist(self, coarse_pose: GraspPose) -> GraspPose:
        """
        第二阶段: 腕部近距精修
        机械臂移动到 coarse_pose 上方后, 用腕部摄像头重新识别角度
        """
        # 腕部视觉检测物体方向
        obj = self.wrist_vis.detect_object()
        if obj is None:
            # 腕部没看到, 沿用粗定位结果
            return coarse_pose

        grasp = self.wrist_vis.plan_grasp(obj)
        # 用腕部检测的高精度角度替换粗角度
        refined = GraspPose(
            x_mm=coarse_pose.x,
            y_mm=coarse_pose.y,
            z_mm=self.grasp_z,
            wrist_roll_deg=grasp.grasp_angle_deg,
            confidence=grasp.confidence,
            obj=obj,
        )
        print(f'[抓取规划] 粗角度={coarse_pose.wrist_roll:.1f}° -> '
              f'精修角度={refined.wrist_roll:.1f}° '
              f'(策略={grasp.strategy}, 置信={grasp.confidence:.2f})')
        return refined

    def plan_grasp(self) -> Optional[GraspPose]:
        """
        完整两阶段抓取规划
        1. 全局视觉定位物体位置
        2. (调用方移动后) 腕部视觉精修角度
        注意: 第2步需在机械臂移动到目标上方后调用 refine_with_wrist
        """
        return self.plan_from_global()


# ===================== 独立测试 =====================
if __name__ == '__main__':
    gv = AIVision(simulate=True)
    wv = WristVision(simulate=True)

    # 生成一个倾斜的物体
    obj = gv.spawn_sim_object(angle=35.0)
    print(f'全局检测: {obj}')

    planner = GraspPlanner(gv, wv, table_z_mm=0)

    # 阶段1: 粗定位
    coarse = planner.plan_grasp()
    print(f'阶段1 粗定位: {coarse}')

    # 阶段2: 腕部精修 (模拟物体在腕部视野中倾斜 35°)
    wv.set_sim_object(angle_deg=33.0)  # 与全局略有差异, 模拟近距更精确
    refined = planner.refine_with_wrist(coarse)
    print(f'阶段2 精修:   {refined}')

    # 测试不同形状物体的策略选择
    print('\n--- 不同形状策略测试 ---')
    for angle, major, minor, name in [
        (20, 80, 15, '细长条'),
        (45, 50, 40, '近方形'),
        (-30, 60, 30, '椭圆'),
    ]:
        wv.set_sim_object(angle_deg=angle, major=major, minor=minor)
        o = wv.detect_object()
        g = wv.plan_grasp(o)
        print(f'{name}: {o} -> {g}')
