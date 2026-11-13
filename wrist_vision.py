"""
腕部视觉 (eye-in-hand) - 夹爪上的近距摄像头
- 当机械臂靠近目标时启动, 精确识别物体方向
- 计算最牢靠的抓取角度 (夹爪旋转角)
- 支持多种抓取策略: 沿长轴 / 沿短轴 / 质心对齐
"""

import math
from typing import Optional, Tuple

try:
    import cv2
    import numpy as np
    HAS_CV = True
except ImportError:
    HAS_CV = False

from ai_vision import analyze_orientation, pca_orientation, DetectedObject


# 抓取策略
GRASP_ALONG_MAJOR = 'major'   # 夹爪沿物体长轴方向 (适合细长物体)
GRASP_ALONG_MINOR = 'minor'   # 夹爪沿物体短轴方向 (适合宽扁物体)
GRASP_CENTER = 'center'       # 仅对心, 不旋转


class GraspResult:
    """抓取规划结果"""

    def __init__(self, grasp_angle_deg: float, confidence: float,
                 strategy: str, obj: DetectedObject = None):
        self.grasp_angle_deg = grasp_angle_deg  # 夹爪需要旋转的角度
        self.confidence = confidence            # 置信度 0~1
        self.strategy = strategy                # 采用的策略
        self.obj = obj                          # 检测到的物体

    def __repr__(self):
        return (f'Grasp(angle={self.grasp_angle_deg:.1f}°, '
                f'conf={self.confidence:.2f}, strategy={self.strategy})')


class WristVision:
    """腕部近距视觉模块"""

    def __init__(self, camera_id: int = 1, simulate: bool = True,
                 gripper_type: str = 'parallel'):
        self.camera_id = camera_id
        self.simulate = simulate or not HAS_CV
        self.gripper_type = gripper_type
        self._cap = None
        self._sim_obj: Optional[DetectedObject] = None

        if not self.simulate and HAS_CV:
            self._cap = cv2.VideoCapture(camera_id)
            if not self._cap.isOpened():
                print('[腕部视觉] 摄像头打开失败, 切换仿真')
                self.simulate = True

    def capture(self):
        if self.simulate:
            return self._simulate_frame()
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def _simulate_frame(self):
        """仿真: 假设物体已在视野中心附近"""
        if not HAS_CV:
            return None
        h, w = 240, 320
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        if self._sim_obj is not None:
            cx, cy = self._sim_obj.cx, self._sim_obj.cy
            rect = ((cx, cy), (self._sim_obj.major_len, self._sim_obj.minor_len),
                    self._sim_obj.angle_deg)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(frame, [box], 0, (0, 255, 0), -1)
        return frame

    def set_sim_object(self, angle_deg: float, major: float = 60, minor: float = 25):
        """设置仿真物体 (腕部视野中心附近)"""
        self._sim_obj = DetectedObject(
            center_px=(160, 120), area=major * minor,
            bbox=(130, 105, 60, 30),
            angle_deg=angle_deg, major_len=major, minor_len=minor,
        )

    def detect_object(self, frame=None,
                      color_range=((35, 80, 80), (85, 255, 255))) -> Optional[DetectedObject]:
        """
        近距检测物体方向
        :return: DetectedObject 或 None
        """
        if frame is None:
            frame = self.capture()
        if frame is None:
            return self._sim_obj

        if not HAS_CV:
            return self._sim_obj

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = color_range
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 200:
            return None

        M = cv2.moments(cnt)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        try:
            angle, major, minor = pca_orientation(cnt)
            if major < minor or major < 1:
                angle, major, minor = analyze_orientation(cnt)
        except Exception:
            angle, major, minor = analyze_orientation(cnt)

        return DetectedObject(
            center_px=(cx, cy), area=area, bbox=(0, 0, 0, 0),
            angle_deg=angle, major_len=major, minor_len=minor,
        )

    def plan_grasp(self, obj: DetectedObject = None,
                   strategy: str = None) -> GraspResult:
        """
        规划抓取角度
        :param obj: 检测到的物体 (None 则自动检测)
        :param strategy: 强制策略 (None 自动选择)
        :return: GraspResult
        """
        if obj is None:
            obj = self.detect_object()
        if obj is None:
            return GraspResult(0.0, 0.0, GRASP_CENTER)

        # 自动策略选择
        if strategy is None:
            ar = obj.aspect_ratio
            if ar > 2.0:
                strategy = GRASP_ALONG_MAJOR   # 细长物体沿长轴抓
            elif ar < 1.3:
                strategy = GRASP_CENTER        # 近方形物体对心即可
            else:
                strategy = GRASP_ALONG_MINOR   # 中等比例沿短轴抓

        # 计算夹爪旋转角度
        if strategy == GRASP_ALONG_MAJOR:
            # 夹爪开口沿长轴 => 夹爪旋转角 = 物体长轴角度
            grasp_angle = obj.angle_deg
            confidence = min(1.0, (obj.aspect_ratio - 1.0) / 3.0 + 0.5)
        elif strategy == GRASP_ALONG_MINOR:
            # 夹爪开口沿短轴 => 旋转角 = 长轴角度 + 90°
            grasp_angle = obj.angle_deg + 90.0
            confidence = min(1.0, 0.7)
        else:  # CENTER
            grasp_angle = 0.0
            confidence = 0.9

        # 归一化到 [-180, 180]
        while grasp_angle > 180:
            grasp_angle -= 360
        while grasp_angle < -180:
            grasp_angle += 360

        return GraspResult(grasp_angle, confidence, strategy, obj)

    def close(self):
        if self._cap:
            self._cap.release()
