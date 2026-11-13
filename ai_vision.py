"""
AI 视觉层 - 桌面物体识别 + 主方向检测
- 颜色分割 + 轮廓分析检测物料
- 计算物体主方向 (PCA / 最小外接矩形), 输出最佳抓取角度
- 支持固定摄像头 (eye-to-hand) 全局定位
- 无摄像头时支持仿真模式
"""

import math
import random
from typing import Optional, Tuple, List

try:
    import cv2
    import numpy as np
    HAS_CV = True
except ImportError:
    HAS_CV = False

import config


class DetectedObject:
    """检测到的物料对象 (含方向信息)"""

    def __init__(self, center_px: Tuple[int, int], area: float,
                 bbox: Tuple[int, int, int, int],
                 angle_deg: float = 0.0,
                 major_len: float = 0.0,
                 minor_len: float = 0.0):
        self.cx, self.cy = center_px          # 像素中心
        self.area = area                      # 面积
        self.x, self.y, self.w, self.h = bbox  # 水平外接矩形
        self.angle_deg = angle_deg            # 主方向角度 (度, 相对水平轴)
        self.major_len = major_len            # 长轴长度 (像素)
        self.minor_len = minor_len            # 短轴长度 (像素)

    @property
    def aspect_ratio(self) -> float:
        """长宽比 (>=1)"""
        if self.minor_len <= 0:
            return 1.0
        return self.major_len / self.minor_len

    def __repr__(self):
        return (f'Object(center=({self.cx},{self.cy}), area={self.area:.0f}, '
                f'angle={self.angle_deg:.1f}°, '
                f'size={self.major_len:.0f}x{self.minor_len:.0f})')


# ============================================================
# 物体方向分析
# ============================================================
def analyze_orientation(contour) -> Tuple[float, float, float]:
    """
    分析轮廓主方向
    :return: (主方向角度度, 长轴长度, 短轴长度)
    """
    if not HAS_CV:
        return (0.0, 0.0, 0.0)

    # 方法1: 最小外接矩形 (对长条物体最稳定)
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    # minAreaRect 的 angle 范围 [-90, 0), 含义是宽度边与水平的夹角
    # 统一为长轴角度
    if w < h:
        angle += 90.0
        major, minor = h, w
    else:
        major, minor = w, h

    # 归一化到 [-90, 90]
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180

    return (angle, major, minor)


def pca_orientation(contour) -> Tuple[float, float, float]:
    """
    用 PCA 分析轮廓主方向 (对不规则形状更鲁棒)
    :return: (主方向角度度, 长轴长度, 短轴长度)
    """
    if not HAS_CV:
        return (0.0, 0.0, 0.0)

    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return (0.0, 0.0, 0.0)

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
    # 第一主成分方向
    angle = math.degrees(math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]))
    major = 2 * math.sqrt(eigenvalues[0, 0])
    minor = 2 * math.sqrt(eigenvalues[1, 0])
    return (angle, major, minor)


def best_grasp_angle(obj: DetectedObject, gripper_type: str = 'parallel') -> float:
    """
    计算最佳抓取角度 (夹爪需要旋转的角度)
    :param obj: 检测到的物体
    :param gripper_type: 'parallel' 平动夹爪 | 'rotary' 旋转夹爪
    :return: 夹爪应旋转的角度 (度)
    """
    if gripper_type == 'parallel':
        # 平动夹爪: 夹爪闭合方向应垂直于物体长轴
        # 即夹爪的 "开口方向" 沿物体长轴
        # 所以夹爪旋转角度 = 物体主方向角度
        return obj.angle_deg
    else:
        # 旋转夹爪 (如两指旋转): 任意角度可抓, 但偏好长轴方向
        return obj.angle_deg


# ============================================================
# 全局视觉 (eye-to-hand, 固定摄像头)
# ============================================================
class AIVision:
    """桌面全局视觉识别模块"""

    def __init__(self, camera_id: int = None, simulate: bool = True):
        self.camera_id = camera_id if camera_id is not None else config.VISION['camera_id']
        self.simulate = simulate or not HAS_CV
        self._cap = None
        self._sim_objects: List[DetectedObject] = []

        if not self.simulate and HAS_CV:
            self._cap = cv2.VideoCapture(self.camera_id)
            if not self._cap.isOpened():
                print('[视觉] 摄像头打开失败, 切换到仿真模式')
                self.simulate = True
                self._cap = None

    # -------------------- 图像采集 --------------------
    def capture(self):
        """采集一帧图像"""
        if self.simulate:
            return self._simulate_frame()
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def _simulate_frame(self):
        """生成仿真画面"""
        if not HAS_CV:
            return None
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 50), (w - 50, h - 50), (100, 100, 100), 2)
        for obj in self._sim_objects:
            # 画旋转矩形表示物体方向
            cx, cy = obj.cx, obj.cy
            rect = ((cx, cy), (obj.major_len, obj.minor_len), obj.angle_deg)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(frame, [box], 0, (0, 255, 0), -1)
        return frame

    # -------------------- 目标检测 --------------------
    def detect(self, frame=None) -> List[DetectedObject]:
        """
        检测物料, 含主方向分析
        :return: 按面积排序的物料列表
        """
        if frame is None:
            frame = self.capture()
        if frame is None:
            return self._sim_objects[:] if self.simulate else []

        if not HAS_CV:
            return self._sim_objects[:]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = config.VISION['target_color_hsv']
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.VISION['min_area']:
                continue
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(cnt)

            # 主方向分析 (优先 PCA, 退化时用 minAreaRect)
            try:
                angle, major, minor = pca_orientation(cnt)
                if major < minor or major < 1:
                    angle, major, minor = analyze_orientation(cnt)
            except Exception:
                angle, major, minor = analyze_orientation(cnt)

            objects.append(DetectedObject(
                center_px=(cx, cy), area=area, bbox=(x, y, w, h),
                angle_deg=angle, major_len=major, minor_len=minor,
            ))

        objects.sort(key=lambda o: o.area, reverse=True)
        return objects

    # -------------------- 抓取决策 --------------------
    def pick_target(self, objects: List[DetectedObject] = None) -> Optional[DetectedObject]:
        """选择抓取目标: 面积最大且在工作区内"""
        if objects is None:
            objects = self.detect()
        if not objects:
            return None

        wx0, wx1 = config.VISION['workspace_x']
        wy0, wy1 = config.VISION['workspace_y']
        for obj in objects:
            if wx0 <= obj.cx <= wx1 and wy0 <= obj.cy <= wy1:
                return obj
        return None

    def pixel_to_mm(self, obj: DetectedObject,
                    workspace_mm: Tuple[float, float] = (200.0, 150.0)) -> Tuple[float, float]:
        """像素坐标 -> 机械臂坐标 (mm)"""
        wx0, wx1 = config.VISION['workspace_x']
        wy0, wy1 = config.VISION['workspace_y']
        px_range = wx1 - wx0
        py_range = wy1 - wy0
        x_mm = (obj.cx - wx0) / px_range * workspace_mm[0]
        y_mm = (obj.cy - wy0) / py_range * workspace_mm[1]
        return (x_mm, y_mm)

    # -------------------- 仿真辅助 --------------------
    def spawn_sim_object(self, x: int = None, y: int = None,
                         angle: float = None) -> DetectedObject:
        """生成虚拟物料 (可指定角度)"""
        if x is None:
            x = random.randint(config.VISION['workspace_x'][0],
                               config.VISION['workspace_x'][1])
        if y is None:
            y = random.randint(config.VISION['workspace_y'][0],
                               config.VISION['workspace_y'][1])
        if angle is None:
            angle = random.uniform(-80, 80)
        obj = DetectedObject(
            center_px=(x, y), area=1200, bbox=(x - 30, y - 15, 60, 30),
            angle_deg=angle, major_len=60, minor_len=30,
        )
        self._sim_objects.append(obj)
        return obj

    def clear_sim_objects(self):
        self._sim_objects.clear()

    def close(self):
        if self._cap:
            self._cap.release()
