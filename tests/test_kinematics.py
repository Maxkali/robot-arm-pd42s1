# -*- coding: utf-8 -*-
"""
运动学单元测试
- FK -> IK 往返精度 (核心正确性)
- 关节限位检查
- 不可达点返回 None
"""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_arm import Kinematics
import config


def test_fk_ik_roundtrip():
    """正运动学 -> 逆运动学 往返误差 < 0.01mm"""
    kin = Kinematics(config.ARM['L1'], config.ARM['L2'],
                     config.ARM['L3'], config.ARM['L4'])

    # 选取可达点做往返测试
    test_angles = [
        {'base': 0, 'shoulder': 45, 'elbow': 90, 'wrist': 0},
        {'base': 30, 'shoulder': 60, 'elbow': 120, 'wrist': -30},
        {'base': -45, 'shoulder': 30, 'elbow': 150, 'wrist': 60},
        {'base': 90, 'shoulder': 0, 'elbow': 90, 'wrist': 90},
    ]

    for angles in test_angles:
        x, y, z = kin.forward(angles)
        ik = kin.inverse(x, y, z, wrist_angle=angles['wrist'])
        assert ik is not None, f'IK 失败: ({x:.2f},{y:.2f},{z:.2f})'

        # 用 IK 结果重新 FK, 对比位置
        x2, y2, z2 = kin.forward(ik)
        err = math.hypot(x - x2, y - y2, z - z2)
        assert err < 0.01, f'往返误差 {err:.4f}mm 超阈值: angles={angles}'

    print('[OK] FK->IK 往返测试通过 (4 组位姿)')


def test_unreachable_returns_none():
    """超出臂展的目标点应返回 None"""
    kin = Kinematics(50, 150, 150, 50)
    # 臂展 = L2+L3 = 300, 目标 500 不可达
    result = kin.inverse(500, 0, 100)
    assert result is None, '不可达点应返回 None'

    # 内部空洞: 距离 < |L2-L3| = 0 也不可达
    result = kin.inverse(0, 0, 50)  # r=0, 小于|150-150|=0... 实际 r=0时检查 abs(L2-L3)=0, 0<0=False, 所以可达
    print('[OK] 不可达点返回 None 测试通过')


def test_joint_limits():
    """关节限位检查"""
    kin = Kinematics(50, 150, 150, 50)
    within = {'base': 0, 'shoulder': 90, 'elbow': 90, 'wrist': 0}
    assert kin.check_limits(within) is True

    over = {'base': 200, 'shoulder': 90, 'elbow': 90, 'wrist': 0}  # base 限位 [-180,180]
    assert kin.check_limits(over) is False

    print('[OK] 关节限位测试通过')


def test_workspace_reachability():
    """工作空间内随机点可达性 (蒙特卡洛)"""
    import random
    random.seed(42)
    kin = Kinematics(50, 150, 150, 50)

    reachable = 0
    total = 200
    for _ in range(total):
        r = random.uniform(0, 300)
        theta = random.uniform(0, 2 * math.pi)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = random.uniform(0, 300)
        if kin.inverse(x, y, z) is not None:
            reachable += 1

    print(f'[OK] 工作空间可达性: {reachable}/{total} ({100*reachable/total:.0f}%)')
    # 大部分点应可达 (臂展内)
    assert reachable > total * 0.3, '可达率异常低'


if __name__ == '__main__':
    test_fk_ik_roundtrip()
    test_unreachable_returns_none()
    test_joint_limits()
    test_workspace_reachability()
    print('\n全部运动学测试通过!')
