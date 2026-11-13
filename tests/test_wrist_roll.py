# -*- coding: utf-8 -*-
"""
腕部翻滚轴 (wrist_roll) 单元测试
- 5-DOF 模式: 第5轴直驱, 与位置解耦
- 4-DOF 降级: 底座旋转补偿
- 翻滚角不影响末端位置 (运动学解耦)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from pd42s1_driver import SerialTransport
from robot_arm import RobotArm, Kinematics


def test_wrist_roll_does_not_affect_position():
    """腕部翻滚角不改变末端位置 (运动学解耦)"""
    kin = Kinematics(config.ARM['L1'], config.ARM['L2'],
                     config.ARM['L3'], config.ARM['L4'])

    base_joints = {'base': 30, 'shoulder': 60, 'elbow': 120, 'wrist': 0}
    # wrist_roll 不是 FK 的输入参数, 不参与位置计算
    p1 = kin.forward(base_joints)
    # 即使附加 wrist_roll 键, FK 也忽略它
    p2 = kin.forward({**base_joints, 'wrist_roll': 90})
    assert abs(p1[0] - p2[0]) < 1e-6
    assert abs(p1[1] - p2[1]) < 1e-6
    assert abs(p1[2] - p2[2]) < 1e-6
    print('[OK] 腕部翻滚与位置解耦测试通过')


def test_set_wrist_roll_5dof():
    """5-DOF 模式: 直接驱动 wrist_roll 电机"""
    config.WRIST_ROLL_AXIS = True
    t = SerialTransport(simulate=True)
    arm = RobotArm(t)

    assert arm.set_wrist_roll(45.0) is True
    angles = arm.get_joint_angles()
    assert abs(angles['wrist_roll'] - 45.0) < 0.1
    # 验证不影响其他关节
    assert abs(angles['base'] - 0.0) < 0.1
    print('[OK] 5-DOF 模式腕部翻滚直驱测试通过')


def test_set_wrist_roll_4dof_fallback():
    """4-DOF 降级模式: 底座旋转补偿翻滚"""
    config.WRIST_ROLL_AXIS = False
    t = SerialTransport(simulate=True)
    arm = RobotArm(t)

    base_before = arm.get_joint_angles()['base']
    arm.set_wrist_roll(30.0)
    base_after = arm.get_joint_angles()['base']
    # 底座应旋转了翻滚角
    assert abs(base_after - (base_before + 30.0)) < 0.1
    print('[OK] 4-DOF 降级底座补偿测试通过')


def test_wrist_roll_in_move_to_xyz():
    """move_to_xyz 传入 wrist_roll 时, 位置到位后设置翻滚"""
    config.WRIST_ROLL_AXIS = True
    t = SerialTransport(simulate=True)
    arm = RobotArm(t)

    ok = arm.move_to_xyz(100, 0, 100, wrist_roll=60.0)
    assert ok is True
    assert abs(arm.get_joint_angles()['wrist_roll'] - 60.0) < 0.1
    print('[OK] move_to_xyz 集成 wrist_roll 测试通过')


def test_pose_includes_wrist_roll():
    """所有预设点位都包含 wrist_roll 键"""
    for name, pose in config.POSES.items():
        assert 'wrist_roll' in pose, f'点位 {name} 缺少 wrist_roll'
    print('[OK] 所有点位包含 wrist_roll 测试通过')


if __name__ == '__main__':
    test_wrist_roll_does_not_affect_position()
    test_set_wrist_roll_5dof()
    test_set_wrist_roll_4dof_fallback()
    test_wrist_roll_in_move_to_xyz()
    test_pose_includes_wrist_roll()
    print('\n全部腕部翻滚轴测试通过!')
