"""
AI 机械臂抓料投炉系统 - 主程序
================================
功能:
  1. 初始化 PD42S1 电机驱动 (支持真实硬件 / 软件仿真)
  2. AI 视觉识别物料
  3. 执行抓料投炉自动化工作流
  4. 急停安全保护

运行:
  python main.py              # 仿真模式 (无硬件)
  python main.py --real       # 连接真实硬件 (需修改 config.py 串口)
"""

import sys
import time
import argparse

from pd42s1_driver import SerialTransport
from robot_arm import RobotArm
from ai_vision import AIVision
from wrist_vision import WristVision
from workflow import Gripper, FurnaceFeedingWorkflow
import config


def print_banner():
    print('=' * 60)
    print('  AI 视觉引导 4-DOF 机械臂 - 抓料投炉自动化系统')
    print('  基于 PD42S1 闭环步进电机驱动器')
    print('=' * 60)
    print(f'  模式: {"软件仿真" if config.SIMULATE else "真实硬件"}')
    print(f'  串口: {config.SERIAL_PORT} @ {config.SERIAL_BAUDRATE}')
    print(f'  电机数: {len(config.MOTOR_ADDR)} ({" ".join(config.MOTOR_ADDR.keys())})')
    print('=' * 60)


def init_system(simulate: bool = None):
    """初始化系统各模块"""
    sim = simulate if simulate is not None else config.SIMULATE

    # 1. 串口传输层
    transport = SerialTransport(
        port=config.SERIAL_PORT,
        baudrate=config.SERIAL_BAUDRATE,
        simulate=sim,
    )

    # 2. 机械臂
    arm = RobotArm(transport)

    # 3. AI 视觉 (全局 + 腕部)
    vision = AIVision(simulate=sim)
    wrist_vis = WristVision(simulate=sim)

    # 4. 夹爪
    gripper = Gripper(simulate=sim)

    # 5. 工作流
    workflow = FurnaceFeedingWorkflow(arm, vision, gripper, wrist_vis)

    return transport, arm, vision, wrist_vis, gripper, workflow


def demo_simulation():
    """仿真模式演示"""
    print_banner()
    transport, arm, vision, wrist_vis, gripper, workflow = init_system(simulate=True)

    # 生成虚拟物料
    print('\n[初始化] 生成 3 个虚拟物料...')
    for _ in range(3):
        obj = vision.spawn_sim_object()
        print(f'  物料 @ 像素({obj.cx},{obj.cy}), 方向 {obj.angle_deg:.0f}°')

    # 演示运动学
    print('\n[运动学测试] 正运动学:')
    home_angles = config.POSES['home']
    xyz = arm.kin.forward(home_angles)
    print(f'  home 位姿 -> 末端 ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm')

    print('\n[运动学测试] 逆运动学:')
    ik = arm.kin.inverse(xyz[0], xyz[1], xyz[2], wrist_angle=0)
    print(f'  ({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f}) -> {ik}')

    # 运行一次完整循环
    print('\n' + '=' * 60)
    print('  开始抓料投炉循环演示')
    print('=' * 60)
    success = workflow.run_cycle()
    print(f'\n循环结果: {"成功" if success else "失败"}')
    print(f'当前状态: {workflow.state}')

    vision.close()
    wrist_vis.close()
    transport.close()
    return success


def real_mode():
    """真实硬件模式"""
    print_banner()
    transport, arm, vision, wrist_vis, gripper, workflow = init_system(simulate=False)

    # 硬件自检
    print('\n[自检] 读取各电机状态...')
    status = arm.read_status()
    for name, sta in status.items():
        print(f'  {name}: {sta}')

    # 使能电机
    print('\n[初始化] 使能所有电机...')
    arm.enable_all(True)
    time.sleep(0.5)

    # 回零 (各关节当前位置设为零点, 实际应先做回零)
    print('[初始化] 标定零点 (当前位置设为零点)...')
    arm.zero_all()

    # 运行
    print('\n按 Enter 开始抓料投炉循环, 输入 q 急停退出...')
    try:
        while True:
            cmd = input('> ').strip().lower()
            if cmd == 'q':
                workflow.stop()
                break
            elif cmd == '':
                workflow.run_cycle()
            elif cmd == 'status':
                print(arm.read_status())
            elif cmd == 'angles':
                print(arm.get_joint_angles())
            elif cmd.startswith('move '):
                # move joint_name angle
                parts = cmd.split()
                if len(parts) == 3:
                    arm.move_joint(parts[1], float(parts[2]))
            else:
                print('命令: [回车]执行循环 | q 退出 | status | angles | move <name> <angle>')
    except KeyboardInterrupt:
        workflow.stop()
    finally:
        arm.enable_all(False)
        transport.close()
        vision.close()
        wrist_vis.close()


def main():
    parser = argparse.ArgumentParser(description='AI 机械臂抓料投炉系统')
    parser.add_argument('--real', action='store_true',
                        help='连接真实硬件 (默认仿真)')
    parser.add_argument('--cycles', type=int, default=1,
                        help='循环次数 (0=无限, 仿真模式)')
    args = parser.parse_args()

    if args.real:
        real_mode()
    else:
        # 仿真模式可连续运行
        if args.cycles == 1:
            demo_simulation()
        else:
            print_banner()
            transport, arm, vision, wrist_vis, gripper, workflow = init_system(simulate=True)
            for _ in range(args.cycles):
                vision.spawn_sim_object()
                workflow.run_cycle()
                vision.clear_sim_objects()
                time.sleep(1)
            vision.close()
            wrist_vis.close()
            transport.close()


if __name__ == '__main__':
    main()
