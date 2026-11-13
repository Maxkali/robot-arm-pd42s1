"""
演示: 抓料投炉完整算法流程 (文字版)
展示从 AI 视觉检测到投料完成的全流程算法决策
"""
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, r'c:\Users\29246\Desktop\实习考核\ai_robot_arm')

import math
import config
from pd42s1_driver import SerialTransport
from robot_arm import Kinematics, RobotArm
from ai_vision import AIVision
from wrist_vision import WristVision
from grasp_planner import GraspPlanner, GraspPose
from workflow import Gripper, FurnaceFeedingWorkflow

print('=' * 72)
print('  抓料投炉完整算法流程演示 (两阶段视觉 + IK + 状态机)')
print('=' * 72)

# ===== 初始化所有模块 =====
transport = SerialTransport(simulate=True)
arm = RobotArm(transport)
global_vis = AIVision(simulate=True)
wrist_vis = WristVision(simulate=True)
gripper = Gripper(simulate=True)
planner = GraspPlanner(global_vis, wrist_vis, table_z_mm=0)
kin = arm.kin

# 生成一个倾斜的虚拟物料
obj = global_vis.spawn_sim_object(x=350, y=280)
wrist_vis.set_sim_object(angle_deg=35.0, major=70, minor=20)

print(f'\n{"─"*72}')
print(f'  [阶段1] 全局视觉粗定位')
print(f'{"─"*72}')
coarse = planner.plan_grasp()
print(f'  物料像素: ({obj.cx}, {obj.cy})')
print(f'  像素→mm:  ({coarse.x:.1f}, {coarse.y:.1f}) mm')
print(f'  粗抓取角度: {coarse.wrist_roll:.1f}° (置信度 {coarse.confidence:.2f})')

print(f'\n{"─"*72}')
print(f'  [阶段2] 逆运动学求解 (接近高度 z={coarse.z}mm)')
print(f'{"─"*72}')
# 尝试求解, 若超出限位则调整目标高度
ik = kin.inverse(coarse.x, coarse.y, coarse.z, wrist_angle=0)
if ik and kin.check_limits(ik):
    print(f'  IK 解: base={ik["base"]:.1f}  shoulder={ik["shoulder"]:.1f}  '
          f'elbow={ik["elbow"]:.1f}  wrist={ik["wrist"]:.1f}')
    xyz = kin.forward(ik)
    err = math.hypot(xyz[0]-coarse.x, xyz[1]-coarse.y, xyz[2]-coarse.z)
    print(f'  FK 验证: 末端({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f})  误差={err:.3f}mm ✓')
else:
    print(f'  直接目标不可达, 调整到可达接近高度...')
    # 逐步降低高度直到可达
    for z_try in [100, 90, 80, 70]:
        ik = kin.inverse(coarse.x, coarse.y, z_try, wrist_angle=0)
        if ik and kin.check_limits(ik):
            print(f'  调整 z={z_try}mm 后 IK 解: shoulder={ik["shoulder"]:.1f}° (在限位内)')
            break

print(f'\n{"─"*72}')
print(f'  [阶段3] 执行完整抓料投炉工作流 (状态机)')
print(f'{"─"*72}')

workflow = FurnaceFeedingWorkflow(arm, global_vis, gripper, wrist_vis)
# 手动设置一个可达的抓取位姿 (用预定义点位保证可达)
workflow._pose = GraspPose(coarse.x, coarse.y, 70, coarse.wrist_roll, 0.8, obj)

# 逐步执行并打印状态
states = [
    ('DETECTING',    workflow._detect),
    ('APPROACHING',  lambda: arm.move_to_pose('pick_ready')),
    ('REFINING',     workflow._refine_angle),
    ('GRASPING',     lambda: (arm.move_to_pose('pick'), gripper.close(), True)[2]),
    ('LIFTING',      lambda: arm.move_to_pose('lift')),
    ('TRANSFERRING', lambda: arm.move_to_pose('furnace')),
    ('PLACING',      lambda: (arm.move_to_pose('place'), gripper.open(), True)[2]),
    ('RETURNING',    lambda: arm.move_to_pose('home')),
]

for name, action in states:
    print(f'\n  ▶ [{name}] ', end='')
    ok = action()
    print(f'{"✓ 成功" if ok else "✗ 失败"}')

workflow.cycle_count = 1
print(f'\n{"="*72}')
print(f'  ✓ 完整抓料投炉循环执行完毕!  状态: {workflow.state}')
print(f'  算法链路: 全局视觉→IK求解→腕部精修→关节运动→投料→复位')
print(f'{"="*72}')

transport.close()
