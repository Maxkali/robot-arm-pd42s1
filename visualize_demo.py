"""
AI 机械臂抓料投炉系统 - 3D 完备过程演示
========================================
逐分镜展示完整投料全过程:
  分镜1: 待机 (home)
  分镜2: AI 视觉识别物料 (检测框)
  分镜3: 移动到物料上方 (pick_ready)
  分镜4: 下降到物料位置 + 夹爪打开 (pick)
  分镜5: 夹爪闭合抓取物料
  分镜6: 提升物料 (lift)
  分镜7: 底座旋转到炉口 (furnace)
  分镜8: 下降到投料位置 (place)
  分镜9: 夹爪打开投料入炉 (物料消失)
  分镜10: 复位回 home

运行: python visualize_demo.py
"""

import sys
import math

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use('TkAgg')

# 修复中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.monospace'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans Mono']

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import numpy as np

sys.path.insert(0, r'c:\Users\29246\Desktop\实习考核\ai_robot_arm')
import config
from robot_arm import Kinematics


# ============================================================
# 机械臂 3D 正运动学 (含底座旋转)
# ============================================================
def fk_3d(joints, L1, L2, L3, L4):
    """返回 5 个关键点 3D 坐标: 底座底/底座顶(肩)/肘/腕/末端"""
    t1 = math.radians(joints['base'])
    t2 = math.radians(joints['shoulder'])
    t3 = math.radians(joints['elbow'])
    t4 = math.radians(joints['wrist'])

    # 在 x-z 平面的局部坐标
    x2 = L2 * math.cos(t2)
    z2 = L2 * math.sin(t2)
    x3 = x2 + L3 * math.cos(t2 + t3)
    z3 = z2 + L3 * math.sin(t2 + t3)
    x4 = x3 + L4 * math.cos(t2 + t3 + t4)
    z4 = z3 + L4 * math.sin(t2 + t3 + t4)

    # 底座旋转
    pts = [
        (0, 0, 0),                              # 底座底
        (0, 0, L1),                             # 肩
        (x2 * math.cos(t1), x2 * math.sin(t1), L1 + z2),  # 肘
        (x3 * math.cos(t1), x3 * math.sin(t1), L1 + z3),  # 腕
        (x4 * math.cos(t1), x4 * math.sin(t1), L1 + z4),  # 末端
    ]
    return pts


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_joints(a, b, t):
    return {k: lerp(a[k], b[k], t) for k in a}


# ============================================================
# 分镜定义: (名称, 目标关节角, 夹爪状态, 物料状态, 描述)
# 物料状态: 'on_table'=在台上, 'held'=被夹爪持有, 'in_furnace'=投入炉中
# ============================================================
L1 = config.ARM['L1']
L2 = config.ARM['L2']
L3 = config.ARM['L3']
L4 = config.ARM['L4']

POSES = config.POSES

# 物料在台上的世界坐标 (X, Y, Z)
MATERIAL_POS = (150.0, 0.0, 10.0)   # 物料台上
FURNACE_POS = (0.0, 180.0, 30.0)    # 炉口位置 (base=90度时)

SHOTS = [
    ('待机',         POSES['home'],       True,  'on_table',  'IDLE - 机械臂待命'),
    ('AI识别',       POSES['home'],       True,  'on_table',  'DETECTING - AI视觉检测物料'),
    ('移动到上方',   POSES['pick_ready'], True,  'on_table',  'PICKING - 移动到物料上方'),
    ('下降+开爪',    POSES['pick'],       True,  'on_table',  'PICKING - 下降, 夹爪打开'),
    ('闭合抓取',     POSES['pick'],       False, 'held',      'PICKING - 夹爪闭合, 抓住物料'),
    ('提升',         POSES['lift'],       False, 'held',      'LIFTING - 提升物料'),
    ('旋转到炉口',   POSES['furnace'],    False, 'held',      'TRANSFERRING - 底座旋转90度到炉口'),
    ('下降投料位',   POSES['place'],      False, 'held',      'PLACING - 下降到投料位置'),
    ('开爪投料',     POSES['place'],      True,  'in_furnace','PLACING - 夹爪打开, 物料投入炉中'),
    ('复位',         POSES['home'],       True,  'in_furnace','RETURNING - 机械臂复位'),
    ('完成',         POSES['home'],       True,  'in_furnace','IDLE - 循环完成, 等待下一物料'),
]

FRAMES_PER_SHOT = 40  # 每个分镜帧数


# ============================================================
# 主演示
# ============================================================
def main():
    kin = Kinematics(L1, L2, L3, L4)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('AI 视觉引导 4-DOF 机械臂 - 抓料投炉全过程演示',
                 fontsize=14, fontweight='bold')

    # 左: 3D 视图
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlim(-220, 220)
    ax.set_ylim(-220, 220)
    ax.set_zlim(0, 250)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('机械臂 3D 视图')
    ax.view_init(elev=25, azim=-60)

    # 右: 状态面板
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('系统状态', fontsize=12, fontweight='bold')

    # ---------- 绘制静态场景 ----------
    # 地面
    xx, yy = np.meshgrid([-220, 220], [-220, 220])
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color='lightgray', alpha=0.3)

    # 物料台 (棕色长方体)
    table_x = [100, 200, 200, 100]
    table_y = [-60, -60, 60, 60]
    table_z = [0, 0, 0, 0]
    verts = [list(zip(table_x, table_y, table_z))]
    ax.add_collection3d(Poly3DCollection(verts, facecolor='saddlebrown', alpha=0.5))
    # 台面
    verts2 = [list(zip(table_x, table_y, [20]*4))]
    ax.add_collection3d(Poly3DCollection(verts2, facecolor='saddlebrown', alpha=0.5))
    ax.text(150, 0, 22, '物料台', ha='center', fontsize=9)

    # 炉子 (红色长方体在 Y 正方向)
    fur_x = [-40, 40, 40, -40]
    fur_y = [160, 160, 220, 220]
    fur_z = [0, 0, 0, 0]
    verts_f = [list(zip(fur_x, fur_y, fur_z))]
    ax.add_collection3d(Poly3DCollection(verts_f, facecolor='firebrick', alpha=0.4))
    verts_f2 = [list(zip(fur_x, fur_y, [120]*4))]
    ax.add_collection3d(Poly3DCollection(verts_f2, facecolor='firebrick', alpha=0.4))
    # 连接侧面 (简化为几条线)
    for i in range(4):
        ax.plot([fur_x[i], fur_x[i]], [fur_y[i], fur_y[i]], [0, 120],
                color='darkred', alpha=0.3)
    ax.text(0, 190, 130, '加热炉', ha='center', fontsize=10,
            color='darkred', fontweight='bold')

    # 炉口 (橙色)
    mouth_x = [-20, 20, 20, -20]
    mouth_y = [160, 160, 160, 160]
    mouth_z = [20, 20, 60, 60]
    verts_m = [list(zip(mouth_x, mouth_y, mouth_z))]
    ax.add_collection3d(Poly3DCollection(verts_m, facecolor='orange', alpha=0.8))
    ax.text(0, 162, 40, '炉口', ha='center', fontsize=8, color='white')

    # ---------- 动态对象 ----------
    # 机械臂
    arm_line, = ax.plot([], [], [], 'b-o', linewidth=3, markersize=6,
                        markerfacecolor='white', label='机械臂')
    # 夹爪 (两条短线)
    grip_l, = ax.plot([], [], [], 'k-', linewidth=3)
    grip_r, = ax.plot([], [], [], 'k-', linewidth=3)
    # 物料球
    material = ax.scatter([MATERIAL_POS[0]], [MATERIAL_POS[1]], [MATERIAL_POS[2]],
                          s=300, c='limegreen', edgecolors='darkgreen',
                          linewidths=2, depthshade=True, label='物料')
    # AI 检测框 (物料上方的方框)
    detect_box, = ax.plot([], [], [], 'r--', linewidth=1.5, alpha=0.7)
    detect_text = ax.text(0, 0, 0, '', fontsize=8, color='red')

    ax.legend(loc='upper left', fontsize=8)

    # 状态文本
    status_text = ax2.text(5, 9.3, '', ha='center', fontsize=11,
                           fontweight='bold', color='blue',
                           transform=ax2.transData)
    shot_text = ax2.text(5, 8.3, '', ha='center', fontsize=10, color='purple')
    joint_text = ax2.text(0.5, 7.8, '', ha='left', fontsize=9,
                          verticalalignment='top')
    grip_status = ax2.text(5, 1.5, '', ha='center', fontsize=11, fontweight='bold')
    mat_status = ax2.text(5, 0.8, '', ha='center', fontsize=9, color='gray')

    cycle_count = [0]
    total_shots = len(SHOTS)

    def get_frame_state(frame):
        """根据全局帧号计算当前分镜和插值"""
        shot_idx = (frame // FRAMES_PER_SHOT) % total_shots
        t = (frame % FRAMES_PER_SHOT) / FRAMES_PER_SHOT
        # 平滑缓动
        t = t * t * (3 - 2 * t)
        return shot_idx, t

    def update(frame):
        shot_idx, t = get_frame_state(frame)
        cur_shot = SHOTS[shot_idx]
        next_shot = SHOTS[(shot_idx + 1) % total_shots]

        shot_name, cur_pose, cur_grip, cur_mat, desc = cur_shot
        _, next_pose, next_grip, next_mat, _ = next_shot

        # 判断是否为垂直直线运动分镜 (下降/提升)
        linear_shots = {'下降+开爪', '闭合抓取', '提升', '下降投料位', '开爪投料'}
        is_linear = shot_name in linear_shots

        if is_linear:
            # 笛卡尔空间直线插值: 末端 x,y 不变, 只改 z (保证垂直下降/提升)
            cur_xyz = fk_3d(cur_pose, L1, L2, L3, L4)[4]
            nxt_xyz = fk_3d(next_pose, L1, L2, L3, L4)[4]
            ix = lerp(cur_xyz[0], nxt_xyz[0], t)
            iy = lerp(cur_xyz[1], nxt_xyz[1], t)
            iz = lerp(cur_xyz[2], nxt_xyz[2], t)
            # IK 求解 (选择与起点最接近的解)
            joints = kin.inverse(ix, iy, iz, wrist_angle=0)
            if joints is None:
                joints = lerp_joints(cur_pose, next_pose, t)
        else:
            # 其他分镜用关节空间插值 (允许弧线运动)
            joints = lerp_joints(cur_pose, next_pose, t)

        # 夹爪状态: 下降分镜一开始就打开, 抓取分镜一开始就闭合
        if shot_name == '下降+开爪':
            grip_open = True   # 下降前已打开
        elif shot_name == '闭合抓取':
            grip_open = False  # 到位立即闭合
        elif shot_name == '开爪投料':
            grip_open = True   # 到位立即打开投料
        else:
            grip_open = cur_grip if t < 0.7 else next_grip

        # 物料状态
        mat_state = cur_mat if t < 0.7 else next_mat

        # 正运动学
        pts = fk_3d(joints, L1, L2, L3, L4)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        arm_line.set_data(xs, ys)
        arm_line.set_3d_properties(zs)

        # 末端 + 夹爪
        end = pts[4]
        wrist = pts[3]
        # 夹爪方向 (沿腕-末端方向)
        dx = end[0] - wrist[0]
        dy = end[1] - wrist[1]
        dz = end[2] - wrist[2]
        dlen = math.hypot(dx, dy, dz) or 1
        dx, dy, dz = dx/dlen, dy/dlen, dz/dlen
        # 垂直方向 (在水平面上垂直于夹爪方向)
        px, py = -dy, dx
        plen = math.hypot(px, py) or 1
        px, py = px/plen, py/plen

        g_len = 18 if grip_open else 7
        glx = [end[0] + px*g_len, end[0] - px*g_len]
        gly = [end[1] + py*g_len, end[1] - py*g_len]
        glz = [end[2], end[2]]
        grip_l.set_data([end[0], end[0] + px*g_len], [end[1], end[1] + py*g_len])
        grip_l.set_3d_properties([end[2], end[2]])
        grip_r.set_data([end[0], end[0] - px*g_len], [end[1], end[1] - py*g_len])
        grip_r.set_3d_properties([end[2], end[2]])

        # 物料位置
        if mat_state == 'on_table':
            mx, my, mz = MATERIAL_POS
        elif mat_state == 'held':
            mx, my, mz = end[0], end[1], end[2] - 5  # 夹爪下方
        else:  # in_furnace
            # 投入炉子后, 物料下落到炉内然后消失
            fall_t = t if cur_mat == 'held' else 1.0
            mz = 40 - fall_t * 40
            mx, my = FURNACE_POS[0], FURNACE_POS[1]
            if fall_t > 0.95:
                mx, my, mz = -999, -999, -999  # 移出视野

        material._offsets3d = (np.array([mx]), np.array([my]), np.array([mz]))

        # AI 检测框 (仅在 AI 识别分镜显示)
        if shot_name == 'AI识别':
            bx = [MATERIAL_POS[0]-30, MATERIAL_POS[0]+30,
                  MATERIAL_POS[0]+30, MATERIAL_POS[0]-30, MATERIAL_POS[0]-30]
            by = [MATERIAL_POS[1]-30, MATERIAL_POS[1]-30,
                  MATERIAL_POS[1]+30, MATERIAL_POS[1]+30, MATERIAL_POS[1]-30]
            bz = [MATERIAL_POS[2]]*5
            detect_box.set_data(bx, by)
            detect_box.set_3d_properties(bz)
            detect_text.set_position((MATERIAL_POS[0], MATERIAL_POS[1]))
            detect_text.set_3d_properties(MATERIAL_POS[2] + 40)
            detect_text.set_text('AI 检测到物料')
        else:
            detect_box.set_data([], [])
            detect_box.set_3d_properties([])
            detect_text.set_text('')

        # 状态面板
        status_text.set_text(f'状态: {desc}')
        shot_text.set_text(f'分镜 {shot_idx+1}/{total_shots}: {shot_name}')

        jt = (f'关节角 (度):\n'
              f'  底座 J1: {joints["base"]:7.1f}\n'
              f'  大臂 J2: {joints["shoulder"]:7.1f}\n'
              f'  小臂 J3: {joints["elbow"]:7.1f}\n'
              f'  腕部 J4: {joints["wrist"]:7.1f}\n\n'
              f'末端坐标 (mm):\n'
              f'  X: {end[0]:7.1f}\n'
              f'  Y: {end[1]:7.1f}\n'
              f'  Z: {end[2]:7.1f}')
        joint_text.set_text(jt)

        g_txt = '打开' if grip_open else '闭合'
        g_col = 'green' if grip_open else 'red'
        grip_status.set_text(f'夹爪: {g_txt}')
        grip_status.set_color(g_col)

        mat_txt = {'on_table': '物料: 在物料台上',
                   'held': '物料: 被夹爪持有',
                   'in_furnace': '物料: 已投入炉中'}[mat_state]
        mat_status.set_text(mat_txt)

        # 循环计数
        if shot_idx == 0 and t < 0.05 and frame > FRAMES_PER_SHOT:
            cycle_count[0] += 1

        return (arm_line, grip_l, grip_r, material, detect_box, detect_text,
                status_text, shot_text, joint_text, grip_status, mat_status)

    anim = FuncAnimation(fig, update,
                         frames=FRAMES_PER_SHOT * total_shots * 3,
                         interval=80, blit=False, repeat=True)

    plt.tight_layout()
    print('[可视化] 3D 演示窗口已打开')
    print('[提示] 拖动鼠标可旋转视角, 滚轮缩放')
    print('[流程] 待机→AI识别→移动→下降→夹取→提升→旋转→投料→复位')
    plt.show()


if __name__ == '__main__':
    main()
