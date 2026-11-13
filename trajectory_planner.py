"""
轨迹规划器 - T 型 (梯形) 速度曲线
================================
核心概念:
  梯形速度曲线 = 加速段(匀加速) + 匀速段(恒速) + 减速段(匀减速)
  用于多关节同步运动: 所有关节同时启动、同时到位, 避免顺序运动的卡顿

  速度 v(t):
    |     ___________
    |    /           \
    |   /             \
    |  /               \
    | /                 \
    +----------------------> t
       ta     tc     td

  位移 s(t): 对 v(t) 积分

用法:
    from trajectory_planner import TrajectoryPlanner
    tp = TrajectoryPlanner(max_speed=600, accel=500)  # RPM, RPM/s
    traj = tp.plan({'base': 0, 'shoulder': 45}, {'base': 30, 'shoulder': 90})
    for t in range(int(traj.duration*100)):
        angles = traj.at(t/100)
        arm.move_joints(angles, block=False)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class JointProfile:
    """单个关节的 T 型速度曲线参数"""
    q0: float = 0.0          # 起始角度 (度)
    q1: float = 0.0          # 目标角度 (度)
    distance: float = 0.0    # 总位移 (度, 绝对值)
    direction: int = 1       # 方向 +1/-1
    v_max: float = 0.0       # 最大速度 (度/秒)
    a: float = 0.0           # 加速度 (度/秒²)
    ta: float = 0.0          # 加速时间 (秒)
    tc: float = 0.0          # 匀速时间 (秒)
    td: float = 0.0          # 减速时间 (秒)
    T: float = 0.0           # 总时间 (秒)
    triangular: bool = False  # 是否三角形(无匀速段)


class TrajectoryPlanner:
    """
    T 型速度曲线轨迹规划器
    支持多关节同步: 以最慢关节的总时间为基准, 其余关节降低速度匹配
    """

    def __init__(self, max_speed_deg_s: float = 360.0,
                 accel_deg_s2: float = 1000.0):
        """
        :param max_speed_deg_s: 最大关节速度 (度/秒)
        :param accel_deg_s2: 加/减速度 (度/秒²)
        """
        self.v_max = max_speed_deg_s
        self.a = accel_deg_s2

    def _plan_single(self, q0: float, q1: float,
                     v_max: float = None, a: float = None) -> JointProfile:
        """规划单关节 T 型曲线"""
        v = v_max if v_max is not None else self.v_max
        acc = a if a is not None else self.a
        dist = abs(q1 - q0)
        direction = 1 if q1 >= q0 else -1

        p = JointProfile(q0=q0, q1=q1, distance=dist, direction=direction,
                         v_max=v, a=acc)

        if dist < 1e-6:
            p.ta = p.tc = p.td = p.T = 0
            return p

        # 加速到 v_max 需要的位移
        s_acc = v ** 2 / (2 * acc)

        if 2 * s_acc <= dist:
            # 有匀速段 (梯形)
            p.ta = v / acc
            p.td = v / acc
            p.tc = (dist - 2 * s_acc) / v
            p.triangular = False
        else:
            # 位移太短, 无法达到 v_max, 退化为三角形
            v_peak = math.sqrt(dist * acc)
            p.ta = v_peak / acc
            p.td = v_peak / acc
            p.tc = 0
            p.triangular = True

        p.T = p.ta + p.tc + p.td
        return p

    def plan(self, start: Dict[str, float], target: Dict[str, float],
             v_max: float = None, a: float = None) -> 'Trajectory':
        """
        规划多关节同步轨迹
        :param start: 起始关节角 {joint: deg}
        :param target: 目标关节角 {joint: deg}
        :return: Trajectory 对象
        """
        # 1. 各关节独立规划
        profiles = {name: self._plan_single(start[name], target[name], v_max, a)
                    for name in start}

        # 2. 找最长总时间作为同步基准
        T_max = max(p.T for p in profiles.values())

        # 3. 对非最长关节, 降低速度使总时间 = T_max (保持加速度不变)
        for name, p in profiles.items():
            if abs(p.T - T_max) < 1e-6 or p.distance < 1e-6:
                continue
            # 重新求解 v 使得 T = T_max, a 不变
            # T = 2*v/a + (s - v²/a)/v = v/a + s/v  (梯形时)
            # 简化: 用二分法求 v
            v_lo, v_hi = 1e-3, p.v_max
            for _ in range(50):
                v_mid = (v_lo + v_hi) / 2
                s_acc = v_mid ** 2 / (2 * p.a)
                if 2 * s_acc <= p.distance:
                    T_mid = 2 * v_mid / p.a + (p.distance - 2 * s_acc) / v_mid
                else:
                    T_mid = 2 * math.sqrt(p.distance / p.a)
                if T_mid < T_max:
                    v_hi = v_mid
                else:
                    v_lo = v_mid

            # 用新速度重新规划
            profiles[name] = self._plan_single(p.q0, p.q1,
                                                v_max=v_lo, a=p.a)

        return Trajectory(profiles, T_max)


class Trajectory:
    """已规划的多关节轨迹, 可查询任意时刻的关节角"""

    def __init__(self, profiles: Dict[str, JointProfile], duration: float):
        self.profiles = profiles
        self.duration = duration

    @staticmethod
    def _position_at(p: JointProfile, t: float) -> float:
        """计算单关节在 t 时刻的角度"""
        if p.distance < 1e-6:
            return p.q0
        if t <= 0:
            return p.q0
        if t >= p.T:
            return p.q1

        if t <= p.ta:
            # 加速段: s = 0.5 * a * t²
            s = 0.5 * p.a * t ** 2
        elif t <= p.ta + p.tc:
            # 匀速段
            s = 0.5 * p.a * p.ta ** 2 + p.v_max * (t - p.ta)
        else:
            # 减速段
            t_dec = t - p.ta - p.tc
            s_acc_cruise = 0.5 * p.a * p.ta ** 2 + p.v_max * p.tc
            s = s_acc_cruise + p.v_max * t_dec - 0.5 * p.a * t_dec ** 2

        return p.q0 + p.direction * s

    def at(self, t: float) -> Dict[str, float]:
        """获取 t 时刻所有关节的目标角度"""
        t = max(0.0, min(t, self.duration))
        return {name: self._position_at(p, t) for name, p in self.profiles.items()}

    def __repr__(self):
        joint_info = ', '.join(
            f'{n}={p.distance:.1f}°' for n, p in self.profiles.items())
        return f'Trajectory(T={self.duration:.3f}s, joints: {joint_info})'


# ===================== 独立测试 =====================
if __name__ == '__main__':
    tp = TrajectoryPlanner(max_speed_deg_s=360, accel_deg_s2=1000)

    # 测试1: 单关节梯形
    p = tp._plan_single(0, 180)
    print(f'单关节 0->180: T={p.T:.3f}s, ta={p.ta:.3f}s, '
          f'tc={p.tc:.3f}s, td={p.td:.3f}s, triangular={p.triangular}')
    assert abs(p.distance - 180) < 0.01

    # 测试2: 多关节同步
    traj = tp.plan({'base': 0, 'shoulder': 0}, {'base': 90, 'shoulder': 180})
    print(f'多关节同步: {traj}')
    # 验证终点
    final = traj.at(traj.duration)
    assert abs(final['base'] - 90) < 0.1
    assert abs(final['shoulder'] - 180) < 0.1
    # 验证起点
    start = traj.at(0)
    assert abs(start['base']) < 0.1
    assert abs(start['shoulder']) < 0.1

    # 测试3: 三角形 (短距离)
    p2 = tp._plan_single(0, 10)
    print(f'短距离 0->10: T={p2.T:.3f}s, triangular={p2.triangular}')
    assert p2.triangular is True

    print('\n[OK] 轨迹规划器测试通过')
