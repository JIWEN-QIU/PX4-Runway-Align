# 固定翼跑道视觉闭环联调说明（2026-04-25）

本文档只描述当前阶段已经稳定下来的闭环联调主线，目标是让下一轮实验直接进入“观察控制量并调参”的状态，而不是再回到早期方案分叉。

## 当前主线

当前项目主线已经不是离线视频验证，也不是继续改 `plane.sdf` 做真实前轮转向，而是：

- Gazebo Classic 中持续接收机载前视图像
- 在线运行现有跑道分割模型
- 输出低维视觉控制接口
- 通过 `runway_ground_align_controller.py` 生成转向命令
- 通过 `runway_ground_motion_assist.py` 在地面低速阶段提供运动辅助
- 先验证闭环的方向符号、中心修正趋势和收敛性

## 当前不再建议重复的方向

- 不要继续往 `Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/plane/plane.sdf` 里硬塞真实轮子和转向关节
- 不要重新纠结 `z:=0` 和 `z:=0.2`
- 不要再用真实螺旋桨推进作为当前地面闭环验证主手段

当前推荐口径保持为：

- PX4 + Gazebo 启动时继续使用 `z:=0.2`
- 闭环联调时 `manual_z:=0`
- 由 `ground_motion_assist` 单独控制地面低速前进

## 当前关键节点

- [launch/runway_ground_align_closed_loop.launch](/home/raychill/PX4-Autopilot/launch/runway_ground_align_closed_loop.launch:1)
- [scripts/runway_ground_align_controller.py](/home/raychill/PX4-Autopilot/scripts/runway_ground_align_controller.py:1)
- [scripts/runway_ground_motion_assist.py](/home/raychill/PX4-Autopilot/scripts/runway_ground_motion_assist.py:1)
- [scripts/runway_rc_override_bridge.py](/home/raychill/PX4-Autopilot/scripts/runway_rc_override_bridge.py:1)
- [scripts/runway_closed_loop_monitor.py](/home/raychill/PX4-Autopilot/scripts/runway_closed_loop_monitor.py:1)

## 推荐闭环参数

当前推荐从下面这组参数起步：

```bash
roslaunch px4 runway_ground_align_closed_loop.launch \
  device:=cpu \
  enable_rc_override:=true \
  actuator_mode:=manual_control \
  manual_axis:=r \
  manual_z:=0 \
  enable_ground_motion_assist:=true \
  ground_assist_speed_override_mps:=0.25 \
  ground_assist_max_speed_mps:=0.35 \
  ground_assist_max_yaw_rate_deg_s:=25 \
  ground_assist_steer_sign:=-1.0 \
  k_heading:=0.04 \
  k_lateral:=0.5 \
  max_steer:=0.6 \
  filter_alpha:=0.5 \
  max_delta:=0.08
```

`runway_ground_align_closed_loop.launch` 当前默认值已经向这个基线对齐。

## 当前新增可观测量

`runway_ground_align_controller.py` 现在会额外发布：

- `/runway_ground_align_controller/heading_term`
- `/runway_ground_align_controller/lateral_term`
- `/runway_ground_align_controller/confidence_scale`
- `/runway_ground_align_controller/steer_cmd_scaled`
- `/runway_ground_align_controller/saturated`
- `/runway_ground_align_controller/rate_limited`

控制器的内部处理链现在可以直接按下面顺序观察：

`heading/lateral error -> heading_term/lateral_term -> steer_cmd_raw -> steer_cmd_scaled -> steer_cmd_limited -> steer_cmd_filtered -> steer_cmd`

## 监控脚本用法

新增脚本：

- [scripts/runway_closed_loop_monitor.py](/home/raychill/PX4-Autopilot/scripts/runway_closed_loop_monitor.py:1)

启动方式：

```bash
rosrun px4 runway_closed_loop_monitor.py
```

默认会周期性输出四组信息：

- `state`
- `vision`
- `ctrl`
- `assist`

其中重点看三条链：

- 视觉误差：`head_deg`、`lat_hw`、`conf`
- 控制压缩链：`raw`、`scaled`、`limited`、`filtered`、`cmd`
- 运动执行链：`steer_used`、`yaw_deg_s`、`speed_mps`

输出中带 `!` 的量表示最近一段时间没有刷新，通常意味着对应 topic 没在正常更新。

## 当前最值得优先判断的问题

### 情况 1：`raw` 本身长期很小

优先怀疑：

- 视觉误差本身太小
- `heading_sign` 或 `lateral_sign` 不合理
- 当前画面几何接口虽然有效，但对控制行的偏差不敏感

### 情况 2：`raw` 不小，但 `limited` 很小

优先看：

- `command_scale`
- `max_steer`
- `min_effective_steer`

### 情况 3：`limited` 不小，但 `cmd` 明显偏小

优先看：

- `filter_alpha`
- `max_delta`
- `rate_limited`

### 情况 4：`cmd` 已经明显，但 `yaw_deg_s` 仍然不明显

优先看：

- `ground_assist_steer_sign`
- `ground_assist_max_yaw_rate_deg_s`
- `/runway_ground_motion_assist/active`
- `/runway_ground_motion_assist/steer_used`

## 当前控制器新增参数

`runway_ground_align_controller.py` 现支持以下调参项：

- `heading_deadband_deg`
- `lateral_deadband`
- `command_scale`
- `min_effective_steer`

其中：

- `command_scale` 用于整体放大控制输出
- `min_effective_steer` 用于在地面辅助节点场景下避免“命令非零但几乎看不出转向”

如果想完全关闭这个补偿，可以显式传：

```bash
min_effective_steer:=0.0
```

## 当前一句话目标

当前阶段的目标不是高保真地面物理，而是先稳定验证：

- 飞机能沿跑道方向前进
- 能根据视觉偏差朝中心修正
- 能确认转向符号正确
- 能在仿真里看出收敛趋势
