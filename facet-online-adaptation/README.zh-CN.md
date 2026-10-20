# Facet Online Adaptation

本包提供有界绝对动作策略、下一步 wrench 辅助预测、双 critic、target actor、
成功干预 BC，以及 UR/Franka 运行接口。

rl 模式采用 Critic 门控的 TD3+BC：Critic 默认回归实际执行动作的折扣回报；
Actor 在 Critic 训练期间保持 BC 种子不变，留出监控 episode 上的各项检查连续通过后，
Q 梯度才作用到 Actor，BC 权重随 RL 生效的 Critic 更新次数从 1.0 退火到 0.3。
`actor_q_weight: 0` 是只去掉 Q 项的等量 BC 对照。详见 `docs/algorithm.md`。使用流程和命令见 [README](README.md)。

`configs/module.example.yaml` 是唯一的任务示例，使用 mock 后端与示例动作框。
硬件连接示例不包含任务几何。真实运行需设置经过标定的 per-task action box、
工作空间、传感器参数与冻结模型标识。
