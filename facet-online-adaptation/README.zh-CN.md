# Facet Online Adaptation

本包提供有界绝对动作策略、下一步 wrench 辅助预测、双 critic、target actor、
成功干预 BC，以及 UR/Franka 运行接口。使用流程和命令见 [README](README.md)。

`configs/module.example.yaml` 是唯一的任务示例，使用 mock 后端与示例动作框。
硬件连接示例不包含任务几何。真实运行需设置经过标定的 per-task action box、
工作空间、传感器参数与冻结模型标识。
