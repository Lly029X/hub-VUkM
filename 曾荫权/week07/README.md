文本识别和意图识别都采用了 联合训练策略，共享一个编码器层；文本识别的 Loss 采用nn.CrossEntropy 交叉熵损失 而实体识别的 Loss 也使用 nn.CrossEntropy 但是它只计算非 Padding 部分



权重失衡 两个任务的 Loss 数量相差较大 
微调 调整 Loss 的权重 寻找模型最佳的平衡点