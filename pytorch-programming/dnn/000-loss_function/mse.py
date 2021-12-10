import torch

y = 0.2  # 实际值，标签
# 预测值:采样自N~(0,1)的10个值
pred = torch.randn(10)
# 计算MSE ，也可以使用mse=F.mse_loss(真实标签值,预测函数(含参数))
mse = (pred - y).norm(2).pow(2)
print(mse)
