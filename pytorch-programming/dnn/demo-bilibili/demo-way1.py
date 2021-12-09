import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# w值为1.0
w = torch.Tensor([1.0])
# 需要计算梯度
w.requires_grad = True


def forward(x):
    # w为Tensor类型，*为Tensor与Tensor之间的数乘，自动把x转换为Tensor类型
    return x * w


def loss(x, y):
    # 求 y_hat
    y_pred = forward(x)
    # 求损失 loss = (y_hat - y) ^ 2
    return (y_pred - y)**2


# 训练前 x = 4 时的 y_hat 值 (y_hat = 4 * w)
print('Predict (before training)', 4, forward(4).item())

for epoch in range(100):
    loss_value = 0
    for x, y in zip(x_data, y_data):
        # 前馈过程，只计算loss，loss(l)是Tensor类型的张量
        loss_value = loss(x, y)
        # 调用张量的成员函数 backward()，自动计算所有梯度，存到w里
        loss_value.backward()
        # 获得梯度 w.grad，用 .item() 将Tensor里的数值拿出来，作为Python的标量
        print('\tgrad:', x, y, w.grad.item())
        # w.grad 是一个Tensor的张量，需要取data数值进行计算，不会建立计算图
        w.data = w.data - 0.01 * w.grad.data
        # 计算时使用的Tensor张量，自动构建计算图，用于求反向传播（梯度）；而更新权重时，需要使用标量
        w.grad.data.zero_()  # 权重w中梯度的数据全都清零

    print("progress:", epoch, loss_value.item())

# 训练后 x = 4 时的 y_hat 值 (y_hat = 4 * w)
print('Predict (after training)', 4, forward(4).item())
