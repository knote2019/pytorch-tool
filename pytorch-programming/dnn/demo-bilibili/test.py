import torch

x = torch.tensor(1.0)  #指定输入x
y = torch.tensor(2.0)  #指定输出y
w = torch.tensor(1.0, requires_grad=True)  #初始化待求的w， requires_grad=True代表需要求导

y_predicted = w * x  #预测值
loss = (y - y_predicted)**2  #计算预测值与真实值间的loss
print(loss)

loss.backward()  #反向传播
print(w.grad)  #打印此时的梯度

with torch.no_grad():
    w -= 0.01 * w.grad  #更新梯度
    print(w)  #打印更新后的梯度
w.grad.zero_()  #梯度清零，防止梯度累加
