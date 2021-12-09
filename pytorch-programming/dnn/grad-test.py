'''
	给某一个变量开启梯度记录，然后投入运算中，获取运算操作并进入计算图中
	经过一系列操作后，利用输出的y进行backward反向计算--获取x的梯度值
	ps: 最后backward时的参数--这里是y，必须是scaler -- 标量才可以计算梯度
'''
import torch

x = torch.tensor([2., 4., 6.])
x.requires_grad = True  # 开启梯度
b = x.T
y = x.T * x * 2  # 开始构建计算图
y.sum().backward()
print(x.grad)
