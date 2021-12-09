import torch
import torch.nn as nn

# affine参数设为True表示weight和bias将被使用
m = nn.BatchNorm2d(2, affine=True)
input = torch.randn(1, 2, 3, 4)
output = m(input)

print(input)
print(m.weight)
print(m.bias)
print(output)
print(output.size())

# https://blog.csdn.net/bigFatCat_Tom/article/details/91619977
