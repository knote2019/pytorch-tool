import torch
import torch.nn as nn

input_data = torch.rand(5, 4, 6, 6)
bn = nn.BatchNorm2d(4, affine=True)
output_data = bn(input_data)

print(bn.weight)
print(bn.bias)
print(output_data.size())
# print(input_data)
# print(output_data)

nn.Softmax