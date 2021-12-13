import torch
import torch.nn as nn

input_data = torch.rand(100, 16, 32, 32)
conv1 = nn.Conv2d(16, 64, (3, 3), stride=(1, 1))
output_data = conv1(input_data)
print(output_data.size())
