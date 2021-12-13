import torch
import torch.nn as nn

input_data = torch.rand(100, 3, 64, 64)
max_pool = nn.MaxPool2d(2, 2)
output_data = max_pool(input_data)
print(output_data.size())
