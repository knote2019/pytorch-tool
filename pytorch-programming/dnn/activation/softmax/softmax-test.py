import torch
from torch.nn import functional as F

y = torch.rand(3, requires_grad=True)
print(y)
# 指定在哪一个维度上进行Softmax操作，比如有两个维度：[batch, feature],
# 第一个维度为batch，第二个维度为feature,feature为一个三个值的向量[1, 2, 3],
# 则我们指定在第二个维度上进行softmax操作，则[1, 2, 3] => [p1, p2, p3]的概率值
# 因为y只有一个维度，所以下面指定的是在dim=0,第一个维度上进行的操作
p = F.softmax(y, dim=0)
print(p)
