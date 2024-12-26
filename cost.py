import torch
import numpy as np
# d= torch.rand((1,20,20))
size = 20
d = []
for j in range(2):
    X = np.random.rand(size ** 2).reshape(size, size)
    X = np.triu(X)
    X += X.T - np.diag(X.diagonal())
    for i in range(size):
        X[i, i] = 10
    # print(X)
    # d.append(torch.FloatTensor(X))
    d.append(X)
# print(X
d=torch.tensor(np.array(d))

pipeline_size = 5
dp_size = 4

d0,d1,d2 = torch.where(d==10)

concat_idx =[]
# for i in range(pipeline_size-1):
#     concat_idx.append((dp_size-1)+dp_size*i)
# concat_idx=torch.LongTensor(concat_idx)
# concat_idx =concat_idx+1
for j in range(2):
    for i in range(pipeline_size - 1):
        concat_idx.append(((dp_size - 1) + dp_size * i)+j*size)

print(concat_idx)
print(d)
concat_idx=torch.LongTensor(concat_idx)
d0_concat_idx = d0[concat_idx]
d1_concat_idx = d1[concat_idx+1]
d2_concat_idx = d2[concat_idx]
d3 = d[d0_concat_idx,d1_concat_idx,d2_concat_idx].reshape((d.shape[0], dp_size))
cost_concat = d3.sum(1)

c = torch.chunk(d,pipeline_size,1)

for dp in c:
    dp = c[0]
    idx_d0, idx_d1, idx_d2 = torch.where(dp==10)
    # idx_d0 = idx_d0.reshape((dp.shape[0], dp.shape[1]))[:, :-1].reshape((dp.shape[0] * (dp.shape[1] - 1)))
    # idx_d1 = idx_d1.reshape((dp.shape[0], dp.shape[1]))[:, :-1].reshape((dp.shape[0] * (dp.shape[1] - 1)))
    # idx_d2 = idx_d2.reshape((dp.shape[0], dp.shape[1]))[:, :-1].reshape((dp.shape[0] * (dp.shape[1] - 1)))
    idx_d0 = idx_d0.reshape((dp.shape[0], dp.shape[1]))
    idx_d1 = idx_d1.reshape((dp.shape[0], dp.shape[1]))
    idx_d2 = idx_d2.reshape((dp.shape[0], dp.shape[1]))
    idx_d1[:,-1] = idx_d1[:,0]-1
    idx_d1 = idx_d1 + 1
    cost_dp = torch.max(dp[idx_d0,idx_d1,idx_d2],1).values
    cost_concat = cost_concat +cost_dp
    # d3 = d[d0, d1, d2].reshape((d.shape[0], d.shape[1] - 1))
# aa
# c = torch.chunk(d,4,1)
# c = torch.split(d,4,1)
# print(d,d.shape)
# print(c[0],c[0].shape)