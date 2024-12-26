from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search
import numpy as np

class TSP(object):
    def __init__(self,graph_size):
        self.SIZE = graph_size
    NAME = 'tsp'
    # SIZE = graph_size

    @staticmethod
    def get_costs(dataset, pi,pipeline_size,dp_size):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"
        # print(dataset.shape)
        pipeline_size = 10
        dp_size = 2
        # Gather dataset in order of tour  # d： 将dataset按照pi顺序排列
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        # print(d.shape)
        # cost = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)


        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        d0,d1,d2 = torch.where(d==10)
        # d0 = d0.reshape((d.shape[0], d.shape[1]))[:,:-1].reshape((d.shape[0]*(d.shape[1]-1)))
        # d1 = d1.reshape((d.shape[0], d.shape[1]))[:,:-1].reshape((d.shape[0]*(d.shape[1]-1)))
        # d2 = d2.reshape((d.shape[0], d.shape[1]))[:,:-1].reshape((d.shape[0]*(d.shape[1]-1)))
        # d1 = d1+1
        # d3 = d[d0,d1,d2].reshape((d.shape[0], d.shape[1]-1))
        # cost = d3.sum(1)


        #  ms config
        pipeline_size = pipeline_size
        dp_size = dp_size
        # cost for ms

        #count idx
        # concat_idx = []
        # for j in range(d.shape[0]):
        #     for i in range(pipeline_size - 1):
        #         concat_idx.append(((dp_size - 1) + dp_size * i) + j * d.shape[1])
        #
        # # print(concat_idx)
        # # print(d)
        #
        # # count pipeline timedelay
        # concat_idx = torch.LongTensor(concat_idx)
        # d0_concat_idx = d0[concat_idx]
        # d1_concat_idx = d1[concat_idx + 1]
        # d2_concat_idx = d2[concat_idx]
        d0_concat_idx = d0.reshape((d.shape[0], d.shape[1]))[:,:-dp_size].reshape((d.shape[0]*(d.shape[1]-dp_size)))
        # d1_concat_idx = d1.reshape((d.shape[0], d.shape[1]))[:,dp_size:].reshape((d.shape[0]*(d.shape[1]-dp_size)))
        d1_concat_idx = d1.reshape((d.shape[0], pipeline_size,dp_size))[:,1:,:].reshape((d.shape[0]*(d.shape[1]-dp_size)))


        d2_concat_idx = d2.reshape((d.shape[0], d.shape[1]))[:,:-dp_size].reshape((d.shape[0]*(d.shape[1]-dp_size)))
        # for p in d[0]:
        #     print(p)
        d3 = d[d0_concat_idx, d1_concat_idx, d2_concat_idx].reshape((d.shape[0], dp_size,pipeline_size-1))
        # cost_concat_allline = d3.sum(2)
        cost_concat_allline = d3.sum(1)  ##  !!!!!!!维度调试
        # cost_concat = torch.max(cost_concat_allline, 1).values

        # cost_concat = torch.max(cost_concat_allline, 1).values + cost_concat_allline.mean(1)
        cost_concat = cost_concat_allline.sum(1)

        c = torch.chunk(d, pipeline_size, 1)

        cost_concat_all = cost_concat
        cost_dp_all = []
        if dp_size==1:
            return cost_concat_all, None

        for dp in c:
        # dp = c[0]
            idx_d0, idx_d1, idx_d2 = torch.where(dp == 10)
            # idx_d0 = idx_d0.reshape((dp.shape[0], dp.shape[1]))[:, :-1].reshape((dp.shape[0] * (dp.shape[1] - 1)))
            # idx_d1 = idx_d1.reshape((dp.shape[0], dp.shape[1]))[:, :-1].reshape((dp.shape[0] * (dp.shape[1] - 1)))
            # idx_d2 = idx_d2.reshape((dp.shape[0], dp.shape[1]))[:, :-1].reshape((dp.shape[0] * (dp.shape[1] - 1)))
            cc = dp.cpu().numpy()
            idx_d0 = idx_d0.reshape((dp.shape[0], dp.shape[1]))
            idx_d1 = idx_d1.reshape((dp.shape[0], dp.shape[1]))
            idx_d2 = idx_d2.reshape((dp.shape[0], dp.shape[1]))
            idx_d1[:, -1] = idx_d1[:, 0] - 1
            idx_d1 = idx_d1 + 1
            dp_delay = dp[idx_d0, idx_d1, idx_d2]
            # cost_dp = torch.max(dp_delay, 1).values
            # cost_dp = dp_delay.mean(1)
            cost_dp = dp_delay.sum(1)
            # cost_dp = torch.max(dp_delay, 1).values + dp_delay.mean(1)
            # cost_dp = torch.max(dp_delay, 1).values + dp_delay.mean(1)

            cost_dp_all.append(cost_dp)
        ts = torch.stack(cost_dp_all)
        # cda_max = ts.max(0).values
        cda_max = ts.sum(0)
        # cda_max = ts.max(0).values + ts.mean(0)

        cost = cost_concat_all + cda_max  # 比例有待修改

        # cost_concat_all = cost_concat_all + cost_dp

        return cost, None


        # return cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)



class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            # 随机生成数据
            # self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

            self.data = []
            for j in range(num_samples):
                X = np.random.rand(size ** 2).reshape(size, size)
                X = np.triu(X)
                X += X.T - np.diag(X.diagonal())
                for i in range(size):
                    X[i, i] = 10
                # print(X)
                self.data.append(torch.FloatTensor(X))


        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
