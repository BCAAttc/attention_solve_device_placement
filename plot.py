import os
import numpy as np
import torch
from utils import load_model
size = 20
# model, _ = load_model('pretrained/tsp_100/')
# model, _ = load_model('outputs/tsp_20/msc20_rollout_20220715T213825/',v=20)
model, _ = load_model('outputs/tsp_20/msc100_rollout_20220725T164725/',v=size)
model.eval()  # Put in evaluation mode to not track gradients
# def general_mc_data(nodes):
#     X = np.random.rand(nodes ** 2).reshape(nodes, nodes)
#     X = np.triu(X)
#     X += X.T - np.diag(X.diagonal())
#     for i in range(nodes):
#         X[i,i]=0
#     print(X)
num_samples = 1

# xy = []
# for j in range(num_samples):
X = np.random.rand(size ** 2).reshape(size, size)
X = np.triu(X)
X += X.T - np.diag(X.diagonal())
for i in range(size):
    X[i, i] = 10
# print(X
xy=torch.FloatTensor(X)

# xy = np.array(general_mc_data(20))
# xy = np.random.rand(100, 2)
# xy1 = np.random.rand(20, 2)
torch.set_printoptions(precision=3)
print(xy,xy.shape)
# print(xy1,xy1.shape)

def make_oracle(model, xy, temperature=1.0):
    num_nodes = xy.shape[0]

    xyt = torch.tensor(xy).float()[None]  # Add batch dimension
    print('xyt', xyt.shape)
    with torch.no_grad():  # Inference only
        embeddings, _ = model.embedder(model._init_embed(xyt))
        print('embeddings',embeddings.shape,embeddings)
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = model._precompute(embeddings)
        print('f', fixed)
    def oracle(tour):
        with torch.no_grad():  # Inference only
            # Input tour with 0 based indices
            # Output vector with probabilities for locations not in tour
            tour = torch.tensor(tour).long()
            print(tour.shape,'tour')
            if len(tour) == 0:
                step_context = model.W_placeholder
            else:

                step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)
                tour = tour + torch.tensor([0, 5, 10, 15])
            print('step_context',step_context.shape,step_context)
            # Compute query = context node embedding, add batch and step dimensions (both 1)
            # query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])
            query = fixed.context_node_projected + model.project_step_context(step_context[None, :])

            print('query',query.shape,query)
            # Create the mask and convert to bool depending on PyTorch version
            mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0

            mask[tour] = 1
            mask = mask[None, None, :]  # Add batch and step dimension

            log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
            # p = torch.softmax(log_p / temperature, -1)[0, 0]
            p = torch.softmax(log_p / temperature, -1)[0]

            print(p,tour)
            # assert (p[tour] == 0).all()
            # assert (p.sum() - 1).abs() < 1e-5
            # assert np.allclose(p.sum().item(), 1)
        return p.numpy()

    return oracle


oracle = make_oracle(model, xy)

sample = False
tour = []
tour_p = []
while (len(tour) < len(xy)//4):
    p = oracle(tour)
    print('p',p.shape,p)
    if sample:
        # Advertising the Gumbel-Max trick
        g = -np.log(-np.log(np.random.rand(*p.shape)))
        i = np.argmax(np.log(p) + g)
        # i = np.random.multinomial(1, p)
    else:
        # Greedy
        i = np.argmax(p,1)
        # _, selected_list = p.max(1)
    # for ii in i:
    tour.append(i)
    tour_p.append(p)

print('tour',tour)
tou = []
for t in tour:
    t = t+[0,5,10,15]
    for tt in t:
        tou.append(tt)
# print(xy.shape)
print('tou',tou)
d = xy.gather(0, torch.tensor(tou).unsqueeze(-1).expand_as(xy))
# print(d,d.shape)
d0,d1 = torch.where(d==10)
dp_size = 4
pipeline_size= 5
d0_concat_idx = d0[dp_size:]
d1_concat_idx = d1[:-dp_size]
# d2_concat_idx = d2.reshape((d.shape[0], d.shape[1]))[:-dp_size].reshape((d.shape[0]*(d.shape[1]-dp_size)))
# d0_concat_idx = d0_concat_idx +
# for p in d[0]:
#     print(p)
d3 = d[d0_concat_idx, d1_concat_idx].reshape( dp_size,pipeline_size-1)
cost_concat_allline = d3.sum(0)
# cost_concat = torch.max(cost_concat_allline, 1).values
# cost_concat = cost_concat_allline.sum(0)
cost_concat = torch.max(cost_concat_allline)

c = torch.chunk(d, pipeline_size, 0)

cost_concat_all = cost_concat
for dp in c:
# dp = c[0]
    idx_d0, idx_d1 = torch.where(dp == 10)
    # idx_d0 = idx_d0.reshape((dp.shape[0], dp.shape[1]))[:, :-1].reshape((dp.shape[0] * (dp.shape[1] - 1)))
    # idx_d1 = idx_d1.reshape((dp.shape[0], dp.shape[1]))[:, :-1].reshape((dp.shape[0] * (dp.shape[1] - 1)))
    # idx_d2 = idx_d2.reshape((dp.shape[0], dp.shape[1]))[:, :-1].reshape((dp.shape[0] * (dp.shape[1] - 1)))
    # idx_d0 = idx_d0
    # idx_d1 = idx_d1
    # idx_d2 = idx_d2.reshape((dp.shape[0], dp.shape[1]))
    idx_d1[-1] = idx_d1[0] - 1
    idx_d1 = idx_d1 + 1
    dp_delay = dp[idx_d0, idx_d1]
    cost_dp = torch.max(dp_delay, 0).values
    cost_concat_all = cost_concat_all + cost_dp

# d0 = d0.reshape((d.shape[0], d.shape[1]))[:,:-1].reshape((d.shape[0]*(d.shape[1]-1)))
# d0 = d0[:-1]
# d1 = d1[:-1]
# # d1 = d1.reshape((d.shape[0], d.shape[1]))[:,:-1].reshape((d.shape[0]*(d.shape[1]-1)))
# # d2 = d2.reshape((d.shape[0], d.shape[1]))[:,:-1].reshape((d.shape[0]*(d.shape[1]-1)))
# # print('d0',d0.cpu().numpy())
# # d0 = d0.cpu().numpy()
# d0 = d0+1
# # d0_1 = d0.re
# # d2 = d2
# d3 = d[d0,d1]
# cost = torch.sum(d3)
cost = cost_concat_all
# print(cost)


np.set_printoptions(suppress=True)
#
# # 贪心算法
# c = np.round(xy.numpy(),4)
# c[c==0]=99
# # print(c)
# tou = [k for k in range(xy.shape[0])]
# # tou.pop(0)
# # print(tou)
# cos = 0
# state = 0
# next = 0
# tour2 = []
# tou.remove(next)
# state = next
# tour2.append(state)
# c[:, state] = 99
#
# while len(tou) != 0 :
#     next = np.argmin(c[state])
#     cos = cos + np.min(c[state])
#
#     tou.remove(next)
#     state = next
#     tour2.append(state)
#     c[:, state] = 99






# cos =cos +
# print('tour',tour)
# print('tour2',tour2)
# print('tanxin:',cos)
print('msc:',cost)

# from matplotlib import pyplot as plt
#
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Rectangle
# from matplotlib.lines import Line2D
#
#
# # Code inspired by Google OR Tools plot:
# # https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py
#
# def plot_tsp(xy, tour, ax1):
#     """
#     Plot the TSP tour on matplotlib axis ax1.
#     """
#
#     ax1.set_xlim(0, 1)
#     ax1.set_ylim(0, 1)
#
#     xs, ys = xy[tour].transpose()
#     xs, ys = xy[tour].transpose()
#     dx = np.roll(xs, -1) - xs
#     dy = np.roll(ys, -1) - ys
#     d = np.sqrt(dx * dx + dy * dy)
#     lengths = d.cumsum()
#
#     # Scatter nodes
#     ax1.scatter(xs, ys, s=40, color='blue')
#     # Starting node
#     ax1.scatter([xs[0]], [ys[0]], s=100, color='red')
#
#     # Arcs
#     qv = ax1.quiver(
#         xs, ys, dx, dy,
#         scale_units='xy',
#         angles='xy',
#         scale=1,
#     )
#
#     ax1.set_title('{} nodes, total length {:.2f}'.format(len(tour), lengths[-1]))
#
#
# fig, ax = plt.subplots(figsize=(10, 10))
# plot_tsp(xy, tour, ax)
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Rectangle
# from matplotlib.lines import Line2D
# from IPython.display import HTML
#
# from celluloid import Camera  # pip install celluloid
#
#
# def format_prob(prob):
#     return ('{:.6f}' if prob > 1e-5 else '{:.2E}').format(prob)
#
#
# def plot_tsp_ani(xy, tour, tour_p=None, max_steps=1000):
#     n = len(tour)
#     fig, ax1 = plt.subplots(figsize=(10, 10))
#     xs, ys = xy[tour].transpose()
#     dx = np.roll(xs, -1) - xs
#     dy = np.roll(ys, -1) - ys
#     d = np.sqrt(dx * dx + dy * dy)
#     lengths = d.cumsum()
#
#     ax1.set_xlim(0, 1)
#     ax1.set_ylim(0, 1)
#
#     camera = Camera(fig)
#
#     total_length = 0
#     cum_log_prob = 0
#     for i in range(n + 1):
#         for plot_probs in [False] if tour_p is None or i >= n else [False, True]:
#             # Title
#             title = 'Nodes: {:3d}, length: {:.4f}, prob: {}'.format(
#                 i, lengths[i - 2] if i > 1 else 0., format_prob(np.exp(cum_log_prob))
#             )
#             ax1.text(0.6, 0.97, title, transform=ax.transAxes)
#
#             # First print current node and next candidates
#             ax1.scatter(xs, ys, s=40, color='blue')
#
#             if i > 0:
#                 ax1.scatter([xs[i - 1]], [ys[i - 1]], s=100, color='red')
#             if i > 1:
#                 qv = ax1.quiver(
#                     xs[:i - 1],
#                     ys[:i - 1],
#                     dx[:i - 1],
#                     dy[:i - 1],
#                     scale_units='xy',
#                     angles='xy',
#                     scale=1,
#                 )
#             if plot_probs:
#                 prob_rects = [Rectangle((x, y), 0.01, 0.1 * p) for (x, y), p in zip(xy, tour_p[i]) if p > 0.01]
#                 pc = PatchCollection(prob_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
#                 ax1.add_collection(pc)
#             camera.snap()
#         if i < n and tour_p is not None:
#             # Add cumulative_probability
#             cum_log_prob += np.log(tour_p[i][tour[i]])
#         if i > max_steps:
#             break
#
#     # Plot final tour
#     # Scatter nodes
#     ax1.scatter(xs, ys, s=40, color='blue')
#     # Starting node
#     ax1.scatter([xs[0]], [ys[0]], s=100, color='red')
#
#     # Arcs
#     qv = ax1.quiver(
#         xs, ys, dx, dy,
#         scale_units='xy',
#         angles='xy',
#         scale=1,
#     )
#     if tour_p is not None:
#         # Note this does not use stable logsumexp trick
#         cum_log_prob = format_prob(np.exp(sum([np.log(p[node]) for node, p in zip(tour, tour_p)])))
#     else:
#         cum_log_prob = '?'
#     ax1.set_title('{} nodes, total length {:.4f}, prob: {}'.format(len(tour), lengths[-1], cum_log_prob))
#
#     camera.snap()
#
#     return camera
#
#
# animation = plot_tsp_ani(xy, tour, tour_p).animate(interval=500)
# animation.save('images/tsp20.gif', writer='imagemagick', fps=2)  # requires imagemagick
# # compress by running 'convert tsp.gif -strip -coalesce -layers Optimize tsp.gif'
# # HTML(animation.to_html5_video())  # requires ffmpeg