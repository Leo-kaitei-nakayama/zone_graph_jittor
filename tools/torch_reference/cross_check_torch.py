"""Cross-framework equivalence check: torch reference vs the jittor port.

Rebuilds the ORIGINAL torch architecture — with the legacy DGL send/recv
message passing implemented as explicit per-node neighbor means, which is
exactly what dgl computed — copies its weights (including BatchNorm
running statistics) into the jittor models, feeds both frameworks the
same inputs, and reports max |difference|.

Comparison runs in eval mode: models.py documents that MPLayer's
BatchNorm sees per-node instead of per-edge batch statistics while
training, so train-mode parity is not expected; eval-mode outputs must
match to float precision.

Requires torch (CPU is fine):  python cross_check_torch.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import numpy as np
import torch
import torch.nn as tnn
import jittor as jt

from models import GraphNetSoftMax, ZoneEncoder, MPLayer, build_adj_norm

torch.manual_seed(0)
rng = np.random.default_rng(0)


# ---- torch reference: the original architecture, dgl replaced by loops ----

class TorchMPLayer(tnn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = tnn.Sequential(tnn.Linear(in_dim, in_dim),
                                  tnn.BatchNorm1d(in_dim),
                                  tnn.LeakyReLU(negative_slope=0.2))
        self.fc2 = tnn.Sequential(tnn.Linear(in_dim, in_dim),
                                  tnn.BatchNorm1d(in_dim),
                                  tnn.LeakyReLU(negative_slope=0.2))
        self.linear = tnn.Linear(in_dim, out_dim)

    def forward(self, h, neighbors):
        msg = self.fc2(self.fc1(h))
        agg = torch.zeros_like(msg)
        for i, nbrs in enumerate(neighbors):
            if nbrs:
                agg[i] = msg[list(nbrs)].mean(0)
        return self.linear(agg)


class TorchGraphNet(tnn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        def post():
            return tnn.Sequential(tnn.BatchNorm1d(in_dim),
                                  tnn.LeakyReLU(negative_slope=0.2))
        self.mp1, self.post_mp1 = TorchMPLayer(in_dim, in_dim), post()
        self.mp2, self.post_mp2 = TorchMPLayer(in_dim, in_dim), post()
        self.mp3, self.post_mp3 = TorchMPLayer(in_dim, in_dim), post()
        self.mp4, self.post_mp4 = TorchMPLayer(in_dim, in_dim), post()
        self.fc1 = tnn.Sequential(tnn.Linear(in_dim, 128),
                                  tnn.BatchNorm1d(128),
                                  tnn.LeakyReLU(negative_slope=0.2))
        self.fc2 = tnn.Sequential(tnn.Linear(128, 128),
                                  tnn.BatchNorm1d(128),
                                  tnn.LeakyReLU(negative_slope=0.2))
        self.fc_final = tnn.Sequential(tnn.Linear(128, out_dim),
                                       tnn.LogSoftmax(dim=1))

    def forward(self, h, neighbors):
        h = self.post_mp1(self.mp1(h, neighbors))
        h = self.post_mp2(self.mp2(h, neighbors))
        h = self.mp3(h, neighbors)
        h = torch.max(h, dim=0, keepdim=True)[0]
        return self.fc_final(self.fc2(self.fc1(h)))


class TorchZoneEncoder(tnn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = tnn.Sequential(tnn.Conv1d(10, 64, 1), tnn.BatchNorm1d(64),
                                    tnn.LeakyReLU(negative_slope=0.2))
        self.conv2 = tnn.Sequential(tnn.Conv1d(64, 128, 1), tnn.BatchNorm1d(128),
                                    tnn.LeakyReLU(negative_slope=0.2))
        self.conv3 = tnn.Sequential(tnn.Conv1d(128, 128, 1))
        self.fc1 = tnn.Sequential(tnn.Linear(128, 128), tnn.BatchNorm1d(128),
                                  tnn.LeakyReLU(negative_slope=0.2))
        self.fc2 = tnn.Sequential(tnn.Linear(128, 128), tnn.BatchNorm1d(128),
                                  tnn.LeakyReLU(negative_slope=0.2))
        self.fc_final = tnn.Sequential(tnn.Linear(128, out_dim), tnn.BatchNorm1d(out_dim),
                                       tnn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = torch.flatten(torch.max(x, 2, keepdim=True)[0], 1, -1)
        return self.fc_final(self.fc1(x))


def copy_weights(torch_model, jt_model):
    state = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()
             if not k.endswith('num_batches_tracked')}
    jt_model.load_parameters(state)


def neighbor_lists(n, edges):
    nbrs = [set() for _ in range(n)]
    for a, b in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    return nbrs


failures = []
def check(name, diff, tol=1e-4):
    ok = diff < tol
    print(("PASS " if ok else "FAIL ") + name + f"  max diff {diff:.2e}")
    if not ok:
        failures.append(name)


N, P = 6, 500
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)]
nbrs = neighbor_lists(N, edges)
x_np = rng.random((N, 10, P)).astype('float32')
h_np = rng.random((N, 128)).astype('float32')
adj = build_adj_norm(N, edges)

# populate non-trivial BatchNorm running statistics before copying; the
# net's head is warmed with a multi-row batch since train-mode BatchNorm
# rejects the [1,128] a single-graph readout produces
t_enc, t_net = TorchZoneEncoder(128), TorchGraphNet(128, 2)
t_mp = TorchMPLayer(128, 128)
with torch.no_grad():
    for _ in range(3):
        t_enc(torch.from_numpy(rng.random((N, 10, P)).astype('float32')))
        h_w = torch.from_numpy(rng.random((N, 128)).astype('float32'))
        h_w = t_net.post_mp1(t_net.mp1(h_w, nbrs))
        h_w = t_net.post_mp2(t_net.mp2(h_w, nbrs))
        t_net.mp3(h_w, nbrs)
        head = torch.from_numpy(rng.random((4, 128)).astype('float32'))
        t_net.fc_final(t_net.fc2(t_net.fc1(head)))
        t_mp(torch.from_numpy(rng.random((N, 128)).astype('float32')), nbrs)
t_enc.eval(); t_net.eval(); t_mp.eval()

j_enc, j_net, j_mp = ZoneEncoder(P, 128), GraphNetSoftMax(128, 2), MPLayer(128, 128)
copy_weights(t_enc, j_enc)
copy_weights(t_net, j_net)
copy_weights(t_mp, j_mp)
j_enc.eval(); j_net.eval(); j_mp.eval()

with torch.no_grad(), jt.no_grad():
    # 1. ZoneEncoder
    out_t = t_enc(torch.from_numpy(x_np)).numpy()
    out_j = j_enc(jt.array(x_np)).numpy()
    check("ZoneEncoder", np.abs(out_t - out_j).max())

    # 2. single MPLayer: loop semantics vs dense adjacency matmul
    out_t = t_mp(torch.from_numpy(h_np), nbrs).numpy()
    out_j = j_mp(jt.array(h_np), adj).numpy()
    check("MPLayer (loops vs matmul)", np.abs(out_t - out_j).max())

    # 3. full GraphNetSoftMax, single graph
    out_t = t_net(torch.from_numpy(h_np), nbrs).numpy()
    out_j = j_net(jt.array(h_np), adj, graph_sizes=[N]).numpy()
    check("GraphNetSoftMax single graph", np.abs(out_t - out_j).max())

    # 4. batched graphs: torch per-graph vs jittor block-diagonal batch
    N2 = 4
    edges2 = [(0, 1), (1, 2), (2, 3)]
    h2_np = rng.random((N2, 128)).astype('float32')
    out_t2 = t_net(torch.from_numpy(h2_np), neighbor_lists(N2, edges2)).numpy()
    A = np.zeros((N + N2, N + N2), dtype='float32')
    A[:N, :N] = adj.numpy()
    A[N:, N:] = build_adj_norm(N2, edges2).numpy()
    out_jb = j_net(jt.concat([jt.array(h_np), jt.array(h2_np)], dim=0),
                   jt.array(A), graph_sizes=[N, N2]).numpy()
    check("batched vs per-graph", np.abs(np.concatenate([out_t, out_t2]) - out_jb).max())

print()
if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("CROSS-CHECK PASSED: torch and jittor outputs match")
