"""Torch-vs-jittor parity on a REAL preprocessed zone graph.

Loads one (ZoneGraph, Extrusion) pair produced by train_preprocess.py,
builds the exact per-zone feature tensor the agent uses, runs it through
the jittor models and a plain-torch reference of the original
architecture with the SAME weights (exported from jittor), and reports
max |difference| on the encoder output and the final log-probabilities.

    python real_check.py --graph <processed>/<seq_id>/train/0_1

Requires torch (CPU is fine) in addition to the project environment.
Comparison runs in eval mode (see models.py on train-mode BatchNorm).
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

parser = argparse.ArgumentParser()
parser.add_argument('--graph', required=True,
                    help='path prefix of a processed pair, e.g. .../train/0_1')
args = parser.parse_args()

import numpy as np
import joblib
import torch
import torch.nn as tnn
import jittor as jt

from objects import *
from models import ZoneEncoder, GraphNetSoftMax
from agent import Agent

zone_graph = joblib.load(args.graph + '_g.joblib')
extrusion = joblib.load(args.graph + '_e.joblib')
zone_graph.encode_with_extrusion(extrusion)
G = zone_graph.zone_graph
n = len(G.nodes)
print(f'loaded real graph: {n} zones, {G.number_of_edges()} edges, '
      f'extrusion zones {extrusion.zone_indices}, bool {extrusion.bool_type}')

# ---- features + adjacency, exactly as agent.encode_zone_graph builds them
pos = nx.get_node_attributes(G, 'shape_positions')
nrm = nx.get_node_attributes(G, 'shape_normals')
cur = nx.get_node_attributes(G, 'in_current')
tgt = nx.get_node_attributes(G, 'in_target')
ext = nx.get_node_attributes(G, 'in_extrusion')
boo = nx.get_node_attributes(G, 'bool')
P = pos[0].shape[0]
def tiled(attr):
    vals = np.array([attr[i] for i in range(n)], dtype='float32')
    return np.broadcast_to(vals[:, None, None], (n, P, 1))
feats = np.concatenate([np.stack([pos[i] for i in range(n)]).astype('float32'),
                        np.stack([nrm[i] for i in range(n)]).astype('float32'),
                        tiled(cur), tiled(tgt), tiled(ext), tiled(boo)], axis=2)
feats = np.ascontiguousarray(feats.transpose(0, 2, 1))

A = np.zeros((n, n), dtype='float32')
for e in G.edges:
    A[e[0]][e[1]] = 1.0
    A[e[1]][e[0]] = 1.0
deg = A.sum(axis=1, keepdims=True)
deg[deg == 0] = 1.0
adj = A / deg
nbrs = [set(np.nonzero(A[i])[0].tolist()) for i in range(n)]

# ---- torch reference of the original architecture --------------------------
class TMP(tnn.Module):
    def __init__(self, d):
        super().__init__()
        blk = lambda: tnn.Sequential(tnn.Linear(d, d), tnn.BatchNorm1d(d),
                                     tnn.LeakyReLU(negative_slope=0.2))
        self.fc1, self.fc2, self.linear = blk(), blk(), tnn.Linear(d, d)
    def forward(self, h):
        msg = self.fc2(self.fc1(h))
        agg = torch.zeros_like(msg)
        for i, nb in enumerate(nbrs):
            if nb:
                agg[i] = msg[list(nb)].mean(0)
        return self.linear(agg)

class TNet(tnn.Module):
    def __init__(self, d, out):
        super().__init__()
        post = lambda: tnn.Sequential(tnn.BatchNorm1d(d),
                                      tnn.LeakyReLU(negative_slope=0.2))
        self.mp1, self.post_mp1 = TMP(d), post()
        self.mp2, self.post_mp2 = TMP(d), post()
        self.mp3, self.post_mp3 = TMP(d), post()
        self.mp4, self.post_mp4 = TMP(d), post()
        fc = lambda i, o: tnn.Sequential(tnn.Linear(i, o), tnn.BatchNorm1d(o),
                                         tnn.LeakyReLU(negative_slope=0.2))
        self.fc1, self.fc2 = fc(d, 128), fc(128, 128)
        self.fc_final = tnn.Sequential(tnn.Linear(128, out), tnn.LogSoftmax(dim=1))
    def forward(self, h):
        h = self.post_mp1(self.mp1(h))
        h = self.post_mp2(self.mp2(h))
        h = self.mp3(h)
        h = torch.max(h, dim=0, keepdim=True)[0]
        return self.fc_final(self.fc2(self.fc1(h)))

class TEnc(tnn.Module):
    def __init__(self, out):
        super().__init__()
        self.conv1 = tnn.Sequential(tnn.Conv1d(10, 64, 1), tnn.BatchNorm1d(64),
                                    tnn.LeakyReLU(negative_slope=0.2))
        self.conv2 = tnn.Sequential(tnn.Conv1d(64, 128, 1), tnn.BatchNorm1d(128),
                                    tnn.LeakyReLU(negative_slope=0.2))
        self.conv3 = tnn.Sequential(tnn.Conv1d(128, 128, 1))
        fc = lambda: tnn.Sequential(tnn.Linear(128, 128), tnn.BatchNorm1d(128),
                                    tnn.LeakyReLU(negative_slope=0.2))
        self.fc1, self.fc2 = fc(), fc()
        self.fc_final = tnn.Sequential(tnn.Linear(128, out), tnn.BatchNorm1d(out),
                                       tnn.LeakyReLU(negative_slope=0.2))
    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x = torch.flatten(torch.max(x, 2, keepdim=True)[0], 1, -1)
        return self.fc_final(self.fc1(x))

# ---- share the jittor agent's weights with the torch reference -------------
agent = Agent('.')
j_enc, j_net = agent.zone_encoder, agent.decision_maker
t_enc, t_net = TEnc(128), TNet(128, 2)
for t_model, j_model in [(t_enc, j_enc), (t_net, j_net)]:
    jstate = j_model.state_dict(to='numpy')
    tstate = t_model.state_dict()
    for k in tstate:
        if k.endswith('num_batches_tracked'):
            continue
        tstate[k] = torch.from_numpy(np.array(jstate[k]))
    t_model.load_state_dict(tstate)
j_enc.eval(); j_net.eval(); t_enc.eval(); t_net.eval()

with torch.no_grad(), jt.no_grad():
    h_j = j_enc(jt.array(feats))
    h_t = t_enc(torch.from_numpy(feats))
    d_enc = np.abs(h_j.numpy() - h_t.numpy()).max()

    out_j = j_net(h_j, jt.array(adj), graph_sizes=[n]).numpy()
    out_t = t_net(h_t).numpy()
    d_net = np.abs(out_j - out_t).max()

p_j = np.exp(out_j)[0]
p_t = np.exp(out_t)[0]
print(f'encoder    max |torch - jittor| = {d_enc:.2e}')
print(f'full net   max |torch - jittor| = {d_net:.2e}')
print(f'P(good)    torch={p_t[1]:.6f}  jittor={p_j[1]:.6f}')
ok = d_enc < 1e-4 and d_net < 1e-4
print('REAL-DATA PARITY ' + ('PASSED' if ok else 'FAILED'))
sys.exit(0 if ok else 1)
