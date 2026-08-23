import sys
sys.path.append('..')

import numpy as np
import jittor as jt
import jittor.nn as nn


def build_adj_norm(num_nodes, edges):
    """Row-normalized dense adjacency (D^-1 A) for mean-aggregation message
    passing. `edges` is an iterable of (i, j) pairs from the networkx zone
    graph; both directions are added, matching the original bidirectional
    dgl graph. A node with no neighbors gets an all-zero row, i.e. its
    aggregated message is zero — same as dgl's update_all mean over an
    empty mailbox."""
    A = np.zeros((num_nodes, num_nodes), dtype='float32')
    for i, j in edges:
        A[i][j] = 1.0
        A[j][i] = 1.0
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return jt.array(A / deg)


class GraphNetSoftMax(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphNetSoftMax, self).__init__()
        self.node_feat_size = in_dim

        self.mp1 = MPLayer(self.node_feat_size, self.node_feat_size)
        self.post_mp1 = nn.Sequential(
                            nn.BatchNorm1d(self.node_feat_size),
                            nn.LeakyReLU(scale=0.2))

        self.mp2 = MPLayer(self.node_feat_size, self.node_feat_size)
        self.post_mp2 = nn.Sequential(
                            nn.BatchNorm1d(self.node_feat_size),
                            nn.LeakyReLU(scale=0.2))

        self.mp3 = MPLayer(self.node_feat_size, self.node_feat_size)
        self.post_mp3 = nn.Sequential(
                            nn.BatchNorm1d(self.node_feat_size),
                            nn.LeakyReLU(scale=0.2))

        # mp4/post_mp4 and post_mp3 are never used in execute(), matching
        # the original torch model; kept so checkpoint keys line up 1:1.
        self.mp4 = MPLayer(self.node_feat_size, self.node_feat_size)
        self.post_mp4 = nn.Sequential(
                            nn.BatchNorm1d(self.node_feat_size),
                            nn.LeakyReLU(scale=0.2))

        self.fc1 = nn.Sequential(nn.Linear(self.node_feat_size, 128),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(scale=0.2))

        self.fc2 = nn.Sequential(nn.Linear(128, 128),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(scale=0.2))

        # Sequential keeps the parameter key 'fc_final.0.*' identical to the
        # torch checkpoint; the original's LogSoftmax moved into execute()
        # because jittor has no LogSoftmax module, only nn.log_softmax.
        self.fc_final = nn.Sequential(nn.Linear(128, out_dim))

    def execute(self, h, adj_norm, graph_sizes=None):
        h = self.mp1(h, adj_norm)
        h = self.post_mp1(h)

        h = self.mp2(h, adj_norm)
        h = self.post_mp2(h)

        h = self.mp3(h, adj_norm)

        h = self.readout(h, graph_sizes)

        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc_final(h)

        return nn.log_softmax(h, dim=1)

    def readout(self, node_feats, graph_sizes=None):
        # Replaces dgl.max_nodes: per-graph elementwise max over nodes.
        # graph_sizes lists the node count of each candidate graph when
        # several are stacked into one block-diagonal batch (the dgl.batch
        # case); None means node_feats holds a single graph.
        if graph_sizes is None:
            return jt.max(node_feats, dim=0, keepdims=True)
        outs = []
        start = 0
        for n in graph_sizes:
            outs.append(jt.max(node_feats[start:start + n], dim=0))
            start += n
        return jt.stack(outs, dim=0)


class MPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MPLayer, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(in_dim, in_dim),
                                 nn.BatchNorm1d(in_dim),
                                 nn.LeakyReLU(scale=0.2),
                                 )

        self.fc2 = nn.Sequential(nn.Linear(in_dim, in_dim),
                                 nn.BatchNorm1d(in_dim),
                                 nn.LeakyReLU(scale=0.2),
                                 )

        self.linear = nn.Linear(in_dim, out_dim)

    def execute(self, h, adj_norm):
        # dgl message = fc2(fc1(h_src)), reduce = mean of incoming messages:
        # the row-normalized adjacency matmul computes both at once.
        # One deviation from the torch original: its message fn ran on
        # edge-stacked features, so BatchNorm saw each node once per
        # outgoing edge; here BatchNorm sees each node once. Identical in
        # eval mode, slightly different batch statistics while training.
        msg = self.fc2(self.fc1(h))
        n = adj_norm.shape[0]
        if msg.shape[0] == n:
            agg = jt.matmul(adj_norm, msg)
        else:
            # Shared-adjacency batch: every candidate extrusion of one zone
            # graph has identical topology, so instead of a block-diagonal
            # [B*N, B*N] matrix (48GB at a few hundred candidates) apply the
            # single [N, N] to a [B, N, C] stack. Same arithmetic, and the
            # concatenated layout means BatchNorm still sees all B*N rows.
            b = msg.shape[0] // n
            agg = jt.matmul(adj_norm, msg.reshape((b, n, -1))).reshape((msg.shape[0], -1))
        return self.linear(agg)


class ZoneEncoder(nn.Module):
    def __init__(self, point_num, out_dim):
        super(ZoneEncoder, self).__init__()
        self.point_num = point_num

        self.conv1 = nn.Sequential(nn.Conv1d(10, 64, 1),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(scale=0.2))

        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(scale=0.2))

        self.conv3 = nn.Sequential(nn.Conv1d(128, 128, 1),
                                   )

        self.fc1 = nn.Sequential(nn.Linear(128, 128),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(scale=0.2))

        # fc2 is never used in execute(), matching the original torch
        # model; kept so checkpoint keys line up 1:1.
        self.fc2 = nn.Sequential(nn.Linear(128, 128),
                                 nn.BatchNorm1d(128),
                                 nn.LeakyReLU(scale=0.2))

        self.fc_final = nn.Sequential(nn.Linear(128, out_dim),
                                      nn.BatchNorm1d(out_dim),
                                      nn.LeakyReLU(scale=0.2))

    def execute(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # jt.max returns the values directly (no (values, indices) tuple),
        # and the keyword is keepdims, not torch's keepdim.
        global_x = jt.max(x, dim=2, keepdims=True)
        x = jt.flatten(global_x, 1, -1)

        x = self.fc1(x)
        x = self.fc_final(x)
        return x
