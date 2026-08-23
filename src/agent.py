import sys
sys.path.append('..')
import os
import numpy as np
import jittor as jt
import jittor.nn as nn
from objects import *
from models import *
import hyperparameters as hp

jt.flags.use_cuda = 1 if jt.has_cuda else 0

def to_numpy(item):
    return item.numpy()

def to_tensor(item):
    return jt.array(np.asarray(item, dtype='float32'))

class Agent():
    def __init__(self, folder=None):
        self.folder = folder

        self.zone_encoder = ZoneEncoder(zone_sample_num, hp.gnn_node_feat_dim)
        self.decision_maker = GraphNetSoftMax(hp.gnn_node_feat_dim, 2)

        self.optim_extrusion = jt.optim.Adam(
            self.zone_encoder.parameters() + self.decision_maker.parameters(),
            lr=hp.learning_rate_optim_extrusion)

    def encode_zone_graph(self, zone_graph):
        """Replaces the DGLGraph of the torch version: returns
        (h, adj_norm) where h is the [N, 128] encoded node features and
        adj_norm the row-normalized adjacency as a numpy array, kept in
        numpy so make_decision can assemble a block-diagonal batch."""
        G = zone_graph.zone_graph
        n = len(G.nodes)

        node_shape_positions = nx.get_node_attributes(G, 'shape_positions')
        node_shape_normals = nx.get_node_attributes(G, 'shape_normals')
        node_cur_state_features = nx.get_node_attributes(G, 'in_current')
        node_tgt_state_features = nx.get_node_attributes(G, 'in_target')
        node_extru_state_features = nx.get_node_attributes(G, 'in_extrusion')
        node_bool_features = nx.get_node_attributes(G, 'bool')

        point_num = node_shape_positions[0].shape[0]

        positions = np.stack(
            [node_shape_positions[i] for i in range(n)]).astype('float32')
        normals = np.stack(
            [node_shape_normals[i] for i in range(n)]).astype('float32')

        def tiled(attr):
            vals = np.array([attr[i] for i in range(n)], dtype='float32')
            return np.broadcast_to(vals[:, None, None], (n, point_num, 1))

        # channel order matches the torch version's cat: positions, normals,
        # current, target, extrusion, bool  ->  [N, point_num, 10]
        features = np.concatenate(
            [positions, normals,
             tiled(node_cur_state_features), tiled(node_tgt_state_features),
             tiled(node_extru_state_features), tiled(node_bool_features)],
            axis=2)
        features = np.ascontiguousarray(features.transpose(0, 2, 1))

        h = self.zone_encoder(jt.array(features))

        # both edge directions, matching the original g.add_edges(src,dst) +
        # g.add_edges(dst,src); mirrors models.build_adj_norm but stays numpy
        A = np.zeros((n, n), dtype='float32')
        for e in G.edges:
            A[e[0]][e[1]] = 1.0
            A[e[1]][e[0]] = 1.0
        deg = A.sum(axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        adj_norm = A / deg

        return h, adj_norm

    def make_decision(self, g_encs):
        # replaces dgl.batch: concatenate node features, per-graph node
        # counts for the readout, and one adjacency for the whole batch
        sizes = [enc[0].shape[0] for enc in g_encs]
        h = jt.concat([enc[0] for enc in g_encs], dim=0)

        # Ranking candidates of a single zone graph (evaluation, and the
        # common case) reuses one topology, so pass that [N, N] alone —
        # MPLayer applies it per graph. A block-diagonal matrix here would
        # be (B*N)^2 and reaches tens of GB at a few hundred candidates.
        first = g_encs[0][1]
        shared = all(enc[1] is first or
                     (enc[1].shape == first.shape and np.array_equal(enc[1], first))
                     for enc in g_encs)
        if shared:
            adj = jt.array(first)
        else:
            total = sum(sizes)
            A = np.zeros((total, total), dtype='float32')
            start = 0
            for h_i, adj_i in g_encs:
                k = adj_i.shape[0]
                A[start:start + k, start:start + k] = adj_i
                start += k
            adj = jt.array(A)

        prob = self.decision_maker(h, adj, graph_sizes=sizes)
        prob = jt.exp(prob)
        return prob

    def update_by_extrusion(self, labels, gs):
        print('update agent weights')

        g_encs = []
        for i in range(len(gs)):
            g_enc = self.encode_zone_graph(gs[i])
            g_encs.append(g_enc)

        prob = self.make_decision(g_encs)

        gathered_prob = []
        for i in range(len(prob)):
            gathered_prob.append(prob[i, int(labels[i].item())])
        prob = jt.stack(gathered_prob)

        gamma = 0.5
        loss = jt.mean(-((1 - prob) ** gamma) * jt.log(prob))
        # jittor's optimizer.step(loss) does zero_grad + backward + step
        self.optim_extrusion.step(loss)

        return loss.item()

    def save_weights(self):
        self.zone_encoder.save(os.path.join(self.folder, "zone_encoder.pkl"))
        self.decision_maker.save(os.path.join(self.folder, "decision_maker.pkl"))

    def save_best_weights(self):
        self.zone_encoder.save(os.path.join(self.folder, "best_zone_encoder.pkl"))
        self.decision_maker.save(os.path.join(self.folder, "best_decision_maker.pkl"))

    def load_weights(self):
        self.zone_encoder.load(os.path.join(self.folder, "zone_encoder.pkl"))
        self.decision_maker.load(os.path.join(self.folder, "decision_maker.pkl"))

    def load_best_weights(self):
        # the torch original loaded the non-best files here by mistake;
        # fixed to load what save_best_weights actually writes
        self.zone_encoder.load(os.path.join(self.folder, "best_zone_encoder.pkl"))
        self.decision_maker.load(os.path.join(self.folder, "best_decision_maker.pkl"))
