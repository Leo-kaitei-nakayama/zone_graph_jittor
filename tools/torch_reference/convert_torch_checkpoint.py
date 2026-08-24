"""Convert PyTorch checkpoints (zone_graph_fix) into jittor weights.

The jittor modules keep the torch attribute names and parameter shapes, so
the state dicts map key-for-key; only torch's num_batches_tracked buffers
are dropped (jittor has no equivalent).

Running the converted weights makes a framework-only comparison possible:
same network, same weights, same data, two frameworks — any difference is
purely numerical.

    python convert_torch_checkpoint.py --torch_dir <fix>/src/train_output \
        --out_dir train_output_from_torch [--best] [--verify]

--verify additionally rebuilds the original architecture in torch (with
DGL's update_all semantics written as explicit neighbor means) and reports
the maximum output difference on random graphs. Needs torch installed.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import jittor as jt
from models import ZoneEncoder, GraphNetSoftMax, build_adj_norm

# not imported from hyperparameters: that module pulls in FreeCAD, which the
# conversion does not need
FEAT_DIM = 128


def torch_state_to_numpy(path):
    # .npz lets the conversion run in an environment without torch: dump the
    # checkpoints from the torch environment first with
    #   python -c "import torch,numpy as np; sd=torch.load('best_zone_encoder.pkl',map_location='cpu'); \
    #              np.savez('best_zone_encoder.npz', **{k:v.numpy() for k,v in sd.items()})"
    if path.endswith('.npz'):
        with np.load(path) as z:
            return {k: z[k] for k in z.files if not k.endswith('num_batches_tracked')}
    import torch
    sd = torch.load(path, map_location='cpu')
    return {k: v.detach().numpy() for k, v in sd.items()
            if not k.endswith('num_batches_tracked')}


def convert(torch_dir, out_dir, best=False, verify=False, point_num=500):
    prefix = 'best_' if best else ''
    enc_path = os.path.join(torch_dir, prefix + 'zone_encoder.pkl')
    dm_path = os.path.join(torch_dir, prefix + 'decision_maker.pkl')
    if not os.path.exists(enc_path) and os.path.exists(enc_path[:-4] + '.npz'):
        enc_path, dm_path = enc_path[:-4] + '.npz', dm_path[:-4] + '.npz'

    enc_state = torch_state_to_numpy(enc_path)
    dm_state = torch_state_to_numpy(dm_path)

    enc = ZoneEncoder(point_num, FEAT_DIM)
    dm = GraphNetSoftMax(FEAT_DIM, 2)

    for name, model, state in (('zone_encoder', enc, enc_state),
                               ('decision_maker', dm, dm_state)):
        own = model.state_dict()
        missing = [k for k in own if k not in state]
        extra = [k for k in state if k not in own]
        mismatched = [k for k in own if k in state
                      and tuple(own[k].shape) != tuple(np.shape(state[k]))]
        print('%s: %d torch tensors -> %d jittor parameters' % (name, len(state), len(own)))
        if missing:
            print('  MISSING in torch checkpoint:', missing)
        if extra:
            print('  EXTRA in torch checkpoint (ignored):', extra)
        if mismatched:
            print('  SHAPE MISMATCH:', [(k, tuple(own[k].shape), np.shape(state[k])) for k in mismatched])
        if missing or mismatched:
            raise SystemExit('conversion aborted: checkpoint does not match the jittor model')
        model.load_parameters(state)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # written under both names so load_weights() and load_best_weights() work
    for p in ('', 'best_'):
        enc.save(os.path.join(out_dir, p + 'zone_encoder.pkl'))
        dm.save(os.path.join(out_dir, p + 'decision_maker.pkl'))
    print('wrote jittor weights to', out_dir)

    if verify:
        verify_against_torch(enc, dm, enc_state, dm_state, point_num)


def verify_against_torch(enc, dm, enc_state, dm_state, point_num):
    import torch
    import torch.nn as tnn

    N = 9
    edges = [(i, (i + 1) % N) for i in range(N)] + [(0, 4), (2, 7)]
    nbrs = [set() for _ in range(N)]
    for a, b in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)

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
            post = lambda: tnn.Sequential(tnn.BatchNorm1d(d), tnn.LeakyReLU(negative_slope=0.2))
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

    t_enc, t_dm = TEnc(FEAT_DIM), TNet(FEAT_DIM, 2)
    t_enc.load_state_dict({k: torch.from_numpy(v) for k, v in enc_state.items()}, strict=False)
    t_dm.load_state_dict({k: torch.from_numpy(v) for k, v in dm_state.items()}, strict=False)
    t_enc.eval(); t_dm.eval()
    enc.eval(); dm.eval()

    rng = np.random.default_rng(0)
    x = rng.random((N, 10, point_num)).astype('float32')
    adj = build_adj_norm(N, edges)

    with torch.no_grad(), jt.no_grad():
        h_t = t_enc(torch.from_numpy(x))
        h_j = enc(jt.array(x))
        d_enc = np.abs(h_t.numpy() - h_j.numpy()).max()
        out_t = t_dm(h_t).numpy()
        out_j = dm(h_j, adj, graph_sizes=[N]).numpy()
        d_net = np.abs(out_t - out_j).max()

    print()
    print('verification with the converted weights (eval mode):')
    print('  encoder    max |torch - jittor| = %.2e' % d_enc)
    print('  full net   max |torch - jittor| = %.2e' % d_net)
    print('  P(good)    torch=%.6f  jittor=%.6f' % (np.exp(out_t)[0][1], np.exp(out_j)[0][1]))
    print('  VERDICT:', 'MATCH' if max(d_enc, d_net) < 1e-4 else 'MISMATCH')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_dir', required=True, type=str)
    parser.add_argument('--out_dir', default='train_output_from_torch', type=str)
    parser.add_argument('--best', action='store_true', help='convert the best_* checkpoints')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--point_num', default=500, type=int)
    args = parser.parse_args()
    convert(args.torch_dir, args.out_dir, args.best, args.verify, args.point_num)
