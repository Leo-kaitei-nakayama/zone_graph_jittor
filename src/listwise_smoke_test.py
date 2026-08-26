"""Smoke test for train_listwise.py — runs without FreeCAD or real data.

Verifies the listwise objective against the unchanged base stack:
scores match Agent.make_decision, the loss equals a hand-computed softmax
cross-entropy, candidate subsampling keeps the GT, a few optimizer steps
drive the GT to rank 1 under evaluation.sort_extrusions_by_agent, and the
best-checkpoint round-trip (what rank_eval.py loads) reproduces scores.

    python listwise_smoke_test.py
"""
import sys, os, types, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _stub(name, **attrs):
    # geometry deps aren't needed for this test; stub them only if absent
    try:
        __import__(name)
    except Exception:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

_stub('FreeCAD', Base=types.SimpleNamespace())
_stub('Part')
_stub('trimesh')
_stub('matplotlib')
_stub('matplotlib.pyplot')
_stub('joblib', load=None, dump=None)

import numpy as np
import networkx as nx
import jittor as jt

from objects import Zone, ZoneGraph, Extrusion, zone_sample_num
from agent import Agent
from evaluation import sort_extrusions_by_agent
from train_listwise import (batched_log_probs, listwise_loss,
                            subsample_candidates, train_step, candidate_cap)

rng = np.random.default_rng(0)
jt.set_global_seed(0)

def make_zone_graph(n_zones):
    zg = ZoneGraph()
    zg.zone_graph = nx.Graph()
    for i in range(n_zones):
        zg.zone_graph.add_node(i)
    for i in range(n_zones - 1):
        zg.zone_graph.add_edge(i, i + 1)
    if n_zones > 3:
        zg.zone_graph.add_edge(0, n_zones - 1)

    for i in range(n_zones):
        z = Zone()
        z.sample_positions = rng.random((zone_sample_num, 3)).astype('float32')
        z.sample_normals = rng.random((zone_sample_num, 3)).astype('float32')
        zg.zones.append(z)
        zg.zone_to_current_label[i] = bool(i % 2)
        zg.zone_to_target_label[i] = bool(i % 3)
    return zg

def make_extrusion(indices, bool_type):
    e = Extrusion()
    e.zone_indices = list(indices)
    e.bool_type = bool_type
    return e

failures = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        failures.append(name)

agent = Agent('.')
zg = make_zone_graph(6)
candidates = [make_extrusion([0], 0), make_extrusion([1, 2], 0),
              make_extrusion([2, 3], 1), make_extrusion([3, 4, 5], 0),
              make_extrusion([5], 1)]
gt_index = 2
gt_hash = candidates[gt_index].hash()

def encode_all():
    g_encs = []
    for e in candidates:
        zg.encode_with_extrusion(e)
        g_encs.append(agent.encode_zone_graph(zg))
    return g_encs

# 1. batched_log_probs agrees with the base Agent.make_decision
agent.eval()
g_encs = encode_all()
log_p = batched_log_probs(agent, g_encs).numpy()
prob = agent.make_decision(g_encs).numpy()
check("log-probs match make_decision", np.allclose(np.exp(log_p), prob, atol=1e-5),
      "max diff %.2e" % np.abs(np.exp(log_p) - prob).max())

# 2. listwise loss equals a hand-computed softmax cross-entropy
loss = listwise_loss(agent, g_encs, gt_index).item()
s = log_p[:, 1]
manual = -(s[gt_index] - (np.log(np.sum(np.exp(s - s.max()))) + s.max()))
check("loss equals manual CE", abs(loss - manual) < 1e-5,
      "loss %.6f manual %.6f" % (loss, manual))

# 3. subsampling keeps the GT, respects the cap, reports the right index
sub, idx = subsample_candidates(list(candidates), gt_hash, 3)
check("subsample caps size", len(sub) == 3, str(len(sub)))
check("subsample keeps GT at reported index", sub[idx].hash() == gt_hash)
sub_all, idx_all = subsample_candidates(list(candidates), gt_hash, 0)
check("no cap keeps all candidates", len(sub_all) == len(candidates))
missing = subsample_candidates(list(candidates), make_extrusion([0, 1, 2], 1).hash(), 3)
check("missing GT -> (None, None)", missing == (None, None))

# 3b. node-budget memory guard
check("small graph keeps full cap", candidate_cap(50, 64, 4000) == 64)
check("large graph shrinks cap", candidate_cap(400, 64, 4000) == 10)
check("huge graph floors at 2", candidate_cap(5000, 64, 4000) == 2)
check("budget disabled -> plain cap", candidate_cap(5000, 64, 0) == 64)
check("no caps at all -> large", candidate_cap(50, 0, 0) >= 10 ** 6)

# 3c. proposal timeout and cache round-trip (uses the real joblib if present)
import time
import train_listwise as tl
if hasattr(sys.modules['joblib'], 'dump') and sys.modules['joblib'].dump is not None:
    tmpcache = tempfile.mkdtemp()
    orig_get_proposals = tl.get_proposals
    tl.get_proposals = lambda zg: time.sleep(5)
    t0 = time.time()
    res = tl.cached_proposals(tmpcache, 'slow_seq', 0, None, timeout=1)
    check("slow proposals time out to empty", res == [] and time.time() - t0 < 3,
          "%.1fs" % (time.time() - t0))
    tl.get_proposals = lambda zg: ['p1', 'p2']
    tl.cached_proposals(tmpcache, 'ok_seq', 0, None, timeout=1)
    tl.get_proposals = lambda zg: (_ for _ in ()).throw(RuntimeError('cache miss'))
    check("proposals served from cache", tl.cached_proposals(tmpcache, 'ok_seq', 0, None) == ['p1', 'p2'])
    check("timed-out step cached as skipped", tl.cached_proposals(tmpcache, 'slow_seq', 0, None) == [])
    tl.get_proposals = orig_get_proposals
else:
    print("SKIP proposal cache checks (joblib stubbed)")

# 4. a few listwise steps drive the GT to rank 1
agent.train()
agent.optim_extrusion.lr = 1e-3
losses = [train_step(agent, zg, candidates, gt_index) for _ in range(40)]
check("loss finite", all(np.isfinite(losses)), str(losses[:3]))
check("loss decreases", losses[-1] < losses[0], "%.4f -> %.4f" % (losses[0], losses[-1]))
agent.eval()
ranked = sort_extrusions_by_agent(list(candidates), zg, agent)
check("GT ranked first after training", ranked[0].hash() == gt_hash,
      "GT rank %d" % next(i for i, e in enumerate(ranked) if e.hash() == gt_hash))

# 5. best-checkpoint round-trip (the rank_eval.py loading path)
tmp = tempfile.mkdtemp()
agent.folder = tmp
agent.save_best_weights()
p1 = agent.make_decision(encode_all()).numpy()
agent2 = Agent(tmp)
agent2.load_best_weights()
agent2.eval()
p2 = agent2.make_decision(encode_all()).numpy()
check("best save/load round-trip", np.allclose(p1, p2, atol=1e-5),
      "max diff %.2e" % np.abs(p1 - p2).max())

print()
if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("ALL CHECKS PASSED")
