"""Smoke test for the Jittor port — runs without FreeCAD or real data.

Synthesizes zone graphs and checks the whole neural corridor:
encode_zone_graph -> make_decision -> sort_extrusions_by_agent ->
update_by_extrusion -> save/load round-trip.

    python smoke_test.py
"""
import sys, os, types
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
from models import build_adj_norm
from agent import Agent, to_tensor
from evaluation import sort_extrusions_by_agent

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

# 1. adjacency semantics
adj = build_adj_norm(4, [(0, 1), (1, 2)]).numpy()
check("adj rows are means", np.allclose(adj[1], [0.5, 0, 0.5, 0]))
check("isolated node row is zero", np.allclose(adj[3], 0))

# 2. encode: shapes
agent = Agent('.')
zg = make_zone_graph(5)
zg.encode_with_extrusion(make_extrusion([1, 2], 0))
h, adj_np = agent.encode_zone_graph(zg)
check("h shape [N,128]", list(h.shape) == [5, 128], str(h.shape))
check("adj shape [N,N]", adj_np.shape == (5, 5))

# 3. decide: batched candidates -> probabilities
cands = [make_extrusion([0], 0), make_extrusion([1, 2], 0), make_extrusion([3, 4], 1)]
g_encs = []
for e in cands:
    zg.encode_with_extrusion(e)
    g_encs.append(agent.encode_zone_graph(zg))
prob = agent.make_decision(g_encs)
p = prob.numpy()
check("prob shape [B,2]", p.shape == (3, 2), str(p.shape))
check("rows sum to 1", np.allclose(p.sum(axis=1), 1.0, atol=1e-4), str(p.sum(axis=1)))
check("probs in (0,1)", bool((p > 0).all() and (p < 1).all()))

# 4. ranking preserves the candidate set and sorts by score
ranked = sort_extrusions_by_agent(cands, zg, agent)
scores = [e.score for e in ranked]
check("ranking keeps all candidates", sorted(id(e) for e in ranked) == sorted(id(e) for e in cands))
check("ranking sorted desc", scores == sorted(scores, reverse=True), str(scores))

# 5. training: loss is finite and decreases on a fixed batch
graphs, labels = [], []
for k in range(4):
    g = make_zone_graph(4 + k)
    g.encode_with_extrusion(make_extrusion([0, 1], k % 2))
    graphs.append(g)
    labels.append(to_tensor([k % 2]))
losses = [agent.update_by_extrusion(labels, graphs) for _ in range(15)]
check("loss finite", all(np.isfinite(losses)), str(losses[:3]))
check("loss decreases", losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}")

# 6. checkpoint round-trip reproduces predictions
agent.save_weights()
p1 = agent.make_decision(g_encs).numpy()
agent2 = Agent('.')
agent2.load_weights()
p2 = agent2.make_decision(g_encs).numpy()
check("save/load round-trip", np.allclose(p1, p2, atol=1e-5), f"max diff {np.abs(p1-p2).max():.2e}")

for f in ("zone_encoder.pkl", "decision_maker.pkl"):
    if os.path.exists(f):
        os.remove(f)

print()
if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("ALL CHECKS PASSED")
