"""Smoke test for ternary_label.py / train_ternary.py — no FreeCAD, no data.

Verifies the paper's labeling rule (all p==0 negatives; argmin-p fallback),
the per-step time budget, example assembly (positive + labeled negatives,
neutrals excluded), and that the assembled batch trains through the
unchanged base Agent.

    python ternary_smoke_test.py
"""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def _stub(name, **attrs):
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
from agent import Agent, to_tensor
import ternary_label as tl
from ternary_label import assign_negatives, label_step
from train_ternary import step_examples

rng = np.random.default_rng(0)
jt.set_global_seed(0)

def make_zone_graph(n_zones):
    zg = ZoneGraph()
    zg.zone_graph = nx.Graph()
    for i in range(n_zones):
        zg.zone_graph.add_node(i)
    for i in range(n_zones - 1):
        zg.zone_graph.add_edge(i, i + 1)

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

# 1. the paper's labeling rule
check("all p==0 become negatives",
      set(assign_negatives({'a': 0, 'b': 0.5, 'c': 0})) == {'a', 'c'})
check("no zeros -> single argmin",
      assign_negatives({'a': 0.4, 'b': 0.1, 'c': 0.9}) == ['b'])
check("empty -> no negatives", assign_negatives({}) == [])

# 2. label_step applies the rule over non-GT candidates
zg = make_zone_graph(6)
gt = make_extrusion([1, 2], 0)
cands = [gt, make_extrusion([0], 0), make_extrusion([2, 3], 1),
         make_extrusion([3, 4, 5], 0), make_extrusion([5], 1)]
preset = {cands[1].hash(): 0.0, cands[2].hash(): 0.5,
          cands[3].hash(): 0.0, cands[4].hash(): 1.0}
orig_cp = tl.completion_probability
tl.completion_probability = lambda zone_graph, e, remaining, rollouts: preset[e.hash()]
labels = label_step(zg, gt, list(cands), remaining=2,
                    rollout_multiplier=1, step_time_limit=0)
check("GT excluded from labeling", gt.hash() not in labels['p'])
check("negatives = the p==0 candidates",
      set(labels['neg_hashes']) == {cands[1].hash(), cands[3].hash()})
check("neutral kept out of negatives", cands[2].hash() not in labels['neg_hashes'])
check("checked/total counted", labels['checked'] == 4 and labels['total'] == 4)

# 2b. time budget marks the step partial but keeps found labels
import time
tl.completion_probability = lambda zone_graph, e, remaining, rollouts: (time.sleep(0.4), 0.0)[1]
capped = label_step(zg, gt, list(cands), remaining=2,
                    rollout_multiplier=1, step_time_limit=0.1)
check("time cap yields partial labels",
      capped['partial'] and 0 < capped['checked'] < capped['total'],
      "checked %d/%d" % (capped['checked'], capped['total']))
tl.completion_probability = orig_cp

# 3. example assembly: positive + labeled negatives only, neutrals excluded
examples = step_examples(zg, gt, cands, labels['neg_hashes'])
labels_out = [l for _, l in examples]
check("one positive first", labels_out[0] == 1 and labels_out.count(1) == 1)
check("all labeled negatives included", labels_out.count(0) == 2)
capped_ex = step_examples(zg, gt, cands, labels['neg_hashes'], max_negatives=1)
check("max_negatives caps", len(capped_ex) == 2)
check("graphs are independent copies",
      examples[0][0] is not examples[1][0] and examples[0][0] is not zg)

# 4. the assembled batch trains through the unchanged base Agent
agent = Agent('.')
gs = [g for g, _ in examples]
ls = [to_tensor([l]) for _, l in examples]
losses = [agent.update_by_extrusion(ls, gs) for _ in range(10)]
check("loss finite", all(np.isfinite(losses)), str(losses[:3]))
check("loss decreases", losses[-1] < losses[0], "%.4f -> %.4f" % (losses[0], losses[-1]))

print()
if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("ALL CHECKS PASSED")
