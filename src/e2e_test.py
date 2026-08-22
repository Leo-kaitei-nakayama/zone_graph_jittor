"""End-to-end pipeline test on real FreeCAD geometry — no dataset needed.

Synthesizes a two-step CAD sequence (an L-shape assembled from two boxes)
in the exact folder layout dataset_fusion.py produces, then runs the whole
pipeline on it:

    ZoneGraph build -> proposal generation -> GT replay (simulate) ->
    negative mining -> short agent training -> heur & agent search

and checks the search reconstructs the target (volumetric IOU ~= 1).

    python e2e_test.py [--work_dir /tmp/zg_e2e]

Needs the project environment (FreeCAD + jittor). Runtime: a few minutes,
dominated by the negative-mining rollouts and jittor's first-run compile.
"""
import sys, os, time, argparse, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', default='/tmp/zg_e2e', type=str)
args = parser.parse_args()

from setup import *
import FreeCAD
import Part
from FreeCAD import Base

import joblib
import numpy as np

from dataset import DataManager
from objects import ZoneGraph, Extrusion
from proposal import get_proposals
from train_preprocess import process_single_data
from agent import Agent, to_tensor
from evaluation import sort_extrusions_by_agent, sort_extrusions_by_heur
from search import SearchSolution, dfs_best_recon

failures = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + str(detail) if detail != "" else ""))
    if not ok:
        failures.append(name)

work = args.work_dir
if os.path.exists(work):
    shutil.rmtree(work)
raw = os.path.join(work, 'raw')
processed = os.path.join(work, 'processed')
seq_id = 'synthetic_L'

# ---- 1. synthesize the raw sequence: L = box1 (z 0..0.5) + box2 on top ----
box1 = Part.makeBox(1.0, 1.0, 0.5)
box2 = Part.makeBox(1.0, 0.5, 0.5, Base.Vector(0, 0, 0.5))
target = box1.fuse(box2).removeSplitter()

def write_step(step, current, extrusion):
    d = os.path.join(raw, seq_id, str(step))
    os.makedirs(d)
    target.exportStep(os.path.join(d, 'target_shape.stp'))
    if current is not None:
        current.exportStep(os.path.join(d, 'current_shape.stp'))
    extrusion.exportStep(os.path.join(d, 'extrusion.stp'))
    # no trailing newline: dataset.py compares the raw file content
    with open(os.path.join(d, 'bool_type.txt'), 'w') as f:
        f.write('addition')

write_step(0, None, box1)
write_step(1, box1, box2)
print('synthetic raw sequence written to', raw)

# ---- 2. raw load: ZoneGraph build + GT extrusion mapping ----
data_mgr = DataManager()
gt_seq, error_type = data_mgr.load_raw_sequence(os.path.join(raw, seq_id), 0, 2)
check("load_raw_sequence", len(gt_seq) == 2, f"steps={len(gt_seq)} error={error_type}")
if not gt_seq:
    print("cannot continue"); sys.exit(1)

zg0 = gt_seq[0][0]
check("zones found", len(zg0.zones) >= 3, f"{len(zg0.zones)} zones")
check("GT step0 maps to zones", len(gt_seq[0][1].zone_indices) > 0, gt_seq[0][1].zone_indices)
check("GT step1 maps to zones", len(gt_seq[1][1].zone_indices) > 0, gt_seq[1][1].zone_indices)

# ---- 3. proposals contain the GT moves; replay reaches the target ----
props = get_proposals(zg0)
prop_keys = {(e.hash(), e.bool_type) for e in props}
check("proposals non-empty", len(props) > 0, f"{len(props)} proposals")
check("GT extrusion is proposable",
      (gt_seq[0][1].hash(), gt_seq[0][1].bool_type) in prop_keys)
check("simulate_sequence replays GT to done", data_mgr.simulate_sequence(gt_seq))

# ---- 4. full preprocessing incl. negative mining ----
process_single_data(seq_id, raw, processed)
train_dir = os.path.join(processed, seq_id, 'train')
pairs = []
step_index = 0
while True:
    try:
        pos_g = joblib.load(os.path.join(train_dir, f'{step_index}_1_g.joblib'))
        pos_e = joblib.load(os.path.join(train_dir, f'{step_index}_1_e.joblib'))
        neg_g = joblib.load(os.path.join(train_dir, f'{step_index}_0_g.joblib'))
        neg_e = joblib.load(os.path.join(train_dir, f'{step_index}_0_e.joblib'))
    except Exception:
        break
    if pos_g and pos_e and neg_g and neg_e:
        pairs.append((pos_g, pos_e, neg_g, neg_e))
    step_index += 1
check("preprocessing produced pos/neg pairs", len(pairs) > 0, f"{len(pairs)} pairs")

# ---- 5. short training on the mined pairs ----
train_out = os.path.join(work, 'train_output')
os.makedirs(train_out)
agent = Agent(train_out)
gs, ls = [], []
for pos_g, pos_e, neg_g, neg_e in pairs:
    pos_g.encode_with_extrusion(pos_e); gs.append(pos_g); ls.append(to_tensor([1]))
    neg_g.encode_with_extrusion(neg_e); gs.append(neg_g); ls.append(to_tensor([0]))
losses = [agent.update_by_extrusion(ls, gs) for _ in range(12)]
check("training loss finite", all(np.isfinite(losses)))
check("training loss decreases", losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}")
agent.save_weights()

# ---- 6. search: heur baseline and the trained agent both rebuild the L ----
for option, ag in [('heur', None), ('agent', agent)]:
    best_sol = SearchSolution()
    start = time.time()
    dfs_best_recon(gt_seq[0][0].copy(), 4, 120, 5, option, best_sol, start,
                   os.path.join(work, 'recon_' + option), ag,
                   cur_step=0, cur_seq=[], visited_graphs=set(),
                   visited_extrusions=set(), data_mgr=DataManager())
    score = best_sol.best_score
    steps = len(best_sol.best_seq) if best_sol.best_seq else 0
    check(f"search[{option}] IOU ~ 1", score > 0.99,
          f"IOU={score:.4f} steps={steps} time={time.time()-start:.1f}s")

print()
if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("END-TO-END PIPELINE PASSED")
