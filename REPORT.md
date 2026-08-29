# Reproduction Report: Zone Graphs in Jittor

Jittor port of [zone_graph_fix](https://github.com/leo-kaitei-nakayama/zone_graph_fix)
(PyTorch + DGL implementation of *"Inferring CAD Modeling Sequences Using
Zone Graphs"*, CVPR 2021), with a verification of the port, a full-dataset
reproduction on Fusion360, an analysis of why the released training recipe
does not reproduce the paper's network ranking quality, and an additive fix
that recovers most of the gap.

All experiments below were run on the Fusion360 reconstruction dataset
(r1.0.0) on a single RTX 4090. The baseline in `src/` is a faithful port —
same architecture, training recipe, and evaluation as the released PyTorch
code; the improvement lives in a separate script (`src/train_listwise.py`)
that does not modify any baseline file.

## 1. Port correctness

The port replaces DGL message passing with an equivalent dense formulation
(`D^-1 A · MLP(H)` with a per-batch shared adjacency) and matches the torch
model key-for-key in checkpoint format. Verification ladder, strongest last
(tools in `tools/torch_reference/`, results recorded there):

| check | result |
|---|---|
| operator-level vs torch (synthetic graphs) | max diff ≤ 6.7e-08 |
| full network, single & batched graphs | max diff 0.0 |
| real preprocessed zone graph | max diff ~1e-06 |
| original torch-trained checkpoint loaded into jittor | encoder 7.9e-06, full net 4.3e-06, P(good) identical to 6 decimals (0.978259) |
| synthetic end-to-end FreeCAD pipeline | reconstruction IOU 1.0 |

Bugs found and handled during the port (fixed on the jittor side without
touching upstream): `Part.Solid` bbox unpicklable in `objects.__getstate__`;
faces deserializing as `Part.Shape` without `curvatureAt`; corrupt STEP
files killing a whole dataset pass; `train.py` referencing an undefined
`data_mgr` (silently skipping all sequences via a bare `except`);
`load_best_weights` loading the non-best files.

## 2. Full-dataset reproduction with the released recipe

Preprocessing yield: 4,693 of 8,625 extrusion pairs (54%) survive
preprocessing under jittor's pipeline; the PyTorch run on the same machine
yielded 4,638 (54%). Split 3,989 / 233 / 469 (train/validate/test, seed 0).

Search-based reconstruction on the test split (jittor, best checkpoint):

| ranker | mean IOU | perfect recon | sec/seq |
|---|---|---|---|
| network | 0.9973 | 99.4% | 7.5 |
| heuristic | 0.9988 | 99.6% | 5.6 |
| random | 0.9842 | 93.3% | 23.5 |

Reconstruction quality matches the torch reproduction; the search
(width 15) compensates for imperfect ranking, so IOU alone does not
discriminate the rankers.

## 3. The ranking metric (paper Fig. 6) does not reproduce

Average relative rank of the GT extrusion among all proposals
(`src/rank_eval.py`, 150 test sequences / 225 steps, eval mode; lower is
better):

| | random | heuristic | network |
|---|---|---|---|
| paper (Fig. 6) | 0.486 | 0.070 | **0.036** |
| torch reproduction | 0.374 | 0.059 | 0.107 |
| jittor reproduction | 0.398 | 0.044 | **0.111** |

Two independent implementations agree to within noise, ruling out a porting
defect: **the released recipe itself does not reproduce the paper's network
ranking, and the paper's core ordering (network beats heuristic) inverts.**
The overall offset of random/heuristic vs the paper comes from the
evaluated step distribution (54% preprocessing yield biased toward simpler
models, and a different test split — the released repository's
`test_ids.txt` is empty), so only within-row comparisons are meaningful.

### Diagnosis

The released code trains a binary classifier: per GT step exactly two
examples — the GT extrusion (label 1) and **one** mined, clearly-bad
negative (label 0; a proposal whose random-rollout hit ratio is ≤ 0.2).
Evaluation instead ranks *all* proposals of a step (dozens to hundreds,
most of them intermediate candidates never seen as training negatives).
The binary task saturates (focal loss 0.58 → 0.02) while ranking stays
mediocre — the objective and the evaluated task are different problems.
The paper (Sec. 5.2) describes ternary positive/negative/neutral labeling;
the released code simplifies this to the single negative.

## 4. Additive fix: listwise training (`src/train_listwise.py`)

Same architecture, checkpoint format, and (unchanged) evaluator; only the
objective changes: for each GT step, score the full proposal set and apply
a softmax cross-entropy over the candidates with the GT as the target —
the score entering the softmax is the same log P(good) the evaluator sorts
by, so training optimizes the evaluated task directly. Practical details:
per-update candidate cap with fresh negatives each epoch, a node budget
guarding GPU memory, an on-disk proposal cache, per-step proposal
timeouts, and optional mid-epoch validation for best-checkpoint selection.

Results (fine-tuned from the baseline's best checkpoint; same unchanged
`rank_eval.py`, same 150 sequences / 225 steps):

| training | network avg relative rank |
|---|---|
| released binary recipe (baseline) | 0.111 |
| listwise, run 1 | 0.068 |
| listwise, run 2 (15 epochs) | 0.074 |
| listwise, run 3 (low-lr fine-polish) | 0.080 |

Three runs land at **≈ 0.07 ± 0.01** (differences are within the ~±0.01
noise of a 225-step evaluation): a reproducible ~35% improvement from
changing only the objective, reaching the paper's *heuristic* level
(0.070) and confirming the diagnosis. It does not yet beat the (unusually
strong, easy-distribution) heuristic of this reproduction (0.044).

We also re-implemented the paper's own Sec. 5.2 ternary labeling
faithfully (`src/ternary_label.py` + `src/train_ternary.py`: every
zero-completion-probability proposal labeled negative via random
completions, neutrals excluded, base focal loss): trained from scratch
with a per-step negative cap of 8, it scores **0.146** — the best
validation rank sum of any run but the worst test relative rank,
overfitting the many small candidate sets while undertraining the large
ones. Neither the released recipe, listwise, nor the described ternary
scheme reaches the paper's 0.036.

## 5. Cross-dataset generalization

Using `src/cross_dataset_eval.py` (Fusion360-trained checkpoints, 300
random shapes per dataset, Table 7 protocol: normalize → build zone graph
→ 120s best-first search; reconstructed = IoU ≥ 0.99):

| dataset | heur | agent | build failed |
|---|---|---|---|
| Fusion360 r1.0.0 (home) | **77.7%** | 70.3% | 12.3% |
| HistCAD-Fusion360 | 72.7% | 63.0% | 17.3% |
| HistCAD-DeepCAD | 68.7% | 65.0% | 27.7% |
| WHUCAD (B-Rep) | 60.0% | 53.0% | 21.0% |

Findings: (a) the home row matches the paper's Table 7 (80%) under the
same protocol, resolving the apparent 54%-vs-80% gap (54% was the
stricter training-preprocessing funnel, the paper's own comparable figure
being 60%); (b) generalization degrades gracefully, dominated by zone
graph construction failures and, on WHUCAD, by non-extrusion features
outside the method's vocabulary — of buildable HistCAD-DeepCAD shapes,
98.1% reconstruct perfectly; (c) the heuristic beats the network on every
dataset (4–10 pp), so the learned ranker is the main cross-dataset
bottleneck, consistent with the Fig. 6 findings above.

## 6. Conclusions

1. The Jittor port is numerically faithful to the released PyTorch code
   and is the appropriate baseline for method comparisons.
2. The released training recipe cannot reproduce the paper's Fig. 6
   network result; the failure is attributable to the
   binary-classification objective, as evidenced by two-implementation
   agreement and by the objective-only listwise fix recovering most of
   the gap.
3. Closing the remaining gap (≈0.07 → ≤0.044 on this data distribution)
   likely requires steps beyond the released method: the paper's ternary
   labeling, hard-negative-aware sampling, training listwise from
   scratch, or richer geometric features. Left as future work.

## 7. Reproducing

```
# baseline (released recipe, faithful port)
python train.py --data_path processed_data --output_path train_output
python rank_eval.py --data_path processed_data --ids_file test_ids.txt \
    --train_output train_output --limit 150

# listwise experiment (additive; new output folder)
python train_listwise.py --data_path processed_data \
    --output_path train_output_listwise --init_from train_output
python rank_eval.py --data_path processed_data --ids_file test_ids.txt \
    --train_output train_output_listwise --limit 150
```
