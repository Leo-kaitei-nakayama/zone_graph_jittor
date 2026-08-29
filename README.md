# zone_graph_jittor

Pure-Jittor port of [zone_graph_fix](https://github.com/leo-kaitei-nakayama/zone_graph_fix)
("Inferring CAD Modeling Sequences Using Zone Graphs", CVPR 2021),
numerically verified against the released PyTorch + DGL implementation.

See **[REPORT.md](REPORT.md)** for the full reproduction report: port
verification, Fusion360 results, the Fig. 6 analysis, and cross-dataset
generalization.

## Requirements

- [Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) (tested with 1.3.11)
- [FreeCAD](https://www.freecadweb.org/) (geometry kernel; set its lib path
  in `src/setup.py`)
- numpy, networkx, trimesh, joblib
- No PyTorch and no DGL anywhere in `src/` — torch appears only in the
  optional verification tools under `tools/`.

Dataset: the [Fusion 360 Gallery reconstruction subset r1.0.0]
(https://github.com/AutodeskAILab/Fusion360GalleryDataset)
(`r1.0.0.zip` + `r1.0.0_extrude_tools.zip`), as in the original README.

## Repository layout

**Baseline port (`src/`)** — same architecture, training recipe and
evaluation as the released code; this is what method comparisons should use.

| file | role |
|---|---|
| `objects.py` | Zone / ZoneGraph / Extrusion, space splitting, heuristic score |
| `utils/` | FreeCAD geometry helpers (space_splitter, face/solid/edge/vector/vertex/combination/file utils) |
| `proposal.py` | candidate extrusion generation (`get_proposals`) |
| `dataset_fusion.py` | stage 1: raw Fusion360 STEP → per-step folders |
| `dataset.py` | data manager: build / load sequences |
| `train_preprocess.py` | stage 2: zone graphs + negative mining → training joblibs (`--workers N` for parallelism) |
| `models.py` | ZoneEncoder (PointNet), MPLayer, GraphNetSoftMax — DGL replaced by dense `D⁻¹A` message passing |
| `agent.py` | encoding, batched scoring, focal-loss update, checkpoints |
| `train.py` | stage 3: the released training recipe |
| `evaluation.py` | random / heuristic / network candidate ranking |
| `search.py` | best-first reconstruction search |
| `experiments/exp_infer/infer.py` | stage 4: test-set reconstruction (IoU, timing) |
| `hyperparameters.py`, `setup.py` | constants; FreeCAD path |

**Evaluation scripts (used for the numbers in REPORT.md)**

| file | role |
|---|---|
| `rank_eval.py` | paper Fig. 6 metric: average relative rank of the GT extrusion |
| `cross_dataset_eval.py` | reconstruction on any folder of STEP/BREP files (DeepCAD, WHUCAD, HistCAD, …), Table 7 protocol |

**Training experiments (additive; never modify the baseline)**

| file | role |
|---|---|
| `train_listwise.py` | listwise objective (softmax CE over all proposals) — best network result, ≈0.07 |
| `ternary_label.py` + `train_ternary.py` | the paper's Sec. 5.2 ternary labeling, re-implemented as described |

**Tests** (run without FreeCAD or data, except e2e): `smoke_test.py`,
`listwise_smoke_test.py`, `ternary_smoke_test.py`, `e2e_test.py`
(e2e needs FreeCAD; builds a synthetic shape and reconstructs it).

**`tools/torch_reference/`** — optional PyTorch comparison scripts, the
numerical proof that the port matches the original (see its README).
The only place torch is used.

## Quickstart

```bash
export FREECAD_LIB_PATH=/path/to/freecad/lib   # or edit src/setup.py
cd src

# 1. preprocess raw Fusion360 data
python dataset_fusion.py --fusion_path <reconstruction dir> \
    --extrusion_path <extrude_tools dir> --output_path ../data/fusion_processed
python train_preprocess.py --data_path ../data/fusion_processed \
    --output_path processed_data --workers 16

# 2. train (released recipe)
python train.py --data_path processed_data --output_path train_output

# 3. evaluate
python rank_eval.py --data_path processed_data --ids_file test_ids.txt \
    --train_output train_output --limit 150          # Fig. 6 metric
(cd experiments/exp_infer && python infer.py --option agent \
    --data_path ../../../data/fusion_processed)       # reconstruction

# cross-dataset generalization (any folder of .step/.stp/.brep)
python cross_dataset_eval.py --step_dir <folder> --option agent --limit 300
python cross_dataset_eval.py --out_dir <crossdata_*> --summary_only

# sanity checks
python smoke_test.py
```

Run-time artifacts (`processed_data/`, `train_output*/`, `proposal_cache/`,
`crossdata_*/`, `*.log`) are intentionally not part of the repository.
