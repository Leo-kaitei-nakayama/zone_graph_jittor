# Pretrained checkpoints (Jittor)

Each sub-folder holds the **best-validation** weights of one training recipe,
saved by the trainers as `best_zone_encoder.pkl` (~0.3 MB) and
`best_decision_maker.pkl` (~1 MB). Training is run from `src/`.

| Folder | Recipe (script) | Fig. 6 average relative rank of the network (lower is better) |
|---|---|---|
| `baseline_from_torch/` | the 9-epoch baseline trained with the original PyTorch code (zone_graph_fix), converted key-for-key to Jittor by `tools/torch_reference/convert_torch_checkpoint.py` (max weight difference 0.0) | 0.107 (measured with torch) |
| `baseline/` | released recipe (`train.py`) | 0.111 |
| `listwise/` | listwise softmax loss (`train_listwise.py`) | 0.074 |
| `listwise_ft/` | low-lr fine-polish of `listwise/` | 0.080 |
| `ternary/` | paper Sec. 5.2 ternary labels (`train_ternary.py`) | 0.146 |

Folders listed here but not present in the repository have not been copied off the
training server yet.

For reference the paper reports 0.036; the random / heuristic baselines of our
evaluation are 0.398 / 0.044.

## Use

```python
import sys; sys.path.append('src')
from agent import Agent
agent = Agent('checkpoints/listwise')   # folder that contains the two .pkl files
agent.load_best_weights()
agent.eval()
```

Evaluate with the unchanged scripts, pointing `--train_output` at a folder:

```
cd src
python rank_eval.py --train_output ../checkpoints/listwise --data_path processed_data
```

Continue training from a checkpoint (weights only, optimizer restarts):

```
python train_listwise.py --init_from ../checkpoints/listwise --output_path train_output_new
```

(`train_ternary.py` accepts the same `--init_from`.)

The cross-dataset table in `REPORT.md` was produced with the `baseline/` weights
(`cross_dataset_eval.py --train_output ../checkpoints/baseline`).
