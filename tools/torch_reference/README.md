# PyTorch reference tools (NOT part of the port)

Nothing in `src/` imports these, and nothing here runs during preprocessing,
training, or inference. **The port itself is pure Jittor** — `src/` contains
no `import torch` at all.

These three scripts exist only to answer one question: *does the Jittor
implementation compute the same function as the original PyTorch + DGL one?*
Proving that requires running the original as a reference, which needs torch.
Running them is optional and requires `pip install torch` in a separate
environment.

| script | what it does |
|---|---|
| `cross_check_torch.py` | Rebuilds the original architecture in torch (DGL's `update_all` semantics as explicit neighbor means), copies identical weights into both implementations, and reports the maximum output difference on synthetic graphs. |
| `real_check.py` | Same comparison on a real preprocessed zone graph produced by `train_preprocess.py`. |
| `convert_torch_checkpoint.py` | Loads a PyTorch checkpoint from the original repository into the Jittor modules (the parameter names and shapes match key-for-key), so both frameworks can be run with the *same trained weights*. Accepts `.npz` dumps so the conversion itself can run without torch. |

Recorded results (jittor 1.3.11 vs torch 2.x, eval mode):

```
ZoneEncoder                     max diff 6.7e-08
MPLayer (DGL loops vs matmul)   max diff 5.2e-08
GraphNetSoftMax, single graph   max diff 0.0
batched vs per-graph            max diff 0.0

with the original repository's trained checkpoint:
  encoder    max diff 7.9e-06
  full net   max diff 4.3e-06
  P(good)    torch 0.978259  |  jittor 0.978259
```
