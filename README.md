# zone_graph_jittor

Jittor port of [zone_graph_fix](https://github.com/leo-kaitei-nakayama/zone_graph_fix)
("Inferring CAD Modeling Sequences Using Zone Graphs", CVPR 2021).

See **[REPORT.md](REPORT.md)** for the full reproduction report: port
verification, Fusion360 results, why the released training recipe misses
the paper's Fig. 6 network number, and the listwise fix.

## Layout: baseline vs experiments

The repository keeps a strict separation so the baseline stays a faithful
reproduction of the released code:

- **`src/` — the baseline port.** Same architecture, training recipe
  (binary GT-vs-mined-negative, focal loss), checkpoint format and
  evaluation as the original PyTorch+DGL code, verified operator-by-operator
  against it (see `tools/torch_reference/README.md`; trained-checkpoint
  agreement to ~1e-6). Train with `src/train.py`, evaluate with
  `src/rank_eval.py` / `src/evaluation.py`. Pure Jittor — no torch imports.

- **`src/train_listwise.py` — an additive experiment.** An alternative
  trainer with a listwise (softmax-over-all-proposals) objective that
  matches the ranking task the evaluation measures. It only *imports* the
  baseline modules and writes to its own output folder
  (`train_output_listwise`), so running it never affects baseline results.
  Its smoke test is `src/listwise_smoke_test.py`.

- **`tools/torch_reference/` — verification only.** Optional scripts that
  compare the port against the original PyTorch implementation; the only
  place torch appears, never imported by `src/`.

Both trainers produce checkpoints in the same format; evaluate either with
the unchanged evaluator:

```
python rank_eval.py --data_path processed_data --ids_file test_ids.txt \
    --train_output <train_output | train_output_listwise> --limit 150
```
