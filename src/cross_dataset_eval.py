"""Cross-dataset generalization evaluation — additive; base untouched.

Runs the (Fusion360-trained) reconstruction pipeline on ANY folder of
.step/.stp files — DeepCAD, WHUCAD, or any other B-Rep dataset exported
to STEP — with no ground-truth sequences required:

    per shape: normalize scale (exactly as dataset_fusion.py does),
    build the zone graph, run the best-first search under a time budget,
    record the best reconstruction IoU and a failure category.

Each shape runs in its own spawned subprocess with a hard timeout (one
pathological shape cannot wedge the run), writing one JSON per shape to
--out_dir/results; already-evaluated shapes are skipped, so the script
is resumable. A summary comparable to the paper's Table 7 is printed at
the end (and can be re-printed anytime with --summary_only).

    python cross_dataset_eval.py --step_dir /path/to/deepcad_steps \
        --option agent --train_output train_output --limit 300

Use --option heur / random for the baselines (no GPU needed for those).
"""

import sys
sys.path.append('..')

import os
import glob
import json
import time
import random
import argparse
import multiprocessing


def evaluate_one(step_file, out_json, option, train_output, max_time,
                 max_step, expand_width, reposition, eval_mode, scratch):
    # heavy imports stay inside: children are spawned (CUDA does not
    # survive fork) and re-import this module during bootstrap
    result = {'file': os.path.basename(step_file), 'status': 'crashed',
              'iou': 0.0, 'zones': 0, 'elapsed': 0.0, 'error': ''}
    t0 = time.time()
    try:
        from objects import ZoneGraph, Part, Base
        from search import SearchSolution, dfs_best_recon

        shape = Part.Shape()
        try:
            shape.read(step_file)
        except Exception as e:
            result.update(status='read_failed', error=str(e)[:200])
            raise SystemExit

        # same normalization as dataset_fusion.py: scale so the bbox
        # diagonal is 1 (the network was trained on shapes at this scale)
        bb = shape.BoundBox
        import math
        w, d, h = bb.XLength, bb.YLength, bb.ZLength
        diag = math.sqrt(w * w + d * d + h * h)
        if diag <= 0:
            result.update(status='read_failed', error='empty bbox')
            raise SystemExit
        if reposition:
            shape.translate(Base.Vector(-(bb.XMin + bb.XMax) / 2,
                                        -(bb.YMin + bb.YMax) / 2,
                                        -(bb.ZMin + bb.ZMax) / 2))
        shape.scale(1.0 / diag, Base.Vector(0, 0, 0))

        zone_graph = ZoneGraph()
        zone_graph.current_shape = None
        zone_graph.target_shape = shape
        try:
            ret, error_type = zone_graph.build()
        except Exception as e:
            result.update(status='build_failed', error=str(e)[:200])
            raise SystemExit
        if not ret:
            result.update(status='build_failed', error=str(error_type))
            raise SystemExit
        result['zones'] = len(zone_graph.zones)

        agent = None
        if option == 'agent':
            from agent import Agent
            agent = Agent(train_output)
            agent.load_best_weights()
            if eval_mode:
                agent.eval()

        best_sol = SearchSolution()
        try:
            dfs_best_recon(zone_graph, max_step, max_time, expand_width,
                           option, best_sol, time.time(), scratch, agent)
        except Exception as e:
            result.update(status='search_failed', error=str(e)[:200])
            raise SystemExit

        iou = best_sol.best_score or 0.0
        result.update(status='ok', iou=float(iou))
    except SystemExit:
        pass
    except Exception as e:
        result['error'] = str(e)[:200]
    finally:
        result['elapsed'] = round(time.time() - t0, 1)
        with open(out_json, 'w') as f:
            json.dump(result, f)


def summarize(results_dir):
    rows = []
    for path in glob.glob(os.path.join(results_dir, '*.json')):
        try:
            with open(path) as f:
                rows.append(json.load(f))
        except Exception:
            pass
    if not rows:
        print('no results yet in', results_dir)
        return
    n = len(rows)
    by_status = {}
    for r in rows:
        by_status.setdefault(r['status'], []).append(r)
    ok = by_status.get('ok', [])
    recon = [r for r in ok if r['iou'] >= 0.99]

    print()
    print('shapes evaluated: %d' % n)
    for status, group in sorted(by_status.items()):
        print('  %-14s %4d  (%.1f%%)' % (status, len(group), 100.0 * len(group) / n))
    if ok:
        print('of searched shapes: mean IoU %.4f | IoU>=0.99: %d (%.1f%% of all, %.1f%% of searched)'
              % (sum(r['iou'] for r in ok) / len(ok),
                 len(recon), 100.0 * len(recon) / n, 100.0 * len(recon) / len(ok)))
        print('mean search time %.1fs | mean zones %.1f'
              % (sum(r['elapsed'] for r in ok) / len(ok),
                 sum(r['zones'] for r in ok) / len(ok)))
    print("paper Table 7 reference (Fusion360, their env): 80%% reconstructable / 20%% not")


def main():
    parser = argparse.ArgumentParser(description='Cross-dataset reconstruction evaluation (additive)')
    parser.add_argument('--step_dir', type=str, required=False, default='',
                        help='folder containing .step/.stp files (searched recursively)')
    parser.add_argument('--option', default='agent', type=str,
                        choices=['agent', 'heur', 'random'])
    parser.add_argument('--train_output', default='train_output', type=str,
                        help='checkpoint folder for --option agent (best_*.pkl is used)')
    parser.add_argument('--out_dir', default='', type=str,
                        help='default: crossdata_<dataset folder name>_<option>')
    parser.add_argument('--max_time', default=120, type=float,
                        help="seconds of search per shape (the paper's budget)")
    parser.add_argument('--max_step', default=15, type=int)
    parser.add_argument('--expand_width', default=15, type=int)
    parser.add_argument('--limit', default=300, type=int,
                        help='evaluate a random sample of this many shapes; 0 = all')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--reposition', action='store_true',
                        help='also translate the bbox center to the origin before scaling')
    parser.add_argument('--no_eval_mode', action='store_true')
    parser.add_argument('--summary_only', action='store_true',
                        help='just re-print the summary for --out_dir and exit')
    args = parser.parse_args()

    if not args.out_dir:
        base = os.path.basename(os.path.normpath(args.step_dir)) or 'dataset'
        args.out_dir = 'crossdata_%s_%s' % (base, args.option)
    results_dir = os.path.join(args.out_dir, 'results')

    if args.summary_only:
        summarize(results_dir)
        return

    if not args.step_dir:
        parser.error('--step_dir is required')
    os.makedirs(results_dir, exist_ok=True)
    scratch = os.path.join(args.out_dir, 'search_out')
    os.makedirs(scratch, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.step_dir, '**', '*.*'), recursive=True))
    # FreeCAD's Shape.read handles STEP, IGES and native BREP by extension
    files = [f for f in files if f.lower().endswith(('.step', '.stp', '.brep'))]
    print('found %d step files' % len(files))
    random.seed(args.seed)
    random.shuffle(files)
    if args.limit:
        files = files[:args.limit]

    for i, step_file in enumerate(files):
        name = os.path.splitext(os.path.basename(step_file))[0]
        out_json = os.path.join(results_dir, name + '.json')
        if os.path.exists(out_json):
            continue
        print('[%d/%d] %s' % (i + 1, len(files), name))
        p = multiprocessing.Process(
            target=evaluate_one,
            args=(step_file, out_json, args.option, args.train_output,
                  args.max_time, args.max_step, args.expand_width,
                  args.reposition, not args.no_eval_mode, scratch))
        p.start()
        p.join(args.max_time + 180)
        if p.is_alive():
            print('  timed out hard, killing')
            p.terminate()
            p.join()
        if not os.path.exists(out_json):
            with open(out_json, 'w') as f:
                json.dump({'file': os.path.basename(step_file),
                           'status': 'timeout_or_crash', 'iou': 0.0,
                           'zones': 0, 'elapsed': args.max_time, 'error': ''}, f)

    summarize(results_dir)


if __name__ == "__main__":
    # CUDA contexts do not survive fork(); spawn children fresh
    multiprocessing.set_start_method('spawn', force=True)
    main()
