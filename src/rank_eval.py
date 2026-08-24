"""Figure 6 of the paper: average relative rank of the GT extrusion.

Jittor port of the zone_graph_fix script of the same name, following the
same protocol so the numbers are directly comparable: for every test
sequence with processed ground truth, each step contributes one datum —
generate all proposals for the step's zone graph, rank them with each
method, locate the GT extrusion by zone-set hash, record rank /
candidate_count. Lower is better.

Paper reference values (Fig 6, Fusion360 test set):
    Ours Net 0.036, Ours Heur 0.070, Random 0.486

    python rank_eval.py --data_path processed_data --ids_file test_ids.txt \
        --train_output train_output [--limit 150]

Point --train_output at a directory produced by convert_torch_checkpoint.py
to score the PyTorch-trained network under jittor.
"""

import argparse
import os
import random

from dataset import DataManager
from proposal import get_proposals
from evaluation import sort_extrusions_by_heur


def relative_rank(ranked, gt_hash):
    for i, extrusion in enumerate(ranked):
        if extrusion.hash() == gt_hash:
            return i / len(ranked)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='processed_data', type=str)
    parser.add_argument('--ids_file', default='test_ids.txt', type=str)
    parser.add_argument('--train_output', default='train_output', type=str)
    parser.add_argument('--limit', default=0, type=int, help='max sequences (0 = all)')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--no_eval_mode', action='store_true',
                        help='keep BatchNorm in training mode, as the released code did')
    args = parser.parse_args()

    random.seed(args.seed)

    from agent import Agent
    from evaluation import sort_extrusions_by_agent

    agent = Agent(args.train_output)
    agent.load_best_weights()
    if not args.no_eval_mode:
        agent.eval()

    data_mgr = DataManager()

    with open(args.ids_file) as f:
        ids = [line.strip() for line in f if line.strip()]

    ranks = {'random': [], 'heur': [], 'agent': []}
    used_sequences = 0

    for seq_id in ids:
        gt_path = os.path.join(args.data_path, seq_id, 'gt')
        if not os.path.isdir(gt_path):
            continue
        try:
            gt_seq = data_mgr.load_processed_sequence(gt_path)
        except Exception:
            continue
        if not gt_seq:
            continue

        used = False
        for zone_graph, gt_extrusion in gt_seq:
            if gt_extrusion is None:
                continue
            try:
                extrusions = get_proposals(zone_graph)
            except Exception as e:
                print('proposals failed for', seq_id, ':', e)
                break
            if len(extrusions) < 2:
                continue
            gt_hash = gt_extrusion.hash()
            if not any(e.hash() == gt_hash for e in extrusions):
                continue

            shuffled = list(extrusions)
            random.shuffle(shuffled)
            r = relative_rank(shuffled, gt_hash)
            if r is not None:
                ranks['random'].append(r)

            r = relative_rank(sort_extrusions_by_heur(list(extrusions), zone_graph), gt_hash)
            if r is not None:
                ranks['heur'].append(r)

            try:
                r = relative_rank(sort_extrusions_by_agent(list(extrusions), zone_graph, agent), gt_hash)
                if r is not None:
                    ranks['agent'].append(r)
            except Exception as e:
                print('agent ranking failed for', seq_id, ':', e)
            used = True

        if used:
            used_sequences += 1
            if used_sequences % 25 == 0:
                print('sequences used so far:', used_sequences)
            if args.limit and used_sequences >= args.limit:
                break

    print()
    print('sequences used:', used_sequences)
    print('paper reference (Fig 6): random 0.486, heur 0.070, net 0.036')
    for method in ('random', 'heur', 'agent'):
        vals = ranks[method]
        if vals:
            print('%-7s steps=%4d  avg relative rank of GT extrusion = %.3f' % (
                method, len(vals), sum(vals) / len(vals)))
        else:
            print('%-7s no data' % method)


if __name__ == '__main__':
    main()
