"""Ternary labeling of proposals, following the paper (Sec. 5.2) — additive.

The paper labels every proposal of a GT step:

  positive = the GT extrusion
  negative = EVERY proposal whose completion probability p equals 0, where
             p is measured with N random modeling-sequence completions
             starting from that proposal (N = number of remaining GT steps);
             if every proposal has p > 0, the single smallest-p proposal
             is the negative
  neutral  = all other proposals — excluded from the loss entirely
             ("there are often multiple approaches to construct a shape")

The released code simplified this to ONE mined negative per step
(min hit-ratio <= 0.2), which is the train_preprocess.py behaviour. This
script restores the paper's scheme on top of the existing preprocessed
data, reusing the same rollout primitive (hit_target_in_path) and the
proposal cache. No base file is modified.

Output: one joblib file per (sequence, step) in --label_path:
    {'neg_hashes': [...], 'p': {hash: p}, 'checked': n, 'total': m,
     'partial': bool}
Steps already labeled are skipped, so the script is resumable and can be
run sharded in parallel (rollouts are CPU-only):

    for i in $(seq 0 15); do
        nohup python ternary_label.py --shard $i --num_shards 16 \
            > label_$i.log 2>&1 &
    done

Training (train_ternary.py) picks up whatever labels exist, so it can be
started before labeling has finished the whole training set.
"""

import sys
sys.path.append('..')

import os
import argparse
import random
import time

import joblib

from dataset import DataManager
from train_preprocess import hit_target_in_path
from train_listwise import cached_proposals
from utils.file_utils import read_file_to_list


def assign_negatives(p_by_hash):
    """The paper's rule, as a pure function over {proposal_hash: p}.
    Every p == 0 proposal is negative; if none are (and at least one
    proposal was measured), the single smallest-p proposal is."""
    if not p_by_hash:
        return []
    zeros = [h for h, p in p_by_hash.items() if p == 0]
    if zeros:
        return zeros
    return [min(p_by_hash, key=p_by_hash.get)]


def completion_probability(zone_graph, extrusion, remaining, rollouts):
    """p for one proposal: apply it, then run random completions with the
    same depth convention as the released generate_neg_steps."""
    try:
        next_zg = zone_graph.update_to_next_zone_graph(extrusion)
    except Exception:
        return 0.0
    hits = 0
    for _ in range(rollouts):
        try:
            if hit_target_in_path(next_zg, 0, remaining):
                hits += 1
        except Exception:
            pass
    return hits / rollouts


def label_step(zone_graph, gt_extrusion, extrusions, remaining,
               rollout_multiplier, step_time_limit):
    """Measure p for every non-GT proposal (under a per-step time budget,
    like the released code's 100s limit) and apply the paper's rule.
    A time-capped step still yields valid labels: any p == 0 proposal
    found so far is a true negative; unmeasured proposals stay neutral."""
    gt_hash = gt_extrusion.hash()
    candidates = [e for e in extrusions if e.hash() != gt_hash]
    random.shuffle(candidates)

    rollouts = max(1, remaining * rollout_multiplier)
    p_by_hash = {}
    start = time.time()
    partial = False
    for e in candidates:
        p_by_hash[e.hash()] = completion_probability(
            zone_graph, e, remaining, rollouts)
        if step_time_limit and time.time() - start > step_time_limit:
            partial = len(p_by_hash) < len(candidates)
            break

    return {
        'neg_hashes': assign_negatives(p_by_hash),
        'p': p_by_hash,
        'checked': len(p_by_hash),
        'total': len(candidates),
        'partial': partial,
    }


def main():
    parser = argparse.ArgumentParser(description='Paper-faithful ternary labeling (additive)')
    parser.add_argument('--data_path', default='processed_data', type=str)
    parser.add_argument('--ids_file', default='train_ids.txt', type=str)
    parser.add_argument('--label_path', default='ternary_labels', type=str)
    parser.add_argument('--cache_path', default='proposal_cache', type=str)
    parser.add_argument('--proposal_timeout', default=600, type=int)
    parser.add_argument('--step_time_limit', default=200, type=int,
                        help='seconds of rollout budget per step; labels found by '
                             'then are kept (0 = unlimited). The released mining '
                             'used 100s for the same reason')
    parser.add_argument('--rollout_multiplier', default=1, type=int,
                        help='rollouts per proposal = multiplier * remaining steps; '
                             '1 matches the paper, 10 matches the released mining')
    parser.add_argument('--shard', default=0, type=int)
    parser.add_argument('--num_shards', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    random.seed(args.seed + args.shard)
    os.makedirs(args.label_path, exist_ok=True)

    data_mgr = DataManager()
    ids = read_file_to_list(args.ids_file)[args.shard::args.num_shards]
    print('shard %d/%d: %d sequences' % (args.shard, args.num_shards, len(ids)))

    done_steps = 0
    for seq_index, seq_id in enumerate(ids):
        try:
            gt_seq = data_mgr.load_processed_sequence(
                os.path.join(args.data_path, seq_id, 'gt'))
        except Exception:
            continue
        for step_index, (zone_graph, gt_extrusion) in enumerate(gt_seq):
            if gt_extrusion is None:
                continue
            out_path = os.path.join(args.label_path,
                                    '%s_%d.joblib' % (seq_id, step_index))
            if os.path.exists(out_path):
                continue
            extrusions = cached_proposals(args.cache_path, seq_id, step_index,
                                          zone_graph, timeout=args.proposal_timeout)
            if len(extrusions) < 2:
                continue
            gt_hash = gt_extrusion.hash()
            if not any(e.hash() == gt_hash for e in extrusions):
                continue
            remaining = len(gt_seq) - step_index
            labels = label_step(zone_graph, gt_extrusion, extrusions, remaining,
                                args.rollout_multiplier, args.step_time_limit)
            joblib.dump(labels, out_path)
            done_steps += 1
            print('[%d/%d] %s step %d: %d negatives / %d checked / %d candidates%s'
                  % (seq_index + 1, len(ids), seq_id, step_index,
                     len(labels['neg_hashes']), labels['checked'], labels['total'],
                     ' (time-capped)' if labels['partial'] else ''))

    print('shard done, labeled %d steps' % done_steps)


if __name__ == "__main__":
    main()
