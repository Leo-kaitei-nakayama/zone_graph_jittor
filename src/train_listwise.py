"""Listwise trainer — an ADDITIVE experiment; no base file is modified.

The released recipe (train.py) trains a binary classifier on two examples
per ground-truth step: the GT extrusion (label 1) and one mined negative
(label 0). Evaluation, however, ranks the GT against ALL proposals of the
zone graph (rank_eval.py, search). Both our torch and jittor runs reproduce
that recipe faithfully and both land at an average relative rank around
0.11 — far from the paper's 0.036 — because the training objective and the
evaluation task are different problems.

This script trains the SAME architecture, with the SAME checkpoint format,
on the exact task the evaluation measures: for each GT step it generates
the full proposal list (the same get_proposals the evaluation uses), scores
every candidate with the network, and applies a softmax cross-entropy over
the candidate set with the GT as the target. The score entering the softmax
is log P(good) — precisely the quantity evaluation.sort_extrusions_by_agent
sorts by — so lowering this loss is literally raising the GT's rank.

The base recipe stays untouched and runnable as the faithful baseline:
train.py, agent.py, models.py, evaluation.py and rank_eval.py are not
changed. This file only imports them. Results from either trainer are
evaluated by the same unchanged rank_eval.py, so the two are directly
comparable and any difference is attributable to the objective alone.

Usage (same data layout and working directory as train.py):

    python train_listwise.py --data_path processed_data \
        --output_path train_output_listwise \
        --init_from train_output          # optional: start from the base
                                          # recipe's best checkpoint
Then evaluate with the unchanged evaluator:

    python rank_eval.py --data_path processed_data --ids_file test_ids.txt \
        --train_output train_output_listwise --limit 150

Proposal generation is pure geometry (independent of the weights), so the
per-(sequence, step) proposal lists are cached on disk after epoch 0; later
epochs skip that cost entirely. The cache directory survives reruns.
"""

import sys
sys.path.append('..')

import os
import argparse
import random
import shutil
import signal

import numpy as np
import joblib
import jittor as jt
import jittor.nn as nn

from dataset import DataManager
from proposal import get_proposals
from evaluation import sort_extrusions_by_agent
from agent import Agent
from utils.file_utils import read_file_to_list, write_list_to_file
import hyperparameters as hp


def batched_log_probs(agent, g_encs):
    """Same batch assembly as Agent.make_decision (agent.py), including the
    shared-adjacency fast path, but returns the network's log-probabilities
    instead of exponentiating them, so the listwise loss can stay in log
    space. Duplicated here rather than modifying the base Agent."""
    sizes = [enc[0].shape[0] for enc in g_encs]
    h = jt.concat([enc[0] for enc in g_encs], dim=0)

    first = g_encs[0][1]
    shared = all(enc[1] is first or
                 (enc[1].shape == first.shape and np.array_equal(enc[1], first))
                 for enc in g_encs)
    if shared:
        adj = jt.array(first)
    else:
        total = sum(sizes)
        A = np.zeros((total, total), dtype='float32')
        start = 0
        for h_i, adj_i in g_encs:
            k = adj_i.shape[0]
            A[start:start + k, start:start + k] = adj_i
            start += k
        adj = jt.array(A)

    return agent.decision_maker(h, adj, graph_sizes=sizes)


def listwise_loss(agent, g_encs, gt_index):
    """Softmax cross-entropy over the candidate set. The per-candidate score
    is log P(good) — the quantity sort_extrusions_by_agent ranks by — so the
    loss is minimized exactly when the GT outscores every other proposal."""
    log_probs = batched_log_probs(agent, g_encs)
    scores = log_probs[:, 1].reshape((1, -1))
    return -nn.log_softmax(scores, dim=1)[0, gt_index]


def candidate_cap(num_zones, max_candidates, max_nodes):
    """Memory guard: one update's activations scale with num_zones *
    num_candidates (every candidate re-encodes every zone with gradients
    held), so on top of the fixed --max_candidates cap, bound the total
    node count of the batch. The floor of 2 (GT + one negative) is the base
    recipe's scale, which fits any graph the base trainer could handle.
    Negatives are resampled every epoch, so a heavily capped step still
    sees different negatives each time."""
    cap = max_candidates if max_candidates else 10 ** 9
    if max_nodes:
        cap = min(cap, max_nodes // max(1, num_zones))
    return max(2, cap)


def subsample_candidates(extrusions, gt_hash, max_candidates):
    """Bound the per-update candidate list. The GT is always kept; negatives
    are drawn uniformly and freshly on every call, so across epochs the
    network still sees the full candidate distribution (unlike the base
    recipe's single fixed hardest negative). Returns (candidates, gt_index),
    or (None, None) when the GT is not among the proposals."""
    gt = None
    negatives = []
    for e in extrusions:
        if gt is None and e.hash() == gt_hash:
            gt = e
        else:
            negatives.append(e)
    if gt is None or not negatives:
        return None, None
    if max_candidates and len(negatives) > max_candidates - 1:
        negatives = random.sample(negatives, max_candidates - 1)
    candidates = negatives + [gt]
    random.shuffle(candidates)
    for i, e in enumerate(candidates):
        if e is gt:
            return candidates, i


def train_step(agent, zone_graph, candidates, gt_index):
    # per-candidate encoding identical to evaluation.sort_extrusions_by_agent,
    # but with gradients enabled
    g_encs = []
    for extrusion in candidates:
        zone_graph.encode_with_extrusion(extrusion)
        g_encs.append(agent.encode_zone_graph(zone_graph))
    loss = listwise_loss(agent, g_encs, gt_index)
    agent.optim_extrusion.step(loss)
    return loss.item()


class _ProposalTimeout(Exception):
    pass


def _raise_proposal_timeout(signum, frame):
    raise _ProposalTimeout()


def cached_proposals(cache_dir, seq_id, step_index, zone_graph, timeout=0):
    """get_proposals depends only on the zone graph geometry, never on the
    network, so each (sequence, step)'s proposal list is computed once and
    reused by every later epoch and by validation. A failure — or exceeding
    `timeout` seconds; some pathological graphs take extremely long, which is
    why the original preprocessing carried its own time limits — caches an
    empty list so the bad step is skipped cheaply forever after. The timeout
    uses SIGALRM, so it only applies on the main thread (where training
    runs) and fires once Python bytecode is executing."""
    path = None
    if cache_dir:
        path = os.path.join(cache_dir, '%s_%d.joblib' % (seq_id, step_index))
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                pass
    try:
        if timeout:
            signal.signal(signal.SIGALRM, _raise_proposal_timeout)
            signal.alarm(timeout)
        try:
            extrusions = get_proposals(zone_graph)
        finally:
            if timeout:
                signal.alarm(0)
    except _ProposalTimeout:
        print('proposals timed out (>%ds) for %s step %d - skipping this step'
              % (timeout, seq_id, step_index))
        extrusions = []
    except Exception as e:
        print('proposals failed for', seq_id, 'step', step_index, ':', e)
        extrusions = []
    if path:
        try:
            joblib.dump(extrusions, path)
        except Exception as e:
            print('proposal cache write failed:', e)
    return extrusions


def validate(agent, data_path, cache_dir, limit=0, timeout=0):
    """Rank sum of the GT extrusion over the validation set — the same
    quantity train.py's validate() computes — used only to pick the best
    checkpoint. One deliberate difference: the networks are switched to eval
    mode for deterministic BatchNorm, so the numbers are self-consistent but
    not comparable 1:1 with the base validationloss.txt (the released code
    validated with the nets left in training mode)."""
    print('validation----------------------------------------')
    agent.eval()
    data_mgr = DataManager()
    total_rank_sum = 0
    steps = 0
    used = 0
    for seq_id in read_file_to_list('validate_ids.txt'):
        try:
            gt_seq = data_mgr.load_processed_sequence(os.path.join(data_path, seq_id, 'gt'))
        except Exception:
            continue
        seq_used = False
        for step_index, (zone_graph, gt_extrusion) in enumerate(gt_seq):
            if gt_extrusion is None:
                continue
            extrusions = cached_proposals(cache_dir, seq_id, step_index, zone_graph,
                                          timeout=timeout)
            if len(extrusions) < 2:
                continue
            gt_hash = gt_extrusion.hash()
            if not any(e.hash() == gt_hash for e in extrusions):
                continue
            ranked = sort_extrusions_by_agent(list(extrusions), zone_graph, agent)
            for i, extrusion in enumerate(ranked):
                if extrusion.hash() == gt_hash:
                    total_rank_sum += i
                    steps += 1
                    break
            seq_used = True
        if seq_used:
            used += 1
            if limit and used >= limit:
                break
    agent.train()
    print('validation rank sum', total_rank_sum, 'over', steps, 'steps',
          '(%d sequences)' % used)
    return total_rank_sum


def train(args):
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)
    if args.cache_path:
        os.makedirs(args.cache_path, exist_ok=True)

    random.seed(args.seed)
    jt.set_global_seed(args.seed)

    agent = Agent(args.output_path)
    if args.init_from:
        agent.zone_encoder.load(os.path.join(args.init_from, 'best_zone_encoder.pkl'))
        agent.decision_maker.load(os.path.join(args.init_from, 'best_decision_maker.pkl'))
        print('initialized from best checkpoint in', args.init_from)
    if args.lr > 0:
        agent.optim_extrusion.lr = args.lr

    data_mgr = DataManager()
    train_ids = read_file_to_list('train_ids.txt')

    train_loss_list = []
    validation_loss_list = []
    min_validation_loss = np.inf

    for epoch_index in range(args.epochs):
        print('epoch', epoch_index, '--------------------------------------------')
        used = 0
        for seq_id in train_ids:
            try:
                gt_seq = data_mgr.load_processed_sequence(os.path.join(args.data_path, seq_id, 'gt'))
            except Exception:
                continue
            seq_used = False
            for step_index, (zone_graph, gt_extrusion) in enumerate(gt_seq):
                if gt_extrusion is None:
                    continue
                extrusions = cached_proposals(args.cache_path, seq_id, step_index, zone_graph,
                                              timeout=args.proposal_timeout)
                if len(extrusions) < 2:
                    continue
                cap = candidate_cap(zone_graph.zone_graph.number_of_nodes(),
                                    args.max_candidates, args.max_nodes)
                candidates, gt_index = subsample_candidates(
                    extrusions, gt_extrusion.hash(), cap)
                if candidates is None:
                    continue
                loss = train_step(agent, zone_graph, candidates, gt_index)
                train_loss_list.append(loss)
                seq_used = True
            if seq_used:
                used += 1
                if used % 25 == 0:
                    recent = train_loss_list[-100:]
                    print('epoch %d  sequences %d  avg loss (last %d updates) %.4f'
                          % (epoch_index, used, len(recent), sum(recent) / len(recent)))
                    write_list_to_file(os.path.join(args.output_path, 'trainloss.txt'),
                                       train_loss_list)
                    agent.save_weights()
                # per-step updates make the validation curve noisy, so an
                # epoch-end-only check can miss the best weights; optionally
                # validate mid-epoch too and keep the best seen anywhere
                if args.validate_every and used % args.validate_every == 0:
                    validation_loss = validate(agent, args.data_path, args.cache_path,
                                               args.validate_limit, args.proposal_timeout)
                    validation_loss_list.append(validation_loss)
                    write_list_to_file(os.path.join(args.output_path, 'validationloss.txt'),
                                       validation_loss_list)
                    if validation_loss <= min_validation_loss:
                        min_validation_loss = validation_loss
                        agent.save_best_weights()
                        print('new best checkpoint (validation rank sum %d)' % validation_loss)
                if args.limit and used >= args.limit:
                    break

        agent.save_weights()
        write_list_to_file(os.path.join(args.output_path, 'trainloss.txt'), train_loss_list)

        validation_loss = validate(agent, args.data_path, args.cache_path,
                                   args.validate_limit, args.proposal_timeout)
        validation_loss_list.append(validation_loss)
        write_list_to_file(os.path.join(args.output_path, 'validationloss.txt'),
                           validation_loss_list)
        if validation_loss <= min_validation_loss:
            min_validation_loss = validation_loss
            agent.save_best_weights()
            print('new best checkpoint (validation rank sum %d)' % validation_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ZoneGraph listwise trainer (additive experiment)')
    parser.add_argument('--data_path', default='processed_data', type=str)
    parser.add_argument('--output_path', default='train_output_listwise', type=str,
                        help='recreated on each run, like train.py; point rank_eval.py here')
    parser.add_argument('--init_from', default='', type=str,
                        help='folder with best_*.pkl from the base recipe to fine-tune from')
    parser.add_argument('--epochs', default=hp.train_epoch_num, type=int)
    parser.add_argument('--max_candidates', default=64, type=int,
                        help='cap per-update candidate list (GT always kept, negatives resampled each epoch); 0 = no cap')
    parser.add_argument('--max_nodes', default=4000, type=int,
                        help='GPU memory guard: cap candidates so candidates * zones '
                             'stays under this node budget (floor of 2 candidates); '
                             '0 disables. Lower it if training still runs out of memory')
    parser.add_argument('--lr', default=0.0, type=float,
                        help='override learning rate; 0 keeps hyperparameters.learning_rate_optim_extrusion')
    parser.add_argument('--limit', default=0, type=int,
                        help='max training sequences per epoch, for quick trials; 0 = all')
    parser.add_argument('--validate_limit', default=0, type=int,
                        help='max validation sequences per epoch; 0 = all')
    parser.add_argument('--validate_every', default=0, type=int,
                        help='additionally validate (and update the best checkpoint) every '
                             'N training sequences within an epoch; 0 = epoch end only')
    parser.add_argument('--proposal_timeout', default=600, type=int,
                        help='seconds allowed per proposal generation; a step exceeding '
                             'it is skipped (and cached as skipped). 0 disables. The '
                             'original preprocessing used time limits for the same reason')
    parser.add_argument('--cache_path', default='proposal_cache', type=str,
                        help='on-disk proposal cache reused across epochs and runs; empty string disables')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    train(args)
