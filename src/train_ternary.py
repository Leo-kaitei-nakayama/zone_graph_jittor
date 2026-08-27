"""Paper-faithful ternary training (Sec. 5.2) — additive; base untouched.

Trains with the SAME loss and batching style as the released recipe
(Agent.update_by_extrusion: focal loss, batches of hp.batch_size), but on
the paper's labels produced by ternary_label.py: per GT step, the positive
plus EVERY p == 0 negative — neutrals excluded. This is the released
train.py with its one-negative simplification undone.

Steps whose label file does not exist yet are skipped, so training can run
on a partially labeled set while ternary_label.py shards are still working;
by default the trainer reports coverage at the start of each epoch.

    python train_ternary.py --data_path processed_data \
        --output_path train_output_ternary

Evaluate with the unchanged evaluator:
    python rank_eval.py --train_output train_output_ternary ...
"""

import sys
sys.path.append('..')

import os
import argparse
import copy
import random
import shutil

import numpy as np
import joblib
import jittor as jt

from dataset import DataManager
from agent import Agent, to_tensor
from train_listwise import cached_proposals, validate
from utils.file_utils import read_file_to_list, write_list_to_file
import hyperparameters as hp


def step_examples(zone_graph, gt_extrusion, extrusions, neg_hashes,
                  max_negatives=0):
    """(graph, label) pairs for one step: the GT as positive plus every
    labeled negative. Graphs are deepcopied because encode_with_extrusion
    mutates node attributes and the batch is encoded later, all at once —
    the same layout the released training data had on disk."""
    neg_set = set(neg_hashes)
    negatives = [e for e in extrusions if e.hash() in neg_set]
    if max_negatives and len(negatives) > max_negatives:
        negatives = random.sample(negatives, max_negatives)

    examples = []
    g = copy.deepcopy(zone_graph)
    g.encode_with_extrusion(gt_extrusion)
    examples.append((g, 1))
    for e in negatives:
        g = copy.deepcopy(zone_graph)
        g.encode_with_extrusion(e)
        examples.append((g, 0))
    return examples


def train(args):
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)

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

    def run_validation():
        nonlocal min_validation_loss
        validation_loss = validate(agent, args.data_path, args.cache_path,
                                   args.validate_limit, args.proposal_timeout)
        validation_loss_list.append(validation_loss)
        write_list_to_file(os.path.join(args.output_path, 'validationloss.txt'),
                           validation_loss_list)
        if validation_loss <= min_validation_loss:
            min_validation_loss = validation_loss
            agent.save_best_weights()
            print('new best checkpoint (validation rank sum %d)' % validation_loss)

    gs = []
    ls = []
    batch_nodes = 0

    def flush_batch():
        nonlocal gs, ls, batch_nodes
        if gs:
            loss = agent.update_by_extrusion(ls, gs)
            train_loss_list.append(loss)
            gs, ls, batch_nodes = [], [], 0

    for epoch_index in range(args.epochs):
        print('epoch', epoch_index, '--------------------------------------------')
        used = 0
        labeled_steps = 0
        skipped_unlabeled = 0
        skipped_large = 0
        for seq_id in train_ids:
            try:
                gt_seq = data_mgr.load_processed_sequence(
                    os.path.join(args.data_path, seq_id, 'gt'))
            except Exception:
                continue
            seq_used = False
            for step_index, (zone_graph, gt_extrusion) in enumerate(gt_seq):
                if gt_extrusion is None:
                    continue
                label_path = os.path.join(args.label_path,
                                          '%s_%d.joblib' % (seq_id, step_index))
                if not os.path.exists(label_path):
                    skipped_unlabeled += 1
                    continue
                try:
                    labels = joblib.load(label_path)
                except Exception:
                    continue
                if not labels['neg_hashes']:
                    continue
                extrusions = cached_proposals(args.cache_path, seq_id, step_index,
                                              zone_graph, timeout=args.proposal_timeout)
                if not extrusions:
                    continue
                # GPU memory guard: a batch's activations scale with its total
                # zone count, so bound both the graph size and the batch's node
                # total. (The base recipe never met the giant graphs — their
                # negative mining timed out, silently dropping those steps.)
                n_zones = zone_graph.zone_graph.number_of_nodes()
                if args.max_zones and n_zones > args.max_zones:
                    skipped_large += 1
                    continue
                for g, label in step_examples(zone_graph, gt_extrusion, extrusions,
                                              labels['neg_hashes'], args.max_negatives):
                    if gs and (len(gs) >= hp.batch_size or
                               batch_nodes + n_zones > args.max_batch_nodes):
                        flush_batch()
                    gs.append(g)
                    ls.append(to_tensor([label]))
                    batch_nodes += n_zones
                labeled_steps += 1
                seq_used = True
            if seq_used:
                used += 1
                if used % 25 == 0:
                    recent = train_loss_list[-100:]
                    if recent:
                        print('epoch %d  sequences %d  steps %d  avg loss (last %d updates) %.4f'
                              % (epoch_index, used, labeled_steps, len(recent),
                                 sum(recent) / len(recent)))
                    write_list_to_file(os.path.join(args.output_path, 'trainloss.txt'),
                                       train_loss_list)
                    agent.save_weights()
                if args.validate_every and used % args.validate_every == 0:
                    run_validation()
                    agent.train()
                if args.limit and used >= args.limit:
                    break

        print('epoch %d done: %d labeled steps used, %d unlabeled, %d over max_zones'
              % (epoch_index, labeled_steps, skipped_unlabeled, skipped_large))
        agent.save_weights()
        write_list_to_file(os.path.join(args.output_path, 'trainloss.txt'), train_loss_list)
        run_validation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ZoneGraph ternary trainer (paper Sec. 5.2, additive)')
    parser.add_argument('--data_path', default='processed_data', type=str)
    parser.add_argument('--output_path', default='train_output_ternary', type=str,
                        help='recreated on each run; point rank_eval.py here')
    parser.add_argument('--label_path', default='ternary_labels', type=str,
                        help='directory written by ternary_label.py')
    parser.add_argument('--cache_path', default='proposal_cache', type=str)
    parser.add_argument('--proposal_timeout', default=600, type=int)
    parser.add_argument('--init_from', default='', type=str,
                        help='optional folder with best_*.pkl to fine-tune from '
                             '(default: from scratch, like the paper)')
    parser.add_argument('--epochs', default=hp.train_epoch_num, type=int)
    parser.add_argument('--lr', default=0.0, type=float,
                        help='override learning rate; 0 keeps the base recipe value')
    parser.add_argument('--max_negatives', default=0, type=int,
                        help='cap negatives per step (resampled each epoch); 0 = all, as the paper')
    parser.add_argument('--max_zones', default=1000, type=int,
                        help='skip steps whose zone graph exceeds this many zones; 0 disables')
    parser.add_argument('--max_batch_nodes', default=4000, type=int,
                        help='flush the batch early once its total zone count would exceed '
                             'this GPU memory budget; 0 disables')
    parser.add_argument('--limit', default=0, type=int)
    parser.add_argument('--validate_limit', default=0, type=int)
    parser.add_argument('--validate_every', default=1000, type=int,
                        help='also validate every N training sequences; 0 = epoch end only')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    train(args)
