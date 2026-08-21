

import sys
sys.path.append('..')

import os
from pathlib import Path
import argparse
import json
import glob
import random
import math
import joblib
import multiprocessing
import time
from collections import defaultdict

from objects import *
from dataset import *

import Part
from FreeCAD import Base
import utils.solid_utils as su
class FusionDataManager:
    
    def __iter__(self):
        return self
    
    def __init__(self, data_path, shuffle=True, start_index=0, augment=False, reposition=False):
        self.num = start_index
        self.augment = augment
        self.reposition = reposition
        
        print("data_path", data_path)
        
        file_list = glob.glob(data_path + "/*.step")
        print(f'[DATA LOADER] Found {len(file_list)} models')
        file_list.sort()
        
        found_sequences = []
        current_sequence = []
        
        for i, f_dir in enumerate(file_list):
            if i < 10e20:
                f_name = f_dir.split('/')[-1]
                segments = f_name.split("_")
                
                if len(segments) == 3:
                    if (current_sequence) >= 1 :
                        found_sequences.append(current_sequence)
                        
                    current_sequence = []
                    
                else:
                    current_sequence.append(f_dir)
        if len(current_sequence) >= 1:
            found_sequences.append(current_sequence)
            
        print('found_sequences', found_sequences)
        self.pairs = []
        
        for seq in (found_sequences):
            target = seq[-1]

            # (current, next, possible target, final target, index, temp_target_count)

            if self.augment:
            # augment data
                for j,temp_target in enumerate(seq[1:]):
                    self.pairs.append((None, seq[0], temp_target, target, f'0_{str(j)}', len(seq[1:]) ))

                for i in range(len(seq)-1):
                    for j,temp_target in enumerate(seq[i+1:]):
                        self.pairs.append((seq[i], seq[i+1], temp_target, target, f'{str(i+1)}_{str(j)}', len(seq[i+1:])))
            else:
            # un-augmented data
                self.pairs.append((None, seq[0], target, target, "0", 1 ))
                for i in range(len(seq)-1):
                    self.pairs.append((seq[i], seq[i+1], target, target, f'{str(i+1)}', 1))

        
        if shuffle:
            random.shuffle(self.pairs)
        
        print(f'[DATA LOADER] Found {len(self.pairs)} extrusion pairs')
        self.pair_count = len(self.pairs)

        print('self.pairs', self.pair_count)
        
    def get_data_by_pair(self, pair):
        try:
            current_dir = pair[0]
            next_dir = pair[1]
            temp_target_dir = pair[2]
            final_target_dir= pair[3]
            index = pair[4]

            count = pair[5]

            segs = next_dir.split('/')[-1].split('_')
            file_name = '_'.join([segs[0],segs[1],segs[2]])

            data = Data()
            data.file_name = file_name
            data.index = index

            data.count = count
            data.target_shape.read(temp_target_dir)

            # calculate normalize scale factor
            final_cad_shape = Part.Shape()
            final_cad_shape.read(final_target_dir)
            bbox = final_cad_shape.BoundBox
            bbox_data = str(bbox).split('BoundBox (')[1].split(')')[0].split(',')
            bbox_data = [float(item) for item in bbox_data]
            w = bbox_data[3]-bbox_data[0]
            d = bbox_data[4]-bbox_data[1]
            h = bbox_data[5]-bbox_data[2]
            x = (bbox_data[3]+bbox_data[0])/2
            y = (bbox_data[4]+bbox_data[1])/2
            z = (bbox_data[5]+bbox_data[2])/2

            diagnal_d = math.sqrt(w*w + d*d + h*h)
            scale_factor = 1 / diagnal_d
            move_vector = Base.Vector(-x, -y, -z)

            if current_dir is None:
                data.current_shape = None

                next_cad_shape = Part.Shape()
                next_cad_shape.read(next_dir)
                data.extrusion_shape = None
                data.bool_type = 0

            else:
                # NOTE: currently extrusion is the subtraction of target and current 
                data.current_shape.read(current_dir)

                next_cad_shape = Part.Shape()
                next_cad_shape.read(next_dir)

                data.extrusion_shape = None
                if su.true_Volume(next_cad_shape) > su.true_Volume(data.current_shape):
                    data.bool_type = 0
                else:
                    data.bool_type = 1

            data.target_shape.scale(scale_factor, Base.Vector(0, 0, 0))
            if data.current_shape:
                data.current_shape.scale(scale_factor, Base.Vector(0, 0, 0))

                # normalize model location
                if self.reposition:
                    data.current_shape.translate(move_vector)

            return data
        except Exception as e:
            print(f"[SKIP] invalid shape at pair index {pair[4]}: {e}")
            return None
    
    def __next__(self):
        if self.num < len(self.pairs):
            num = self.num
            picked = self.pairs[num]
            self.num += 1

            res = self.get_data_by_pair(picked)
            if res:
                return num, res
            else:
                # NOTE: when one of the model pairs is not valid, return None

                return num, None
        else:
            raise StopIteration

def write_step_data(data, processed_fusion_path):
    """Write one pair's shapes into <output>/<design_id>/<step_index>/."""
    step_path = os.path.join(processed_fusion_path, data.file_name, str(data.index))
    os.makedirs(step_path, exist_ok=True)

    if data.current_shape and not data.current_shape.isNull():
        data.current_shape.exportStep(f"{str(step_path)}/current_shape.stp")

    data.target_shape.exportStep(f"{str(step_path)}/target_shape.stp")

    with open(f"{str(step_path)}/bool_type.txt", "w+") as f:
        f.write('addition' if data.bool_type == 0 else 'subtraction')


def process_design_extrusions(design_id, fusion_path, extrusion_path, processed_fusion_path, reposition):
    """Normalize one design's GT extrusion tools into its step folders."""
    model_dir = os.path.join(fusion_path, design_id + ".step")

    final_cad_shape = Part.Shape()
    final_cad_shape.read(model_dir)
    bbox = final_cad_shape.BoundBox
    bbox_data = str(bbox).split('BoundBox (')[1].split(')')[0].split(',')
    bbox_data = [float(item) for item in bbox_data]
    w = bbox_data[3]-bbox_data[0]
    d = bbox_data[4]-bbox_data[1]
    h = bbox_data[5]-bbox_data[2]
    x = (bbox_data[3]+bbox_data[0])/2
    y = (bbox_data[4]+bbox_data[1])/2
    z = (bbox_data[5]+bbox_data[2])/2

    diagnal_d = math.sqrt(w*w + d*d + h*h)
    scale_factor = 1 / diagnal_d
    move_vector = Base.Vector(-x, -y, -z)

    exts = glob.glob(os.path.join(extrusion_path, design_id + "*.step"))
    exts.sort()

    for j, ext in enumerate(exts):
        ext_shape = Part.Shape()
        ext_shape.read(ext)

        # normalize model location
        if (reposition):
            ext_shape.translate(move_vector)

        # normalize model scale
        ext_shape.scale(scale_factor, Base.Vector(0, 0, 0))

        out_dir = os.path.join(processed_fusion_path, design_id, str(j), "extrusion.stp")
        try:
            ext_shape.exportStep(out_dir)
        except:
            pass


def process_single_design(design_id, pairs, fusion_path, extrusion_path, processed_fusion_path, reposition):
    """Convert one design: all its step pairs, then its extrusion tools.

    Runs inside a worker process so that a STEP file that hangs the OCC reader
    only stalls this design, not the whole conversion.
    """
    # get_data_by_pair only reads self.reposition/self.augment, so skip
    # FusionDataManager.__init__ (which re-scans the whole raw directory).
    mgr = FusionDataManager.__new__(FusionDataManager)
    mgr.reposition = reposition
    mgr.augment = False
    mgr.num = 0

    for pair in pairs:
        data = mgr.get_data_by_pair(pair)
        if data:
            write_step_data(data, processed_fusion_path)

    process_design_extrusions(design_id, fusion_path, extrusion_path, processed_fusion_path, reposition)
    print('design', design_id, 'converted,', len(pairs), 'steps')


def group_pairs_by_design(pairs):
    designs = defaultdict(list)
    for pair in pairs:
        segs = pair[1].split('/')[-1].split('_')
        designs['_'.join(segs[0:3])].append(pair)
    return designs


def marker_path(processed_fusion_path, design_id):
    return os.path.join(processed_fusion_path, '.markers', design_id)


def mark_done(processed_fusion_path, design_id):
    with open(marker_path(processed_fusion_path, design_id), 'w') as f:
        f.write('done\n')


def preprocess_fusion_data(fusion_path, extrusion_path, processed_fusion_path, reposition, num_workers=1, design_timeout=0):
    """
    Convert the raw Fusion360 reconstruction data, one worker process per
    design, num_workers at a time. Some STEP files hang the OpenCascade reader
    outright (no exception, just an infinite loop in native code), so each
    design gets a deadline and is killed and skipped when it exceeds it.

    Every attempted design (converted, failed, or timed out) is recorded under
    <output_path>/.markers; re-running the same command resumes and skips them.
    Delete the .markers folder to reconvert from scratch.
    """
    dm = FusionDataManager(fusion_path, shuffle=False, start_index=0, augment=False, reposition=reposition)

    os.makedirs(processed_fusion_path, exist_ok=True)
    os.makedirs(os.path.join(processed_fusion_path, '.markers'), exist_ok=True)

    designs = group_pairs_by_design(dm.pairs)
    pending = [d for d in sorted(designs) if not os.path.exists(marker_path(processed_fusion_path, d))]
    skipped = len(designs) - len(pending)
    if skipped > 0:
        print('resume: skipping', skipped, 'already attempted designs')
    print('converting', len(pending), 'designs with', num_workers, 'workers')

    total = len(pending)
    running = []  # (process, deadline, design_id)
    done_count = 0
    while pending or running:
        while pending and len(running) < num_workers:
            design_id = pending.pop(0)
            pairs = designs[design_id]
            timeout = design_timeout if design_timeout > 0 else 600 + 60 * len(pairs)
            worker = multiprocessing.Process(target=process_single_design, name="process_single_design",
                                             args=(design_id, pairs, fusion_path, extrusion_path, processed_fusion_path, reposition))
            worker.start()
            running.append((worker, time.time() + timeout, design_id))

        time.sleep(1)

        still_running = []
        for worker, deadline, design_id in running:
            if not worker.is_alive():
                worker.join()
            elif time.time() > deadline:
                print('design', design_id, 'timed out, killing worker')
                worker.terminate()
                worker.join()
            else:
                still_running.append((worker, deadline, design_id))
                continue
            mark_done(processed_fusion_path, design_id)
            done_count += 1
            print('progress:', done_count, '/', total, 'designs attempted')
        running = still_running

    print('all designs converted !')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion_path', default='../data/fusion/reconstruction', type=str)
    parser.add_argument('--extrusion_path', default='../data/extrude', type=str)
    parser.add_argument('--output_path', default='../data/fusion_processed', type=str)
    parser.add_argument('--reposition', default=False, type=bool)
    parser.add_argument('--num_workers', default=max(1, multiprocessing.cpu_count() - 2), type=int,
                        help='designs converted concurrently (default: cpu count - 2)')
    parser.add_argument('--design_timeout', default=0, type=int,
                        help='seconds allowed per design; 0 = automatic (600 + 60 per step)')

    args = parser.parse_args()

    preprocess_fusion_data(args.fusion_path, args.extrusion_path, args.output_path, args.reposition, args.num_workers, args.design_timeout)
