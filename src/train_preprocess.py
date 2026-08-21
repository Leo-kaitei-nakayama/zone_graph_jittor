import sys
sys.path.append('..')

import numpy as np
import os

import argparse
import numpy as np
from dataset import *
from objects import *
from proposal import *
import multiprocessing
import shutil
import time
import matplotlib.pyplot as plt
import copy
import joblib


def hit_target_in_path(zone_graph, depth, max_depth):
    if zone_graph.is_done():
        return True

    if depth == max_depth:
        return False

    next_extrusions = get_proposals(zone_graph)
    if len(next_extrusions) == 0:
        return False
    random.shuffle(next_extrusions)
    next_zone_graph = zone_graph.update_to_next_zone_graph(next_extrusions[0])
    ret = hit_target_in_path(next_zone_graph, depth+1, max_depth)
    
    return ret