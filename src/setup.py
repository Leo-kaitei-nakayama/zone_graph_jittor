"""
Setup functions for the running environment

Point FREECAD_LIB_PATH at the lib directory of a FreeCAD installation
(e.g. <conda env>/lib). Override without editing this file:

    export FREECAD_LIB_PATH=/path/to/freecad/lib
"""

import os
import sys

FREECAD_LIB_PATH = os.environ.get(
    "FREECAD_LIB_PATH", "/home/zhangkaicheng/miniconda3/envs/zonegraphs/lib")

sys.path.append(FREECAD_LIB_PATH)
