# -*- coding: utf-8 -*-

from collections import namedtuple
import os

cur_dir = os.path.dirname(__file__)
data_path = os.path.join(cur_dir, "data/us_images_resized/")

sym_dict = {
        "DATA_PATH" : data_path,
        "LABELS_FILE" : os.path.join(data_path, "labels.txt"),
        "RESULTS_FILE" : os.path.join(data_path, "output.txt"),
        "INPUT_SIZE" : 256
        }

SYMBOLS = namedtuple("SYMBOLS", sym_dict.keys())(*sym_dict.values())