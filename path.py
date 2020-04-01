# -*- coding: utf-8 -*-

from collections import namedtuple
import os

cur_dir = os.path.dirname(__file__)
RESIZED = False
img_dir = "us_images"
if RESIZED:
    img_dir += "_resized"
    
data_path = os.path.join(cur_dir, "data/" + img_dir + "/")

sym_dict = {
        "DATA_PATH" : data_path,
        "LABELS_FILE" : os.path.join(data_path, "labels.txt"),
        "RESULTS_FILE" : os.path.join(data_path, "output.txt"),
        "INPUT_SIZE" : 256,
        "CROPPED" : True,
        "RESIZED" : RESIZED,
        "PX_TO_MM" :  0.26458
        }

SYMBOLS = namedtuple("SYMBOLS", sym_dict.keys())(*sym_dict.values())