# -*- coding: utf-8 -*-

from collections import namedtuple
import os

cur_dir = os.path.dirname(__file__)
RESIZED = True
img_dir = "us_images"
if RESIZED:
    img_dir += "_resized"
    
data_path = os.path.join(cur_dir, "data/" + img_dir + "/")

sym_dict = {
        "DATA_PATH" : data_path,
        "LABELS_FILE" : os.path.join(data_path, "labels.txt"),
        "TRAIN_RESULTS_FILE" : os.path.join(data_path, "train_log.txt"),
        "TEST_RESULTS_FILE" : os.path.join(data_path, "test_log.txt"),
        "CV_RESULTS_FILE" : os.path.join(data_path, "cv_results.txt"),
        "INPUT_SIZE" : 256,
        "CROPPED" : True, # Set this flag to always true
        "RESIZED" : RESIZED,
        "PX_TO_MM" :  0.103
        }

SYMBOLS = namedtuple("SYMBOLS", sym_dict.keys())(*sym_dict.values())