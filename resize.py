#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:09:08 2020

@author: nageswara
"""

from path import SYMBOLS
from PIL import Image
import os

def resize():
    orig_imgs_path = os.path.join(__file__, "/data/us_images")
    resized_imgs_path =  os.path.join(__file__, "/data/us_images_resized")
    for filename in os.listdir(orig_imgs_path):
        if filename.endswith(".jpg"):
            file = Image.open(os.path.join(orig_imgs_path, filename))
            resized = file.resize((SYMBOLS.INPUT_SIZE, SYMBOLS.INPUT_SIZE))
            filename = os.path.join(resized_imgs_path, filename)
            open(filename, 'a').close()
            resized.save(filename)
            

if __name__ == "__main__":
    resize()