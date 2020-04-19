#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:57:55 2020

@author: nageswara
"""

# -*- coding: utf-8 -*-
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from data import UltrasoundData
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
from path import SYMBOLS
import random

class UltrasoundDataloader():
    def __init__(self, params):
        self.dataset = UltrasoundData(params)
        
        self.dataset_size = dataset_size = self.dataset.datasize
        self.indices = indices = list(range(dataset_size))
        val_split = params.val_split
        test_split = params.test_split
        
        val_split = int(np.floor(val_split * dataset_size))
        test_split = int(np.floor(test_split * dataset_size))
        
        if not params.cv:
            self.shuffle_indices()
            (test_indices, val_indices, train_indices) = \
                (indices[:test_split], \
                 indices[test_split:test_split+val_split], 
                 indices[test_split+val_split:])
            
            # sampler to pick train and val dataset
            self.train_sampler = SubsetRandomSampler(train_indices)
            self.valid_sampler = SubsetRandomSampler(val_indices)
            self.test_sampler = SubsetRandomSampler(test_indices)
        
        def collate(batch):
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
            img_paths = [item[2] for item in batch]
            batch_size = len(data)
            padded_batch = torch.zeros(batch_size, 3, \
                            self.dataset.max_height, self.dataset.max_width)
            for i, img in enumerate(data):
                (c, h, w) = img.shape
                padded_batch[i, :, :h, :w] = img
            
            target =  torch.FloatTensor(target)
            target = target.view((batch_size, 1))
            return [padded_batch, target, img_paths]

        self.common_params = {'batch_size' : params.batch_size, 
                              'num_workers' : params.num_workers,
                              }
        
        # If we'are using cropped images then padding is needed
        if SYMBOLS.CROPPED:
            self.common_params['collate_fn'] = collate

        self.params = params

    def get_samplers(self, prop):
        size = len(self.indices)
        test_size = int(np.floor(self.params.test_split*size))
        test_indices = self.indices[:test_size]
        
        # take out test part 
        train_val_indices = self.indices[test_size:]
        #val_size = int(np.floor(self.params.val_split*size))
        train_size = len(train_val_indices)
        val_low_bound = int(np.floor(prop * train_size))
        val_upp_bound = int(np.floor((prop + self.params.val_split) * train_size))
        # represents K-fold set of validation
        val_indices = train_val_indices[val_low_bound:val_upp_bound]
        
        # remaining are for training
        train_indices = train_val_indices[0:val_low_bound]
        train_indices.extend(train_val_indices[val_upp_bound:])
        
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(val_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)
        
    def load_train_val_data(self, ix=0):
        if self.params.cv:
            self.get_samplers(ix)
            
        self.train_loader = DataLoader(self.dataset, sampler=self.train_sampler,\
                                  **self.common_params)
        self.val_loader = DataLoader(self.dataset, sampler=self.valid_sampler,\
                                     **self.common_params)
        return self.train_loader, self.val_loader

    
    def load_test_data(self):
        self.dataset.split = 'test'
        self.test_loader = DataLoader(self.dataset, sampler=self.test_sampler,\
                                 **self.common_params)
        return self.test_loader
        
    def shuffle_indices(self):
        indices = self.indices
        random.Random(0).shuffle(indices)
        self.indices = indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params = parser.parse_args()
    params.test_split = 0.2
    params.val_split = 0.1
    params.batch_size = 4
    params.num_workers = 1
    params.predict_only_avg = False
    params.aug_by_crop = True
    params.in_mm = True
    params.cv = False

    dataloader = UltrasoundDataloader(params)
    
    size = 0; ii = 0
    for imgs, dias, paths in dataloader.load_train_data():
        size += params.batch_size
        for jj in range(params.batch_size):
            if ii == 0:
                img = imgs[jj]
                img = img.numpy()
                img = np.transpose(img, axes=[1,2,0])
                img *= (0.229)
                img += (0.485)
                img *= 255.0
                img = img.astype(np.uint8)
                plt.figure()
                plt.imshow(img.squeeze())
        break
            
    print(size)
