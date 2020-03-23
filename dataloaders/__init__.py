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

class UltrasoundDataloader():
    def __init__(self, params):
        self.dataset = UltrasoundData(params)
        
        dataset_size = self.dataset.datasize
        indices = list(range(dataset_size))
        val_split = params.val_split
        test_split = params.test_split
        
        val_split = int(np.floor(val_split * dataset_size))
        test_split = int(np.floor(test_split * dataset_size))
        
        np.random.seed(0)
        np.random.shuffle(indices)
        (test_indices, val_indices, train_indices) = (indices[:test_split], \
        indices[test_split:test_split+val_split], indices[test_split+val_split:])
        
        # sampler to pick train and val dataset
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(val_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)
        
        def collate(batch):
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
            img_paths = [item[2] for item in batch]
            batch_size = len(data)
            # TODO: assumption that we've only one channel
            padded_batch = torch.zeros(batch_size, 3, \
                            self.dataset.max_height, self.dataset.max_width)
            for i, img in enumerate(data):
                (c, h, w) = img.shape
                padded_batch[i, :, :h, :w] = img
                
            target =  torch.FloatTensor(target)
            return [padded_batch, target, img_paths]

        self.common_params = {'batch_size' : params.batch_size, 
                   'num_workers' : params.num_workers,
                   #'collate_fn' : collate
                   }

  
    def load_train_data(self):
        
        self.train_loader = DataLoader(self.dataset, sampler = self.train_sampler, \
                                  **self.common_params)
        return self.train_loader

    def load_val_data(self):
        self.val_loader = DataLoader(self.dataset, sampler = self.valid_sampler, \
                                     **self.common_params)
        return self.val_loader
    
    def load_test_data(self):
        self.dataset.split = 'test'
        self.test_loader = DataLoader(self.dataset, sampler = self.test_sampler,\
                                 **self.common_params)
        return self.test_loader
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params = parser.parse_args()
    params.test_split = 0.2
    params.batch_size = 4
    params.num_workers = 1
    params.predict_only_avg = True

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
        ii += 1
            
    print(size)