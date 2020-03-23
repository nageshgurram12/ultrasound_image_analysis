# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch
from path import SYMBOLS
import os
from PIL import Image
from torchvision import transforms as tr

class UltrasoundData(Dataset):
    
    def __init__(self, params, split='train'):
        super().__init__()
        
        if 'base_dir' not in params:
            self.base_dir = SYMBOLS.DATA_PATH
        else:
            self.base_dir = params.base_dir
        
        
        self.labels_file = SYMBOLS.LABELS_FILE
        
        self.images = []
        self.targets = []
        self.box_coords = []
        self.max_width = float('-inf')
        self.max_height = float('-inf')
        
        with open(self.labels_file) as f:
            samples = f.readlines()
        
        for sample in samples:
            (image, diameters) = sample.split(" ")
            self.images.append(image)
            
            diameters = [round(float(d),2) for d in diameters.split(",")]
            average = sum(diameters)/len(diameters)
            diameters.append(average)
            if params.predict_only_avg:
                self.targets.append([average])
            else:
                self.targets.append(diameters)
            
            '''
            co_ords = [int(float(c)) for c in box.rstrip()[:-1].split(",")]
            if co_ords[2] > self.max_width:
                self.max_width = co_ords[2]
            
            if co_ords[3] > self.max_height:
                self.max_height = co_ords[3]
                
            self.box_coords.append(co_ords)
            '''
            
        self.datasize = len(self.images)
        
        self.split = split
        
        # Define the augmentation pipeline
        self.train_aug_pipeline = tr.Compose([
            tr.Grayscale(3),
            #tr.Resize((params.input_size,params.input_size)),
            tr.RandomHorizontalFlip(),
            tr.RandomVerticalFlip(),
            tr.RandomCrop((params.input_size,params.input_size), pad_if_needed=True),
            tr.RandomRotation(45),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
        self.test_aug_pipeline = tr.Compose([
            tr.Grayscale(3),
            #tr.Resize((params.input_size,params.input_size)),    
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
                
        
    def __len__(self):
        return self.datasize
    
    def __getitem__(self, index):
        image_path = self.images[index]
        abs_image_path = os.path.join(self.base_dir, image_path)
        image = Image.open(abs_image_path).convert("L")
        
        target = self.targets[index]
        
        # crop according to the box
        '''
        box = self.box_coords[index]
        (width, height) = image.size
        cwidth = box[0] + box[2]
        if cwidth > width:
            cwidth = width
        
        image = image.crop((box[0], box[1], cwidth, box[1]+box[3]))
        '''
        if self.split == 'train':
            image = self.train_aug_pipeline(image)
        else:
            image = self.test_aug_pipeline(image)
        
        target =  torch.FloatTensor(target)
        return [image, target, image_path]


    