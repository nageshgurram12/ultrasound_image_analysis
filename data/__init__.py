# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch
from path import SYMBOLS
import os
from PIL import Image
from torchvision import transforms as tr
import numpy as np
import sys

class UltrasoundData(Dataset):
    
    def __init__(self, params, split='train'):
        super().__init__()
        
        if not(SYMBOLS.CROPPED or SYMBOLS.RESIZED):
            print("Either Images has to be cropped or resized")
            sys.exit()
        
        if params.aug_by_crop != SYMBOLS.CROPPED:
            print("Both of these should be equal")
            sys.exit()
            
        self.base_dir = SYMBOLS.DATA_PATH
        self.labels_file = SYMBOLS.LABELS_FILE
        
        self.images = []
        self.targets = []
        self.box_coords = []
        self.max_width = float('-inf')
        self.max_height = float('-inf')
        
        with open(self.labels_file) as f:
            samples = f.readlines()
        
        for sample in samples:
            (image, diameters, box) = sample.split(" ")
                
            self.images.append(image)
            
            diameters = [round(float(d),2) for d in diameters.split(",")]
            # reversing the list as these represent right to left
            diameters.reverse()
            average = round(sum(diameters)/len(diameters),2)
            diameters.append(average)
            self.targets.append(diameters)
            

            if SYMBOLS.CROPPED:
                co_ords = [int(float(c)) for c in box.rstrip().split(",")]
                if co_ords[2] > self.max_width:
                    self.max_width = co_ords[2]
                
                if co_ords[3] > self.max_height:
                    self.max_height = co_ords[3]
                    
                self.box_coords.append(co_ords)
            
            
        self.datasize = len(self.images)
        
        self.split = split
        self.params = params
        
        # Define the augmentation pipeline
        self.train_aug_pipeline = tr.Compose([
            # We need 3 channels as models expect images in 3-dim
            tr.Grayscale(3), 
            #tr.Resize((params.input_size,params.input_size)),
            tr.RandomHorizontalFlip(),
            #tr.RandomVerticalFlip(),
            #tr.RandomCrop((params.input_size,params.input_size), \
            #pad_if_needed=True),
            #tr.RandomRotation(45),
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
        self.test_aug_pipeline = tr.Compose([
            tr.Grayscale(3),
            #tr.Resize((params.input_size,params.input_size)),    
            tr.ToTensor(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
                
    
    def get_cropped_image(self, image, choice):
        (width, height) = image.size
        if choice == 0:
            image = image.crop((0, 0, width/3, height))
        elif choice == 1:
            image = image.crop((width/3, 0, 2*width/3, height))
        elif choice == 2:
            image = image.crop((2*width/3, 0, width, height))
        
        return image
        
    def __len__(self):
        return self.datasize
    
    def __getitem__(self, index):
        image_path = self.images[index]
        abs_image_path = os.path.join(self.base_dir, image_path)
        image = Image.open(abs_image_path).convert("L")
        
        target = self.targets[index] # 3 diameters and average
        
        # crop according to the box
        if SYMBOLS.CROPPED:
            box = self.box_coords[index]
            (width, height) = image.size
            cwidth = box[0] + box[2]
            if cwidth > width:
                cwidth = width
            
            image = image.crop((box[0], box[1], cwidth, box[1]+box[3]))
        else:
            # -- We supply entire image to model--
            
            # no need of collate fn in dataloader
            if self.params.predict_only_avg:
                target = [target[3]]
            target =  torch.FloatTensor(target) 
        
        # Generate 3 images for 3 diameter cross sections and give random
        if self.params.aug_by_crop and self.split != 'test':
            # take cropped image with p=0.2 and original image with p=0.4
            choice = np.random.choice(4, 1, p=[0.2,0.2,0.2,0.4])[0]
            image = self.get_cropped_image(image, choice)
            target = target[choice] # last is average in targets
            #print(image_path + " " + str(choice) + " " + str(target))
            
            
        if self.split == 'train':
            image = self.train_aug_pipeline(image)
        else:
            image = self.test_aug_pipeline(image)
                
        return [image, target, image_path]


    