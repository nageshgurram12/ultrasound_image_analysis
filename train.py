# -*- coding: utf-8 -*-
import argparse
import torch
import copy
import torch.optim as optim
import torch.nn as nn
from dataloaders import UltrasoundDataloader
from models import DiameterEstimation
from path import SYMBOLS
import numpy as np
import timeit

class Trainer():
    def __init__(self, params):
        # create dataloader
        self.dataloader = UltrasoundDataloader(params)
        
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if not params.cv:
            self.init_model(params)

        self.params = params
     
    def init_model(self, params):
        #initialize the model
        model = DiameterEstimation(params)
        model.to(self.device)
        
        # Get parameters which are not froze to give to optimizer
        params_to_update = model.parameters()
        print("Params to learn:")
        if params.pretrained:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        
        self.model = model
        
        # Observe that all parameters are being optimized
        self.optimizer = optim.Adam(params_to_update, lr=params.lr)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10)
        self.criterion = nn.SmoothL1Loss()
        
    def train(self, prop=0):
        '''
        prop represents the proportion part to take out for K-fold CV
        Ex: if test_split=0.1, then props are [0,0.1,0.2...]
        '''
        params = self.params
        best_val_loss = float('inf')
        train_data_loader, val_data_loader = \
                    self.dataloader.load_train_val_data(prop)
        
        out = open(SYMBOLS.RESULTS_FILE, "a")
        out.write("-------- Training Results  --------- \n")
        for epoch in range(params.epochs):
            print('Epoch {}/{}'.format(epoch, params.epochs - 1))
            print('-' * 10)
              
            # set model in train mode
            self.model.train()
              
            running_loss = 0
            count = 0
            for images, diameters, img_paths  in train_data_loader:
                #images, diameters = (sample['image'], sample['label'])
                images = images.to(self.device)
                diameters = diameters.to(self.device)
                  
                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                # get predictions and calcualte loss
                if self.params.backbone == 'inceptionv3':
                    predicted, aux_pred = self.model(images)
                    loss1 = self.criterion(predicted, diameters)
                    loss2 = self.criterion(aux_pred, diameters)
                    loss = loss1 + 0.4*loss2
                else:
                    predicted = self.model(images)
                    loss = self.criterion(predicted, diameters)
                
                # backpropagate
                loss.backward()                
                self.optimizer.step()
                
                # loss.item gives mean loss over batch
                running_loss += loss.item() * images.size(0)
                count += images.size(0)
                
                if epoch == params.epochs - 1:
                    copy_pred = copy.deepcopy(predicted.detach().cpu().numpy())
                    copy_dia = copy.deepcopy(diameters.cpu().numpy())
                    for ix in range(len(copy_dia)):
                        out.write("{} \t {} \t {} \n". \
                            format(img_paths[ix], str(copy_dia[ix]), \
                                    str(copy_pred[ix])) )
                
                
            train_loss = running_loss / count
            print('{} Loss: {:.4f} '. \
                  format("Training ", train_loss))
            
            val_loss = self.val(val_data_loader, params, epoch)
            #self.scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
        
        # set model weights based on val loss
        self.model.load_state_dict(best_model_wts)
        
        out.close()
        return train_loss, val_loss
    
                    
    def val(self,  data_loader, params, epoch):
        self.model.eval()
        total_loss = 0
        count = 0
        if epoch == params.epochs - 1:
            out = open(SYMBOLS.RESULTS_FILE, "a")
            out.write("-------- Validation Results  --------- \n")
            
        for images, diameters, img_paths in data_loader:
            images = images.to(self.device)
            diameters = diameters.to(self.device)
            
            with torch.no_grad():
                # get predictions and calcualte loss
                predicted = self.model(images)
                loss = self.criterion(predicted, diameters)
                
                total_loss += loss.item() * images.size(0)
                count += images.size(0)
                                
                if epoch == params.epochs - 1:
                    copy_pred = copy.deepcopy(predicted.cpu().numpy())
                    copy_dia = copy.deepcopy(diameters.cpu().numpy())
                    for ix in range(len(copy_dia)):
                        out.write("{} \t {} \t {} \n". \
                            format(img_paths[ix], str(copy_dia[ix]), \
                                    str(copy_pred[ix])) )
                
        loss_per_image =  total_loss/ count
        print('{} Loss: {:.4f}'.format("Validation",  loss_per_image))
        return loss_per_image
    
    def test(self, phase='test'):
        
        self.model.eval()
        total_loss = 0
        data_loader = self.dataloader.load_test_data()
        count = 0
        # write results to RESULTS_FILE
        
        if self.params.cv:
            out_file = SYMBOLS.CV_RESULTS_FILE            
        else:
            out_file = SYMBOLS.RESULTS_FILE
            
        with open(out_file, "a") as out:
            out.write("Image \t Actual Diameters \t Predicted Diameters \n")
            out.write("--" * 20 + "\n")
                
            for images, diameters, img_paths in data_loader:
                #images, diameters = (sample['image'], sample['label'])
                images = images.to(self.device)
                diameters = diameters.to(self.device)
                
                with torch.no_grad():
                    # get predictions and calcualte loss
                    predicted = self.model(images)
                    loss = self.criterion(predicted, diameters)
                    
                    copy_predicted = predicted.cpu().numpy()
                    copy_diameters = diameters.cpu().numpy()
                    if self.params.cv:
                        res = np.concatenate((copy_predicted, copy_diameters), \
                                             axis=1)
                        self.cv_results = np.concatenate((self.cv_results, res), \
                                                         axis=0)
                    for ix in range(len(img_paths)):
                        out.write("{} \t {} \t {} \n". \
                        format(img_paths[ix], str(copy_diameters[ix]), \
                                str(copy_predicted[ix])) )
                    
                    total_loss += loss.item() * images.size(0)
                    count += images.size(0)
        
        loss_per_image =  total_loss/ count
        print('{} Loss: {:.4f}'.format("Testing",  loss_per_image))
        
        return loss_per_image
        
def  main():
    parser = argparse.ArgumentParser(description="Diameter Esitmation")
    
    # model argumets
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18','vgg19','inceptionv3'],
                        help="Choose the backbone model")
    parser.add_argument('--diameters', type=int, default=4,
                        help="Output responses to estimate")
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument('--predict-only-avg', action='store_true',
                        help="Predict only average of 3 cross sections")
    init_group.add_argument('--predict-only-centre', action='store_true',
                        help="Predict only centre diameter")
    
    parser.add_argument('--aug-by-crop', action='store_true',
                        help="Crop image vertically at 3 sections to augment, \
                        But for test images, it always predicts avg")
    parser.add_argument('--in-mm', action='store_true',
                        help="Convert to pixels to mm")
    
    # training hyper params
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--input-size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--val-split', type=float, default=0.1,
                        help="In K-fold cv, this represents 1/K")
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--pretrained', default=True,  
                        action='store_true')
    parser.add_argument('--cv', action='store_true', \
                        help='Cross-Validation with test_split as 1/K')
    #optimizers hyper params
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    
    params = parser.parse_args()
    
    
    if params.cv:
        #datasize = trainer.dataloader.dataset_size
        #train_size = (1-params.test_split) * datasize
        K = 0
        total_train_loss, total_val_loss, total_test_loss = (0,0,0)
        trainer = Trainer(params)
        trainer.dataloader.shuffle_indices()
        
        trainer.cv_results = np.empty((0,2))
        for ix in np.arange(0,1,params.test_split):
            # we train model on every k-fold separately
            trainer.init_model(params)       
                
            train_loss, val_loss = trainer.train(ix)
            total_train_loss += train_loss
            total_val_loss += val_loss
            
            total_test_loss += trainer.test()
            print("Split: " + str(ix) + "Test loss: "  + str(total_test_loss))
            K += 1
        
        print("---- Cross Validation ---- \n")
        print("Average Train Loss: {:.4f}".format(total_train_loss/K))
        #print("Average Val Loss: {:.4f}".format(total_val_loss/K))
        
        print("Average CV Loss: {:.4f}".format(total_test_loss/K))
        
    else:
        trainer = Trainer(params)
        trainer.train()
        
        trainer.test()
    
    
if __name__ == "__main__":
    main()