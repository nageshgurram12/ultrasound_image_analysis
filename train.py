# -*- coding: utf-8 -*-
import argparse
import torch
import copy
import torch.optim as optim
import torch.nn as nn
from dataloaders import UltrasoundDataloader
from models import DiameterEstimation
from path import SYMBOLS

class Trainer():
    def __init__(self, params):
        # create dataloader
        self.dataloader = UltrasoundDataloader(params)
        
        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #initialize the model
        model = DiameterEstimation(params)
        model.to(device)
        
        # Get parameters which are not froze to give to optimizer
        params_to_update = model.parameters()
        print("Params to learn:")
        if params.pretrained:
            params_to_update = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        
        
        # Observe that all parameters are being optimized
        self.optimizer = optim.SGD(params_to_update, lr=params.lr, \
                                   momentum=params.momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10)
        self.criterion = nn.MSELoss()

        self.model = model
        self.params = params
        self.device = device
        
    def train(self):
        
        params = self.params
        best_val_loss = float('inf')
        train_data_loader = self.dataloader.load_train_data()
        
        for epoch in range(params.epochs):
            print('Epoch {}/{}'.format(epoch, params.epochs - 1))
            print('-' * 10)
              
            # set model in train mode
            self.model.train()
              
            running_loss = 0            
            for images, diameters, img_paths  in train_data_loader:
                #images, diameters = (sample['image'], sample['label'])
                images = images.to(self.device)
                diameters = diameters.to(self.device)
                  
                # zero the parameter gradients
                self.optimizer.zero_grad()
                  
                output_dia = self.model(images)
                loss = self.criterion(output_dia, diameters)
                
                # backpropagate
                loss.backward()                
                self.optimizer.step()
                
                # loss.item gives mean loss over batch
                running_loss += loss.item() * images.size(0)
            
            epoch_loss = running_loss / len(train_data_loader.dataset)
            print('{} Loss: {:.4f} '. \
                  format("Training ", epoch_loss))
            
            val_loss = self.test('val')
            self.scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
        
        # set model weights based on val loss
        self.model.load_state_dict(best_model_wts)
            
    def test(self, phase='test'):
        
        self.model.eval()
        total_loss = 0
        if phase == 'test':
            data_loader = self.dataloader.load_test_data()
        else:
            data_loader = self.dataloader.load_val_data()
        
        # write results to RESULTS_FILE
        out_file = SYMBOLS.RESULTS_FILE
        with open(out_file, "w") as out:
            if phase == 'test':
                out.write("Image \t Actual Diameters \t Predicted Diameters \n")
                out.write("--" * 20 + "\n")
                
            for images, diameters, img_paths in data_loader:
                #images, diameters = (sample['image'], sample['label'])
                images = images.to(self.device)
                diameters = diameters.to(self.device)
                
                with torch.no_grad():
                    output_dia = self.model(images)
                    if phase == 'test':
                        for ix in range(len(img_paths)):
                            out.write("{} \t {} \t {} \n". \
                            format(img_paths[ix], str(output_dia[ix].numpy()), \
                                   str(diameters[ix].numpy())))
                    
                    loss = self.criterion(output_dia, diameters)
                    total_loss += loss.item() * images.size(0)
        
        loss_per_image =  total_loss/ len(data_loader.dataset)
        if phase == 'test':
            print('{} Loss: {:.4f}'.format("Testing",  loss_per_image))
        else:
            print('{} Loss: {:.4f}'.format("Validation",  loss_per_image))
        
        return loss_per_image
        
        
def  main():
    parser = argparse.ArgumentParser(description="Diameter Esitmation")
    
    # model argumets
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18','resnet50','vgg19'],
                        help="Choose the backbone model")
    parser.add_argument('--diameters', type=int, default=4,
                        help="Output responses to estimate")
    parser.add_argument('--predict-only-avg', type=bool, action='store_true',
                        help="Predict only average of 3 cross sections")
    parser.add_argument('--aug-by-crop', type=bool, action='store_true',
                        help="Crop image vertically at 3 sections to augment")
    
    # training hyper params
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--input-size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--pretrained', type=bool, default=True,  
                        action='store_true')
    
    #optimizers hyper params
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    
    params = parser.parse_args()
    
    trainer = Trainer(params)
    trainer.train()
    trainer.test()
    
    
if __name__ == "__main__":
    main()