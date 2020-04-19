# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models
import argparse

class DiameterEstimation(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        # set for only last FC layer
        if params.predict_only_avg or params.aug_by_crop or params.predict_only_centre:
            out = 1
        else:
            out = params.diameters
            
        # Define the network by using pretrained models
        if params.backbone == 'vgg19':
            self.backbone = models.vgg19_bn(pretrained=params.pretrained)
            
            if params.pretrained:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                    
            self.backbone.classifier[6] = nn.Linear(4096, out)
            
        elif params.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=params.pretrained)
         
            # set requires_grad flag to false if pretrained model
            if params.pretrained:
                layer = 8
                counter = 0
                for child in self.backbone.children():
                    counter += 1
                    #print(str(counter) + "--" + str(child))
                    if counter < layer:
                        for param in child.parameters():
                            param.requires_grad = False

                    
                self.backbone.fc = nn.Linear(self.backbone.fc.in_features, \
                                             out)
        
        elif params.backbone == 'inceptionv3':
            self.backbone = models.inception_v3(pretrained=params.pretrained)
            
            if params.pretrained:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            self.backbone.AuxLogits.fc = nn.Linear( \
                    768, out)
            self.backbone.fc = nn.Linear( \
                    2048, out)
        
        self.params = params
        
    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    params = parser.parse_args()
    params.backbone = 'resnet50'
    params.pretrained = True
    params.predict_only_avg = False
    params.aug_by_crop = True
    
    params.diameters = 4
    
    img = torch.rand(1,3,512,512)
    model = DiameterEstimation(params)
    out = model(img)
    print(out.shape)