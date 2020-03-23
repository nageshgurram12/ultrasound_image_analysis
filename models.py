# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models
import argparse

class DiameterEstimation(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        # Define the network by using pretrained models
        if params.backbone == 'vgg19':
            self.backbone = models.vgg19_bn(params.pretrained)
        elif params.backbone == 'resnet18':
            self.backbone = models.resnet18(params.pretrained)
        elif params.backbone == 'resnet50':
            self.backbone = models.resnet50(params.pretrained)
         
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
                
            # set for only last FC layer
            if params.predict_only_avg:
                out = 1
            else:
                out = params.diameters
                
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, \
                                         out)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    params = parser.parse_args()
    params.backbone = 'resnet50'
    params.pretrained = True
    params.diameters = 4
    
    img = torch.rand(1,3,512,512)
    model = DiameterEstimation(params)
    out = model(img)
    print(out.shape)