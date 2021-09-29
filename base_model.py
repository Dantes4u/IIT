import torch.nn as nn
import torch
import timm

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.features = timm.models.efficientnet.efficientnet_b1(pretrained=True, in_chans=config['sequence']*3)
        in_features = self.features.classifier.in_features
        self.features.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features, 1)
        self.sigm = nn.Sigmoid()
        #self.layers = nn.ModuleList()
        #self.layers.append(nn.Linear(in_features, 1))
        print(self)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable params: {total_params}")
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        #x = self.sigm(x)
        #output = [layer(x) for layer in self.layers]
        return x

class BaseModel2(nn.Module):
    def __init__(self, config):
        super().__init__()
        config['sequence'] = 3
        self.features = timm.models.efficientnet.efficientnet_b1(pretrained=True, in_chans=config['sequence']*3)
        in_features = self.features.classifier.in_features
        self.features.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features, 1)
        self.sigm = nn.Sigmoid()
        #self.layers = nn.ModuleList()
        #self.layers.append(nn.Linear(in_features, 1))
        print(self)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable params: {total_params}")
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        #x = self.sigm(x)
        #output = [layer(x) for layer in self.layers]
        return x
