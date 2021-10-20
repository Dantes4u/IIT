import torch.nn as nn
import torch
import timm

from timm import create_model


class BaseModel(nn.Module):
    def __init__(self, output_sizes, config):
        super().__init__()
        self.features = create_model('mobilenetv2_120d', pretrained=True, num_classes=0, global_pool='')
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.layers = nn.ModuleList()
        for output_size in output_sizes:
            self.layers.append(nn.Linear(1280, output_size))

    def forward(self, x):
        x = self.pool(self.features(x))
        output = [layer(x) for layer in self.layers]
        return output


class BaseModel2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.features = timm.models.efficientnet.efficientnet_b2(pretrained=True)
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

class BaseModel3(nn.Module):
    def __init__(self, output_sizes, config):
        super().__init__()
        # self.features = create_model('mobilenetv2_100', pretrained=True, num_classes=0, global_pool='')
        self.features = timm.models.efficientnet.efficientnet_b3_pruned(pretrained=True)
        # self.features.load_state_dict(torch.load('mobilenetv2_100.pth'), strict=False)
        in_features = self.features.classifier.in_features
        self.features.classifier = nn.Identity()
        # self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.layers = nn.ModuleList()
        for output_size in output_sizes:
            self.layers.append(nn.Linear(in_features, output_size))
        print(self)
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable params: {total_params}")

    def forward(self, x):
        x = self.features(x)
        output = [layer(x) for layer in self.layers]
        return output