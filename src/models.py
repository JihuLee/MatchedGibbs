import torch
import torch.nn as nn
import bayesian_torch
import torchvision

class MLP(nn.Module):
    def __init__(self, num_layer, input_dim, rep_dim, output_dim, lmda_init=None):
        super(MLP, self).__init__()

        if lmda_init is not None: 
            self.lmda = nn.Parameter(torch.tensor([lmda_init]), requires_grad=True)

        self.acti = nn.ReLU()
        layers = []

        if num_layer == 0:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            for i in range(num_layer+1):
                if i == 0:
                    layers.append(nn.Linear(input_dim, rep_dim))
                elif i < num_layer:
                    layers.append(self.acti)
                    layers.append(nn.Linear(rep_dim, rep_dim))
                else:
                    layers.append(self.acti)
                    layers.append(nn.Linear(rep_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

class BayesianMLP(nn.Module):
    def __init__(self, num_layer, input_dim, rep_dim, output_dim, lmda_init=None):
        super().__init__()
        if lmda_init is not None: 
            self.lmda = nn.Parameter(torch.tensor([lmda_init]), requires_grad=True)
            
        self.acti = nn.ReLU()
        layers = []
        
        if num_layer == 0:
            layers.append(bayesian_torch.layers.LinearReparameterization(
                in_features=input_dim,
                out_features=output_dim,
                prior_mean=0,
                prior_variance=1.0,
                bias=True,
            ))
        else:
            layers.append(bayesian_torch.layers.LinearReparameterization(
                in_features=input_dim,
                out_features=rep_dim,
                prior_mean=0,
                prior_variance=1.0,
                bias=True,
            ))
            
            for _ in range(num_layer - 1):
                layers.append(self.acti)
                layers.append(bayesian_torch.layers.LinearReparameterization(
                    in_features=rep_dim,
                    out_features=rep_dim,
                    prior_mean=0,
                    prior_variance=1.0,
                    bias=True,
                ))
            
            layers.append(self.acti)
            layers.append(bayesian_torch.layers.LinearReparameterization(
                in_features=rep_dim,
                out_features=output_dim,
                prior_mean=0,
                prior_variance=1.0,
                bias=True,
            ))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        kl = 0.0
        for module in self.net:
            if isinstance(module, bayesian_torch.layers.LinearReparameterization):
                x, _kl = module(x)
                kl += _kl
            elif isinstance(module, nn.ReLU):
                x = module(x)
            else:
                x = module(x) if not isinstance(x, tuple) else x[0]
        return x, kl

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ResNet18_Encoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.resnet.avgpool = Identity()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        res = self.resnet(x)
        res = res.view(-1, 512, 8, 8)
        res = self.avg(res)
        return res.view(-1, 512)