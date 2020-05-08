import torch.nn as nn
import pickle
import torch


# For now neuarl net tries to predict only one value from output
class FacialDetectModel(nn.Module):
    def __init__(self, num_classes):
        super(FacialDetectModel, self).__init__()
        model = pickle.load(open('inception_model.dt', 'rb'))
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, 512)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 512)
        self.facial_model = model
        self.head_1 = nn.Linear(512,num_classes)

    def forward(self,x):
        if self.facial_model.training:
            out,_ = self.facial_model(x)
        else:
            out = self.facial_model(x)
        out = torch.sigmoid(self.head_1(out))
        return out