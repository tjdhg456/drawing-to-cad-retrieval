import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from glob import glob
from copy import deepcopy

## Network
class MVCNN(nn.Module):
    def __init__(self, emb_size, saved_path=None, infer=False):
        super(MVCNN, self).__init__()
        self.infer = infer

        ## 2D Outline Model
        self.model_2D = deepcopy(models.resnet34(pretrained=True))
        self.model_2D.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if emb_size != 1000:
            self.model_2D.fc = nn.Linear(512, emb_size)

        if saved_path is not None:
            self.model_2D.load_state_dict(torch.load(saved_path))

        ## 2D rendered Model
        self.model_3D = deepcopy(self.model_2D)
        self.model_3D_1 = nn.Sequential(*list(self.model_3D.children())[:7])
        self.model_3D_2 = nn.Sequential(*list(self.model_3D.children())[7:-1])
        self.model_3D_fc = self.model_3D.fc

    def forward(self, input1, input2, input3=None):
        if self.infer == False:
            result_2D = self.model_2D(input1)

            N, V, C, H, W = input2.size()

            input2 = input2.view(-1, C, H, W)
            result_3D_P = self.model_3D_1(input2)
            result_3D_P = result_3D_P.view((N, V, *result_3D_P.size()[1:]))
            result_3D_P = torch.max(result_3D_P, dim=1)[0]
            result_3D_P = self.model_3D_2(result_3D_P).view(N,-1)
            result_3D_P = self.model_3D_fc(result_3D_P)

            input3 = input3.view(-1, C, H, W)
            result_3D_N = self.model_3D_1(input3)
            result_3D_N = result_3D_N.view((-1, V, *result_3D_N.size()[1:]))
            result_3D_N = torch.max(result_3D_N, dim=1)[0]
            result_3D_N = self.model_3D_2(result_3D_N).view(N,-1)
            result_3D_N = self.model_3D_fc(result_3D_N)

            return result_2D, result_3D_P, result_3D_N

        else:
            result_2D = self.model_2D(input1)

            N, V, C, H, W = input2.size()
            input2 = input2.view(-1, C, H, W)
            result_3D_P = self.model_3D_1(input2)
            result_3D_P = result_3D_P.view((-1, V, *result_3D_P.size()[1:]))
            result_3D_P = torch.max(result_3D_P, dim=1)[0]
            result_3D_P = self.model_3D_2(result_3D_P)

            return result_2D, result_3D_P

