import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from collections import OrderedDict
import torch

"""
Cx: convolution layer x
Sx: subsample layer x
Fx: full connection layer x

layer(output) (maps)@(size x size)  (ReLU(x): max{0, x}) :=
INPUT: 32x32 -> C1: 6@28X28, convolution layer with 6 feature maps,
                             each unit connected to a 5x5 neighborhood in the input,
                             size of feature map = 28x28
             -> S2: 6@14x14, sub-sampling layer with 6 feature maps of size 14x14,
                             each unit in each map is connected to a 2x2 neighborhood
                             in the corresponding feature map in C1
                             Four inputs to a unit in S2 are added, then multiplied by
                             a 'trainable coefficient', and added to a 'trainable bias',
                             the result is then passed through a sigmoid function
                             (trainable coefficient: learnable parameters.weight,
                              trainable bias: learnable parameters.bias)
             -> C3: 16@10x10, each unit in each of its 16 feature maps is connected to
                              several 5x5 neighborhood at identical locations in a subset
                              of S2's feature maps, a more detailed correspondence is
                              shown in pdf
             -> S4: 16@5x5, sub-sampling layer with 16 feature maps of size 5x5, each unit
                            is connected to a 2x2 neighborhood in the corresponding map in
                            C3
             -> C5: 120, convolution layer with 120 feature maps, each unit is connected
                         to a 5x5 neighborhood on all 16 of S4's feature maps.
                         Because the size of S4 is also 5x5, the size of C5's feature maps
                         is 1x1
             -> F6: 84, passed through an sigmoid function, f(a) = Atanh(Sa)
             -> OUTPUT: 10
"""


class sub_pool2d(nn.Module):
    def __init__(self, kernel_size=(2, 2), stride=2, params=6):
        super(sub_pool2d, self).__init__()
        self.stride = stride
        self.params = params
        self.kernel_size = kernel_size
        self.w_height = kernel_size[0]
        self.w_width = kernel_size[1]
        w = torch.randn(params, dtype=torch.float)
        b = torch.randn(params, dtype=torch.float)
        flag = 14 if params == 6 else 5
        ww = torch.zeros(params * flag * flag, dtype=torch.float)
        bb = torch.zeros(params * flag * flag, dtype=torch.float)
        
        for i in range(ww.shape[0]):
            ww[i] = w[i // flag ** 2]
            bb[i] = b[i // flag ** 2]
        
        ww = torch.reshape(ww, (params, flag, flag))
        bb = torch.reshape(bb, (params, flag, flag))
        ww = torch.unsqueeze(ww, 0)
        bb = torch.unsqueeze(bb, 0)
        
        self.W = nn.Parameter(ww, requires_grad=True)
        self.B = nn.Parameter(bb, requires_grad=True)
    
    def forward(self, x):
        return self.W * f.avg_pool2d(x, kernel_size=(2, 2), stride=2) + self.B


class RBF(nn.Module):
    
    def __init__(self, in_features: int or tuple, out_features: int or tuple):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
    
    def forward(self, input: torch.Tensor):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1)
        return distances


class CS1(nn.Module):
    def __init__(self):
        super(CS1, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1)
        self.tanh = nn.Tanh()
        self.pooling = sub_pool2d(kernel_size=(2, 2), stride=2)
    
    def forward(self, raw):
        output = self.conv(raw)
        output = self.tanh(output)
        output = self.pooling(output)
        
        return output


class CS123(nn.Module):
    def __init__(self):
        super(CS123, self).__init__()
        
        self.c123 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(5, 5))
    
    def forward(self, raw):
        output = self.c123(raw)
        return output


class CS124(nn.Module):
    def __init__(self):
        super(CS124, self).__init__()
        
        self.c124 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(5, 5))
    
    def forward(self, raw):
        output = self.c124(raw)
        return output


class CS126(nn.Module):
    def __init__(self):
        super(CS126, self).__init__()
        
        self.c126 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=(5, 5))
    
    def forward(self, raw):
        output = self.c126(raw)
        return output


class CS3(nn.Module):
    def __init__(self):
        super(CS3, self).__init__()
        
        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5))),
            ('relu2', nn.ReLU()),
            ('s2', sub_pool2d(kernel_size=(2, 2), stride=2))
        ]))
    
    def forward(self, raw):
        output = self.c2(raw)
        return output


class S2(nn.Module):
    def __init__(self):
        super(S2, self).__init__()
        
        self.tanh = nn.Tanh()
        self.pooling = sub_pool2d(kernel_size=(2, 2), stride=2, params=16)
    
    def forward(self, raw):
        output = self.tanh(raw)
        output = self.pooling(output)
        
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()
        
        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5))),
            ('tanh3', nn.Tanh())
        ]))
    
    def forward(self, raw):
        output = self.c3(raw)
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()
        
        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(in_features=120, out_features=84)),
            ('tanh4', nn.Tanh())
        ]))
    
    def forward(self, raw):
        output = self.f4(raw)
        return output


class OUTPUT(nn.Module):
    def __init__(self):
        super(OUTPUT, self).__init__()
        
        self.f5 = nn.Sequential(OrderedDict([
            ('rbf', RBF(in_features=84, out_features=10)),
            ('sig', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, raw):
        output = self.f5(raw)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.c1 = CS1()
        self.middleLayer11 = CS123()
        self.middleLayer12 = CS123()
        self.middleLayer13 = CS123()
        self.middleLayer14 = CS123()
        self.middleLayer15 = CS123()
        self.middleLayer16 = CS123()

        self.middleLayer21 = CS124()
        self.middleLayer22 = CS124()
        self.middleLayer23 = CS124()
        self.middleLayer24 = CS124()
        self.middleLayer25 = CS124()
        self.middleLayer26 = CS124()
        self.middleLayer27 = CS124()
        self.middleLayer28 = CS124()
        self.middleLayer29 = CS124()

        self.middleLayer3 = CS126()
        self.s2 = S2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = OUTPUT()
        
        self.model = nn.Sequential()
    
    def forward(self, img):
        output = self.c1(img)
        
        num_output = output.detach().numpy()  # shape: (1, 6, 5, 5)
        y1 = self.middleLayer11(torch.from_numpy(num_output[:, 0:3])).detach().numpy()
        y2 = self.middleLayer12(torch.from_numpy(num_output[:, 1:4])).detach().numpy()
        y3 = self.middleLayer13(torch.from_numpy(num_output[:, 2:5])).detach().numpy()
        y4 = self.middleLayer14(torch.from_numpy(num_output[:, 3:])).detach().numpy()
        y5 = self.middleLayer15(torch.from_numpy(num_output[:, [0, 4, 5]])).detach().numpy()
        y6 = self.middleLayer16(torch.from_numpy(num_output[:, [0, 1, 4]])).detach().numpy()

        y7 = self.middleLayer21(torch.from_numpy(num_output[:, 0:4])).detach().numpy()
        y8 = self.middleLayer22(torch.from_numpy(num_output[:, 1:5])).detach().numpy()
        y9 = self.middleLayer23(torch.from_numpy(num_output[:, 2:])).detach().numpy()
        y10 = self.middleLayer24(torch.from_numpy(num_output[:, [0, 3, 4, 5]])).detach().numpy()
        y11 = self.middleLayer25(torch.from_numpy(num_output[:, [0, 1, 4, 5]])).detach().numpy()
        y12 = self.middleLayer26(torch.from_numpy(num_output[:, [0, 1, 2, 5]])).detach().numpy()
        y13 = self.middleLayer27(torch.from_numpy(num_output[:, [0, 1, 3, 4]])).detach().numpy()
        y14 = self.middleLayer28(torch.from_numpy(num_output[:, [1, 2, 4, 5]])).detach().numpy()
        y15 = self.middleLayer29(torch.from_numpy(num_output[:, [0, 2, 3, 5]])).detach().numpy()

        y16 = self.middleLayer3(torch.from_numpy(num_output)).detach().numpy()

        y = np.concatenate((y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16),
                           axis=1)
        output = torch.from_numpy(y)

        output = self.s2(output)
        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        
        output = self.f5(output)
        return output

