import torch.nn as nn
from collections import OrderedDict


class CS1(nn.Module):
    def __init__(self):
        super(CS1, self).__init__()
        
        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=1)),
            ('tanh1', nn.Tanh()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))
    
    def forward(self, raw):
        output = self.c1(raw)
        return output


class CS2(nn.Module):
    def __init__(self):
        super(CS2, self).__init__()
        
        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))),
            ('tanh2', nn.Tanh()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))
    
    def forward(self, raw):
        output = self.c2(raw)
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
            ('f5', nn.Linear(in_features=84, out_features=20)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, raw):
        output = self.f5(raw)
        return output


class LeNet5(nn.Module):
    """
    Input - 256x32x32x1
    Output - 256x20
    """
    
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.c1 = CS1()
        self.c2_1 = CS2()
        self.c2_2 = CS2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = OUTPUT()
        
        self.model = nn.Sequential()
    
    def forward(self, img):
        output = self.c1(img)
        x = self.c2_1(output)
        output = self.c2_2(output)
        output += x
        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        
        output = self.f5(output)
        return output

