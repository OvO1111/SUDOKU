"""import torch
import cv2, os, re
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from optimizer import MD_with_pnorm

from pyimagesearch.models import LeNet5


class sub_pool2d(nn.Module):
    def __init__(self, kernel_size=(2, 2), stride=2):
        super(sub_pool2d, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.w_height = kernel_size[0]
        self.w_width = kernel_size[1]
        self.W = nn.Parameter(torch.randn(1), requires_grad=True)
        self.B = nn.Parameter(torch.randn(1), requires_grad=True)
    
    def forward(self, x):
        in_height = x.size(0)
        in_width = x.size(1)
        
        out_height = int((in_height - self.w_height) / self.stride) + 1
        out_width = int((in_width - self.w_width) / self.stride) + 1
        
        out = torch.zeros((out_height, out_width))
        
        for i in range(out_height):
            for j in range(out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out[i, j] = self.W * torch.sum(x[start_i: end_i, start_j: end_j]) + self.B
        return out


if __name__ == "__main__":
    print("=" * 10 + "MyMaxPool2D" + "=" * 10)
    x = torch.randn((1, 1, 6, 8), requires_grad=True)
    pool = sub_pool2d()
    y = pool(x)
    c = torch.mean(y)
    c.backward()
    
    print("=" * 10 + "nn.MaxPool2d" + "=" * 10)
    x2 = x.detach().view(1, 1, 6, 8)
    x2.requires_grad = True
    pool = nn.MaxPool2d((2, 2), 2)
    y2 = pool(x2)
    c2 = torch.mean(y2)
    c2.backward()
    
    print("=" * 10 + "leNet" + "=" * 10)
    z = torch.randn((256, 1, 32, 32), requires_grad=True)
    model = LeNet5()
    zz = model(z)

train_dst = MNIST('./data/mnist',
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()
                  ]))
test_dst = MNIST('./data/mnist',
                 train=False,
                 download=True,
                 transform=transforms.Compose([
                     transforms.Resize((32, 32)),
                     transforms.ToTensor()
                 ]))


class MyDataset(Data.Dataset):
    def __init__(self, root, cat, transform=None):
        super(MyDataset, self).__init__()
        self.tags = []
        self.labels = []
        self.path = root
        self.category = cat
        self.transform = transform
        self.__initiate()
    
    def __initiate(self):
        self.path += self.category
        
        pattern = re.compile(r'\d')
        for root, _, filenames in os.walk(self.path):
            for file in filenames:
                self.tags.append(root + file)
                if not re.match(pattern, file[1]):
                    self.labels.append(int(file[0]))
                else:
                    self.labels.append(0)
    
    def __getitem__(self, index):
        # img = Image.open(self.tags[index]).convert('1')
        img = cv2.imread(self.tags[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        img = cv2.resize(img, (32, 32))
        img = img.astype("float") / 255.0
        img = np.reshape(img, (1, 32, 32))
        lab = self.labels[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        im_tensor = torch.from_numpy(img)
        la_tensor = torch.tensor(lab, dtype=torch.long)
        return im_tensor, la_tensor
    
    def __len__(self):
        return len(self.labels)


model = LeNet5()
model = model.double()
#
dataset = MyDataset('./data/zh', cat='/train/')
loader = Data.DataLoader(dataset, batch_size=256, shuffle=True)
# loader = Data.DataLoader(train_dst, batch_size=256, shuffle=True)
model.train()
optm = MD_with_pnorm(model.parameters(), lr=1e-4, p=5)
criterion = torch.nn.CrossEntropyLoss()
for i, (data, label) in enumerate(loader):
    print(i)
    param = []
    param_dict = []
    '''for para in model.named_parameters():
        print(para)
        print(para[1].size())
        print('*******************')'''
    optm.zero_grad()
    output = model(data)
    weight_img = output.detach().numpy()
    '''print()'''
    loss = criterion(output, label)
    '''print(loss, '\n', loss.size())'''
    loss.backward()
    optm.step()
print('end mark')

a = np.array([223.245, 221.24, 11.12])
a = torch.from_numpy(a)
p = 10
pw = 1/p*(2/p-1)
a = torch.sum(a)
pow = torch.pow(a, pw)
print(pow)"""
