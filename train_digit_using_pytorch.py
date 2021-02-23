import cv2
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST

from pyimagesearch.models import LeNet5
from utils import save_figure

model_path = 'output/'
lr = 1e-4
train_batch = 256
test_batch = 1024
cores = 8
epochs = 16

'''os.remove('output/loss')
os.mkdir('output/loss')'''

mdl, itr = 'torch0.ckpt', 0
for _, _, filenames in os.walk(model_path):
    for filename in filenames:
        if '.ckpt' in filename:
            if int(filename.lstrip('torch').rstrip('.ckpt')) > itr:
                mdl = filename
                itr = int(filename.lstrip('torch').rstrip('.ckpt'))

model_path += mdl


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
                    self.labels.append(int(file[0])+10)
                else:
                    self.labels.append(10)
    
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

        im_tensor = torch.from_numpy(img).float()
        la_tensor = torch.tensor(lab, dtype=torch.long)
        return im_tensor, la_tensor
    
    def __len__(self):
        return len(self.labels)


def train(epoch, modl):
    print(f'[PyTorch Train] Epoch [{epoch+1}/{epochs}]')
    loss_this_epoch = []
    dts, lbs = [], []
    for i, (data, label) in enumerate(data_train_loader):
        dts.append(data)
        lbs.append(label)

    for i, (data, label) in enumerate(data_train_loader2):
        dts.append(data)
        lbs.append(label)
        
    modl.train()
    ttl_len = len(dts) if len(dts) == len(lbs) else 0
    idx_list = [i for i in range(ttl_len)]
    random.shuffle(idx_list)
    for i in range(ttl_len):
        if not i % 24:
            print(f'[PyTorch Train] Batch [{i+1}/{ttl_len}]')
        data, label = dts[idx_list[i]], lbs[idx_list[i]]
        optm.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        '''wi = output.detach().numpy()
        wi = -wi
        cv2.imshow('i', wi)
        cv2.waitKey(0)'''
        loss.backward()
        optm.step()
        loss_this_epoch.append(loss.detach().cpu().item())
    
    # torch.save(model, model_path)
    print(f"Training Loss: {np.mean(loss_this_epoch):.8f}")
    return loss_this_epoch


def test():
    model.eval()
    avg_loss, total_correct = 0., 0
    for i, (data, label) in enumerate(data_test_loader):
        output = model(data)
        avg_loss += criterion(output, label).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(label.view_as(pred)).sum()

    avg_loss /= len(data_test)
    mnist_acc = float(total_correct) / (len(data_test))
    print('[PyTorch Test] Arabic test case: Avg. Loss: %f, Accuracy: %f' % (avg_loss, float(total_correct) / (len(data_test))))

    avg_loss, total_correct = 0., 0
    for i, (data, label) in enumerate(data_test_loader2):
        output = model(data)
        avg_loss += criterion(output, label).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(label.view_as(pred)).sum()
    
    avg_loss /= len(zh_test)
    zh_acc = float(total_correct) / (len(zh_test))
    print('[PyTorch Test] Chinese test case: Loss: %f, Accuracy: %f' % (avg_loss, float(total_correct) / (len(zh_test))))
    
    return mnist_acc, zh_acc


def get_dat():
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
    train_zh_dst = MyDataset('./data/zh', cat='/train/')
    test_zh_dst = MyDataset('./data/zh', cat='/test/')
    return train_dst, test_dst, train_zh_dst, test_zh_dst


data_train, data_test, zh_train, zh_test = get_dat()

data_train_loader = Data.DataLoader(data_train, batch_size=train_batch, shuffle=True, num_workers=cores)
data_test_loader = Data.DataLoader(data_test, batch_size=test_batch, num_workers=cores)

data_train_loader2 = Data.DataLoader(zh_train, batch_size=train_batch, shuffle=True, num_workers=cores)
data_test_loader2 = Data.DataLoader(zh_test, batch_size=test_batch, num_workers=cores)

if os.path.exists(model_path):
    model = torch.load(model_path)
else:
    model = LeNet5()

optm = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9, .99))
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    ix, temp = 0., 0
    los = []
    cur_loss = []
    
    for ep in range(epochs):
        if np.log2(ep + 1) == ix:
            ix += 1.
            cur_loss.append(train(ep, model))
            temp = np.mean(cur_loss[-1])
        else:
            temp = np.mean(train(ep, model))
        los.append(temp)
    
    save_figure(int(ix), los, cur_loss)
    acc = test()
    
    '''with open('output/out1.txt', 'w+') as out:
        arabic = out.readline()+' '+str(acc[0])
        zh = out.readline()+' '+str(acc[1])
        out.write(arabic)
        out.write('\n')
        out.write(zh)'''

    if os.path.exists(model_path):
        os.remove(model_path)
    torch.save(model, 'output/torch'+str(itr+epochs)+'.ckpt')
    

