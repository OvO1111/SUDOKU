import os
import torch
import matplotlib.pyplot as plt

import utils


def main():
    img_path = 'data/img/'
    model_path = 'output/'
    correct, le = 0, 0
    debug = -1
    
    mdl, itr = 'torch0.ckpt', 0
    for _, _, filenames in os.walk(model_path):
        for filename in filenames:
            if '.ckpt' in filename:
                if int(filename.lstrip('torch').rstrip('.ckpt')) > itr:
                    mdl = filename
                    itr = int(filename.lstrip('torch').rstrip('.ckpt'))
    
    model_path += mdl
    model = torch.load(model_path)
    
    for _, _, files in os.walk(img_path):
        for file in files:
            if 'png.' in file:
                os.remove(img_path+file)
    
    for root, _, filenames in os.walk(img_path):
        le = len(filenames)
        for filename in filenames:
            if utils.network_test(model, root+filename, debug) % 10 == int(filename[-5]):
                correct += 1
    
    print(correct/le)
    
    utils.visualization('pytorch', debug)
    
    with open('output/out1.txt', 'r') as out:
        arabic_acc = [float(a) for a in out.readline().split()]
        zh_acc = [float(a) for a in out.readline().split()]
        total_acc = [(arabic_acc[i]+zh_acc[i])/2 for i in range(len(arabic_acc))]
    
    legends = ['MNIST dataset accuracy', 'Chinese dataset accuracy', 'Avg. accuracy']
    plt.scatter(utils.generator(pos=0., itr=len(arabic_acc)), utils.convert(arabic_acc), alpha=0.6, c='blue')
    plt.scatter(utils.generator(pos=0.5, itr=len(zh_acc)), utils.convert(zh_acc), alpha=0.6, c='orange')
    plt.scatter(utils.generator(pos=1., itr=len(total_acc)), utils.convert(total_acc), alpha=0.6, c='green')
    
    plt.title('Testcase accuracy trend in 128 epochs')
    plt.legend(legends)
    
    with open('output/out2.txt', 'r') as out:
        arabic_acc = [float(a) for a in out.readline().split()]
        zh_acc = [float(a) for a in out.readline().split()]
        total_acc = [(arabic_acc[i]+zh_acc[i])/2 for i in range(len(arabic_acc))]
    plt.scatter(utils.generator(pos=5., itr=len(arabic_acc)), utils.convert(arabic_acc), alpha=0.6, c='blue')
    plt.scatter(utils.generator(pos=5.5, itr=len(zh_acc)), utils.convert(zh_acc), alpha=0.6, c='orange')
    plt.scatter(utils.generator(pos=6., itr=len(total_acc)), utils.convert(total_acc), alpha=0.6, c='green')
    
    plt.xticks([0.5, 5.5],
               [r'LeNet-5', r'Optimized LeNet-5'])
    plt.yticks(utils.convert([50, 75, 90, 95, 100]),
               ['0.50', '0.75', '0.90', '0.95', '1.00'])
    plt.axhline(utils.convert([50])[0], 0, 5.5, linestyle='--', alpha=0.3, c='gray', linewidth=0.5)
    plt.axhline(utils.convert([75])[0], 0, 5.5, linestyle='--', alpha=0.3, c='gray', linewidth=0.5)
    plt.axhline(utils.convert([90])[0], 0, 5.5, linestyle='--', alpha=0.3, c='gray', linewidth=0.5)
    plt.axhline(utils.convert([95])[0], 0, 5.5, linestyle='--', alpha=0.3, c='gray', linewidth=0.5)
    plt.axhline(utils.convert([100])[0], 0, 5.5, linestyle='--', alpha=0.3, c='gray', linewidth=0.5)
    
    plt.savefig('output/out.png')


if __name__ == '__main__':
    main()
