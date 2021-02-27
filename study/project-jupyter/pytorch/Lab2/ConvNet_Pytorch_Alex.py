import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as dset
import torchvision.transforms as T

import matplotlib.pyplot as plt

batch_size = 64

mnist_train = dset.MNIST('../data', train=True, download=True, transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
mnist_test = dset.MNIST('../data', train=False, download=True, transform=T.ToTensor())
loader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)

class Model_Real(nn.Module):
    def __init__(self, num_classes):
        super(Model_Real, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.network(x)

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.network = nn.Sequential(
            #######################################################################
            #                       Fill in your code here                        #
            #######################################################################
            raise NotImplementedError
            #######################################################################
            #                                End                                  #
            #######################################################################
        )

    def forward(self, x):
        #######################################################################
        #                       Fill in your code here                        #
        #######################################################################
        raise NotImplementedError
        return None
        #######################################################################
        #                                End                                  #
        #######################################################################

model = Model(10)

#######################################################################
#                       Fill in your code here                        #
#######################################################################
raise NotImplementedError
lossfn = None
optimizer = None
#######################################################################
#                                End                                  #
#######################################################################


def train(model, loss_fn, optimizer, hist, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        for t, (x, y) in enumerate(loader_train):
            optimizer.zero_grad()

            #######################################################################
            #                       Fill in your code here                        #
            #######################################################################
            raise NotImplementedError
            #######################################################################
            #                                End                                  #
            #######################################################################
            if t % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, t * len(x), len(loader_train.dataset),
                           100. * t / len(loader_train), loss.data[0]))

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, y in loader:
        scores = model(x)
        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

costs = []
train(model, lossfn, optimizer, costs, num_epochs=5)
check_accuracy(model, loader_test)

plt.plot(costs)
plt.show()
