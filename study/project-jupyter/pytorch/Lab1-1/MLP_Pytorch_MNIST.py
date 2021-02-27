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
loader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

dtype = torch.FloatTensor
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    torch.cuda.random.manual_seed()

class Flatten(nn.Module):
    def forward(self, x):
        #######################################################################
        #                       Fill in your code here                        #
        #######################################################################
        raise NotImplementedError
        return None
        #######################################################################
        #                                End                                  #
        #######################################################################

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #######################################################################
        #                       Fill in your code here                        #
        #######################################################################
        raise NotImplementedError
        #######################################################################
        #                                End                                  #
        #######################################################################

    def forward(self, x):
        #######################################################################
        #                       Fill in your code here                        #
        #######################################################################
        raise NotImplementedError
        return None
        #######################################################################
        #                                End                                  #
        #######################################################################

model = Model()
model.type(dtype)

#######################################################################
#                       Fill in your code here                        #
#######################################################################
raise NotImplementedError
lossfn = None
optimizer = None
#######################################################################
#                                End                                  #
#######################################################################

cost = []
def train(model, loss_fn, optimizer, num_epochs=1):
    for epoch in range(num_epochs):
        for t, (x, y) in enumerate(loader_train):
            x_var = x.type(dtype)
            y_var = y.type(dtype).long()

            #######################################################################
            #                       Fill in your code here                        #
            #######################################################################
            raise NotImplementedError
            #######################################################################
            #                                End                                  #
            #######################################################################

        cost.append(loss.data[0])
        if (epoch + 1) % (num_epochs / 10) == 0:
            print('Epoch = %d, loss = %.4f' % (epoch + 1, cost[-1]))

def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    for x, y in loader:
        x_var = x.type(dtype)

        scores = model(x_var)
        _, preds = scores.max(1)
        num_correct += (preds == y.long()).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

train(model, lossfn, optimizer, num_epochs=50)
check_accuracy(model, loader_test)
plt.plot(cost)
plt.show()