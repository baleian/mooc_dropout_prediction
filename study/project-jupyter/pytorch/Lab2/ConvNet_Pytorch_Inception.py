import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets as dset
from torchvision import transforms as T

# Training settings
batch_size = 64

mnist_train = dset.MNIST('../data', train=True, download=True, transform=T.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
mnist_test = dset.MNIST('../data', train=False, download=True, transform=T.ToTensor())
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
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


model = Net()
#######################################################################
#                       Fill in your code here                        #
#######################################################################
raise NotImplementedError
loss_fn = None
optimizer = None
#######################################################################
#                                End                                  #
#######################################################################

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #######################################################################
        #                       Fill in your code here                        #
        #######################################################################
        raise NotImplementedError
        #######################################################################
        #                                End                                  #
        #######################################################################
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        # sum up batch loss
        test_loss += loss_fn(output, target)
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss.data[0], correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(0, 5):
    train(epoch)
    test()