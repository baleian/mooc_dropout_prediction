import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#######################################################################
# 1                                                                   #
# data generation                                                     #
#######################################################################
class dset:
    def __init__(self, class_size, seed=1):
        np.random.seed(seed)
        self.class_size = class_size
    
    def create(self):
        raise NotImplementedError
    
    def get_data(self, training_rate=0.75):
        data = np.append(self.X.copy(), self.labels.copy(), axis=1)
        np.random.shuffle(data)
        training_size = int(training_rate * self.class_size * self.classes)
        tr_x = data[:training_size, :-self.classes]
        tr_y = data[:training_size, -self.classes:]
        te_x = data[training_size:, :-self.classes]
        te_y = data[training_size:, -self.classes:]

        return tr_x, tr_y, te_x, te_y

    def show_data(self):
        plt.scatter(self.X[:,0], self.X[:,1], c=self.T, s=30, alpha=0.5)
        plt.show()

"""
1.a
generating three classes of data mostly non-overlapping each other (3NO)
"""
class dset_3NO(dset):
    def __init__(self, class_size, seed=1):
        dset.__init__(self, class_size, seed)
        self.classes = 3

    def create(self):
        #######################################################################
        #                       Fill in your code here                        #
        #######################################################################
        self.X = None
        self.T = None
        raise NotImplementedError
        #######################################################################
        #                                End                                  #
        #######################################################################
        self.labels = np.zeros((self.class_size * self.classes, self.classes))
        for i in range(self.class_size * self.classes):
            self.labels[i, self.T[i]] = 1

"""
1.b
a circle-shaped data in another circle-shaped data (CC)
"""

class dset_CC(dset):
    def __init__(self, class_size, radius=5, noise=0.2, seed=1):
        dset.__init__(self, class_size, seed)
        self.classes = 2
        self.radius = radius
        self.noise = noise

    def create(self):
        #######################################################################
        #                       Fill in your code here                        #
        #######################################################################
        self.X = None
        self.T = None
        raise NotImplementedError
        #######################################################################
        #                                End                                  #
        #######################################################################
        self.labels = np.zeros((self.class_size * self.classes, self.classes))
        for i in range(self.class_size * self.classes):
            self.labels[i, self.T[i]] = 1

"""
1.c
spiral data (SP)
"""
class dset_SP(dset):
    def __init__(self, class_size, radius=5, noise=0.2, seed=1):
        dset.__init__(self, class_size, seed)
        self.classes = 3
        self.noise = noise
        self.radius = radius
    
    def create(self):
        #######################################################################
        #                       Fill in your code here                        #
        #######################################################################
        self.X = None
        self.T = None
        raise NotImplementedError
        #######################################################################
        #                                End                                  #
        #######################################################################
        self.labels = np.zeros((self.class_size * self.classes, self.classes))
        for i in range(self.class_size * self.classes):
            self.labels[i, self.T[i]] = 1

"""
1.d
MNIST data
"""
class dset_MNIST:
    def __init__(self):
        self.classes = 10
        self.training_size = 1500
        self.test_size = 150
    def create(self):
        mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
        loader_train = DataLoader(mnist_train, batch_size=10, shuffle=True)
        mnist_test = datasets.MNIST('../data', train=False, download=True, transform=transforms.ToTensor())
        loader_test = DataLoader(mnist_test, batch_size=10, shuffle=True)


        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            shuffle=True)

    def get_data(self):
        tr_x = np.array([])
        tr_y = np.array([])
        te_x = np.array([])
        te_y = np.array([])

        print ("Training Data Loading")
        for idx, (d, t) in enumerate(self.train_loader):
            if idx == 0:
                tr_x = d.numpy().squeeze().reshape(1, -1).copy()
                y = np.zeros((1,10))
                y[0, t.numpy()[0]] = 1
                tr_y = y
            else:
                tr_x = np.vstack((tr_x, d.numpy().squeeze().reshape(1, -1).copy()))
                y = np.zeros((1, 10))
                y[0, t.numpy()[0]] = 1
                tr_y = np.vstack((tr_y, y))
            if idx % 1000 == 0:
                print ('Training data number: %d'%idx)
            if idx == self.training_size-1:
                break
        print ("Done!")

        print ("Test Data Loading")
        for idx, (d, t) in enumerate(self.test_loader):
            if idx == 0:
                te_x = d.numpy().squeeze().reshape(1, -1).copy()
                y = np.zeros((1,10))
                y[0, t.numpy()[0]] = 1
                te_y = y
            else:
                te_x = np.vstack((te_x, d.numpy().squeeze().reshape(1, -1).copy()))
                y = np.zeros((1, 10))
                y[0, t.numpy()[0]] = 1
                te_y = np.vstack((te_y, y))
            if idx == self.test_size-1:
                break
        print ("Done!")

        return tr_x, tr_y, te_x, te_y