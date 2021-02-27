import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

#######################################################################
# 1                                                                   #
# data generation                                                     #
#######################################################################
"""
1.a
generating three classes of data mostly non-overlapping each other (3NO)
"""
class dset_3NO(Dataset):
    def __init__(self, class_size):
        self.classess = 3
        X1 = np.random.randn(class_size, 2) + np.array([0, -2])
        X2 = np.random.randn(class_size, 2) + np.array([2, 2])
        X3 = np.random.randn(class_size, 2) + np.array([-2, 2])
        X = np.vstack([X1, X2, X3])
        T = np.array([0] * class_size + [1] * class_size + [2] * class_size)
        self.len = class_size * self.classess
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(T)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

"""
1.b
a circle-shaped data in another circle-shaped data (CC)
"""
class dset_CC(Dataset):
    def __init__(self, class_size, radius=5, noise=0.2):
        self.classess = 2
        r1 = np.random.uniform(0, 0.5 * radius, class_size)
        theta1 = np.random.uniform(0, 2 * np.pi, class_size)
        x1 = r1 * np.sin(theta1) + np.random.uniform(-radius, radius, class_size) * noise
        y1 = r1 * np.cos(theta1) + np.random.uniform(-radius, radius, class_size) * noise

        r2 = np.random.uniform(0.7 * radius, radius, class_size)
        theta2 = np.random.uniform(0, 2 * np.pi, class_size)
        x2 = r2 * np.sin(theta2) + np.random.uniform(-radius, radius, class_size) * noise
        y2 = r2 * np.cos(theta2) + np.random.uniform(-radius, radius, class_size) * noise
        X = np.vstack([
            np.hstack([
                x1.reshape(x1.shape[0], 1),
                y1.reshape(y1.shape[0], 1)
            ]),
            np.hstack([
                x2.reshape(x2.shape[0], 1),
                y2.reshape(y2.shape[0], 1)
            ])
        ])
        T = np.array([0] * class_size + [1] * class_size)
        self.len = class_size * self.classess
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(T)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

"""
1.c
spiral data (SP)
"""
class dset_SP(Dataset):
    def __init__(self, class_size, radius=5, noise=0.2):
        self.classes = 3
        r = np.linspace(0.0, radius, class_size)
        t0 = np.linspace(0, 4 * np.pi / self.classes, class_size)
        t1 = np.linspace(4 * np.pi / self.classes, 8 * np.pi / self.classes, class_size)
        t2 = np.linspace(8 * np.pi / self.classes, 12 * np.pi / self.classes, class_size)

        x0 = r * np.cos(t0) + np.random.randn(class_size) * noise
        y0 = r * np.sin(t0) + np.random.randn(class_size) * noise
        x1 = r * np.cos(t1) + np.random.randn(class_size) * noise
        y1 = r * np.sin(t1) + np.random.randn(class_size) * noise
        x2 = r * np.cos(t2) + np.random.randn(class_size) * noise
        y2 = r * np.sin(t2) + np.random.randn(class_size) * noise
        x0 = x0.reshape(x0.shape[0], 1)
        x1 = x1.reshape(x1.shape[0], 1)
        x2 = x2.reshape(x2.shape[0], 1)
        y0 = y0.reshape(y0.shape[0], 1)
        y1 = y1.reshape(y1.shape[0], 1)
        y2 = y2.reshape(y2.shape[0], 1)

        X0 = np.hstack((x0, y0))
        X1 = np.hstack((x1, y1))
        X2 = np.hstack((x2, y2))

        X = np.vstack((X0, X1, X2))
        T = np.array([0] * class_size + [1] * class_size + [2] * class_size)

        self.len = class_size * self.classes
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(T)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len