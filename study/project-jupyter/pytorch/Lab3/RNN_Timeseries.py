import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn

data_csv = pd.read_csv('../data/lstm.csv', usecols=[1])
# data_csv = pd.read_csv('../data/lstm2.csv', usecols=[5])

data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))

def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

data_X, data_Y = create_dataset(dataset)
train_size = int(len(data_X) * 0.9)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

train_X = train_X.reshape(-1, 1, 2)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
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
        #######################################################################
        #                                End                                  #
        #######################################################################

#######################################################################
#                       Fill in your code here                        #
#######################################################################
raise NotImplementedError
model = None
lossfn = None
optimizer = None
#######################################################################
#                                End                                  #
#######################################################################

for e in range(1000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    #######################################################################
    #                       Fill in your code here                        #
    #######################################################################
    raise NotImplementedError
    #######################################################################
    #                                End                                  #
    #######################################################################
    if (e + 1) % 100 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))

model = model.eval()

data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
pred_test = model(data_X)

pred_test = pred_test.view(-1).data.numpy()

plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset, 'b', label='real')
plt.legend(loc='best')
plt.show()