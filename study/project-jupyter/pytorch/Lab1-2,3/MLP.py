import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("error")

def _safelog(x):
    x[x <= 0] = 4e-18
    return np.log(x)

def _safeexp(x):
    x[x >= 100] = 100
    x[x <= -100] = -100
    return np.exp(x)

def _safediv(x, y):
    y[y==0] = 0.000000001
    return x / y


class nn_Linear:
    def __init__(self, input_dim, output_dim, std=1e-2):
        self.weight = np.random.randn(input_dim, output_dim) * std
        self.bias = np.zeros((1, output_dim))
        self.gradWeight = np.zeros_like(self.weight)
        self.gradBias = np.zeros_like(self.bias)

    def forward(self, x):
        return np.dot(x, self.weight) + self.bias
   
    def backward(self, x, gradOutput):
        self.gradWeight = np.dot(x.T, gradOutput)
        self.gradBias = np.sum(gradOutput, axis=0, keepdims=True)
        return np.dot(gradOutput, self.gradWeight.T)
    
    def getParameters(self):
        params = [self.weight, self.bias]
        gradParams = [self.gradWeight, self.gradBias]
        return params, gradParams


class nn_Sigmoid:
    def forward(self, x):
        return 1 / (1+_safeexp(-x))

    def backward(self, x, gradOutput):
        gv = 1 / (1+_safeexp(-x))
        return np.multiply(np.multiply(gv, 1-gv), gradOutput)


class nn_ReLU:
    def forward(self, x):
        return np.maximum(x, 0)
    
    def backward(self, x, gradOutput):
        return np.multiply((x > 0), gradOutput)


class nn_MSECriterion:
    def forward(self, predictions, labels):
        n = labels.shape[0]
        return np.sum(np.square(predictions - labels)) / n
    
    def backward(self, predictions, labels): 
        n = labels.shape[0]
        return (2 * (predictions - labels)) / n


class nn_CrossEntropyLoss:
    def forward(self, predictions, labels):
        n = labels.shape[0]
        exp = _safeexp(predictions)
        softmax = exp / np.sum(exp, axis=1, keepdims=True)
        return -np.sum(np.multiply(labels, _safelog(softmax))) / n
    
    def backward(self, predictions, labels):
        n = labels.shape[0]
        exp = _safeexp(predictions)
        softmax = exp / np.sum(exp, axis=1, keepdims=True)
        return (softmax - labels) / n


class nn_BCECriterion:
    def forward(self, predictions, labels):
        n = labels.shape[0]
        return -np.sum(np.multiply(labels, _safelog(predictions)) + np.multiply(1-labels, _safelog(1-predictions))) / n
    
    def backward(self, predictions, labels):
        n = labels.shape[0]
        return -1 * (_safediv(labels, predictions) - _safediv(1-labels, 1-predictions)) / n

    
class nn_Net:
    def __init__(self, input_dim, output_dim):
        self.layer = nn_Linear(input_dim, output_dim)

    def forward(self, xi):
        self.a0 = self.layer.forward(xi)
        return self.a0
    
    def backward(self, xi, da0):
        dx = self.layer.backward(xi, da0)
        return dx

    def predict(self, xi):
        return self.forward(xi)

    def gradUpdate(self, lr):
        self.layer.weight -= np.multiply(lr, self.layer.gradWeight)
        self.layer.bias -= np.multiply(lr, self.layer.gradBias)


class nn_1layerNet:
    def __init__(self, input_dim, output_dim):
        self.layer = nn_Linear(input_dim, output_dim)
        self.activ = nn_ReLU()

    def forward(self, xi):
        self.a0 = self.layer.forward(xi)
        self.a1 = self.activ.forward(self.a0)
        return self.a1
    
    def backward(self, xi, da0):
        da1 = self.activ.backward(self.a0, da0)
        dx = self.layer.backward(xi, da1)
        return dx

    def predict(self, xi):
        return np.argmax(self.forward(xi), axis=1)

    def gradUpdate(self, lr):
        self.layer.weight -= lr * self.layer.gradWeight
        self.layer.bias -= lr * self.layer.gradBias

        
class nn_2layerNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layer1 = nn_Linear(input_dim, hidden_dim)
        self.layer2 = nn_Linear(hidden_dim, output_dim)
        self.activ = nn_ReLU()

    def forward(self, xi):
        self.a0 = self.layer1.forward(xi)
        self.a1 = self.activ.forward(self.a0)
        self.a2 = self.layer2.forward(self.a1)
        self.a3 = self.activ.forward(self.a2)
        return self.a3

    def backward(self, xi, da0):
        da1 = self.activ.backward(self.a2, da0)
        da2 = self.layer2.backward(self.a1, da1)
        da3 = self.activ.backward(self.a0, da2)
        dx = self.layer1.backward(xi, da3)
        return dx

    def predict(self, xi):
        return np.argmax(self.forward(xi), axis=1)

    def gradUpdate(self, lr):
        self.layer1.weight -= lr * self.layer1.gradWeight
        self.layer1.bias -= lr * self.layer1.gradBias
        self.layer2.weight -= lr * self.layer2.gradWeight
        self.layer2.bias -= lr * self.layer2.gradBias


def GD(x, y, net, lossfn, lr=1e-2, num_epoch=400):
    costs = []
    for epoch in range(0, num_epoch):
        y_hat = net.forward(x)
        loss = lossfn.forward(y_hat, y)
        da0 = lossfn.backward(y_hat, y)
        net.backward(x, da0)
        net.gradUpdate(lr)
        if (epoch+1) % (num_epoch / 10) == 0:
            print ('Epoch %d, loss=%f'%(epoch + 1, loss))
        costs.append(loss)

    plt.plot(costs)
    plt.show()

def SGD(x, y, batch_size, net, lossfn, lr=1e-2, num_epoch=400):
    costs = []
    for epoch in range(0, num_epoch):
        loss = 0
        for i in range(0, int(x.shape[0] / batch_size)):
            xi = x[i*batch_size:(i+1)*batch_size, :]
            yi = y[i*batch_size:(i+1)*batch_size, :]
            y_hat = net.forward(xi)
            loss += lossfn.forward(y_hat, yi)
            da0 = lossfn.backward(y_hat, yi)
            net.backward(xi, da0)
            net.gradUpdate(lr)
            
        if (epoch+1) % (num_epoch / 10) == 0:
            print ('Epoch %d, loss=%f'%(epoch + 1, loss / (x.shape[0] / batch_size)))
        costs.append(loss / (x.shape[0] / batch_size))

    plt.plot(costs)
    plt.show()