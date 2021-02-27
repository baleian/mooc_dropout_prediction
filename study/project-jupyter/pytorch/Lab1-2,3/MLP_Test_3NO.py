import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import Dataset as dset
import MLP as mlp

#######################################################################
# 1.f                                                                 #
# classification with synthetic/real datasets                         #
#######################################################################

"""
1.f.iii
training with gradient descent using 3NO
"""
data = dset.dset_3NO(500)
data.create()
training_rate = 0.75
x, y, tx, ty = data.get_data(training_rate)
dataset_size = int(data.classes * data.class_size * training_rate)

lossfn = mlp.nn_MSECriterion()
NN = mlp.nn_1layerNet(2, data.classes)
mlp.GD(x, y, NN, lossfn, num_epoch=20000, lr=1e-2)

"""
1.f.iv
testing with 3NO
"""
test_acc = 0
test_num = data.class_size * data.classes - dataset_size
for i, test in enumerate(tx):
    if NN.predict(test) == np.argmax(ty[i, :]):
        test_acc = test_acc + 1

print ('Test Accuracy = %.2f%%' % (float(test_acc)/test_num*100))

img_size = 300
maxVal = np.max(x, axis=0)
minVal = np.min(x, axis=0)
img = np.zeros((img_size,img_size, 3))
pX, pY = np.arange(img_size), np.arange(img_size)
for px in pX:
    for py in pY:
        pdata = np.array([[px/(float(img_size)-1) * (maxVal[0]-minVal[0]) + minVal[0], py/(float(img_size)-1) * (maxVal[1]-minVal[1]) + minVal[1]]])
        i = NN.predict(pdata)
        img[py, px, :] = np.zeros((1,3))
        img[py, px, i] = 1

ox = np.zeros_like(tx)
olabel = np.zeros(tx.shape[0])
for i, ent in enumerate(tx):
    ox[i, 0] = int(((ent[0] - minVal[0]) / (maxVal[0]-minVal[0]))*(float(img_size)-1))
    ox[i, 1] = int(((ent[1] - minVal[1]) / (maxVal[1]-minVal[1]))*(float(img_size)-1))
    olabel[i] = np.argmax(ty[i])

plt.imshow(img, extent=[0,img_size,0,img_size], origin='lower')
plt.scatter(ox[:,0], ox[:,1], c=olabel, s=100, alpha=0.8)
plt.show()