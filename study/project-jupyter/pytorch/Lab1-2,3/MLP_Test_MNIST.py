import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import Dataset as dset
import MLP as mlp

#######################################################################
# 1.f                                                                 #
# classification with synthetic/real datasets                         #
#######################################################################

"""
1.f.v
training with SGD using CC
"""
data = dset.dset_MNIST()
data.create()
x, y, tx, ty = data.get_data()
dataset_size = data.training_size

lossfn = mlp.nn_MSECriterion()
NN = mlp.nn_2layerNet(784, 5, data.classes)
NN.activ = mlp.nn_ReLU()
mlp.SGD(x, y, 10, NN, lossfn, num_epoch=100, lr=1e-2)

"""
1.f.iv
testing with CC
"""
test_acc = 0
test_num = data.test_size
error_img = np.zeros((10, 10, 784))
error_label = np.zeros((10, 10)) - 1
j = np.zeros(10,).astype(int)

for i, test in enumerate(tx):
    predict_label = NN.predict(test)
    true_label = np.argmax(ty[i, :])
    if predict_label == true_label:
        test_acc = test_acc + 1

    elif predict_label != true_label:
        error_img[true_label, j[true_label]] = test
        error_label[true_label, j[true_label]] = predict_label
        if j[true_label] < 9:
            j[true_label] += 1

print ('Test Accuracy = %.2f%%' % (float(test_acc)/test_num*100))

sqrtimg = int(np.ceil(np.sqrt(error_img.shape[2])))

fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(10, 10)
gs.update(wspace=0.05, hspace=0.05)

for j, imgs in enumerate(error_img):
    for i, img in enumerate(imgs):
        ax = plt.subplot(gs[j * 10 + i])
        ax.text(8, 9, int(error_label[j, i]), verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=15)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))

plt.show()
