import torch
import torch.nn as nn

import unidecode
import string
import random

import matplotlib.pyplot as plt

all_characters = string.printable
n_characters = len(all_characters)

file = unidecode.unidecode(open('../data/shakespeare.txt').read())
file_len = len(file)

chunk_len = 200

def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

def random_training_set():
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(Model, self).__init__()
        #######################################################################
        #                       Fill in your code here                        #
        #######################################################################
        raise NotImplementedError
        #######################################################################
        #                                End                                  #
        #######################################################################

    def forward(self, input, hidden):
        #######################################################################
        #                       Fill in your code here                        #
        #######################################################################
        raise NotImplementedError
        return None
        #######################################################################
        #                                End                                  #
        #######################################################################

    def init_hidden(self):
        #######################################################################
        #                       Fill in your code here                        #
        #######################################################################
        raise NotImplementedError
        return None
        #######################################################################
        #                                End                                  #
        #######################################################################

n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005

#######################################################################
#                       Fill in your code here                        #
#######################################################################
raise NotImplementedError
model = None
optimizer = None
lossfn = None
#######################################################################
#                                End                                  #
#######################################################################

all_losses = []
loss_avg = 0

def train(inp, target):
    #######################################################################
    #                       Fill in your code here                        #
    #######################################################################
    raise NotImplementedError
    #######################################################################
    #                                End                                  #
    #######################################################################

    return loss.data[0] / chunk_len

def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    hidden = model.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = model(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted

for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set())
    loss_avg += loss

    if epoch % print_every == 0:
        print('[(%d %d%%) %.4f]' % (epoch, float(epoch) / n_epochs * 100, loss))

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0

for i in range(10):
    print(evaluate(input("Input Starter: "), 100))

plt.figure()
plt.plot(all_losses)
plt.show()