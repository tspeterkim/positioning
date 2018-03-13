import numpy as np

import torch
from torch.autograd import Variable

from model import RNN
from utils import *

def eval_model(rnn, data_loader):
    # Test the Model
    correct = 0
    total = 0
    for batch_X, batch_y in data_loader:
        points = Variable(torch.from_numpy(batch_X))
        outputs = rnn(points)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.shape[0]
        correct += (predicted == torch.from_numpy(batch_y)).sum()
    return correct / total


def test_model(args):
    # Hyper Parameters
    sequence_length = args.seq_len
    input_size = args.input_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_classes = args.num_classes
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    # Load back the best performing model
    rnn = RNN('LSTM', input_size, hidden_size, num_layers, num_classes)
    rnn.load_state_dict(torch.load('rnn.pkl'))

    train_dataset = create_dataset('data/train/', timesteps=sequence_length)
    train_loader = dataloader(train_dataset, batch_size=batch_size)
    test_dataset = create_dataset('data/test/', timesteps=sequence_length)
    test_loader = dataloader(test_dataset, batch_size=batch_size)

    # print('training accuracy = %.4f, test accuracy = %.4f' % (eval_model(rnn, train_loader), eval_model(rnn, test_loader)))
    print('training accuracy = %.4f' % eval_model(rnn, train_loader))
    print('test accuracy = %.4f' % eval_model(rnn, test_loader))
