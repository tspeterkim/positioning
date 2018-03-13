import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import RNN
from evaluate import eval_model
from utils import *

def train_model(args):
    # Hyper Parameters
    sequence_length = args.seq_len
    input_size = args.input_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_classes = args.num_classes
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    # Create the dataset
    train_dataset = create_dataset('data/train/', timesteps=sequence_length)
    train_loader = dataloader(train_dataset, batch_size=batch_size)
    test_dataset = create_dataset('data/test/', timesteps=sequence_length)
    test_loader = dataloader(test_dataset, batch_size=batch_size)

    # Define model
    rnn = RNN('LSTM', input_size, hidden_size, num_layers, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    # Train the Model
    i = 0 # updates
    best_test_acc = 0.0
    for epoch in range(num_epochs):
        # Generate random batches every epoch
        train_loader = dataloader(train_dataset, batch_size=batch_size)
        for batch_X, batch_y in train_loader:
            points = Variable(torch.from_numpy(batch_X))
            labels = Variable(torch.from_numpy(batch_y))

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print ('Epoch [%d/%d], Loss: %.4f' % (epoch+1, num_epochs, loss.data[0]))
            if i % 100 == 0: # every 100 updates, evaluate on test set
                # print("training accuracy = %.4f" % eval_model(rnn, train_loader))
                test_acc = eval_model(rnn, test_loader)
                print("test accuracy = %.4f" % test_acc)
                if test_acc > best_test_acc:
                    print ("best test accuracy found")
                    best_test_acc = test_acc
                    torch.save(rnn.state_dict(), 'rnn_best.pkl')
            i += 1
