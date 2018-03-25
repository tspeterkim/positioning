import numpy as np
from sklearn.metrics import f1_score
import itertools
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, accuracy_score, recall_score

import torch
from torch.autograd import Variable

from model import RNN
from utils import *

def eval_model(rnn, data_loader):
    # Test the Model
    correct, total = 0, 0
    for batch_X, batch_y in data_loader:
        points = Variable(torch.from_numpy(batch_X))
        labels = torch.from_numpy(batch_y)
        if next(rnn.parameters()).is_cuda:
            points, labels = points.cuda(), labels.cuda()
        outputs = rnn(points)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return correct / total

def print_confusion_matrix(rnn, data_loader):
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    y_test, y_pred = [], []
    for batch_X, batch_y in data_loader:
        points = Variable(torch.from_numpy(batch_X))
        labels = torch.from_numpy(batch_y)
        if next(rnn.parameters()).is_cuda:
            points, labels = points.cuda(), labels.cuda()
        outputs = rnn(points)
        _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum()
        y_test.append(labels.numpy())
        y_pred.append(predicted.numpy())
    # return correct / total
    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)

    cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])
    print(f1_score(y_test, y_pred, average='macro'))
    # plot_confusion_matrix(cnf_matrix, ['laying down','sitting','standing','walking'], title='Confusion Matrix')



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
    dropout = args.dropout

    # Load back the best performing model
    rnn = RNN('LSTM', input_size, hidden_size, num_layers, num_classes, dropout)
    if args.cuda:
        rnn = rnn.cuda()
    rnn.load_state_dict(torch.load(args.model_path))

    # train_dataset = create_dataset('data/train/', timesteps=sequence_length)
    # train_loader = dataloader(train_dataset, batch_size=batch_size)
    test_dataset = create_dataset('data/test/', timesteps=sequence_length)
    test_loader = dataloader(test_dataset, batch_size=batch_size)

    print('-'*50)
    # print('training accuracy = %.4f, test accuracy = %.4f' % (eval_model(rnn, train_loader), eval_model(rnn, test_loader)))
    # print('training accuracy = %.4f' % eval_model(rnn, train_loader))
    print('test accuracy = %.4f' % eval_model(rnn, test_loader))
    # print('test f1-score = %.4f' % get_f1score(rnn, test_loader))
    print_confusion_matrix(rnn, test_loader)
