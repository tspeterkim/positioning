from train import train_model
from evaluate import test_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--seq_len', type=int, default=250)
parser.add_argument('--input_size', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--learning_rate', type=int, default=0.001)
args = parser.parse_args()

if args.train:
    train_model(args)
elif args.test:
    test_model(args)
