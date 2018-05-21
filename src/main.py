"""Main pipeline of DA-RNN.

@author Zhenye Na 05/21/2018

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).


"""

import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.autograd import Variable

# Parameters settings
parser = argparse.ArgumentParser(description="DA-RNN")

# Dataset setting
parser.add_argument('--dataroot', type=str, default="../nasdaq/", help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers [2]')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size [128]')

# Encoder / Decoder parameters setting
parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
parser.add_argument('--ntimetep', type=int, default=10, help='the number of time steps in the window T [10]')

# Training parameters setting
parser.add_argument('--niters', type=int, default=200, help='number of epochs to train')
parser.add_argument('--resume', type=bool, default=False, help='resume training or not')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--cuda', action='store_true', help='enables cuda')


parser.add_argument('--manualSeed', type=int, help='manual seed')







