"""DA-RNN model initialization.

@author Zhenye Na 05/21/2018

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).

"""
from ops import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable

import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')

# self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)


class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T, input_size, encoder_num_hidden, parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.encoder_num_hidden)

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(in_features=2 * self.encoder_num_hidden, out_features=1, bias=True)

    def forward(self, X):
        """forward.

        Args:
            X

        """
        # Eq. 8, parameters not in nn.Linear but to be learnt
        v_e = torch.nn.Parameter(data=torch.empty(self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        U_e = torch.nn.Parameter(data=torch.empty(self.T, self.T).uniform_(0, 1) , requires_grad=True)

        # hidden, cell: initial states with dimention hidden_size
        h_n = self._init_states(X)
        s_n = self._init_states(X)

        for t in range(self.T):
            # batch_size * input_size * (2*hidden_size + T - 1)
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

        # e = F.tanh(..)
        # alpha = F.softmax(x.view(-1, self.input_size))




    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        if self.parallel:
            initial_states = Variable(torch.zeros(X.size(0), self.encoder_num_hidden)).cuda()
        else:
            initial_states = Variable(torch.zeros(X.size(0), self.encoder_num_hidden))
        return initial_states


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, T, decoder_num_hidden):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.T = T


    def forward():
        """forward."""
        pass







class DA_rnn(nn.Module):
    """da_rnn."""

    def __init__(self, X, y, T, encoder_num_hidden, decoder_num_hidden, batch_size, learning_rate=0.001, epochs, parallel=False):
        """da_rnn initialization."""
        super(DA_rnn, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T

        # Loss function
        self.criterion = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        # Read dataset
        self.X_train, self.y_train, self.X_test, self.y_test, _, _ = train_val_test_split(X, y, False)
        self.total_timestamps = X.shape[0]
        self.train_timestamps = X_train.shape[0]
        self.test_timestamps = X_test.shape[0]
        self.input_size = X.shape[1]

    def train(self):
        """training process."""

        n_iter = 0

        for epoch in tqdm(range(self.epochs)):
            if self.shuffle:
                ref_idx = np.random.permutation(self.total_timestamps - self.T)
            else:
                ref_idx = np.array([range(self.total_timestamps - self.T)])

            for idx in ref_idx:
                indices = list(range(idx, idx + self.batch_size + 1))
                x = np.zeros((self.T - 1, len(indices), self.input_size))
                # y_history =
                y_gt = self.y_train[indices[-1] + self.T]

                n_iter += 1

                if n_iter % 5 == 0 and n_iter > 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9






    def val(self):
        """validation."""
        pass

    def test(self):
        """test."""
        pass

