"""DA-RNN model initialization.

@author Zhenye Na 05/21/2018

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).

"""
from ops import *
from torch.autograd import Variable

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T,
                 input_size,
                 encoder_num_hidden,
                 parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.parallel = parallel
        self.T = T

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.encoder_num_hidden)

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T - 1, out_features=1, bias=True)

    def forward(self, X):
        """forward.

        Args:
            X

        """
        X_tilde = Variable(X.data.new(
            X.size(0), self.T - 1, self.input_size).zero_())
        X_encoded = Variable(X.data.new(
            X.size(0), self.T - 1, self.encoder_num_hidden).zero_())

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # hidden, cell: initial states with dimention hidden_size
        h_n = self._init_states(X)
        s_n = self._init_states(X)

        for t in range(self.T - 1):
            # batch_size * input_size * (2*hidden_size + T - 1)
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T - 1))

            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size))

            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])

            # encoder LSTM
            self.encoder_lstm.flatten_parameters()
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = Variable(X.data.new(
            1, X.size(0), self.encoder_num_hidden).zero_())
        return initial_states


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_num_hidden + encoder_num_hidden, encoder_num_hidden),
                                        nn.Tanh(),
                                        nn.Linear(encoder_num_hidden, 1))
        self.lstm_layer = nn.LSTM(
            input_size=1, hidden_size=decoder_num_hidden)
        self.fc = nn.Linear(encoder_num_hidden + 1, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)

        self.fc.weight.data.normal_()

    def forward(self, X_encoed, y_prev):
        """forward."""
        d_n = self._init_states(X_encoed)
        c_n = self._init_states(X_encoed)

        for t in range(self.T - 1):

            x = torch.cat((d_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           X_encoed), dim=2)

            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T - 1))
            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoed)[:, 0, :]
            if t < self.T - 1:
                # Eqn. 15
                # batch_size * 1
                y_tilde = self.fc(
                    torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))
                # 1 * batch_size * decoder_num_hidden
                d_n = final_states[0]
                # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]
        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))

        return y_pred

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder.

        Args:
            X
        Returns:
            initial_hidden_states

        """
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        initial_states = Variable(X.data.new(
            1, X.size(0), self.decoder_num_hidden).zero_())
        return initial_states


class DA_rnn(nn.Module):
    """da_rnn."""

    def __init__(self, X, y, T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 parallel=False):
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
        self.X = X
        self.y = y

        self.Encoder = Encoder(input_size=X.shape[1],
                               encoder_num_hidden=encoder_num_hidden,
                               T=T)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T)

        # Loss function
        self.criterion = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate)

        # Read dataset
        self.X_train, self.y_train, self.X_test, self.y_test, _, _ = train_val_test_split(
            X, y, False)

        # self.y_train = self.y_train.reshape(self.y_train.shape[0], -1)
        # self.y_test = self.y_test.reshape(self.y_test.shape[0], -1)

        self.total_timesteps = self.X.shape[0]
        self.train_timesteps = self.X_train.shape[0]
        self.test_timesteps = self.X_test.shape[0]
        self.input_size = self.X.shape[1]

    def train(self):
        """training process."""
        self.loss = []
        self.y_pred = []
        self.true = []
        n_iter = 0

        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))

            idx = 0

            while (idx < self.train_timesteps):
                # get the indices of X_train
                indices = ref_idx[idx:(idx + self.batch_size)]
                # x = np.zeros((self.T - 1, len(indices), self.input_size))
                self.x = np.zeros((len(indices), self.T - 1, self.input_size))
                y_prev = np.zeros((len(indices), self.T - 1))
                y_gt = self.y_train[indices + self.T]

                # format x into 3D tensor
                for bs in range(len(indices)):
                    self.x[bs, :, :] = self.X[indices[bs]:(indices[bs] + self.T - 1), :]
                    y_prev[bs, :] = self.y[indices[bs]:(indices[bs] + self.T - 1)]

                n_iter += 1
                idx += self.batch_size

                # zero gradients
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                input_weighted, input_encoded = self.Encoder(
                    Variable(torch.from_numpy(self.x).type(torch.FloatTensor)))
                y_pred = self.Decoder(input_encoded, Variable(
                    torch.from_numpy(y_prev).type(torch.FloatTensor)))

                y_true = Variable(torch.from_numpy(
                    y_gt).type(torch.FloatTensor))

                self.y_true = y_true
                self.y_pred = y_pred

                loss = self.criterion(y_pred, y_true)
                self.loss.append(loss.data[0])
                loss.backward()

                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                if n_iter % 500 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                if n_iter % 10 == 0:
                    print("Iterations: ", n_iter, "\tLoss: ", self.loss[-1])

            if epoch % 2 == 0:
                y_train_pred = self.test(on_train=True)
                y_test_pred = self.test(on_train=False)
                y_pred = np.concatenate((y_train_pred, y_test_pred))
                plt.figure()
                plt.plot(range(1, 1 + len(self.y_train)),
                         self.y_train, label="True")
                plt.plot(range(self.T, len(y_train_pred) + self.T),
                         y_train_pred, label='Predicted - Train')
                plt.plot(range(self.T + len(y_test_pred), len(self.y_test) + 1),
                         y_test_pred, label='Predicted - Test')
                plt.legend(loc='upper left')
                plt.savefig("plot_" + str(epoch) + ".png")
                plt.close(fig)

            # Save files in last iterations
            if epoch == self.epochs - 1:
                np.savetxt('../loss.txt', np.array(self.loss), delimiter=',')
                np.savetxt('../y_pred.txt',
                           np.array(self.y_pred), delimiter=',')
                np.savetxt('../y_true.txt',
                           np.array(self.y_true), delimiter=',')

    def val(self):
        """validation."""
        pass

    def test(self, on_train=False):
        """test."""
        if on_train:
            y_pred = np.zeros(self.train_timesteps - self.T + 1)
        else:
            y_pred = np.zeros(self.X_test.shape[0] - self.test_timesteps)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: (i + self.batch_size)]

            if on_train:
                X = np.zeros((len(batch_idx), self.T - 1, self.X_train.shape[1]))
            else:
                X = np.zeros((len(batch_idx), self.T - 1, self.X_test.shape[1]))

            y_history = np.zeros((len(batch_idx), self.T - 1))
            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X_train[range(
                        batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y_train[range(
                        batch_idx[j], batch_idx[j] + self.T - 1)]
                else:
                    X[j, :, :] = self.X_test[range(
                        batch_idx[j] + self.test_timesteps - self.T, batch_idx[j] + self.test_timesteps - 1), :]
                    y_history[j, :] = self.y_test[range(
                        batch_idx[j] + self.test_timesteps - self.T, batch_idx[j] + self.test_timesteps - 1)]

            if on_train:
                y_history = Variable(torch.from_numpy(
                    y_history).type(torch.FloatTensor))
                _, input_encoded = self.Encoder(
                    Variable(torch.from_numpy(self.x).type(torch.FloatTensor)))
                y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded,
                                                               y_history).cpu().data.numpy()[:, 0]
                i += self.batch_size
            else:
                y_history = Variable(torch.from_numpy(
                    y_history).type(torch.FloatTensor))
                _, input_encoded = self.Encoder(
                    Variable(torch.from_numpy(self.x).type(torch.FloatTensor)))
                y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded,
                                                               y_history).cpu().data.numpy()[:, 0]
                i += self.batch_size
        return y_pred
