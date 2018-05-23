# Implementation of DA-RNN

> *Get hands-on experience of implementation of RNN (LSTM) in Pytorch;*  
> *Get familiar with Finacial data with Deep Learning;*

## Table of Contents

- [Dataset](#dataset)
    - [Download](#download)
    - [Description](#description)
- [DA-RNN](#da-rnn)
    - [LSTM](#lstm)
    - [Attention Mechanism](#attention-mechanism)
    - [Model](#model)
    - [Experiments and Parameters Settings](#experiments-and-parameters-settings)
        - [NASDAQ 100 Stock dataset]()
        - [Training procedure & Parameters Settings](#training-procedure--parameters-settings)
- [References](#references)



## Dataset
### Download
[NASDAQ 100 stock data](http://cseweb.ucsd.edu/~yaq007/NASDAQ100_stock_data.html)

### Description
This dataset is a subset of the full `NASDAQ 100 stock dataset` used in <sup>[1]</sup>. It includes 105 days' stock data starting from July 26, 2016 to December 22, 2016. Each day contains 390 data points except for 210 data points on November 25 and 180 data points on Decmber 22.

Some of the corporations under `NASDAQ 100` are not included in this dataset because they have too much missing data. There are in total 81 major coporations in this dataset and we interpolate the missing data with linear interpolation.

In <sup>[1]</sup>, the first 35,100 data points are used as the training set and the following 2,730 data points are used as the validation set. The last 2,730 data points are used as the test set.

## DA-RNN

In the paper [*"A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"*](https://arxiv.org/pdf/1704.02971.pdf). 

They proposes a novel dual-stage attention-based recurrent neural network (DA-RNN) for time series prediction. In the ﬁrst stage, an input attention mechanism is introduced to adaptively extract relevant driving series (a.k.a., input features) at each time step by referring to the previous encoder hidden state. In the second stage, a temporal attention mechanism is introduced to select relevant encoder hidden states across all time steps.

For the objective, a square loss is used. With these two attention mechanisms, the DA-RNN can adaptively select the most relevant input features and capture the long-term temporal dependencies of a time series. A graphical illustration of the proposed model is shown in Figure 1.


<figure>
    <img src="https://github.com/Zhenye-Na/DA-RNN/blob/master/fig/fig1.png?raw=true" width="100%" class="center">
    <figcaption><center>Figure 1: Graphical illustration of the dual-stage attention-based recurrent neural network.</center></figcaption>
</figure>
  
  
  
The Dual-Stage Attention-Based RNN (a.k.a. DA-RNN) model belongs to the general class of Nonlinear Autoregressive Exogenous (NARX) models, which predict the current value of a time series based on historical values of this series plus the historical values of multiple exogenous time series.

### LSTM

Recursive Neural Network model has been used in this paper. RNN models are powerful to exhibit quite sophisticated dynamic temporal structure for sequential data. RNN models come in many forms, one of which is the Long-Short Term Memory(LSTM) model that is widely applied in language models. 


### Attention Mechanism

Attention mechanism performs feature selection as the paper mentioned, the model can keep only the most useful information at each temporal stage.

### Model

DA-RNN model includes two LSTM networks with attention mechanism (an encoder and a decoder). 

In the encoder, they introduced a novel input attention mechanism that can adaptively select the relevant driving series. In the decoder, a temporal attention mechanism is used to automatically select relevant encoder hidden states across all time steps.

### Experiments and Parameters Settings

#### NASDAQ 100 Stock dataset
> In the NASDAQ 100 Stock dataset, we collected the stock prices of 81 major corporations under NASDAQ 100, which are used as the driving time series. The index value of the NASDAQ 100 is used as the target series. The frequency of the data collection is minute-by-minute. This data covers the period from July 26, 2016 to December 22, 2016, 105 days in total. Each day contains 390 data points from the opening to closing of the market except that there are 210 data points on November 25 and 180 data points on December 22. In our experiments, we use the ﬁrst 35,100 data points as the training set and the following 2,730 data points as the validation set. The last 2,730 data points are used as the test set. This dataset is publicly available and will be continuously enlarged to aid the research in this direction.


#### Training procedure & Parameters Settings
|                 Category                |                                       Description                                      |
|:---------------------------------------:|:--------------------------------------------------------------------------------------:|
|           Optimization method           |      minibatch stochastic gradient descent (SGD) together with the Adam optimizer      |
|   number of time steps in the window T  |                                         T = 10                                         |
| size of hidden states for the encoder m |                                     m = p = 64, 128                                    |
| size of hidden states for the decoder p |                                     m = p = 64, 128                                    |
|            Evaluation Metrics           | $$O(y_T , \hat{y_T} ) = \frac{1}{N} \sum \limits_{i=1}^{N} (y_T^i , \hat{y_T}^i)^2  $$ |

## References

[1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell. [*"A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"*](https://arxiv.org/pdf/1704.02971.pdf). arXiv preprint arXiv:1704.02971 (2017).  
[2] Chandler Zuo. [*"A PyTorch Example to Use RNN for Financial Prediction"*](http://chandlerzuo.github.io/blog/2017/11/darnn). (2017).  
[3] YitongCU. [*"Dual Staged Attention Model for Time Series prediction
"*](https://github.com/YitongCU/Duel-staged-Attention-for-NYC-Weather-prediction).  
[4] Pytorch Forum. [*"Why 3d input tensors in LSTM?"*](https://discuss.pytorch.org/t/why-3d-input-tensors-in-lstm/4455).

