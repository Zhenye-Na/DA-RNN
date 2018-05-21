# Implementation of DA-RNN

> *Get hands-on experience of implementation of RNN (LSTM) in Pytorch;*  
> *Get familiar with Finacial data with Deep Learning;*

## 1. Dataset
### 1.1 Download
[NASDAQ 100 stock data](http://cseweb.ucsd.edu/~yaq007/NASDAQ100_stock_data.html)

### 1.2 Description
This dataset is a subset of the full `NASDAQ 100 stock dataset` used in <sup>[1]</sup>. It includes 105 days' stock data starting from July 26, 2016 to December 22, 2016. Each day contains 390 data points except for 210 data points on November 25 and 180 data points on Decmber 22.

Some of the corporations under `NASDAQ 100` are not included in this dataset because they have too much missing data. There are in total 81 major coporations in this dataset and we interpolate the missing data with linear interpolation.

In <sup>[1]</sup>, the first 35,100 data points are used as the training set and the following 2,730 data points are used as the validation set. The last 2,730 data points are used as the test set.


## References

[1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell. [*"A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"*](https://arxiv.org/pdf/1704.02971.pdf). arXiv preprint arXiv:1704.02971 (2017).