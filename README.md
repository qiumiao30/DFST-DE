# DFST-DE
# Introduction
Efficient wind power cluster management and sustainable industry development require an optimized maintenance approach integrating condition monitoring. However, the intricate spatiotemporal correlations and non-stationarity of the data continue to pose significant challenges in extracting valuable insights from the supervisory control and data acquisition (SCADA) system. Hence, this paper proposes a novel approach for **D**ynamically **F**using **S**patio-**T**emporal information with **D**ifference **E**xcitation (DFST-DE) in wind power systems' condition monitoring. First, it extracts spatial dependencies by integrating sparse graph structures with Gumbel Softmax regularization and incorporates temporal information using a hybrid adaptive self-attention mechanism. This combined approach effectively captures comprehensive spatiotemporal correlations. Second, it proposes a Spatio-Temporal Difference Excitation module to mitigate non-stationarity in time series data by smoothing trend changes. The proposed DFST-DE approach achieves superior performance by effectively fusing spatial, temporal and excitation information. Finally, it optimizes anomaly detection accuracy and efficiency by dynamically adjusting the threshold based on anomaly score statistics. Experimental results on real-world wind farm datasets demonstrate that the proposed approach outperforms other established methods in detecting early abnormal conditions in wind turbines.


# Model Overview

![model](https://github.com/qiumiao30/SLMR/blob/master/image/model.png)

# Geting Started
To clone this repo:
```bash
git clone https://github.com/qiumiao30/DFST-DE.git && cd DFST-DE
```


## 1. Install Dependencies(Recomend Virtualenv)

- python>=3.7
- torch>=1.9


## 2. Params

> - --dataset :  default "WT01".
> - --lookback : Windows size, default 10.
> - --normalize : Whether to normalize, default True.
> - --epochs : default 10
> - --bs : Batch Size, default 256
> - --init_lr : init learning rate, default 1e-3
> - --val_split : val dataset, default 0.1
> - --dropout : 0.3

## 5. run

```python
python train.py --Params "value" --Parmas "value" ......
```
