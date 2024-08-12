import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import os
import torch.nn as nn

class QuantileLoss(nn.Module):
    ## From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

    def __init__(self, quantiles):
        ##takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                   (q-1) * errors,
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))): # np.isnan(data),返回bool,sum(bool),np.any()，判断是否有空值
        data = np.nan_to_num(data) # 使用0代替数组x中的nan元素，使用有限的数字代替inf元素（比较大的数）

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler

def get_data_dim(dataset):
    """
    :param dataset: Name of dataset
    :return: Number of dimensions in data
    """
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif str(dataset).startswith("machine"):
        return 38
    elif dataset == "SERVERMACHINEDATASET":
        return 38
    elif dataset == "SWAT":
        return 51
    elif dataset == "WADI":
        return 118
    elif dataset == "PSM":
        return 25
    elif dataset == "WIND":
        return 10
    elif dataset == "WINDNEW":
        return 37
    elif dataset == "KDD":
        return 41
    elif dataset == "WT03":
        return 10
    elif dataset == "WT13":
        return 34
    elif dataset == "WT23":
        return 10
    elif dataset == "ICE":
        return 27
    elif dataset == "OMI":
        return 19
    elif str(dataset).startswith("omi"):
        return 19
    else:
        raise ValueError("unknown dataset " + str(dataset))

def get_target_dims(dataset):
    """
    :param dataset: Name of dataset
    :return: index of data dimension that should be modeled (forecasted and reconstructed),
                     returns None if all input dimensions should be modeled
    """
    if dataset == "SMAP":
        return [0]
    elif dataset == "MSL":
        return [0]
    elif dataset == "SMD":
        return None
    elif dataset == "SERVERMACHINEDATASET":
        return None
    elif dataset == "SWAT":
        return None
    elif dataset == "WADI":
        return None
    elif dataset == "PSM":
        return None
    elif dataset == "KDD":
        return None
    elif dataset == "ICE":
        return None
    elif dataset == "WT03":
        return None
    elif dataset == "WT13":
        return None
    elif dataset == "WT23":
        return None
    elif dataset == "WIND":
        return None
    elif dataset == "WINDNEW":
        return None
    elif dataset == "OMI":
        return None
    else:
        raise ValueError("unknown dataset " + str(dataset))


def get_data(dataset, max_train_size=None, max_test_size=None,
             normalize=False, spec_res=False, train_start=0, test_start=0):
    """
    Get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    prefix = "../datasets"
    if str(dataset).startswith("machine"):
        prefix += "/ServerMachineDataset/processed"
    elif dataset in ["MSL", "SMAP"]:
        prefix += "/data/processed"
    elif dataset == "SERVERMACHINEDATASET":
        prefix += "/ServerMachineDataset/processed"
    elif dataset == "SWAT":
        prefix += "/SWAT/processed"
    elif dataset == "WADI":
        prefix += "/WADI/processed"
    elif dataset == "WIND":
        prefix += "/WIND/processed"
    elif dataset == "WINDNEW":
        prefix += "/WINDNEW/processed"
    elif dataset == "PSM":
        prefix += "/PSM/processed"
    elif dataset == "WT03":
        prefix += "/WT03/processed"
    elif dataset == "WT13":
        prefix += "/WT13/processed"
    elif dataset == "WT23":
        prefix += "/WT23/processed"
    elif dataset == "KDD":
        prefix += "/KDD/processed"
    elif dataset == "ICE":
        prefix += "/ICE/processed"
    elif str(dataset).startswith("omi"):
        prefix += "/OMI/processed"
    #elif dataset == "OMI":
    #    prefix += "/OMI/processed"
    else:
        raise Exception(f'Dataset "{dataset}" not available.')
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print("load data of:", dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset) # 维度(特征)

    train = pd.read_csv(f'{prefix}/{dataset}_train.csv')
    #train.drop(["2B_AIT_002_PV"], axis=1, inplace=True)
    # train_data = train.values.reshape((-1, x_dim))[train_start:train_end, :]
    try:
        test = pd.read_csv(f'{prefix}/{dataset}_test.csv')
        #test.drop(["2B_AIT_002_PV"], axis=1, inplace=True)
        # test_data = test.values.reshape((-1, x_dim))[test_start:test_end, :]
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        test_label = pd.read_csv(f'{prefix}/{dataset}_test_label.csv')
        test_label = test_label.values.reshape((-1))[test_start:test_end]
    except (KeyError, FileNotFoundError):
        test_label = None
#    try:
#        test_label = pd.read_csv(f'{prefix}/{dataset}_test.csv').iloc[:,-1]
        # test_label = test_label.values.reshape((-1))[test_start:test_end]
#    except (KeyError, FileNotFoundError):
#        test_label = None

    # f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
    # train_data = pickle.load(f).values.reshape((-1, x_dim))[train_start:train_end, :]
    # f.close()
    # try:
    #     f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
    #     test_data = pickle.load(f).values.reshape((-1, x_dim))[test_start:test_end, :]
    #     f.close()
    # except (KeyError, FileNotFoundError):
    #     test_data = None
    # try:
    #     f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
    #     test_label = pickle.load(f).values.reshape((-1))[test_start:test_end]
    #     f.close()
    # except (KeyError, FileNotFoundError):
    #     test_label = None

    train_cols = train.columns.tolist()
    test_cols = test.columns.tolist()

    cat_cols = []
    for col in (train_cols and test_cols):
        if test[col].nunique() <= 5 and train[col].nunique() <= 5:
            cat_cols.append(col)
    series_cols = [i for i in (train_cols and test_cols)]

    if normalize:
        train_data, scaler = normalize_data(train, scaler=None)
        test_data, _ = normalize_data(test, scaler=scaler)

    train_data = pd.DataFrame(train_data, columns=train_cols)
    test_data = pd.DataFrame(test_data, columns=test_cols)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", None if test_label is None else test_label.shape)
    return (train_data, None), (test_data, test_label), series_cols

# class SlidingWindowDataset(Dataset):
#     def __init__(self, df, window_size, target_dim=None, horizon=1, interval=3):
#         self.window_size = window_size  ##窗口大小
#         self.interval = interval  ##间隔
#         self.horizon = horizon   ##视野，标签（target）大小
#         df = pd.DataFrame(df)
#         df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
#         self.data = df
#         self.df_length = len(df)
#         self.x_end_idx = self.get_x_end_idx()
#
#     def __getitem__(self, index):
#         hi = self.x_end_idx[index]
#         lo = hi - self.window_size
#         train_data = self.data[lo: hi]
#         target_data = self.data[hi:hi + self.horizon]
#         x = torch.from_numpy(train_data).type(torch.float)
#         y = torch.from_numpy(target_data).type(torch.float)
#         return x, y
#
#     def __len__(self):
#         print(len(self.x_end_idx))
#         return len(self.x_end_idx)
#
#     def get_x_end_idx(self):
#         # each element `hi` in `x_index_set` is an upper bound for get training data获取训练数据上界
#         # training data range: [lo, hi), lo = hi - window_size
#         x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
#         x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
#         return x_end_idx

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1, stride=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon
        self.stride = stride
        self.data_x, self.data_y = self.get_xy()

    def get_xy(self):
        x_list, y_list = [], []
        seq_len, feature = self.data.shape

        for index in range(0, seq_len - self.window - self.horizon + 1, self.stride):
            x = self.data[index:index + self.window, :]
            y = self.data[index + self.window: index + self.window + self.horizon, :]
            x_list.append(x)
            y_list.append(y)
        data_x = torch.stack(x_list)
        data_y = torch.stack(y_list)

        return data_x, data_y

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        return x, y

    def __len__(self):
        return len(self.data_x)

def create_data_loaders(train_dataset, val_dataset, batch_size, val_split=0.1, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if val_split is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()


def load(model, PATH, device="cpu"):
    """
    Loads the model's parameters from the path mentioned
    :param PATH: Should contain pickle file
    """
    model.load_state_dict(torch.load(PATH, map_location=device))


def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"


def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1


def adjust_anomaly_scores(scores, dataset, is_train, lookback):
    """
    Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
    :param scores: anomaly_scores
    :param dataset: name of dataset
    :param is_train: if scores is from train set
    :param lookback: lookback (window size) used in model
    """

    # Remove errors for time steps when transition to new channel (as this will be impossible for model to predict)
    if dataset.upper() not in ['SMAP', 'MSL']:
        return scores

    adjusted_scores = scores.copy()
    if is_train:
        md = pd.read_csv(f'../datasets/data/{dataset.lower()}_train_md.csv')
    else:
        md = pd.read_csv('../datasets/data/labeled_anomalies.csv')
        md = md[md['spacecraft'] == dataset.upper()]

    md = md[md['chan_id'] != 'P-2']

    # Sort values by channel
    md = md.sort_values(by=['chan_id'])

    # Getting the cumulative start index for each channel
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    sep_cuma = sep_cuma[:-1]
    buffer = np.arange(1, 20)
    i_remov = np.sort(np.concatenate((sep_cuma, np.array([i+buffer for i in sep_cuma]).flatten(),
                                      np.array([i-buffer for i in sep_cuma]).flatten())))
    i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
    i_remov = np.sort(np.unique(i_remov))
    if len(i_remov) != 0:
        adjusted_scores[i_remov] = 0

    # Normalize each concatenated part individually
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    s = [0] + sep_cuma.tolist()
    for c_start, c_end in [(s[i], s[i+1]) for i in range(len(s)-1)]:
        e_s = adjusted_scores[c_start: c_end+1]

        e_s = (e_s - np.min(e_s))/(np.max(e_s) - np.min(e_s))
        adjusted_scores[c_start: c_end+1] = e_s

    return adjusted_scores


from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) *
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                self.end_learning_rate for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) *
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) +
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr

class Spectralresidual():

    def __init__(self, wl=3):
        self.wl = wl

    def series_filter(self, values, kernel_size=3):
        """
        Filter a time series. Practically, calculated mean value inside kernel size.
        As math formula, see https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html.
        :param values:
        :param kernel_size:
        :return: The list of filtered average
        """
        filter_values = np.cumsum(values, dtype=float)

        filter_values[kernel_size:] = filter_values[kernel_size:] - filter_values[:-kernel_size]
        filter_values[kernel_size:] = filter_values[kernel_size:] / kernel_size

        for i in range(1, kernel_size):
            filter_values[i] /= i + 1

        return filter_values

    def get_sr(self, data):

        freq = np.fft.fft(data)

        logamp = np.log(np.sqrt(freq.real ** 2 + freq.imag ** 2) + 0.00000001)

        phase = np.angle(freq)

        sr = logamp - self.series_filter(logamp, kernel_size=self.wl)

        res = np.fft.ifft(np.exp(sr+1j*phase))

        return np.sqrt(res.real**2 + res.imag**2)

