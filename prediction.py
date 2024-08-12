import json

from eval_methods import *
from utils import *
from tqdm import tqdm
from scipy.stats import rankdata, iqr, trim_mean
import numpy as np
#from diagnosis import hit_att, ndcg_score, ndcg

from eval_methods import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


class Predictor:
    """MTAD-GAT predictor class.

    :param model: MTAD-GAT model (pre-trained) used to forecast and reconstruct
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param pred_args: params for thresholding and predicting anomalies

    """

    def __init__(self, model, window_size, n_features, pred_args, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args
        self.k = pred_args["k"]
        self.adjust_score = pred_args["adjust_score"]
        self.summary_file_name = summary_file_name

    def get_f1_scores(self, total_err_scores, topk=0):
        total_features = total_err_scores.shape[1]
        topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=1)[:, -topk:]

        total_topk_err_scores = []

        for i, indexs in enumerate(topk_indices):
            sum_score = [score for k, score in
                     enumerate(sorted([total_err_scores[i, index] for j, index in enumerate(indexs)]))]
            sum_score = sum(sum_score) / len(sum_score)

            total_topk_err_scores.append(sum_score)

        return total_topk_err_scores

    def get_score(self, values):
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :return np array of anomaly scores + dataframe with prediction for each channel and global anomalies
        """

        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        preds = []
        recons = []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)

                # y_hat，在MSL数据集上shape为[batch_size，1]
                y_hat = self.model(x).to(device) # 预测值
                preds.append(y_hat.detach().cpu().numpy()) # 拼接预测值.

        preds = np.concatenate(preds, axis=0) # MSL数据集shape(train set shape-win, 1)
        # actual = values.detach().cpu().numpy()[self.window_size:] # value shape:(tran set shape -win, features)
        actual = values.detach().cpu().numpy()[self.window_size:]  # value shape:(train set shape -win, features)

        if self.target_dims is not None:
            actual = actual[:, self.target_dims] # msl数据集为shape（train set shape-win, 1）

        anomaly_scores = np.zeros_like(actual)
        df = pd.DataFrame()
        for i in range(preds.shape[1]):
            df[f"Forecast_{i}"] = preds[:, i]
            df[f"True_{i}"] = actual[:, i]
            a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2)

            if self.scale_scores:
                # np_arr = np.abs(np.subtract(preds[:, i] - actual[:, i]))  # np.subtract:减法
                #
                # err_median = np.median(np_arr)  # 计算中位数
                # err_iqr = iqr(np_arr)  # 计算沿指定轴的数据的四分位数范围。
                #
                # a_score = (np_arr-err_median)/(err_iqr+0.00001)
                #
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                a_score = (a_score - median) / (0.01 + iqr)

            anomaly_scores[:, i] = a_score
            df[f"A_Score_{i}"] = a_score

        anomaly_scores_max = np.max(anomaly_scores, 1)
        anomaly_scores_mean = np.mean(anomaly_scores, 1)
        anomaly_scores_topk = self.get_f1_scores(anomaly_scores, topk=3)
        df['A_Score_Global_max'] = anomaly_scores_max
        df['A_Score_Global_mean'] = anomaly_scores_mean
        df['A_Score_Global_topk'] = anomaly_scores_topk

        return df, anomaly_scores

    def get_score_final(self, test_result, val_result):

        # feature_num = len((test_result[0][0].tolist()))
        feature_num = test_result.shape[1]
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)  # best,label
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1) # val_max

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':#阈值范围【0,1，0.01】【0,0。01，0.02,。。。，1】，F1
            info = top1_best_info
        elif self.env_config['report'] == 'val': # 预测跟标签差值， mse，rmse。。。差值=score，val_max
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')

    def predict_anomalies(self, train, test, true_anomalies, load_scores=False, save_output=True,
                          scale_scores=False):
        """ Predicts anomalies

        :param train: 2D array of train multivariate time series data
        :param test: 2D array of test multivariate time series data
        :param true_anomalies: true anomalies of test set, None if not available
        :param save_scores: Whether to save anomaly scores of train and test
        :param load_scores: Whether to load anomaly scores instead of calculating them
        :param save_output: Whether to save output dataframe
        :param scale_scores: Whether to feature-wise scale anomaly scores
        """

        if load_scores:
            print("Loading anomaly scores")

            train_pred_df = pd.read_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")

            train_anomaly_scores_max = train_pred_df['A_Score_Global_max'].values
            test_anomaly_scores_max = test_pred_df['A_Score_Global_max'].values

        else:
            train_pred_df, train_anomaly = self.get_score(train)
            test_pred_df, test_anomaly = self.get_score(test)

            # self.get_score_final(test_anomaly, train_anomaly)

            train_anomaly_scores_max = train_pred_df['A_Score_Global_max'].values
            test_anomaly_scores_max = test_pred_df['A_Score_Global_max'].values

            train_anomaly_scores_mean = train_pred_df['A_Score_Global_mean'].values
            test_anomaly_scores_mean = test_pred_df['A_Score_Global_mean'].values

            train_anomaly_scores_topk = train_pred_df['A_Score_Global_topk'].values
            test_anomaly_scores_topk = test_pred_df['A_Score_Global_topk'].values

            train_anomaly_scores_max = adjust_anomaly_scores(train_anomaly_scores_max, self.dataset, True, self.window_size)
            test_anomaly_scores_max = adjust_anomaly_scores(test_anomaly_scores_max, self.dataset, False, self.window_size)
            train_anomaly_scores_mean = adjust_anomaly_scores(train_anomaly_scores_mean, self.dataset, True, self.window_size)
            test_anomaly_scores_mean = adjust_anomaly_scores(test_anomaly_scores_mean, self.dataset, False, self.window_size)
            train_anomaly_scores_topk = adjust_anomaly_scores(train_anomaly_scores_topk, self.dataset, True, self.window_size)
            test_anomaly_scores_topk = adjust_anomaly_scores(test_anomaly_scores_topk, self.dataset, False, self.window_size)

            # Update df
            # train_pred_df['A_Score_Global'] = train_anomaly_scores
            # test_pred_df['A_Score_Global'] = test_anomaly_scores

        if self.use_mov_av:
            # smoothing_window = int(self.batch_size * self.window_size * 0.001)
            smoothing_window = 3
            train_anomaly_scores_max_s = pd.DataFrame(train_anomaly_scores_max).ewm(span=smoothing_window).mean().values.flatten()
            test_anomaly_scores_max_s = pd.DataFrame(test_anomaly_scores_max).ewm(span=smoothing_window).mean().values.flatten()

            train_anomaly_scores_mean_s = pd.DataFrame(train_anomaly_scores_mean).ewm(span=smoothing_window).mean().values.flatten()
            test_anomaly_scores_mean_s = pd.DataFrame(test_anomaly_scores_mean).ewm(span=smoothing_window).mean().values.flatten()

            train_anomaly_scores_topk_s = pd.DataFrame(train_anomaly_scores_topk).ewm(span=smoothing_window).mean().values.flatten()
            test_anomaly_scores_topk_s = pd.DataFrame(test_anomaly_scores_topk).ewm(span=smoothing_window).mean().values.flatten()

            train_pred_df['A_Score_Global_max'] = train_anomaly_scores_max_s
            test_pred_df['A_Score_Global_max'] = test_anomaly_scores_max_s

            train_pred_df['A_Score_Global_mean'] = train_anomaly_scores_mean_s
            test_pred_df['A_Score_Global_mean'] = test_anomaly_scores_mean_s

            train_pred_df['A_Score_Global_topk'] = train_anomaly_scores_topk_s
            test_pred_df['A_Score_Global_topk'] = test_anomaly_scores_topk_s

        # Find threshold and predict anomalies at feature-level (for plotting and diagnosis purposes)
        out_dim = self.n_features if self.target_dims is None else len(self.target_dims)
        all_preds = np.zeros((len(test_pred_df), out_dim))
        for i in range(out_dim):
            train_feature_anom_scores = train_pred_df[f"A_Score_{i}"].values
            test_feature_anom_scores = test_pred_df[f"A_Score_{i}"].values
            epsilon = find_epsilon(train_feature_anom_scores, reg_level=1)

            train_feature_anom_preds = (train_feature_anom_scores >= epsilon).astype(int)
            test_feature_anom_preds = (test_feature_anom_scores >= epsilon).astype(int)

            train_pred_df[f"A_Pred_{i}"] = train_feature_anom_preds
            test_pred_df[f"A_Pred_{i}"] = test_feature_anom_preds

            train_pred_df[f"Thresh_{i}"] = epsilon
            test_pred_df[f"Thresh_{i}"] = epsilon

            all_preds[:, i] = test_feature_anom_preds

        # Global anomalies (entity-level) are predicted using aggregation of anomaly scores across all features
        # These predictions are used to evaluate performance, as true anomalies are labeled at entity-level
        # Evaluate using different threshold methods: brute-force, epsilon and peaks-over-treshold
        e_eval = epsilon_eval(train_anomaly_scores_max_s, test_anomaly_scores_max_s, true_anomalies, k=0, reg_level=self.reg_level)
        p_eval = pot_eval(train_anomaly_scores_max_s, test_anomaly_scores_max_s, true_anomalies, k=0,
                          q=self.q, level=self.level, dynamic=self.dynamic_pot)
        if true_anomalies is not None:
            bf_eval = bf_search(test_anomaly_scores_max_s, true_anomalies, k=100, start=0, end=80, step_num=300, verbose=False)
            bf_eval2 = bf_search(test_anomaly_scores_mean_s, true_anomalies, k=100, start=0, end=80, step_num=300, verbose=False)
            bf_eval3 = bf_search(test_anomaly_scores_topk_s, true_anomalies, k=100, start=0, end=80, step_num=300, verbose=False)
            bf_eval_adjust = bf_search(test_anomaly_scores_max, true_anomalies, self.k, start=0, end=80, step_num=300, verbose=False)
            bf_eval_adjust_topk = bf_search(test_anomaly_scores_topk, true_anomalies, self.k, start=0, end=80, step_num=300, verbose=False)
            bf_eval_adjust_mean = bf_search(test_anomaly_scores_mean, true_anomalies, self.k, start=0, end=80, step_num=300, verbose=False)
        else:
            bf_eval = {}
            bf_eval2 = {}
            bf_eval3 = {}

        # bf_eval.update(hit_att(test_anomaly_scores_max, true_anomalies))
        # bf_eval.update(ndcg(test_anomaly_scores_max, true_anomalies))

        print(f"Results using epsilon method:\n {e_eval}")
        print(f"Results using peak-over-threshold method:\n {p_eval}")
        print(f"Results using best f1 score search (MAX):\n {bf_eval}")
        print(f"Results using best f1 score search (MEAN):\n {bf_eval2}")
        print(f"Results using best f1 score search (TOPK):\n {bf_eval3}")
        print(f"Results using best f1 score search (Max_adjust):\n {bf_eval_adjust}")
        print(f"Results using best f1 score search (Topk_adjust):\n {bf_eval_adjust_topk}")
        print(f"Results using best f1 score search (Mean_adjust):\n {bf_eval_adjust_mean}")

        for k, v in e_eval.items():
            if not type(e_eval[k]) == list:
                e_eval[k] = float(v)
        for k, v in p_eval.items():
            if not type(p_eval[k]) == list:
                p_eval[k] = float(v)
        for k, v in bf_eval.items():
            bf_eval[k] = float(v)
        for k, v in bf_eval2.items():
            bf_eval2[k] = float(v)
        for k, v in bf_eval3.items():
            bf_eval3[k] = float(v)
        for k, v in bf_eval_adjust.items():
            bf_eval_adjust[k] = float(v)
        for k, v in bf_eval_adjust_topk.items():
            bf_eval_adjust_topk[k] = float(v)
        for k, v in bf_eval_adjust_mean.items():
            bf_eval_adjust_mean[k] = float(v)

        # Save
        # summary = {"epsilon_result": e_eval, "bf_result": bf_eval}
        summary = {"epsilon_result": e_eval, "pot_result": p_eval, "bf_result_max": bf_eval, "bf_result_mean": bf_eval2, "bf_result_topk": bf_eval3, "bf_result_max_adjust": bf_eval_adjust,
                   "bf_result_adjust_topk": bf_eval_adjust_topk, "bf_result_max_adjust_mean": bf_eval_adjust_mean}
        with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
            json.dump(summary, f, indent=2)

        if save_output:
            global_bf = bf_eval["threshold"]
            test_pred_df["A_True_Global"] = true_anomalies
            train_pred_df["Thresh_Global"] = global_bf
            test_pred_df["Thresh_Global"] = global_bf
            train_pred_df[f"A_Pred_Global"] = (train_anomaly_scores_max_s >= global_bf).astype(int)
            test_preds_global = (test_anomaly_scores_max_s >= global_bf).astype(int)
            # Adjust predictions according to evaluation strategy
            # if true_anomalies is not None:
            # if self.adjust_score:
            #     test_preds_global = adjust_predicts(None, true_anomalies, global_bf, pred=test_preds_global)
            test_pred_df[f"A_Pred_Global"] = test_preds_global

            print(f"Saving output to {self.save_path}/<train/test>_output.pkl")
            train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")
            test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")

        print("-- Done.")



