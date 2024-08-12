import json
from datetime import datetime
import torch.nn as nn
import random
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import os

from args import get_parser
from utils import *
from model.cnn_gru import CNN_GRU
from prediction import Predictor
from training import Trainer
# from model.m_cnn import Res2NetBottleneck
from model.m_cnn2 import Res2NetBottleneck
from model.se_cnn import Res2NetBottleneck
from model.LGAT import LGAT
from model.mtad_gat import MTAD_GAT
from model.TSformer import TSformer
from model.transformer import TransformerEncoder
from model.transformerpackage import TransformerEncoderPackage
from model.LSTNet import LSTNet
from model.convformer import ForcastConvTransformer
from model.time_tcn import TCN
from model.PatchTST import PatchTST

torch.manual_seed(0)
if __name__ == "__main__":

    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

    random.seed(args.seed)               # 设置随机数生成器的种子，是每次随机数相同
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)         # 为CPU设置种子用于生成随机数,以使得结果是确定的
    torch.cuda.manual_seed(args.seed)    # 为GPU设置种子用于生成随机数,以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False      # 网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速，适用场景是网络结构以及数据结构固定
    torch.backends.cudnn.deterministic = False   # 为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    os.environ['PYTHONHASHSEED'] = str(args.seed) # 为0则禁止hash随机化，使得实验可复现。

    # if args.seed is not None:
    #     random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     cudnn.deterministic = True
    #     warnings.warn('You have chosen to seed training. '
    #                   'This will turn on the CUDNN deterministic setting, '
    #                   'which can slow down your training considerably! '
    #                   'You may see unexpected behavior when restarting '
    #                   'from checkpoints.')
    #
    # if args.use_cuda is not None:
    #     warnings.warn('You have chosen a specific GPU. This will completely '
    #                   'disable data parallelism.')

    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    #index = args.index
    depth = args.depth
    header = args.header
    kernel_size = args.kernel_size
    dropout = args.dropout
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    args_summary = str(args.__dict__)
    print(args_summary)

    if dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        (x_train, _), (x_test, y_test), series_cols = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "SERVERMACHINEDATASET":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "SWAT":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "ICE":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "WADI":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "PSM":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "KDD":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "WIND":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "WT03":
         output_path = f'output/{dataset}'
         (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "WT13":
         output_path = f'output/{dataset}'
         (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "WT23":
         output_path = f'output/{dataset}'
         (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "WINDNEW":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "OMI":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(f"omi-{index}", normalize=normalize)
        #(x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sr = Spectralresidual()

    # x_train_sr = np.zeros([x_train.shape[0], len(series_cols)])
    # x_test_sr = np.zeros([x_test.shape[0], len(series_cols)])

    # for i in series_cols:
    #     x_train_sr = sr.get_sr(x_train[i].values)
    #     x_train[i] = x_train[i] + x_train_sr
    #     x_test_sr = sr.get_sr(x_test[i].values)
    #     x_test[i] = x_test[i] + x_test_sr

    # x_train + x_train__sr
    # x_train = x_train + x_train_sr
    # x_test = x_test + x_test_sr

    x_train = x_train.values
    x_test = x_test.values
    y_test = y_test

    ### x_train_sr
    # x_train = x_train_sr
    # x_test = x_test_sr

    ### x_train_original

    # 转换为tensor
    if args.val_split is not None:
        dataset_size = len(x_train)
        split = int(np.floor(args.val_split * dataset_size))
        train_data, val_data = x_train[:split], x_train[split:]

    x_train = torch.from_numpy(train_data).float()
    x_val = torch.from_numpy(val_data).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1] # 特征

    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims, stride=args.stride)
    val_dataset = SlidingWindowDataset(x_val, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, batch_size, shuffle_dataset, test_dataset=test_dataset
    )
    if args.model == "CNN_GRU":
        model = CNN_GRU(
            n_features,
            out_dim,
            kernel_size=args.kernel_size,
            gru_n_layers=args.gru_n_layers,
            gru_hid_dim=args.gru_hid_dim,
            forecast_n_layers=args.fc_n_layers,
            forecast_hid_dim=args.fc_hid_dim,
            dropout=args.dropout,
        )
    elif args.model == "PatchTST":
        model = PatchTST()
    elif args.model == "m_cnn":
        model = Res2NetBottleneck(inplanes=51, planes=51, downsample=None, stride=1, scales=3, groups=1, se=False,
                              norm_layer=None)
    elif args.model == "m_cnn2":
        model = Res2NetBottleneck(inplanes=51, planes=51 * 6, downsample=None, stride=1, scales=6, groups=6, se=False,
                                  norm_layer=None)
    elif args.model == "se_cnn":
        model = Res2NetBottleneck(
            inplanes=n_features,
            planes=128,
            out_dim=out_dim,
            window=window_size,
            gru_hid_dim=args.gru_hid_dim,
            gru_n_layers=args.gru_n_layers,
            fc_hid_dim=args.fc_hid_dim,
            fc_n_layers=args.fc_n_layers,
            recon_hid_dim=args.recon_hid_dim,
            recon_n_layers=args.recon_n_layers,
            downsample=True,
            dropout=args.dropout,
            stride=1,
            scales=4,
            groups=6,
            se=True,
            norm_layer=None,
        )
    elif args.model == "LGAT":
        model = LGAT(
            n_features,
            window_size,
            out_dim,
            kernel_size=args.kernel_size,
            feat_gat_embed_dim=args.feat_gat_embed_dim,
            use_gatv2=args.use_gatv2,
            gru_n_layers=args.gru_n_layers,
            gru_hid_dim=args.gru_hid_dim,
            forecast_n_layers=args.fc_n_layers,
            forecast_hid_dim=args.fc_hid_dim,
            dropout=args.dropout,
            alpha=args.alpha,
        )

    elif args.model == "MTGAT":
        model = MTAD_GAT(
            n_features,
            window_size,
            out_dim,
            kernel_size=args.kernel_size,
            use_gatv2=args.use_gatv2,
            feat_gat_embed_dim=args.feat_gat_embed_dim,
            time_gat_embed_dim=args.time_gat_embed_dim,
            gru_n_layers=args.gru_n_layers,
            gru_hid_dim=args.gru_hid_dim,
            forecast_n_layers=args.fc_n_layers,
            forecast_hid_dim=args.fc_hid_dim,
            dropout=args.dropout,
            alpha=args.alpha
        )

    elif args.model == "TSformer":
        model = TSformer(
            n_features,
            window_size,
            gru_hid_dim=args.gru_hid_dim,
            gru_n_layers=args.gru_n_layers,
            num_hiddens=512,
            norm_shape=[args.lookback, 512],
            norm_shape_gat=[args.lookback, n_features],
            forecast_hid_dim=args.fc_hid_dim,
            out_dim=out_dim,
            forecast_n_layers=args.fc_n_layers,
            ffn_num_input=512,
            ffn_num_gat=n_features,
            ffn_num_hiddens=1024,
            num_heads=8,
            num_layers=3,
            dropout=0.1,
            alpha=0.2)

    elif args.model == "transformer":
        model = TransformerEncoder(
            n_features,
            window_size,
            num_hiddens = 256,
            norm_shape = [args.lookback, 256],
            num_heads = 8,
            num_layers = 3,
            ffn_num_hiddens = 256,
            dropout = 0.3,
            use_bias = False,
        )
    elif args.model == "transformerpackage":
        model = TransformerEncoderPackage(
            n_features,
            d_model=512,
            num_layers=3,
            nhead=8
        )
    elif args.model == "LSTNet":
        model = LSTNet(n_features=n_features, window_size=window_size, kernel_size=kernel_size, gru_hid_dim=args.gru_hid_dim,
                       )
    elif args.model == "convformer":
        model = ForcastConvTransformer(
            k=n_features, headers=header, depth=depth, seq_length=window_size, kernel_size=kernel_size, mask_next=False, mask_diag=False, dropout_proba=dropout, num_tokens=None
        )
    elif args.model == "time_tcn":
        model = TCN(n_features, args.hid_dim, [args.hid_dim] * 2, seq_length=window_size, kernel_size=7, dropout=0.1)
    else:
        pass
    # print(model)

    import torch.nn.functional as F
    def js_div(p_output, q_output, get_softmax=True):
        """
        Function that measures JS divergence between target and output logits:
        """
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        if get_softmax:
            p_output = F.softmax(p_output)
            q_output = F.softmax(q_output)
        log_mean_output = ((p_output + q_output) / 2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


    # optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    forecast_criterion = nn.MSELoss()
    # forecast_criterion = nn.L1Loss()
    # forecast_criterion = QuantileLoss([0.1,0.5,0.9])
    # # forecast_criterion = ms_loss
    # forecast_criterion = js_div
    # forecast_criterion = nn.SmoothL1Loss(size_average=None, reduce=None, reduction='mean', beta=1.0)

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss:.5f}")

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.90, 0.03),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
        "SWAT": (0.88, 0.12),
        "WADI": (0.99, 0.001),
        "WIND": (0.99, 0.001),
        "SERVERMACHINEDATASET": (0.95, 0.05),
        "ICE": (0.93, 0.07),
        "PSM": (0.75, 0.25),
        "WINDNEW": (0.80, 0.20),
        "KDD": (0.95, 0.05),
        "WT03": (0.95, 0.05),
        "WT13": (0.95, 0.05),
        "WT23": (0.95, 0.05),
        "OMI": (0.9925, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "WINDNEW": 1, "KDD": 1, "SERVERMACHINEDATASET": 1, "ICE": 0, "MSL": 0,
                      "SMD-1": 1, "PSM": 1, "SMD-2": 1, "WT03": 1, "WT13": 1, "WT23": 1,
                      "SMD-3": 1, "WADI": 1, "SWAT":1, "WIND":1, "OMI":1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
        "k": args.k,
        "adjust_score": args.adjust_score,
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )

    label = y_test[window_size:] if y_test is not None else None


    predictor.predict_anomalies(x_train, x_test, label)

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

