import argparse
import datetime
import importlib
import itertools
import json
import os
import random
import traceback
import uuid
from pathlib import Path

import numpy as np
import psutil
import torch

from utils.out_utils import reload_args, save_obj


def save_args_to_json(args, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def init_dl_program(
        args,
        use_cudnn=False,
        deterministic=False,
        benchmark=False,
        use_tf32=False,
        max_threads=None
):
    device_name = args.gpu
    seed = args.seed
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    if args.use_gpu:
        devices = []
        for t in reversed(device_name):
            t_device = torch.device(f'cuda:{t}')
            devices.append(t_device)
            if t_device.type == 'cuda':
                assert torch.cuda.is_available()
                torch.cuda.set_device(t_device)
                if seed is not None:
                    torch.cuda.manual_seed(seed)
        devices.reverse()
        torch.backends.cudnn.enabled = use_cudnn
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark

        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = use_tf32
            torch.backends.cuda.matmul.allow_tf32 = use_tf32

        return devices if len(devices) > 1 else devices[0]
    return None


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_args(args, n=3, k=0, v=0):
    if k == 0:
        k = max([len(k) for k, v in vars(args).items()]) + 4
    if v == 0:
        v = max([len(str(v)) for k, v in vars(args).items()]) + 4
    items = list(vars(args).items())
    items.sort(key=lambda x: x[0])
    for i in range(0, len(items), n):
        line = ""
        for j in range(n):
            if i + j < len(items):
                key, value = items[i + j]
                line += f"| \033[92m {key:<{k}} \033[94m{str(value):>{v}} \033[0m"
        line += "|"
        print(line)


def get_all_models():
    model_path = Path("./models")
    models = [f"models.{x.stem}.Model" for x in model_path.iterdir() if
              x.is_file() and x.name.endswith(".py") and x.stem not in ['CEP', '__init__']]
    models.sort()
    return models


def parse_args():
    parser = argparse.ArgumentParser(description='Continuous Evolution Pool')

    # basic settings
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--exp', type=str, default='exp_pure')
    parser.add_argument('--model', type=str, default='none', help='data')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--data', type=str, default='ETTh2', help='data')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--out_path', type=str, default='./out/', help='out path of experiment')
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    parser.add_argument('--target', type=int, default=None, help='target feature in the data file')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='result', help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=30, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--prefetch_factor', type=int, default=2000, help='prefetch_factor')
    parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--comment', type=str, default='', help='your comment')

    # online settings
    parser.add_argument('--online_learning', type=str, default='full')
    parser.add_argument('--test_bsz', type=int, default=1)
    parser.add_argument('--n_inner', type=int, default=1)
    parser.add_argument('--delay_fb', action='store_true', default=True, help='use delayed feedback')

    # model settings
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    # parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--seg_len', type=int, default=30,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # CEP
    parser.add_argument('--forecaster', type=str, default='none', help='')
    parser.add_argument('--replace_ratio', type=float, default=0.2, help='')
    parser.add_argument('--mean_b', type=float, default=3.0, help='')
    parser.add_argument('--std_b', type=float, default=0.0, help='')
    parser.add_argument('--wait_time', type=int, default=15, help='')
    parser.add_argument('--die_ratio', type=float, default=1.5, help='')
    parser.add_argument('--fast_ratio', type=float, default=0.8, help='')
    parser.add_argument('--pre_shift', action='store_true', default=True, help='')
    parser.add_argument('--opt_mode', type=str, default='fade', help='')
    parser.add_argument('--opt_scale', type=float, default=0.5, help='')
    parser.add_argument('--opt_wait', type=float, default=20, help='')
    parser.add_argument('--scope', type=str, default='seq', help='')

    # FSNET/OneNet
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--learning_rate_w', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--learning_rate_bias', type=float, default=0.001, help='optimizer learning rate')

    args = parser.parse_args()
    args.debug = True

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.test_bsz = args.batch_size if args.test_bsz == -1 else args.test_bsz
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    return args


def args_data_adjust(args):
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'LULL', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'LUFL', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'LULL', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'MUFL', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'WTH': {'data': 'WTH.csv', 'T': 'DewPointFarenheit', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
        'ECL': {'data': 'ECL.csv', 'T': 'MT_104', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
        'Exchange': {'data': 'exchange_rate.csv', 'T': 'OT', 'M': [8, 8, 8], 'S': [1, 1, 1]},
        'Traffic': {'data': 'traffic.csv', 'T': '16', 'M': [862, 862, 862], 'S': [1, 1, 1]},
    }

    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        if args.target is None:
            args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    return args


def run(args):
    try:
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)

        print('Args in experiment:')
        print_args(args, 3)
        os.makedirs(args.out_path, exist_ok=True)

        Exp = getattr(importlib.import_module('exp.{}'.format(args.exp)), 'Exp_TS2VecSupervised')

        metrics, preds, trues, mae, mse = [], [], [], [], []

        for ii in range(args.itr):
            print('\n ====== Run {} ====='.format(ii))
            uid = uuid.uuid4().hex[:4]
            setting = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + uid
            args.setting = setting

            init_dl_program(args)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            m, mae_, mse_, p, t = exp.test(setting)

            metrics.append(m)
            preds.append(p)
            trues.append(t)
            mae.append(mae_)
            mse.append(mse_)
            torch.cuda.empty_cache()

        folder_path = Path(args.out_path) / 'result' / setting

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path / 'metrics.npy', np.array(metrics))
        np.save(folder_path / 'preds.npy', np.array(preds))
        np.save(folder_path / 'trues.npy', np.array(trues))
        np.save(folder_path / 'mae.npy', np.array(mae))
        np.save(folder_path / 'mse.npy', np.array(mse))

        if args.debug and args.model == 'models.CEP.net':
            save_obj(exp.model.traj, folder_path / 'traj.pkl')
            save_obj(exp.model.sample_token, folder_path / 'sample_token.pkl')
            save_obj(exp.model.f_hist, folder_path / 'f_hist.pkl')

        args.error = metrics
        save_args_to_json(args, folder_path / 'args.json')
    except:
        print_args(args, 2)
        print('>>>>>>>error: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        traceback.print_exc()
        exit()
    return metrics[0][1], metrics[0][0]


def run_train_only(args):
    print('Args in experiment:')
    print_args(args, 3)
    os.makedirs(args.out_path, exist_ok=True)

    Exp = getattr(importlib.import_module('exp.{}'.format(args.exp)), 'Exp_TS2VecSupervised')

    for ii in range(args.itr):
        print('\n ====== Run {} ====='.format(ii))
        uid = uuid.uuid4().hex[:4]
        setting = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + uid
        args.setting = setting

        init_dl_program(args)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

    folder_path = Path(args.out_path) / 'result' / setting
    folder_path.mkdir(exist_ok=True)
    del args.device
    save_args_to_json(args, folder_path / 'args.json')


def cartesian_product_dict(test_space, key_order=None):
    if key_order is None:
        key_order = list(test_space.keys())

    values_combinations = itertools.product(*(test_space[key] for key in key_order))

    result = []
    for combination in values_combinations:
        result.append(dict(zip(key_order, combination)))

    return result


def filtered_by_rules(args):
    rules = [
        (args.model == 'models.SegRNN.Model' or args.forecaster == 'models.SegRNN.Model') and (args.pred_len == 1),
    ]
    return any(rule for rule in rules)


def filter_args_list(arg_list):
    res = []
    for args in arg_list:
        if not filtered_by_rules(args):
            res.append(args)
    return res


def modify_list(arg_list):
    for args in arg_list:
        if args.model == 'models.TimesNet.Model' or args.forecaster == 'models.TimesNet.Model':
            args.e_layers = 2
            args.d_layers = 1
            args.factor = 3
            args.d_model = 16
            args.d_ff = 32
            args.top_k = 5

    return arg_list


def get_arg_list(args, combine_space=None, alter_space=None, r_path=None, s_path=None, sf=False):
    arg_list = [args]
    if combine_space:
        new_arg_list = []
        for cur_args in arg_list:
            for args_dict in cartesian_product_dict(combine_space):
                new_args = argparse.Namespace(**vars(cur_args))
                for k, v in args_dict.items():
                    setattr(new_args, k, v)
                new_arg_list.append(new_args)
        arg_list = new_arg_list

    if alter_space:
        new_arg_list = []
        for cur_args in arg_list:
            for key, values in alter_space.items():
                for value in values:
                    new_args = argparse.Namespace(**vars(cur_args))
                    setattr(new_args, key, value)
                    new_arg_list.append(new_args)
        arg_list = new_arg_list

    for cur_args in arg_list:
        args_data_adjust(cur_args)

    arg_list = filter_args_list(arg_list)
    arg_list = modify_list(arg_list)
    arg_list = reload_args(arg_list, r_path, s_path)
    if sf:
        random.shuffle(arg_list)

    return arg_list


def set_args_str(args_str, args):
    for line in args_str.split('\n'):
        line = line.strip()
        if line:
            try:
                key, value = line.split(maxsplit=1)
                value = float(value)
                setattr(args, key, value)
            except:
                print(traceback.format_exc())
