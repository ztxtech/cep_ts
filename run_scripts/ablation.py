import os
import sys

from joblib import Parallel, delayed

from utils.main_utils import parse_args, run, get_arg_list, get_all_models

if __name__ == '__main__':
    sys.path.append("./")
    path = './out'

    os.makedirs(path, exist_ok=True)

    args = parse_args()
    args.features = 'S'
    args.debug = True
    args.num_workers = 1
    args.exp = 'exp_cep'
    args.comment = 'ablation'

    r_path = os.path.join(path, 'result')
    s_path = os.path.join(path, args.checkpoints)

    combine_space = {
        'data': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'WTH', 'ECL', 'Exchange', 'Traffic'],
        'model': ['models.CEP.net'],
        'forecaster': get_all_models(),
        'pred_len': [30, 60],
        'fast_ratio': [0.0, 0.8, 1.0],
        'die_ratio': [0.0, 1.5],
        'opt_mode': ['flat', 'fade'],
        'pre_shift': [False, True]
    }

    args_list = get_arg_list(args=args, combine_space=combine_space, r_path=r_path,
                             s_path=s_path)

    print(f'Total {len(args_list)} args')

    parallel = True
    workers = 12

    if parallel:
        Parallel(n_jobs=workers, prefer='processes')(
            delayed(run)(args) for args in args_list)
    else:
        for args in args_list:
            run(args)
