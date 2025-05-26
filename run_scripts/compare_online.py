import os
import sys

from joblib import Parallel, delayed

from utils.main_utils import parse_args, run, get_arg_list

if __name__ == '__main__':
    sys.path.append("./")
    path = './out'

    os.makedirs(path, exist_ok=True)

    args = parse_args()
    args.features = 'S'
    args.debug = True
    args.num_workers = 1

    r_path = os.path.join(path, 'result')
    s_path = os.path.join(path, args.checkpoints)

    combine_space = {
        'exp': ['exp_fsnet', 'exp_onenet_fsnet', 'exp_er', 'exp_derpp'],
        'data': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'WTH', 'ECL', 'Exchange', 'Traffic'],
        'pred_len': [1, 30, 60]
    }

    args_list = get_arg_list(args=args, combine_space=combine_space, r_path=r_path,
                             s_path=s_path)
    print(f'Total {len(args_list)} args')

    parallel = True
    workers = 2

    if parallel:
        Parallel(n_jobs=workers, prefer='processes')(
            delayed(run)(args) for args in args_list)
    else:
        for args in args_list:
            run(args)
