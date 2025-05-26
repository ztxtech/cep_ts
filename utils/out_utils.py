import concurrent
import json
import os
import pathlib
import shutil

import dill


def get_settings(r_path):
    r_path = pathlib.Path(r_path)
    folders = [folder for folder in r_path.iterdir() if folder.is_dir()]
    return folders


def check_folder(folder, file_name='args.json', silent=True):
    channel_png_path = folder / file_name
    if not silent:
        print(channel_png_path, ', exist: ', channel_png_path.exists())
    return channel_png_path.exists()


def delete_folder(path, folder):
    path = pathlib.Path(path)
    folder_name = folder.name
    folder_to_delete = path / folder_name
    if folder_to_delete.exists() and folder_to_delete.is_dir():
        shutil.rmtree(folder_to_delete)
        print(f'delete {folder_to_delete}')


def filter_setting_worker(folder, r_path, s_path):
    if not check_folder(folder):
        delete_folder(r_path, folder)
        delete_folder(s_path, folder)


def filter_setting(r_path, s_path):
    if not os.path.exists(r_path):
        return
    folders = get_settings(r_path)
    worker = 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        executor.map(filter_setting_worker, folders, [r_path] * len(folders), [s_path] * len(folders))


def get_args_worker(folder):
    args_json_path = folder / 'args.json'
    if args_json_path.exists():
        with open(args_json_path, 'r') as f:
            args = json.load(f)
        return args
    else:
        print(args_json_path, 'not exist')
        return None


def get_args_list(r_path):
    if not os.path.exists(r_path):
        return []
    folders = get_settings(r_path)
    worker = 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        args_list = list(executor.map(get_args_worker, folders))
    return [args for args in args_list if args is not None]


def dict_equals(d_a, d_b):
    keys = set(d_a).intersection(d_b)
    for k in keys:
        if k == 'use_gpu':
            continue
        if d_a[k] != d_b[k]:
            return False
    return True


def filter_unrun_args_list(cur_args_list, hist_args_list):
    res = []
    for cur_args in cur_args_list:
        args = vars(cur_args)
        for hist_args in hist_args_list:
            if dict_equals(args, hist_args):
                break
        else:
            res.append(cur_args)
    return res


def delete_wrong_checkpoint(r_path, s_path):
    if not os.path.exists(r_path):
        if os.path.exists(s_path):
            shutil.rmtree(s_path)
        return
    if not os.path.exists(s_path):
        s_settings = []
    else:
        s_settings = get_settings(s_path)
        s_settings = [s_setting.name for s_setting in s_settings]

    r_settings = get_settings(r_path)
    r_settings = [r_setting.name for r_setting in r_settings]

    wrong_settings = set(s_settings).difference(set(r_settings))
    for wrong_setting in wrong_settings:
        delete_folder(s_path, pathlib.Path(s_path) / wrong_setting)

    for setting in r_settings:
        args_path = pathlib.Path(r_path) / setting / 'args.json'
        if not args_path.exists():
            delete_folder(r_path, pathlib.Path(r_path) / setting)


def reload_args(cur_args_list, r_path=None, s_path=None):
    if not r_path:
        r_path = './out/result'
    if not s_path:
        s_path = './out/checkpoints'
    if not os.path.exists(r_path) and not os.path.exists(s_path):
        print('load path not exist')
        return cur_args_list
    else:
        delete_wrong_checkpoint(r_path, s_path)
        filter_setting(r_path, s_path)
        args_list = get_args_list(r_path)
        res = filter_unrun_args_list(cur_args_list, args_list)
        print(f'load {len(cur_args_list) - len(res)} args from {r_path}, remain {len(res)} args')
        return res


def save_obj(x, path):
    with open(path, 'wb') as f:
        dill.dump(x, f)


def load_obj(path):
    with open(path, 'rb') as f:
        return dill.load(f)


def args2name(args):
    def fl(x):
        return x[0] + x[-1]

    meaningful_attributes = ['exp', 'data', 'model', 'forecaster', 'pred_len', 'fast_ratio', 'die_ratio', 'opt_mode',
                             'pre_shift']
    remove_patterns = ['exp_', 'models.', '.net', '.Model']

    name_parts = []
    for attr in meaningful_attributes:
        name_parts.append(fl(attr))
        name_parts.append(str(getattr(args, attr)))

    full_name = '_'.join(name_parts)

    for pattern in remove_patterns:
        full_name = full_name.replace(pattern, '')

    return full_name
