import argparse
import json

from utils.main_utils import run

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    path_args = parser.parse_args()
    config_path = path_args.config_path
    del path_args

    with open(config_path, 'r', encoding='utf-8') as f:
        args = json.load(f)

    args = argparse.Namespace(**args)

    run(args)
