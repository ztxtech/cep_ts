from utils.main_utils import parse_args, run, args_data_adjust

if __name__ == '__main__':
    args = parse_args()
    args = args_data_adjust(args)
    run(args)
