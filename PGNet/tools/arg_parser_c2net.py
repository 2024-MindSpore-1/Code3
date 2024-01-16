import argparse

import yaml

from .modelarts_adapter.c2net import modelarts_setup
from .arg_parser import _parse_options, _merge_options


def create_parser():
    parser = argparse.ArgumentParser(description="Training Config", add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="",
        required=True,
        help="YAML config file specifying default arguments (default=" ")",
    )
    parser.add_argument(
        "-o",
        "--opt",
        nargs="+",
        help="Options to change yaml configuration values, "
        "e.g. `-o system.distribute=False eval.dataset.dataset_root=/my_path/to/ocr_data`",
    )
    # modelarts
    group = parser.add_argument_group("modelarts")
    group.add_argument("--enable_modelarts", type=bool, default=False, help="Run on modelarts platform (default=False)")
    group.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        help="Target device, only used on modelarts platform (default=Ascend)",
    )
    # The url are provided by modelart, usually they are S3 paths
    group.add_argument("--multi_data_url", type=str, default="", help="path to multi dataset")
    group.add_argument("--data_url", type=str, default="", help="path to dataset")
    group.add_argument("--ckpt_url", type=str, default="", help="pre_train_model path in obs")
    group.add_argument("--pretrain_url", type=str, default="", help="pre_train_model paths in obs")
    group.add_argument("--train_url", type=str, default="", help="model folder to save/load")

    # 智算平台的额外参数
    group.add_argument("--model_url", type=str, default="", help="一个奇怪的参数")
    group.add_argument("--grampus_code_file_name", type=str, default="", help="一个奇怪的参数")

    # args = parser.parse_args()

    return parser


def parse_args_and_config():
    """
    Return:
        args: command line argments
        cfg: train/eval config dict
    """
    parser = create_parser()
    args = parser.parse_args()  # CLI args

    modelarts_setup(args)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        # TODO: check validity of config arguments to avoid invalid config caused by typo.
        # _check_cfgs_in_parser(cfg, parser)
        # parser.set_defaults(**cfg)
        # parser.set_defaults(config=args_config.config)

    if args.opt:
        options = _parse_options(args.opt)
        cfg = _merge_options(cfg, options)

    return args, cfg
