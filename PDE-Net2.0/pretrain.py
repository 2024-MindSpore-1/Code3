import argparse
from src.utils import init_env, get_config
from train import single_train, mkdir
from src.dataset import DataGenerator


def pretrain(config):
    data_generator = DataGenerator(config=config)
    single_train(config=config, step_num=1, data_generator=data_generator)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="burgers PDENet2.0 pretrain")
    parser.add_argument("--mode", type=str, default="PYNATIVE", choices=["PYNATIVE"], help="Running in PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["CPU", "GPU", "Ascend"],
                        help="The target device to run, support 'CPU', 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()
    init_env(env_args=args)

    my_config = get_config(args.config_file_path)
    my_config['device_target'] = args.device_target
    my_config['context_mode'] = args.mode
    my_config['device_id'] = args.device_id
    mkdir(config=my_config)
    pretrain(config=my_config)
