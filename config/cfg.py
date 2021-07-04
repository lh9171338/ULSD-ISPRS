import os
from yacs.config import CfgNode
import argparse


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, help='gpu id')
    parser.add_argument('-o', '--order', type=int, choices=[1, 2, 3, 4, 5, 6], help='order of Bezier curve')
    parser.add_argument('-v', '--version', type=str, help='version')
    parser.add_argument('-d', '--dataset_name', type=str, help='dataset name')
    parser.add_argument('-t', '--type', type=str, help='type')
    parser.add_argument('-l', '--last_epoch', type=int, help='last epoch')
    parser.add_argument('-s', '--save_image', action='store_true', help='save image')
    parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate')
    parser.add_argument('-c', '--save_checkpoint', action='store_true', help='save temporary files')
    parser.add_argument('-m', '--model_name', type=str, help='model name')
    parser.add_argument('--config_path', type=str, default='config', help='config path')
    parser.add_argument('--config_file', type=str, default='default.yaml', help='config filename')

    opts = parser.parse_args()
    opts_dict = vars(opts)
    opts_list = []
    for key, value in zip(opts_dict.keys(), opts_dict.values()):
        if value is not None:
            opts_list.append(key)
            opts_list.append(value)

    yaml_file = os.path.join(opts.config_path, opts.config_file)
    cfg = CfgNode.load_cfg(open(yaml_file))
    cfg.merge_from_list(opts_list)

    for key, value in zip(cfg.dataset_dict.keys(), cfg.dataset_dict.values()):
        if cfg.dataset_name in value:
            cfg.type = key
            break
    if cfg.model_name == '':
        cfg.version = f'{cfg.type}-{cfg.version}-{cfg.order}'
        cfg.model_name = f'{cfg.version}.pkl'
    else:
        cfg.version = '.'.join(cfg.model_name.split('.')[:-1])
    cfg.log_path = f'{cfg.log_path}/{cfg.version}'
    cfg.raw_dataset_path = os.path.join(cfg.raw_dataset_path, cfg.dataset_name + '_raw')
    cfg.train_dataset_path = os.path.join(cfg.train_dataset_path, cfg.dataset_name + f'_{cfg.order}')
    cfg.test_dataset_path = os.path.join(cfg.test_dataset_path, cfg.dataset_name)
    cfg.groundtruth_path = os.path.join(cfg.groundtruth_path, cfg.dataset_name)
    cfg.output_path = os.path.join(cfg.output_path, cfg.dataset_name + f'_{cfg.version}')
    cfg.figure_path = os.path.join(cfg.figure_path, cfg.dataset_name)

    cfg.image_size = tuple(cfg.image_size)
    cfg.heatmap_size = tuple(cfg.heatmap_size)
    cfg.freeze()

    # Print cfg
    for k, v in cfg.items():
        print(f'{k}: {v}')

    return cfg

