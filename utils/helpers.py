# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
辅助工具函数：配置加载与保存
"""

import os
import yaml
import json
import argparse
import torch


def save_config(args, save_path):
    """保存配置参数到YAML文件

    Args:
        args: argparse.Namespace 对象
        save_path: 保存路径
    """
    # 将Namespace转换为字典
    config = vars(args).copy()

    # 处理无法序列化的对象
    for key, value in config.items():
        # 将 torch.device 对象转换为字符串
        if isinstance(value, torch.device):
            config[key] = str(value)
        # 将类对象或函数对象转换为字符串名称
        elif hasattr(value, '__name__'):
            config[key] = value.__name__
        # 处理Path对象
        elif hasattr(value, 'as_posix'):
            config[key] = str(value)

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Warning: Failed to save config to {save_path}. Error: {e}")
        # 降级尝试保存为 JSON
        json_path = save_path.replace('.yaml', '.json')
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=4)


def load_config(config_path):
    """从文件加载配置

    Args:
        config_path: 配置文件路径 (.yaml 或 .json)

    Returns:
        argparse.Namespace 对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ext = os.path.splitext(config_path)[1]

    with open(config_path, 'r') as f:
        if ext in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif ext == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")

    # 转换为 Namespace 对象
    args = argparse.Namespace(**config_dict)
    return args


class AverageMeter(object):
    """计算并存储平均值和当前值"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
