# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志记录工具
"""

import logging
import os
import sys


def setup_logger(name, save_file, distributed_rank=0, level=logging.INFO):
    """设置日志记录器

    Args:
        name: logger名称
        save_file: 日志保存路径
        distributed_rank: 分布式训练的rank (仅rank 0打印)
        level: 日志级别

    Returns:
        logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.hasHandlers():
        return logger

    # 只有主进程记录日志
    if distributed_rank > 0:
        return logger

    # 创建格式化器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 控制台 Handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 2. 文件 Handler
    if save_file:
        # 确保目录存在
        log_dir = os.path.dirname(save_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        fh = logging.FileHandler(save_file, mode='w')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
