#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedMM: Uncertainty-aware Multimodal Federated Learning
主入口文件
"""

import os
import random
import argparse
import numpy as np
import torch
from datetime import datetime

from trainers.fedmm_trainer import FedMMTrainer
# from trainers.creamfl_trainer import CreamFLTrainer
from trainers.fedmm_trainer_ssl import FedMMTrainerSSL
from trainers.fedssl_moe_trainer import FedSSLMoeTrainer  # NEW
# from trainers.car_mfl_trainer import *
# from trainers.fedmsplit_trainer import FedMSplitTrainerLite  # NEW (lite)

# ===== NEW: strict FedMSplit/FMTL trainer =====
# from trainers.fedmsplit_trainer_fmtl import FedMSplitTrainerFMTL

from utils.logger import setup_logger
from utils.helpers import load_config, save_config

# ===== CHANGED: import partial centralized entry =====
from trainers.centralized_trainer import run_centralized_training, run_partial_centralized_training

# from trainers.poe_moe_trainer import PoEMoETrainer

# MOSEI trainers
from trainers.mosei_fedmm_ssl_trainer import MOSEIFedMMSSLTrainer
from trainers.mosei_fedssl_moe_trainer import MOSEIFedSSLMoeTrainer
# from trainers.mosei_creamfl_trainer import MOSEICreamFLTrainer
# from trainers.mosei_car_mfl_trainer import MOSEICARMFLTrainer
# from trainers.mosei_fedavg_trainer import MOSEIFedAvgTrainer
# from trainers.mosei_fedmsplit_trainer import MOSEIFedMSplitLiteTrainer
from trainers.mosei_centralized_trainer import run_mosei_centralized_training


def args_parser():
    parser = argparse.ArgumentParser(description='FedMM: Multimodal Federated Learning')

    # ====================  ====================
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='directory')
    parser.add_argument('--exp_name', type=str, default=None, help='')

    # ==================== dataset ====================
    parser.add_argument('--dataset', type=str, default='flickr30k',
                        choices=['mscoco', 'flickr30k', 'mosei'],
                        help='dataset')

    parser.add_argument('--data_root', type=str, default=None,
                        help='data_root）')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='cache_dir')

    parser.add_argument('--train_ratio', type=float, default=0.7, help='train_ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='val_ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='test_ratio')

    parser.add_argument('--clip_cache_dir', type=str, default=None,
                        help='Directory for offline cached CLIP features')

    # ==================== FL ====================
    parser.add_argument('--num_rounds', type=int, default=100, help='num_rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='local_epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')

    parser.add_argument('--num_multimodal_clients', type=int, default=1, help='num_multimodal_clients (K)')
    parser.add_argument('--num_image_clients', type=int, default=1, help='num_image_clients (L)')
    parser.add_argument('--num_text_clients', type=int, default=1, help='num_text_clients (M)')

    parser.add_argument('--partition', type=str, default='dirichlet',
                        choices=['iid', 'noniid', 'dirichlet'],
                        help='partition')
    parser.add_argument('--alpha', type=float, default=0.2, help='Dirichlet')
    parser.add_argument('--mm_ratio', type=float, default=0.2, help='MFL_ratio')

    # ==================== MODEL ====================
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='clip_model')
    parser.add_argument('--embed_dim', type=int, default=512, help='embed_dim')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent_dim')
    parser.add_argument('--num_classes', type=int, default=80, help='num_classes')

    # ==================== TRAIN ====================
    parser.add_argument('--lr', type=float, default=1e-4, help='lr')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['none', 'step', 'cosine'],
                        help='lr_scheduler')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--min_delta', type=float, default=0.001)

    # ==================== LOSS ====================
    parser.add_argument('--beta', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.07)

    # ==================== BASELINES ====================
    parser.add_argument('--method', type=str, default='fedssl_moe',
                        choices=[
                            'centralized',
                            'partial_centralized',  # ===== NEW =====
                            'fedavg',
                            'fedmm',
                            'fedmm_ssl',
                            'fedssl_moe',
                            'creamfl',
                            'local',
                            'car_mfl',
                            'poe_moe',
                            'fedmsplit_lite',   # existing lite
                            'fedmsplit_fmtl',   # NEW: strict FedMSplit/FMTL
                        ],
                        help='method')

    # ==================== TASK ====================
    parser.add_argument('--task', type=str, default='retrieval',
                        choices=['classification', 'retrieval', 'regression'],
                        help='task')

    # ==================== CreamFL  ====================
    parser.add_argument('--public_size', type=int, default=4096)
    parser.add_argument('--public_batch_size', type=int, default=128)
    parser.add_argument('--creamfl_server_lr', type=float, default=None)
    parser.add_argument('--creamfl_distill_steps', type=int, default=1)
    parser.add_argument('--creamfl_kd_weight', type=float, default=1.0)
    parser.add_argument('--lcr_weight', type=float, default=0.1)
    parser.add_argument('--lcr_steps_per_epoch', type=int, default=1)

    # ==================== car_mfl  ====================
    parser.add_argument('--car_mfl_alpha', type=float, default=0.3)

    # ==================== PoE/MoE  ====================
    parser.add_argument('--moe_gate', type=str, default='public_loss',
                        choices=['public_loss', 'uncertainty'])
    parser.add_argument('--moe_tau', type=float, default=3.0)

    # ==================== FedMSplit-lite  ====================
    parser.add_argument('--fedmsplit_weight_by', type=str, default='num_samples',
                        choices=['num_samples', 'uniform', 'uncertainty_inv'],
                        help='FedMSplit-lite blockwise ')

    # ==================== NEW: FedMSplit-FMTL  ====================
    parser.add_argument('--clients_per_round', type=int, default=10,
                        help='FedMSplit-FMTL: clients_per_round')
    parser.add_argument('--use_bandit', type=int, default=1,
                        help='FedMSplit-FMTL: use_bandit')
    parser.add_argument('--bandit_gamma', type=float, default=0.9,
                        help='FedMSplit-FMTL:  bandit_gamma')
    parser.add_argument('--lambda_rel', type=float, default=0.0,
                        help='FedMSplit-FMTL: lambda_rel')
    parser.add_argument('--fedmsplit_topk', type=int, default=0,
                        help='FedMSplit-FMTL: fedmsplit_topk')
    parser.add_argument('--eval_num_clients', type=int, default=0,
                        help='FedMSplit-FMTL: eval_num_clients')

    # ====================  ====================
    parser.add_argument('--num_audio_clients', type=int, default=1)
    parser.add_argument('--num_vision_clients', type=int, default=1)

    parser.add_argument('--ssl_dropout_p', type=float, default=0.2)
    parser.add_argument('--ssl_noise_std', type=float, default=0.0)
    parser.add_argument('--ssl_weight', type=float, default=1.0)
    parser.add_argument('--ssl_reg_weight', type=float, default=1.0)

    # ==================== LOG ====================
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()
    # normalize use_bandit to bool
    args.use_bandit = bool(args.use_bandit)
    return args


def apply_dataset_default_paths(args):
    DEFAULTS = {
        "mscoco": {"data_root": "/home/userdata/data/MSCOCO", "clip_cache_dir": "/home/userdata/data/MSCOCO"},
        "flickr30k": {"data_root": "/home/weiyi/Desktop/FL POE/data", "clip_cache_dir": "/home/userdata/data/flicker30k_clip/"},
        "mnist": {"data_root": "./data", "clip_cache_dir": "./cache"},
        "mosei": {"data_root": "/home/userdata/data/cmumosei", "clip_cache_dir": "./cache"},
    }
    d = DEFAULTS.get(args.dataset, {})
    if args.data_root is None:
        args.data_root = d.get("data_root", "./data")
    if args.clip_cache_dir is None:
        args.clip_cache_dir = d.get("clip_cache_dir", "./cache")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_results_txt(args, result_dict):
    result_path = os.path.join(args.exp_dir, "results.txt")
    with open(result_path, "w") as f:
        f.write("Experiment: {}\n".format(args.exp_name))
        f.write("Dataset: {}\n".format(args.dataset))
        f.write("Method: {}\n\n".format(args.method))

        if "best_round" in result_dict:
            f.write("Best Round: {}\n".format(result_dict["best_round"]))
        f.write("Total Rounds: {}\n\n".format(args.num_rounds))

        if "best_val_metric" in result_dict:
            f.write("Validation Metrics:\n")
            f.write("  val_metric: {:.4f}\n\n".format(result_dict["best_val_metric"]))

        if "final_test_metrics" in result_dict:
            f.write("Test Metrics:\n")
            for k, v in result_dict["final_test_metrics"].items():
                f.write("  {}: {:.4f}\n".format(k, v))


def main():
    args = args_parser()
    args = apply_dataset_default_paths(args)

    # dataset-specific task override (keep your original behavior)
    if args.dataset == "mosei":
        args.task = "regression"

    set_seed(args.seed)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')
    args.device = device

    # ===== FIX: auto-generate exp_name if None =====
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"{args.dataset}_{args.method}_{timestamp}"

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.exp_dir = exp_dir

    logger = setup_logger(args.exp_name, os.path.join(exp_dir, 'train.log'))
    logger.info('Aug Setup:{}'.format(args))
    save_config(args, os.path.join(exp_dir, 'config.yaml'))

    # -------------------- dataset-first dispatch --------------------
    if args.dataset == "mosei":
        # MOSEI centralized
        if args.method == "centralized":
            run_mosei_centralized_training(args, logger)
            logger.info('Centralized(MOSEI) finished!')
            return

        # partial_centralized
        if args.method == "partial_centralized":
            raise ValueError("MOSEI not supported partial_centralized")

        if args.method == "fedavg":
            trainer = MOSEIFedAvgTrainer(args, logger)
        elif args.method == "fedmsplit_lite":
            trainer = MOSEIFedMSplitLiteTrainer(args, logger)
        elif args.method == "creamfl":
            trainer = MOSEICreamFLTrainer(args, logger)
        elif args.method == "car_mfl":
            trainer = MOSEICARMFLTrainer(args, logger)
        elif args.method == "fedmm_ssl":
            trainer = MOSEIFedMMSSLTrainer(args, logger)
        elif args.method == "fedssl_moe":
            trainer = MOSEIFedSSLMoeTrainer(args, logger)
        else:
            raise ValueError(
                "MOSEI not supported method={}".format(args.method)
            )

    else:
        # --------------------  --------------------
        if args.method == 'centralized':
            run_centralized_training(args, logger)
            logger.info('Centralized finished!')
            return

        if args.method == 'partial_centralized':
            run_partial_centralized_training(args, logger)
            logger.info('Partial centralized finished!')
            return

        if args.method in ['fedavg', 'fedmm']:
            trainer = FedMMTrainer(args, logger)
        elif args.method == 'fedmm_ssl':
            trainer = FedMMTrainerSSL(args, logger)
        elif args.method == 'fedssl_moe':
            trainer = FedSSLMoeTrainer(args, logger)
        elif args.method == 'creamfl':
            trainer = CreamFLTrainer(args, logger)
        elif args.method == 'local':
            trainer = FedMMTrainer(args, logger)
        elif args.method == 'car_mfl':
            trainer = CARMFLTrainer(args, logger)
        elif args.method == 'poe_moe':
            trainer = PoEMoETrainer(args, logger)
        elif args.method == 'fedmsplit_lite':
            trainer = FedMSplitTrainerLite(args, logger)
        elif args.method == 'fedmsplit_fmtl':
            trainer = FedMSplitTrainerFMTL(args, logger)
        else:
            raise ValueError('Unknown method: {}'.format(args.method))

    result = trainer.train()

    if result is not None:
        save_results_txt(args, result)

    logger.info('finished!')


if __name__ == '__main__':
    main()