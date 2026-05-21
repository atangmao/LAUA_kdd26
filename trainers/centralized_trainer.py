#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import os
import copy
import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from trainers.base_trainer import BaseTrainer
from models.model import FedMMModel
from losses.losses import FedMMCriterion
from server.fed_server import FedServer
from data import get_dataset, FederatedDataSplitter


class CentralizedTrainer(BaseTrainer):
    """"""

    def __init__(self, args, logger=None, train_dataset_override=None):
        super().__init__(args, logger)

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.server = None  #

        self.train_dataset_override = train_dataset_override

        # for sweep timing
        self.actual_epochs = 0
        self.total_time_sec = 0.0
        self.epoch_times_sec = []

    def setup(self):
        self.log('Setting up Centralized trainer...')
        self.log('Loading datasets...')
        train_dataset, val_dataset, test_dataset, collate_fn = get_dataset(self.args)

        if self.train_dataset_override is not None:
            train_dataset = self.train_dataset_override

        self.log('Creating global model...')
        self.model = FedMMModel(self.args).to(self.device)

        self.criterion = FedMMCriterion(self.args)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        if self.args.lr_scheduler == 'cosine':
            #
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=int(getattr(self.args, "epochs", 1)))
        else:
            self.scheduler = None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=False,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=False,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=False,
        )

        self.server = FedServer(self.model, self.args, self.logger)
        self.log('Centralized setup completed!')

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            img_feat = batch['image'].to(self.device)
            txt_feat = batch['text'].to(self.device)

            if self.args.task == 'classification':
                target = batch['label'].to(self.device)
            else:
                target = None

            output, mu_joint, log_var_joint, modality_outputs = self.model(
                img_feat=img_feat,
                txt_feat=txt_feat,
                sample=False
            )

            if self.args.task == 'retrieval':
                img_proj = self.model.task_head(modality_outputs['image']['mu'])
                txt_proj = self.model.task_head(modality_outputs['text']['mu'])
            else:
                img_proj, txt_proj = None, None

            loss, _ = self.criterion(
                output, target, mu_joint, log_var_joint, modality_outputs,
                img_proj=img_proj, txt_proj=txt_proj
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = img_feat.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {"loss": avg_loss}

    def train_round(self):
        # BaseTrainer.train()
        #
        out = self.train_one_epoch()

        #
        self.actual_rounds = int(getattr(self, "actual_rounds", 0)) + 1
        return out

    # def train(self):
    #     self.setup()
    #
    #     epochs = int(getattr(self.args, "epochs", 1))
    #     start = time.time()
    #     self.epoch_times_sec = []
    #     self.actual_epochs = 0
    #
    #     for ep in range(epochs):
    #         t0 = time.time()
    #         _ = self.train_one_epoch()
    #         self.epoch_times_sec.append(time.time() - t0)
    #         self.actual_epochs = ep + 1
    #
    #     self.total_time_sec = time.time() - start
    #     return

    def evaluate(self, test=False):
        loader = self.test_loader if test else self.val_loader
        return self.server.evaluate(loader, self.criterion)

    def get_model_state(self):
        return copy.deepcopy(self.model.state_dict())

    def set_model_state(self, state_dict):
        self.model.load_state_dict(state_dict)


def _write_timing_json(args, method: str, executed_epochs: int, total_time_sec: float, epoch_times_sec=None):
    if epoch_times_sec is None:
        epoch_times_sec = []
    timing_out = {
        "method": method,
        "executed_rounds": int(executed_epochs),   #
        "total_time_sec": float(total_time_sec),
        "round_times_sec": [float(x) for x in epoch_times_sec],
    }
    with open(os.path.join(args.exp_dir, "timing.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(timing_out, f, ensure_ascii=False, indent=2)


def run_centralized_training(args, logger):
    trainer = CentralizedTrainer(args, logger)
    trainer.train()

    test_metrics = trainer.evaluate(test=True)

    _write_timing_json(
        args,
        method="centralized",
        executed_epochs=trainer.actual_epochs,
        total_time_sec=trainer.total_time_sec,
        epoch_times_sec=trainer.epoch_times_sec,
    )

    logger.info(
        "centralized : "
        + " | ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                     for k, v in test_metrics.items()])
    )
    return test_metrics, trainer.actual_epochs, trainer.total_time_sec


def run_partial_centralized_training(args, logger):
    """

    """
    #
    full_train, _, _, _ = get_dataset(args)

    #
    splitter = FederatedDataSplitter(
        num_multimodal_clients=1,  #
        num_text_clients=0,
        num_audio_clients=0,
        num_image_clients=0,
        num_vision_clients=0,
        partition=args.partition,
        alpha=args.alpha,
        seed=args.seed,
        mm_ratio=args.mm_ratio,
    )

    #
    splits = splitter.split(full_train, labels=None)

    #
    mm_indices = []
    for _, info in splits.items():
        mod = info.get("modality")
        if mod == "both":
            mm_indices.extend(info.get("indices", []))

    mm_indices = sorted(set(mm_indices))  #

    if len(mm_indices) == 0:
        raise ValueError("partial_centralized got empty multimodal subset. Check num_multimodal_clients/mm_ratio/partition/alpha.")

    if logger:
        logger.info(f"[partial_centralized] train subset size={len(mm_indices)}/{len(full_train)}")

    #
    partial_train = Subset(full_train, mm_indices)

    #
    trainer = CentralizedTrainer(args, logger, train_dataset_override=partial_train)
    trainer.setup()  #
    trainer.train()  #

    #
    test_metrics = trainer.evaluate(test=True)

    #
    _write_timing_json(
        args,
        method="partial_centralized",
        executed_epochs=trainer.actual_epochs,
        total_time_sec=trainer.total_time_sec,
        epoch_times_sec=trainer.epoch_times_sec,
    )

    logger.info(
        "partial_centralized : "
        + " | ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                      for k, v in test_metrics.items()])
    )
    return test_metrics, trainer.actual_epochs, trainer.total_time_sec