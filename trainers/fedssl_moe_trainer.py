# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedSSL + PoE + MoE Trainer
- client internal fusion: keep FedMMModel PoE (in model.py)
- cross-client fusion: MoE aggregation on server (gating weights from public loss or uncertainty)
- unimodal clients: use SSL versions (ImageOnlyClientSSL/TextOnlyClientSSL) from fedmm_trainer_ssl.py
"""

import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from trainers.base_trainer import BaseTrainer
from models.model import FedMMModel
from losses.losses import FedMMCriterion
from server.fed_server import FedServer
from data import get_dataset, FederatedDataSplitter, Flickr30KSubset
from clients.multimodal_client import MultimodalClient

# 关键：导入你已经实现的 SSL 单模态客户端
from trainers.fedmm_trainer_ssl import ImageOnlyClientSSL, TextOnlyClientSSL


class FedSSLMoeTrainer(BaseTrainer):
    def __init__(self, args, logger=None):
        super().__init__(args, logger)
        self.clients = []
        self.server = None
        self.criterion = None
        self.val_loader = None
        self.test_loader = None
        self.public_loader = None

    def setup(self):
        self.log("Setting up FedSSL+PoE+MoE trainer...")

        train_dataset, val_dataset, test_dataset, collate_fn = get_dataset(self.args)

        global_model = FedMMModel(self.args)
        self.criterion = FedMMCriterion(self.args)

        splitter = FederatedDataSplitter(
            num_multimodal_clients=self.args.num_multimodal_clients,
            num_image_clients=self.args.num_image_clients,
            num_text_clients=self.args.num_text_clients,
            partition=self.args.partition,
            alpha=self.args.alpha,
            seed=self.args.seed,
            mm_ratio=self.args.mm_ratio,
        )
        client_data_splits = splitter.split(train_dataset, labels=None)

        SubsetClass = Flickr30KSubset  # 你原 PoEMoETrainer 也是这么写的，这里保持一致

        self.clients = []
        for client_id, data_info in client_data_splits.items():
            indices = data_info['indices']
            modality = data_info['modality']

            client_train = SubsetClass(train_dataset, indices, modality)
            client_val = SubsetClass(val_dataset, indices[:len(indices)//5], modality) if val_dataset else None

            if modality == 'both':
                c = MultimodalClient(
                    client_id, global_model, client_train, client_val,
                    self.args, self.logger, collate_fn=collate_fn
                )
            elif modality == 'image':
                c = ImageOnlyClientSSL(
                    client_id=client_id,
                    model=global_model,
                    train_dataset=client_train,
                    val_dataset=client_val,
                    args=self.args,
                    logger=self.logger,
                    collate_fn=collate_fn
                )
            else:
                c = TextOnlyClientSSL(
                    client_id=client_id,
                    model=global_model,
                    train_dataset=client_train,
                    val_dataset=client_val,
                    args=self.args,
                    logger=self.logger,
                    collate_fn=collate_fn
                )

            self.clients.append(c)

        self.server = FedServer(global_model, self.args, self.logger)

        self.val_loader = DataLoader(
            val_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn
        )

        # public subset for gating
        pub_size = int(getattr(self.args, 'public_size', 4096))
        pub_bs = int(getattr(self.args, 'public_batch_size', max(64, self.args.batch_size)))

        g = torch.Generator().manual_seed(self.args.seed)
        pub_size = min(pub_size, len(train_dataset))
        pub_indices = torch.randperm(len(train_dataset), generator=g)[:pub_size].tolist()
        public_dataset = Subset(train_dataset, pub_indices)

        self.public_loader = DataLoader(
            public_dataset,
            batch_size=pub_bs,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        self.log("FedSSL+PoE+MoE setup completed.")

    def evaluate(self, test=False):
        loader = self.test_loader if test else self.val_loader
        return self.server.evaluate(loader, self.criterion)

    def get_model_state(self):
        return self.server.get_global_params()

    def set_model_state(self, state_dict):
        self.server.global_model.load_state_dict(state_dict)

    # ---------------- MoE gating ----------------

    @torch.no_grad()
    def _client_public_loss(self, client):
        model = client.model
        model.eval().to(self.device)

        total_loss, total_n = 0.0, 0
        for batch in self.public_loader:
            img_feat = batch['image'].to(self.device)
            txt_feat = batch['text'].to(self.device)
            target = batch['label'].to(self.device) if self.args.task == 'classification' else None

            output, mu_joint, log_var_joint, modality_outputs = model(
                img_feat=img_feat, txt_feat=txt_feat, sample=False
            )

            if self.args.task == 'retrieval':
                img_proj = model.task_head(modality_outputs['image']['mu'])
                txt_proj = model.task_head(modality_outputs['text']['mu'])
            else:
                img_proj, txt_proj = None, None

            loss, _ = self.criterion(
                output, target, mu_joint, log_var_joint, modality_outputs,
                img_proj=img_proj, txt_proj=txt_proj
            )
            bs = img_feat.size(0)
            total_loss += float(loss.item()) * bs
            total_n += bs

        return total_loss / max(1, total_n)

    def _compute_moe_weights(self):
        mode = getattr(self.args, "moe_gate", "public_loss")  # 'public_loss' or 'uncertainty'
        tau = float(getattr(self.args, "moe_tau", 3.0))
        eps = 1e-8

        if mode == "uncertainty":
            scores = []
            for c in self.clients:
                u = float(c.get_uncertainty())
                scores.append(1.0 / (u + eps))
            w = torch.tensor(scores, dtype=torch.float32)
            w = w / (w.sum() + eps)
            return w.tolist()

        losses = [self._client_public_loss(c) for c in self.clients]
        scores = torch.exp(-tau * torch.tensor(losses, dtype=torch.float32))
        w = scores / (scores.sum() + eps)

        if self.logger:
            self.logger.info(f"[FedSSL+PoE+MoE] public losses: {losses}")
            self.logger.info(f"[FedSSL+PoE+MoE] moe weights: {w.tolist()}")

        return w.tolist()

    def _moe_aggregate(self, moe_weights):
        client_params = [c.get_model_params() for c in self.clients]
        w = np.array(moe_weights, dtype=np.float64)
        w = w / (w.sum() + 1e-12)

        aggregated = copy.deepcopy(client_params[0])
        for k in aggregated.keys():
            aggregated[k] = torch.zeros_like(aggregated[k], dtype=torch.float32)
            for i, p in enumerate(client_params):
                aggregated[k] += float(w[i]) * p[k].float()
            aggregated[k] = aggregated[k].to(client_params[0][k].dtype)

        self.server.global_model.load_state_dict(aggregated)
        return aggregated

    # ---------------- Round ----------------

    def train_round(self):
        # 1) dispatch
        global_params = self.server.get_global_params()
        for c in self.clients:
            c.set_model_params(global_params)

        # 2) local train
        round_losses, round_uncerts = [], []
        for c in self.clients:
            for _ in range(self.args.local_epochs):
                loss, unc = c.local_train(self.criterion)
            round_losses.append(loss)
            round_uncerts.append(unc)

        # 3) MoE gating + aggregate
        moe_w = self._compute_moe_weights()
        self._moe_aggregate(moe_w)

        return {
            "loss": sum(round_losses) / max(1, len(round_losses)),
            "uncertainty": sum(round_uncerts) / max(1, len(round_uncerts))
        }