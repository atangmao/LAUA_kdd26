# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedMMTrainerSSL

"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trainers.base_trainer import BaseTrainer
from models.model import FedMMModel
from losses.losses import FedMMCriterion, KLDivergenceLoss
from clients.multimodal_client import MultimodalClient
from clients.unimodal_client import ImageOnlyClient, TextOnlyClient
from server.fed_server import FedServer
from data import get_dataset, FederatedDataSplitter, Flickr30KSubset


# ------------------------- SimCLR / NT-Xent -------------------------

def nt_xent_loss(z1, z2, temperature=0.07):
    """
    SimCLR NT-Xent / InfoNCE (batch )
    z1, z2: [B, D]
    """
    if z1.dim() != 2 or z2.dim() != 2:
        raise ValueError(f"nt_xent_loss expects 2D tensors, got {z1.shape}, {z2.shape}")
    if z1.size(0) != z2.size(0):
        raise ValueError(f"batch mismatch: {z1.size(0)} vs {z2.size(0)}")

    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    logits = (z @ z.t()) / temperature  # [2B, 2B]

    #
    diag = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(diag, -1e9)

    #
    labels = torch.cat(
        [torch.arange(B, 2 * B, device=z.device), torch.arange(0, B, device=z.device)],
        dim=0
    )  # [2B]

    return F.cross_entropy(logits, labels)


def augment_features(x, dropout_p=0.2, noise_std=0.0):
    """

    """
    x = F.dropout(x, p=dropout_p, training=True)
    if noise_std and noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x


# ------------------------- New unimodal clients (SSL) -------------------------

class ImageOnlyClientSSL(ImageOnlyClient):
    """

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #
        self.kl_loss_fn = KLDivergenceLoss()

    def local_train(self, criterion):
        #
        if self.args.task != 'retrieval':
            return super().local_train(criterion)

        self.model.train()
        total_loss = 0.0
        total_uncertainty = 0.0
        num_batches = 0

        tau = getattr(self.args, "ssl_temperature", getattr(self.args, "temperature", 0.07))
        p = getattr(self.args, "ssl_dropout_p", 0.2)
        noise = getattr(self.args, "ssl_noise_std", 0.0)

        for batch in self.train_loader:
            img_feat = batch['image'].to(self.device)

            self.optimizer.zero_grad()

            #
            v1 = augment_features(img_feat, dropout_p=p, noise_std=noise)
            v2 = augment_features(img_feat, dropout_p=p, noise_std=noise)

            #
            _, mu1, logv1, _ = self.model(img_feat=v1, txt_feat=None, sample=True)
            _, mu2, logv2, _ = self.model(img_feat=v2, txt_feat=None, sample=True)

            #
            loss_ssl = nt_xent_loss(mu1, mu2, temperature=tau)

            #
            loss_kl = 0.5 * (self.kl_loss_fn(mu1, logv1) + self.kl_loss_fn(mu2, logv2))

            loss = loss_ssl + self.args.beta * loss_kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += float(loss.item())

            #
            uncertainty = torch.exp(logv1).sum(dim=-1).mean().item()
            total_uncertainty += uncertainty

            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_uncertainty = total_uncertainty / max(num_batches, 1)
        self.accumulated_uncertainty = avg_uncertainty
        self.train_loss_history.append(avg_loss)

        return avg_loss, avg_uncertainty


class TextOnlyClientSSL(TextOnlyClient):
    """ """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_loss_fn = KLDivergenceLoss()

    def local_train(self, criterion):
        if self.args.task != 'retrieval':
            return super().local_train(criterion)

        self.model.train()
        total_loss = 0.0
        total_uncertainty = 0.0
        num_batches = 0

        tau = getattr(self.args, "ssl_temperature", getattr(self.args, "temperature", 0.07))
        p = getattr(self.args, "ssl_dropout_p", 0.2)
        noise = getattr(self.args, "ssl_noise_std", 0.0)

        for batch in self.train_loader:
            txt_feat = batch['text'].to(self.device)

            self.optimizer.zero_grad()

            v1 = augment_features(txt_feat, dropout_p=p, noise_std=noise)
            v2 = augment_features(txt_feat, dropout_p=p, noise_std=noise)

            _, mu1, logv1, _ = self.model(img_feat=None, txt_feat=v1, sample=True)
            _, mu2, logv2, _ = self.model(img_feat=None, txt_feat=v2, sample=True)

            loss_ssl = nt_xent_loss(mu1, mu2, temperature=tau)
            loss_kl = 0.5 * (self.kl_loss_fn(mu1, logv1) + self.kl_loss_fn(mu2, logv2))
            loss = loss_ssl + self.args.beta * loss_kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            uncertainty = torch.exp(logv1).sum(dim=-1).mean().item()
            total_uncertainty += uncertainty
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_uncertainty = total_uncertainty / max(num_batches, 1)
        self.accumulated_uncertainty = avg_uncertainty
        self.train_loss_history.append(avg_loss)

        return avg_loss, avg_uncertainty


# ------------------------- Trainer -------------------------

class FedMMTrainerSSL(BaseTrainer):
    """

    """
    def __init__(self, args, logger=None):
        super().__init__(args, logger)
        self.clients = []
        self.server = None
        self.criterion = None
        self.test_loader = None
        self.val_loader = None

    def setup(self):
        self.log('Setting up FedMMTrainerSSL...')

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

        #
        if self.args.task == 'classification' and hasattr(train_dataset, 'id_to_label'):
            labels = [train_dataset.id_to_label.get(train_dataset.image_ids[i], 0)
                      for i in range(len(train_dataset))]
        else:
            labels = None

        client_data_splits = splitter.split(train_dataset, labels)
        split_info = splitter.get_client_info(client_data_splits)
        self.log(f'Data split info: {split_info}')

        SubsetClass = Flickr30KSubset

        self.clients = []
        for client_id, data_info in client_data_splits.items():
            indices = data_info['indices']
            modality = data_info['modality']

            client_train_data = SubsetClass(train_dataset, indices, modality)
            client_val_data = SubsetClass(val_dataset, indices[:len(indices) // 5], modality) if val_dataset else None

            if modality == 'both':
                client = MultimodalClient(
                    client_id=client_id,
                    model=global_model,
                    train_dataset=client_train_data,
                    val_dataset=client_val_data,
                    args=self.args,
                    logger=self.logger,
                    collate_fn=collate_fn
                )
            elif modality == 'image':
                client = ImageOnlyClientSSL(
                    client_id=client_id,
                    model=global_model,
                    train_dataset=client_train_data,
                    val_dataset=client_val_data,
                    args=self.args,
                    logger=self.logger,
                    collate_fn=collate_fn
                )
            else:
                client = TextOnlyClientSSL(
                    client_id=client_id,
                    model=global_model,
                    train_dataset=client_train_data,
                    val_dataset=client_val_data,
                    args=self.args,
                    logger=self.logger,
                    collate_fn=collate_fn
                )

            self.clients.append(client)
            self.log(f'Created client {client_id}: {modality}, {len(indices)} samples')

        self.server = FedServer(global_model, self.args, self.logger)

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

        self.log('Setup completed (SSL)!')

    def train_round(self):
        round_losses = []
        round_uncertainties = []

        global_params = self.server.get_global_params()
        for client in self.clients:
            client.set_model_params(global_params)

        for client in self.clients:
            #
            #
            for _ in range(self.args.local_epochs):
                loss, uncertainty = client.local_train(self.criterion)
            round_losses.append(loss)
            round_uncertainties.append(uncertainty)

        if self.args.method == 'fedavg':
            self.server.fedavg_aggregate(self.clients)
        elif self.args.method in ['fedmm', 'fedmm_ssl']:
            self.server.uncertainty_aggregate(self.clients)
        elif self.args.method == 'fedprox':
            self.server.fedprox_aggregate(self.clients, mu=self.args.mu)
        elif self.args.method == 'local':
            pass
        else:
            raise ValueError(f'FedMMTrainerSSL does not support method: {self.args.method}')

        avg_loss = sum(round_losses) / max(len(round_losses), 1)
        avg_uncertainty = sum(round_uncertainties) / max(len(round_uncertainties), 1)

        return {
            'loss': avg_loss,
            'uncertainty': avg_uncertainty,
            'client_losses': round_losses,
            'client_uncertainties': round_uncertainties
        }

    def evaluate(self, test=False):
        loader = self.test_loader if test else self.val_loader
        return self.server.evaluate(loader, self.criterion)

    def get_model_state(self):
        return self.server.get_global_params()

    def set_model_state(self, state_dict):
        self.server.global_model.load_state_dict(state_dict)