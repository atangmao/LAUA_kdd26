# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedMM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoders import ModalityEncoder


class ProductOfExperts(nn.Module):
    """  """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, mus, log_vars, masks=None):
        """
        Args:
            mus: list of [B, D]
            log_vars: list of [B, D]
            masks: list of [B, 1] or None
        """
        precisions = []

        for mu, log_var in zip(mus, log_vars):
            var = torch.exp(log_var) + self.eps
            precision = 1.0 / var
            precisions.append(precision)

        if masks is not None:
            precisions = [p * m for p, m in zip(precisions, masks)]

        precision_joint = sum(precisions) + self.eps
        mu_joint = sum(p * mu for p, mu in zip(precisions, mus)) / precision_joint
        log_var_joint = torch.log(1.0 / precision_joint + self.eps)

        return mu_joint, log_var_joint


class Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z):
        return self.net(z)


class ProjectionHead(nn.Module):
    def __init__(self, latent_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, proj_dim)
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


class FedMMModel(nn.Module):
    """FedMM """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.latent_dim = args.latent_dim

        self.image_encoder = ModalityEncoder(
            modality='image',
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim
        )

        self.text_encoder = ModalityEncoder(
            modality='text',
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim
        )

        self.poe = ProductOfExperts()

        if args.task == 'classification':
            self.task_head = Classifier(args.latent_dim, args.num_classes)
        else:
            self.task_head = ProjectionHead(args.latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img_feat=None, txt_feat=None, sample=False):
        mus, log_vars, masks = [], [], []
        modality_outputs = {}

        device = img_feat.device if img_feat is not None else txt_feat.device
        batch_size = img_feat.size(0) if img_feat is not None else txt_feat.size(0)

        # -------- Image --------
        if img_feat is not None:
            mu, log_var, h = self.image_encoder(img_feat)
            mus.append(mu)
            log_vars.append(log_var)
            masks.append(torch.ones(batch_size, 1, device=device))
            modality_outputs['image'] = {'mu': mu, 'log_var': log_var, 'h': h}
        else:
            mus.append(torch.zeros(batch_size, self.latent_dim, device=device))
            log_vars.append(torch.ones(batch_size, self.latent_dim, device=device) * 1e8)
            masks.append(torch.zeros(batch_size, 1, device=device))

        # -------- Text --------
        if txt_feat is not None:
            mu, log_var, h = self.text_encoder(txt_feat)
            mus.append(mu)
            log_vars.append(log_var)
            masks.append(torch.ones(batch_size, 1, device=device))
            modality_outputs['text'] = {'mu': mu, 'log_var': log_var, 'h': h}
        else:
            mus.append(torch.zeros(batch_size, self.latent_dim, device=device))
            log_vars.append(torch.ones(batch_size, self.latent_dim, device=device) * 1e8)
            masks.append(torch.zeros(batch_size, 1, device=device))

        # -------- PoE Fusion --------
        mu_joint, log_var_joint = self.poe(mus, log_vars, masks)

        z = self.reparameterize(mu_joint, log_var_joint) if sample else mu_joint
        output = self.task_head(z)

        return output, mu_joint, log_var_joint, modality_outputs