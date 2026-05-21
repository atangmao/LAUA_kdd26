# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """  """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 最后一层不加激活和dropout
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class ImageEncoder(nn.Module):
    """


    """

    def __init__(self, embed_dim=512, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = MLPEncoder(embed_dim, hidden_dim, hidden_dim, num_layers, dropout)

    def forward(self, x):
        """

        """
        return self.encoder(x)


class TextEncoder(nn.Module):
    """


    """

    def __init__(self, embed_dim=512, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = MLPEncoder(embed_dim, hidden_dim, hidden_dim, num_layers, dropout)

    def forward(self, x):
        """

        """
        return self.encoder(x)


class GaussianHead(nn.Module):
    """


    """

    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        #
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, h):
        """

        """
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        #
        log_var = torch.clamp(log_var, min=-10, max=10)
        return mu, log_var


class ModalityEncoder(nn.Module):
    """

    """

    def __init__(self, modality, embed_dim=512, hidden_dim=256, latent_dim=128,
                 num_layers=2, dropout=0.1):
        super().__init__()

        self.modality = modality

        if modality == 'image':
            self.backbone = ImageEncoder(embed_dim, hidden_dim, num_layers, dropout)
        elif modality == 'text':
            self.backbone = TextEncoder(embed_dim, hidden_dim, num_layers, dropout)
        else:
            raise ValueError(f'Unknown modality: {modality}')

        self.gaussian_head = GaussianHead(hidden_dim, latent_dim)

    def forward(self, x):
        """

        """
        h = self.backbone(x)
        mu, log_var = self.gaussian_head(h)
        return mu, log_var, h

    def get_distribution(self, x):
        """  """
        mu, log_var, h = self.forward(x)
        std = torch.exp(0.5 * log_var)
        dist = torch.distributions.Normal(mu, std)
        return dist, mu, log_var, h
