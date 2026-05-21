# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    """KL

    KL(q(z|x) || p(z))， p(z) = N(0, I)
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, mu, log_var):
        """
        Args:
            mu:  [B, D]
            log_var:  [B, D]
        Returns:
            kl_loss: KL
        """
        # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        if self.reduction == 'mean':
            return kl.mean()
        elif self.reduction == 'sum':
            return kl.sum()
        else:
            return kl


class DistillationLoss(nn.Module):
    """


    """

    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, mu_student, mu_teacher, conf_student, conf_teacher):
        """
        Args:
            mu_student:  [B, D]
            mu_teacher:  [B, D]（stop_grad）
            conf_student:  [B, 1]
            conf_teacher:  [B, 1]
        Returns:
            distill_loss:
        """

        weight = torch.sigmoid(conf_teacher - conf_student)

        if self.loss_type == 'mse':
            loss = F.mse_loss(mu_student, mu_teacher.detach(), reduction='none')
        elif self.loss_type == 'cosine':
            loss = 1 - F.cosine_similarity(mu_student, mu_teacher.detach(), dim=-1, eps=1e-8)
            loss = loss.unsqueeze(-1)
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

        #
        weighted_loss = (weight * loss).mean()
        return weighted_loss


class MutualDistillationLoss(nn.Module):
    """


    """

    def __init__(self):
        super().__init__()
        self.distill_loss = DistillationLoss(loss_type='mse')

    def forward(self, modality_outputs):
        """
        Args:
            modality_outputs: dict {
                'image': {'mu': ..., 'log_var': ..., 'h': ...},
                'text': {'mu': ..., 'log_var': ..., 'h': ...}
            }
        Returns:
            loss:
        """
        if len(modality_outputs) < 2:
            #
            return torch.tensor(0.0, device=list(modality_outputs.values())[0]['mu'].device)

        #
        img_out = modality_outputs.get('image')
        txt_out = modality_outputs.get('text')

        if img_out is None or txt_out is None:
            return torch.tensor(0.0)

        #
        var_img = torch.exp(img_out['log_var']).mean(dim=-1, keepdim=True)
        var_txt = torch.exp(txt_out['log_var']).mean(dim=-1, keepdim=True)
        conf_img = 1.0 / (var_img + 1e-8)
        conf_txt = 1.0 / (var_txt + 1e-8)

        #
        loss_img2txt = self.distill_loss(txt_out['mu'], img_out['mu'], conf_txt, conf_img)
        loss_txt2img = self.distill_loss(img_out['mu'], txt_out['mu'], conf_img, conf_txt)

        return loss_img2txt + loss_txt2img


class ContrastiveLoss(nn.Module):
    """（CLIP-style InfoNCE）


    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_features, txt_features):
        """
        Args:
            img_features:  [B, D]
            txt_features:  [B, D]
        Returns:
            loss:
        """
        #
        logits = torch.matmul(img_features, txt_features.T) / self.temperature

        #
        batch_size = img_features.size(0)
        labels = torch.arange(batch_size, device=img_features.device)

        #
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2


class FedMMCriterion(nn.Module):
    """FedMM

    L = L_task + β * L_KL + γ * L_distill
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.beta = args.beta
        self.gamma = args.gamma

        #
        if args.task == 'classification':
            self.task_loss = nn.CrossEntropyLoss()
        else:  # retrieval
            self.task_loss = ContrastiveLoss(temperature=args.temperature)

        #
        self.kl_loss = KLDivergenceLoss()

        #
        self.distill_loss = MutualDistillationLoss()

    def forward(self, output, target, mu_joint, log_var_joint, modality_outputs,
                img_proj=None, txt_proj=None):
        """
        Args:
            output:
            target:
            mu_joint:
            log_var_joint:
            modality_outputs:
            img_proj:
            txt_proj:
        Returns:
            total_loss:
            loss_dict:
        """
        loss_dict = {}

        #
        if self.args.task == 'classification':
            loss_task = self.task_loss(output, target)
        else:
            if img_proj is not None and txt_proj is not None:
                loss_task = self.task_loss(img_proj, txt_proj)
            else:
                loss_task = torch.tensor(0.0, device=mu_joint.device)
        loss_dict['task'] = loss_task.item()

        # 2.
        loss_kl = self.kl_loss(mu_joint, log_var_joint)
        loss_dict['kl'] = loss_kl.item()

        # 3.
        loss_distill = self.distill_loss(modality_outputs)
        loss_dict['distill'] = loss_distill.item()

        #
        total_loss = loss_task + self.beta * loss_kl + self.gamma * loss_distill
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict
