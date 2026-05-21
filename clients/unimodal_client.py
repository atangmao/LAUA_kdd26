# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import torch
from clients.base_client import BaseClient
from data.collate import flickr_collate_fn


class ImageOnlyClient(BaseClient):
    """"""

    def __init__(self, client_id, model, train_dataset, val_dataset=None,
                 args=None, logger=None, collate_fn=None):
        super().__init__(
            client_id=client_id,
            client_type='image_only',
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            logger=logger,
            collate_fn=collate_fn
        )
        self.accumulated_uncertainty = 0.0

    def local_train(self, criterion):
        """"""
        self.model.train()
        total_loss = 0.0
        total_uncertainty = 0.0
        num_batches = 0

        for batch in self.train_loader:
            img_feat = batch['image'].to(self.device)

            if self.args.task == 'classification':
                target = batch['label'].to(self.device)
            else:
                target = None

            self.optimizer.zero_grad()

            #
            output, mu_joint, log_var_joint, modality_outputs = self.model(
                img_feat=img_feat,
                txt_feat=None,  #
                sample=True
            )

            #
            loss, loss_dict = criterion(
                output, target, mu_joint, log_var_joint, modality_outputs,
                img_proj=None, txt_proj=None
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            uncertainty = torch.exp(log_var_joint).sum(dim=-1).mean().item()
            total_uncertainty += uncertainty
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_uncertainty = total_uncertainty / num_batches
        self.accumulated_uncertainty = avg_uncertainty
        self.train_loss_history.append(avg_loss)

        return avg_loss, avg_uncertainty

    def local_eval(self, criterion):
        """  """
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                img_feat = batch['image'].to(self.device)

                if self.args.task == 'classification':
                    target = batch['label'].to(self.device)
                else:
                    target = None

                output, mu_joint, log_var_joint, modality_outputs = self.model(
                    img_feat=img_feat,
                    txt_feat=None,
                    sample=False
                )

                loss, _ = criterion(
                    output, target, mu_joint, log_var_joint, modality_outputs
                )

                total_loss += loss.item() * img_feat.size(0)

                if self.args.task == 'classification':
                    pred = output.argmax(dim=1)
                    total_correct += (pred == target).sum().item()

                total_samples += img_feat.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples if self.args.task == 'classification' else None

        return avg_loss, accuracy

    def get_uncertainty(self):
        return self.accumulated_uncertainty


class TextOnlyClient(BaseClient):
    """  """

    def __init__(self, client_id, model, train_dataset, val_dataset=None,
                 args=None, logger=None, collate_fn=None):
        super().__init__(
            client_id=client_id,
            client_type='text_only',
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            logger=logger,
            collate_fn=collate_fn
        )
        self.accumulated_uncertainty = 0.0

    def local_train(self, criterion):
        """  """
        self.model.train()
        total_loss = 0.0
        total_uncertainty = 0.0
        num_batches = 0

        for batch in self.train_loader:
            txt_feat = batch['text'].to(self.device)

            if self.args.task == 'classification':
                target = batch['label'].to(self.device)
            else:
                target = None

            self.optimizer.zero_grad()

            #
            output, mu_joint, log_var_joint, modality_outputs = self.model(
                img_feat=None,  #
                txt_feat=txt_feat,
                sample=True
            )

            loss, loss_dict = criterion(
                output, target, mu_joint, log_var_joint, modality_outputs,
                img_proj=None, txt_proj=None
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            uncertainty = torch.exp(log_var_joint).sum(dim=-1).mean().item()
            total_uncertainty += uncertainty
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_uncertainty = total_uncertainty / num_batches
        self.accumulated_uncertainty = avg_uncertainty
        self.train_loss_history.append(avg_loss)

        return avg_loss, avg_uncertainty

    def local_eval(self, criterion):
        """  """
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                txt_feat = batch['text'].to(self.device)

                if self.args.task == 'classification':
                    target = batch['label'].to(self.device)
                else:
                    target = None

                output, mu_joint, log_var_joint, modality_outputs = self.model(
                    img_feat=None,
                    txt_feat=txt_feat,
                    sample=False
                )

                loss, _ = criterion(
                    output, target, mu_joint, log_var_joint, modality_outputs
                )

                total_loss += loss.item() * txt_feat.size(0)

                if self.args.task == 'classification':
                    pred = output.argmax(dim=1)
                    total_correct += (pred == target).sum().item()

                total_samples += txt_feat.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples if self.args.task == 'classification' else None

        return avg_loss, accuracy

    def get_uncertainty(self):
        return self.accumulated_uncertainty
