# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedMM
"""

import torch
from torch.utils.data import DataLoader, Subset

from trainers.base_trainer import BaseTrainer
from models.model import FedMMModel
from losses.losses import FedMMCriterion
from clients.multimodal_client import MultimodalClient
from clients.unimodal_client import ImageOnlyClient, TextOnlyClient
from server.fed_server import FedServer
from data import get_dataset, FederatedDataSplitter, COCOSubset, Flickr30KSubset
from data.collate import flickr_collate_fn


class FedMMTrainer(BaseTrainer): 
    """FedMM"""

    def __init__(self, args, logger=None):
        super().__init__(args, logger)

        self.clients = []
        self.server = None
        self.criterion = None
        self.test_loader = None
        self.val_loader = None

    def setup(self):
        """"""
        self.log('Setting up FedMM trainer...')

        # 1.
        self.log('Loading datasets...')
        train_dataset, val_dataset, test_dataset, collate_fn = get_dataset(self.args)

        # 2.
        self.log('Creating global model...')
        global_model = FedMMModel(self.args)

        # 3.
        self.criterion = FedMMCriterion(self.args)

        # 4.
        self.log('Splitting data for federated learning...')
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

        # 5.
        self.log('Creating clients...')
        SubsetClass = Flickr30KSubset

        for client_id, data_info in client_data_splits.items():
            indices = data_info['indices']
            modality = data_info['modality']

            #
            client_train_data = SubsetClass(train_dataset, indices, modality)
            client_val_data = SubsetClass(val_dataset, indices[:len(indices) // 5], modality) if val_dataset else None

            #
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
                client = ImageOnlyClient(
                    client_id=client_id,
                    model=global_model,
                    train_dataset=client_train_data,
                    val_dataset=client_val_data,
                    args=self.args,
                    logger=self.logger,
                    collate_fn=collate_fn
                )
            else:  # text
                client = TextOnlyClient(
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

        # 6.
        self.log('Creating server...')
        self.server = FedServer(global_model, self.args, self.logger)

        # 7.
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
            num_workers=0,  #
            collate_fn=collate_fn,
            pin_memory=False,
        )

        self.log('Setup completed!')

    def train_round(self):
        """  """
        round_losses = []
        round_uncertainties = []

        # 1.
        global_params = self.server.get_global_params()
        for client in self.clients:
            client.set_model_params(global_params)

        # 2.
        for client in self.clients:
            for local_epoch in range(self.args.local_epochs):
                loss, uncertainty = client.local_train(self.criterion)

            round_losses.append(loss)
            round_uncertainties.append(uncertainty)

        # 3.
        if self.args.method == 'fedavg':
            self.server.fedavg_aggregate(self.clients)
        elif self.args.method == 'fedmm':
            self.server.uncertainty_aggregate(self.clients)  #
        elif self.args.method == 'fedprox':
            self.server.fedprox_aggregate(self.clients, mu=self.args.mu)
        elif self.args.method == 'local':
            pass  #
        else:
            raise ValueError(f'FedMMTrainer does not support method: {self.args.method}')

        avg_loss = sum(round_losses) / len(round_losses)
        avg_uncertainty = sum(round_uncertainties) / len(round_uncertainties)

        return {
            'loss': avg_loss,
            'uncertainty': avg_uncertainty,
            'client_losses': round_losses,
            'client_uncertainties': round_uncertainties
        }

    def evaluate(self, test=False):
        """  """
        if test:
            loader = self.test_loader
        else:
            loader = self.val_loader
        return self.server.evaluate(loader, self.criterion)

    def get_model_state(self):
        """  """
        return self.server.get_global_params()

    def set_model_state(self, state_dict):
        """ """
        self.server.global_model.load_state_dict(state_dict)
