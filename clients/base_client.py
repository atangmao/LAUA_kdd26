# -*- coding: utf-8 -*-
"""

"""

import copy
import torch
from torch.utils.data import DataLoader


class BaseClient:
    """BaseClient"""

    def __init__(self, client_id, client_type, model, train_dataset,
                 val_dataset=None, args=None, logger=None, collate_fn=None):
        """

        """
        self.client_id = client_id
        self.client_type = client_type
        self.args = args
        self.logger = logger
        self.device = args.device


        self.model = copy.deepcopy(model).to(self.device)


        if collate_fn is None:
            from data.collate import flickr_collate_fn
            collate_fn = flickr_collate_fn
        self.collate_fn = collate_fn

        if self.logger is not None:
            name = getattr(self.collate_fn, "__name__", str(self.collate_fn))
            self.logger.info(f"Client {self.client_id} uses collate_fn: {name}")

        #
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_fn
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=self.collate_fn
            )
        else:
            self.val_loader = None

        #
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        #
        self.train_loss_history = []
        self.num_samples = len(train_dataset)

    def get_model_params(self):

        return copy.deepcopy(self.model.state_dict())

    def set_model_params(self, params):

        self.model.load_state_dict(params)

    def get_num_samples(self):

        return self.num_samples

    def get_uncertainty(self):

        raise NotImplementedError

    def local_train(self, criterion):

        raise NotImplementedError

    def local_eval(self, criterion):

        raise NotImplementedError