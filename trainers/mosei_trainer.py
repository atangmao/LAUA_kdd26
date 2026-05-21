# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader

from trainers.base_trainer import BaseTrainer
from server.fed_server import FedServer
from data import get_dataset, FederatedDataSplitter

from models.mosei_model import MOSEIRegressor
from clients.mosei_clients import MOSEIAllClient, MOSEITextClient, MOSEIAudioClient, MOSEIVisionClient


class MOSEIFedTrainer(BaseTrainer):
    """

    """

    def __init__(self, args, logger=None):
        super().__init__(args, logger)
        self.clients = []
        self.server = None
        self.val_loader = None
        self.test_loader = None

    def setup(self):
        self.log("Setting up MOSEIFedTrainer...")

        #
        self.args.task = "regression"

        train_dataset, val_dataset, test_dataset, collate_fn = get_dataset(self.args)

        global_model = MOSEIRegressor(
            bert_name=getattr(self.args, "bert_name", "bert-base-uncased"),
            hidden_dim=getattr(self.args, "hidden_dim", 256),
            freeze_bert=getattr(self.args, "freeze_bert", False),
        )

        splitter = FederatedDataSplitter(
            num_multimodal_clients=self.args.num_multimodal_clients,
            num_text_clients=self.args.num_text_clients,
            num_audio_clients=getattr(self.args, "num_audio_clients", 0),
            num_vision_clients=getattr(self.args, "num_vision_clients", 0),
            num_image_clients=0,  # MOSEI
            partition=self.args.partition,
            alpha=self.args.alpha,
            seed=self.args.seed,
            mm_ratio=self.args.mm_ratio,
        )

        client_data_splits = splitter.split(train_dataset, labels=None)
        self.log(f"Data split info: {splitter.get_client_info(client_data_splits)}")

        # client:
        self.clients = []
        for client_id, info in client_data_splits.items():
            indices = info["indices"]
            modality = info["modality"]

            #
            from torch.utils.data import Subset
            client_train = Subset(train_dataset, indices)

            #
            client_val = None
            if val_dataset is not None:
                client_val = Subset(val_dataset, indices[: max(1, len(indices)//5)])

            if modality == "both":
                c = MOSEIAllClient(client_id, global_model, client_train, client_val,
                                   args=self.args, logger=self.logger, collate_fn=collate_fn)
                #
                c.blocks = ["text_encoder", "audio_encoder", "vision_encoder", "task_head"]
            elif modality == "text":
                c = MOSEITextClient(client_id, global_model, client_train, client_val,
                                    args=self.args, logger=self.logger, collate_fn=collate_fn)
                c.blocks = ["text_encoder", "task_head"]
            elif modality == "audio":
                c = MOSEIAudioClient(client_id, global_model, client_train, client_val,
                                     args=self.args, logger=self.logger, collate_fn=collate_fn)
                c.blocks = ["audio_encoder", "task_head"]
            elif modality == "vision":
                c = MOSEIVisionClient(client_id, global_model, client_train, client_val,
                                      args=self.args, logger=self.logger, collate_fn=collate_fn)
                c.blocks = ["vision_encoder", "task_head"]
            else:
                raise ValueError(f"Unknown modality for MOSEI: {modality}")

            self.clients.append(c)
            self.log(f"Created client {client_id}: {modality}, {len(indices)} samples")

        self.server = FedServer(global_model, self.args, self.logger)

        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False,
                                     num_workers=0, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False,
                                      num_workers=0, collate_fn=collate_fn)

        self.log("MOSEIFedTrainer setup completed.")

    def train_round(self):
        # dispatch global
        global_params = self.server.get_global_params()
        for c in self.clients:
            c.set_model_params(global_params)

        # local train
        losses = []
        for c in self.clients:
            for _ in range(self.args.local_epochs):
                loss, _ = c.local_train(criterion=None)
            losses.append(loss)

        # aggregate
        if self.args.method == "fedavg":
            self.server.fedavg_aggregate(self.clients)
        elif self.args.method == "fedprox":
            self.server.fedprox_aggregate(self.clients, mu=getattr(self.args, "mu", 0.01))
        elif self.args.method == "fedmsplit_lite":
            self.server.blockwise_aggregate(self.clients, weight_by=getattr(self.args, "fedmsplit_weight_by", "num_samples"))
        else:
            raise ValueError(f"MOSEIFedTrainer does not support method: {self.args.method}")

        return {"loss": sum(losses) / max(1, len(losses))}

    def evaluate(self, test=False):
        loader = self.test_loader if test else self.val_loader
        # criterion
        return self.server.evaluate(loader, criterion=None)

    def get_model_state(self):
        return self.server.get_global_params()

    def set_model_state(self, state_dict):
        self.server.global_model.load_state_dict(state_dict, strict=True)