# -*- coding: utf-8 -*-
import torch
import time
from torch.utils.data import DataLoader, Subset

from trainers.base_trainer import BaseTrainer
from server.fed_server import FedServer
from data import get_dataset, FederatedDataSplitter

from models.mosei_model import MOSEIRegressor
from clients.mosei_clients import MOSEIAllClient, MOSEITextClient, MOSEIAudioClient, MOSEIVisionClient


class MOSEIBaseTrainer(BaseTrainer):
    """

    """
    def __init__(self, args, logger=None):
        super().__init__(args, logger)
        self.clients = []
        self.server = None
        self.val_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.collate_fn = None

    def setup(self):
        self.log("Setting up MOSEIBaseTrainer...")

        #
        self.args.task = "regression"

        train_dataset, val_dataset, test_dataset, collate_fn = get_dataset(self.args)
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        self.collate_fn = collate_fn

        global_model = MOSEIRegressor(
            bert_name=getattr(self.args, "bert_name", "bert-base-uncased"),
            hidden_dim=getattr(self.args, "hidden_dim", 256),
            freeze_bert=getattr(self.args, "freeze_bert", False),
        )

        splitter = FederatedDataSplitter(
            num_multimodal_clients=getattr(self.args, "num_multimodal_clients", 0),
            num_text_clients=getattr(self.args, "num_text_clients", 0),
            num_audio_clients=getattr(self.args, "num_audio_clients", 0),
            num_vision_clients=getattr(self.args, "num_vision_clients", 0),
            num_image_clients=0,
            partition=self.args.partition,
            alpha=self.args.alpha,
            seed=self.args.seed,
            mm_ratio=self.args.mm_ratio,
        )

        splits = splitter.split(train_dataset, labels=None)
        if self.logger:
            self.logger.info(f"Data split info: {splitter.get_client_info(splits)}")

        self.clients = []
        for client_id, info in splits.items():
            idx = info["indices"]
            modality = info["modality"]   # both/text/audio/vision

            client_train = Subset(train_dataset, idx)

            client_val = None
            if val_dataset is not None:
                client_val = Subset(val_dataset, idx[:max(1, len(idx)//5)])

            if modality == "both":
                c = MOSEIAllClient(client_id, global_model, client_train, client_val,
                                   args=self.args, logger=self.logger, collate_fn=collate_fn)
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
                raise ValueError(f"Unknown modality: {modality}")

            self.clients.append(c)
            self.log(f"Created MOSEI client {client_id}: {modality}, {len(idx)} samples")

        self.server = FedServer(global_model, self.args, self.logger)

        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False,
                                     num_workers=0, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False,
                                      num_workers=0, collate_fn=collate_fn)

        self.log("MOSEIBaseTrainer setup completed.")

    def train(self):
        """
        Generic federated training loop for MOSEI:
        - calls self.setup()
        - loops rounds and calls self.train_round()
        - evaluates every eval_interval
        - early stopping by args.patience/min_delta on val main metric (metrics["metric"])
        - records self.actual_rounds and self.total_time
        """
        self.setup()

        self.actual_rounds = 0
        self.total_time = 0.0

        num_rounds = int(getattr(self.args, "num_rounds", 1))
        eval_interval = int(getattr(self.args, "eval_interval", 1))
        patience = int(getattr(self.args, "patience", 8))
        min_delta = float(getattr(self.args, "min_delta", 0.0))

        best_metric = -float("inf")
        best_round = -1
        bad_rounds = 0

        start_time = time.time()

        for r in range(num_rounds):
            out = self.train_round() or {}
            train_loss = out.get("loss", None)

            # completed one round
            self.actual_rounds = r + 1

            do_eval = (eval_interval > 0) and ((r + 1) % eval_interval == 0)
            if do_eval:
                val_metrics = self.evaluate(test=False) or {}
                val_metric = float(val_metrics.get("metric", -float(val_metrics.get("loss", 0.0))))

                improved = val_metric > (best_metric + min_delta)
                if improved:
                    best_metric = val_metric
                    best_round = r + 1
                    bad_rounds = 0
                else:
                    bad_rounds += 1

                # log line (match your existing style)
                if train_loss is None:
                    self.log(
                        f"Round {r+1}/{num_rounds} - Val Metric: {val_metric:.4f} "
                        f"- Best: {best_metric:.4f} - BadRounds: {bad_rounds}/{patience}"
                    )
                else:
                    self.log(
                        f"Round {r+1}/{num_rounds} - Train Loss: {float(train_loss):.4f} "
                        f"- Val Metric: {val_metric:.4f} - Best: {best_metric:.4f} "
                        f"- BadRounds: {bad_rounds}/{patience}"
                    )

                if bad_rounds >= patience:
                    self.log(
                        f"Early stopping at round {r+1} "
                        f"(best round: {best_round}, best metric: {best_metric:.4f})."
                    )
                    break
            else:
                # optional minimal logging if you want
                if int(getattr(self.args, "log_interval", 1)) > 0 and ((r + 1) % int(self.args.log_interval) == 0):
                    if train_loss is not None:
                        self.log(f"Round {r+1}/{num_rounds} - Train Loss: {float(train_loss):.4f}")

        self.total_time = time.time() - start_time
        self.log(f"Training completed. actual_rounds={self.actual_rounds}, total_time_sec={self.total_time:.2f}")
        return

    def evaluate(self, test=False):
        loader = self.test_loader if test else self.val_loader
        return self.server.evaluate(loader, criterion=None)

    def get_model_state(self):
        return self.server.get_global_params()

    def set_model_state(self, state_dict):
        self.server.global_model.load_state_dict(state_dict, strict=True)