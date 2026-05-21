# -*- coding: utf-8 -*-
import copy
import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from trainers.base_trainer import BaseTrainer
from server.fed_server import FedServer
from data import get_dataset
from models.mosei_model import MOSEIRegressor


class MOSEICentralizedTrainer(BaseTrainer):
    def __init__(self, args, logger=None, train_dataset_override=None):
        super().__init__(args, logger)
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.server = None
        self.train_dataset_override = train_dataset_override

        # for timing/rounds
        self.actual_epochs = 0
        self.total_time = 0.0

    def setup(self):
        self.args.task = "regression"
        train_dataset, val_dataset, test_dataset, collate_fn = get_dataset(self.args)
        if self.train_dataset_override is not None:
            train_dataset = self.train_dataset_override

        local_bert_path = "/home/weiyi/Desktop/FL_POE/bert-base-uncased"
        self.model = MOSEIRegressor(
            bert_name=getattr(self.args, "bert_name", local_bert_path),
            hidden_dim=getattr(self.args, "hidden_dim", 256),
            freeze_bert=getattr(self.args, "freeze_bert", False),
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn
        )

        self.server = FedServer(self.model, self.args, self.logger)

    def train_round(self):
        self.model.train()
        mse = torch.nn.MSELoss()

        total, n = 0.0, 0
        for batch in self.train_loader:
            texts = batch.get("texts", None)
            audio = batch.get("audio", None)
            vision = batch.get("vision", None)
            audio_mask = batch.get("audio_mask", None)
            vision_mask = batch.get("vision_mask", None)
            y = batch["labels"].to(self.device)

            if texts is not None:
                _ = self.model.encode_text(texts, device=self.device)  # optional / keep if needed
            # NOTE: current forward uses text_cls, not text_inputs

            if audio is not None:
                audio = audio.to(self.device)
            if vision is not None:
                vision = vision.to(self.device)
            if audio_mask is not None:
                audio_mask = audio_mask.to(self.device)
            if vision_mask is not None:
                vision_mask = vision_mask.to(self.device)

            text_cls = batch["text_cls"].to(self.device)
            pred = self.model(
                text_cls=text_cls, text_inputs=None, audio=audio, vision=vision,
                audio_mask=audio_mask, vision_mask=vision_mask
            )
            loss = mse(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            bs = y.size(0)
            total += loss.item() * bs
            n += bs

        return {"loss": total / max(1, n)}

    def evaluate(self, test=False):
        loader = self.test_loader if test else self.val_loader
        return self.server.evaluate(loader, criterion=None)

    def get_model_state(self):
        return copy.deepcopy(self.model.state_dict())

    def set_model_state(self, state_dict):
        self.model.load_state_dict(state_dict, strict=True)

    def train(self):
        self.setup()
        start = time.time()

        # centralized "rounds" == local epochs
        epochs = int(getattr(self.args, "epochs", 1))
        for ep in range(epochs):
            self.train_round()
            self.actual_epochs = ep + 1

        self.total_time = time.time() - start
        final_metrics = self.evaluate(test=True)
        return final_metrics, self.actual_epochs, self.total_time


def run_mosei_centralized_training(args, logger):
    t = MOSEICentralizedTrainer(args, logger)
    return t.train()


def run_mosei_partial_centralized_training(args, logger):
    """
    Centralized training but only on the multimodal-client subset of the training set
    induced by FederatedDataSplitter (same seed/partition/mm_ratio/client counts).
    Val/Test remain full.
    """
    from torch.utils.data import Subset
    from data import FederatedDataSplitter, get_dataset

    # 1) load full datasets
    full_train, val_dataset, test_dataset, collate_fn = get_dataset(args)

    # 2) reproduce the federated split (must match FL runs)
    splitter = FederatedDataSplitter(
        num_multimodal_clients=getattr(args, "num_multimodal_clients", 0),
        num_text_clients=getattr(args, "num_text_clients", 0),
        num_audio_clients=getattr(args, "num_audio_clients", 0),
        num_vision_clients=getattr(args, "num_vision_clients", 0),
        num_image_clients=0,
        partition=args.partition,
        alpha=args.alpha,
        seed=args.seed,
        mm_ratio=args.mm_ratio,
    )
    splits = splitter.split(full_train, labels=None)

    # 3) collect multimodal ("both") indices
    mm_indices = []
    for _, info in splits.items():
        if info.get("modality") == "both":
            mm_indices.extend(info.get("indices", []))

    # safety
    mm_indices = sorted(set(mm_indices))
    if len(mm_indices) == 0:
        raise ValueError(
            "partial_centralized got empty multimodal subset. "
            "Check num_multimodal_clients/mm_ratio/partition settings."
        )

    partial_train = Subset(full_train, mm_indices)

    # 4) train centralized on this subset (val/test unchanged inside trainer)
    t = MOSEICentralizedTrainer(args, logger, train_dataset_override=partial_train)
    return t.train()  # returns (final_metrics, actual_epochs, total_time)