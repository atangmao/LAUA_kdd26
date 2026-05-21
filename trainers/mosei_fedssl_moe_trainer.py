# -*- coding: utf-8 -*-
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from trainers.mosei_base_trainer import MOSEIBaseTrainer


@torch.no_grad()
def _public_mse_of_model(model, public_loader, device):
    """ """
    model.eval().to(device)
    mse_sum, n = 0.0, 0

    for batch in public_loader:
        texts = batch.get("texts", None)
        audio = batch.get("audio", None)
        vision = batch.get("vision", None)
        audio_mask = batch.get("audio_mask", None)
        vision_mask = batch.get("vision_mask", None)
        y = batch["labels"].to(device)
        if y.dim() == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)

        text_inputs = None
        if texts is not None and hasattr(model, "encode_text"):
            text_inputs = model.encode_text(texts, device=device)
        else:
            text_inputs = texts

        if audio is not None: audio = audio.to(device)
        if vision is not None: vision = vision.to(device)
        if audio_mask is not None: audio_mask = audio_mask.to(device)
        if vision_mask is not None: vision_mask = vision_mask.to(device)

        pred = model(text_inputs=text_inputs, audio=audio, vision=vision,
                     audio_mask=audio_mask, vision_mask=vision_mask)
        if pred.dim() == 2 and pred.size(-1) == 1:
            pred = pred.squeeze(-1)

        diff = pred - y
        mse_sum += (diff * diff).sum().item()
        n += y.size(0)

    return mse_sum / max(1, n)


def _moe_aggregate_state_dicts(client_states, weights):
    """ """
    w = torch.tensor(weights, dtype=torch.float64)
    w = (w / (w.sum() + 1e-12)).tolist()

    agg = copy.deepcopy(client_states[0])
    for k in agg.keys():
        t = None
        for i, sd in enumerate(client_states):
            x = sd[k].float()
            t = (w[i] * x) if t is None else (t + w[i] * x)
        agg[k] = t.to(dtype=client_states[0][k].dtype)
    return agg


class MOSEIFedSSLMoeTrainer(MOSEIBaseTrainer):
    """

    """

    def setup(self):
        super().setup()

        pub_size = min(int(getattr(self.args, "public_size", 4096)), len(self.train_dataset))
        pub_bs = int(getattr(self.args, "public_batch_size", max(64, self.args.batch_size)))
        g = torch.Generator().manual_seed(self.args.seed)
        pub_idx = torch.randperm(len(self.train_dataset), generator=g)[:pub_size].tolist()
        public_dataset = Subset(self.train_dataset, pub_idx)

        self.public_loader = DataLoader(
            public_dataset,
            batch_size=pub_bs,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn,
        )

    def train_round(self):
        # 1) dispatch global
        gp = self.server.get_global_params()
        for c in self.clients:
            c.set_model_params(gp)

        # 2) local train
        losses, uncs = [], []
        for c in self.clients:
            for _ in range(self.args.local_epochs):
                loss, unc = c.local_train(criterion=None)
            losses.append(loss)
            uncs.append(unc)

        # 3) public loss gate
        public_losses = []
        for c in self.clients:
            pl = _public_mse_of_model(c.model, self.public_loader, self.device)
            public_losses.append(float(pl))

        tau = float(getattr(self.args, "moe_tau", 3.0))
        gate = torch.softmax(-tau * torch.tensor(public_losses, dtype=torch.float32), dim=0).tolist()

        if self.logger:
            self.logger.info(f"[MOE] public_losses={public_losses}")
            self.logger.info(f"[MOE] gate_weights={gate}")

        # 4) aggregate by gate
        client_states = [c.get_model_params() for c in self.clients]
        new_sd = _moe_aggregate_state_dicts(client_states, gate)
        self.server.global_model.load_state_dict(new_sd, strict=True)

        return {
            "loss": sum(losses) / max(1, len(losses)),
            "uncertainty": sum(uncs) / max(1, len(uncs)),
            "public_loss": sum(public_losses) / max(1, len(public_losses)),
        }