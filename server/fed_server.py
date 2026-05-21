# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""

import copy
from collections import OrderedDict

import torch
import numpy as np
import torch.nn.functional as F


class FedServer:
    """  """

    def __init__(self, global_model, args, logger=None):
        self.global_model = global_model
        self.args = args
        self.logger = logger
        self.device = args.device
        self.round_history = []

    def get_global_params(self):
        return copy.deepcopy(self.global_model.state_dict())

    # ------------------------- Basic Aggregation -------------------------

    def aggregate(self, client_params_list, client_weights):
        """  """
        weights = np.array(client_weights, dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)

        aggregated_params = copy.deepcopy(client_params_list[0])

        for key in aggregated_params.keys():
            aggregated_params[key] = torch.zeros_like(
                aggregated_params[key], dtype=torch.float32
            )
            for i, params in enumerate(client_params_list):
                aggregated_params[key] += weights[i] * params[key].float()
            aggregated_params[key] = aggregated_params[key].to(client_params_list[0][key].dtype)

        self.global_model.load_state_dict(aggregated_params)
        return aggregated_params

    def fedavg_aggregate(self, clients):
        client_params_list = [c.get_model_params() for c in clients]
        client_weights = [c.get_num_samples() for c in clients]
        return self.aggregate(client_params_list, client_weights)

    def uncertainty_aggregate(self, clients, eps=1e-8):
        """   """
        client_params_list = [c.get_model_params() for c in clients]

        weights = []
        uncerts = []
        for c in clients:
            n_samples = c.get_num_samples()
            uncertainty = float(c.get_uncertainty()) + eps
            uncerts.append(float(c.get_uncertainty()))
            weights.append(n_samples / uncertainty)

        if self.logger:
            self.logger.info(f'unc: {uncerts}')
            self.logger.info(f'weights: {weights}')

        return self.aggregate(client_params_list, weights)

    def fedprox_aggregate(self, clients, mu=0.01):
        #
        return self.fedavg_aggregate(clients)

    def selective_aggregate(self, clients, threshold=0.5):
        selected_clients = [c for c in clients if c.get_uncertainty() < threshold]
        if len(selected_clients) == 0:
            if self.logger:
                self.logger.warning('no state, use all clients')
            selected_clients = clients
        return self.uncertainty_aggregate(selected_clients)

    # ------------------------- FedMSplit-lite: Blockwise Selective Aggregation -------------------------

    def _infer_blocks_from_state_dict(self, state_dict):
        #
        PREFIXES = [
            # ---- MOSEI ----
            ("text_encoder", "text_encoder."),
            ("text_encoder", "text_mlp."),
            ("audio_encoder", "audio_mlp."),
            ("vision_encoder", "vision_mlp."),
            ("task_head", "fusion."),

            # ----  Flickr/COCO ----
            ("image_encoder", "image_encoder."),
            ("text_encoder", "text_encoder."),
            ("task_head", "task_head."),
        ]

        key2block = {}
        for k in state_dict.keys():
            blk = None
            for b, pref in PREFIXES:
                if k.startswith(pref):
                    blk = b
                    break
            key2block[k] = blk
        return key2block

    def blockwise_aggregate(self, clients, weight_by="num_samples"):
        """

        """
        global_sd = self.global_model.state_dict()
        key2block = self._infer_blocks_from_state_dict(global_sd)

        #
        client_states = [(c, c.get_model_params()) for c in clients]

        new_sd = OrderedDict()

        for k, g in global_sd.items():
            blk = key2block.get(k, None)

            #
            if blk is None:
                new_sd[k] = g
                continue

            owners = [(c, sd) for (c, sd) in client_states if hasattr(c, "blocks") and (blk in c.blocks)]

            #
            if len(owners) == 0:
                new_sd[k] = g
                continue

            #
            weights = []
            for (c, _) in owners:
                if weight_by == "uniform":
                    w = 1.0
                elif weight_by == "num_samples":
                    w = float(c.get_num_samples())
                elif weight_by == "uncertainty_inv":
                    u = float(c.get_uncertainty())
                    w = 1.0 / (u + 1e-6)
                else:
                    raise ValueError(f"Unknown weight_by: {weight_by}")
                weights.append(w)

            wsum = sum(weights) + 1e-12

            #
            agg = None
            for w, (c, sd) in zip(weights, owners):
                t = sd[k].to(g.device)
                coef = float(w / wsum)
                agg = (coef * t) if agg is None else (agg + coef * t)

            new_sd[k] = agg.to(dtype=g.dtype)

        self.global_model.load_state_dict(new_sd, strict=True)
        return self.get_global_params()

    # ------------------------- Eval -------------------------

    def _extract_image_ids(self, batch):
        """
        Flickr30kDataset/collate_fn  batch['id']  image_id
        """
        if 'image_id' in batch:
            x = batch['image_id']
        elif 'id' in batch:
            x = batch['id']
        else:
            raise KeyError("Batch no image_id/id ")

        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        return list(x)

    def evaluate(self, test_loader, criterion):
        self.global_model.eval()
        self.global_model.to(self.device)

        # -------------------- MOSEI / regression --------------------
        # -------------------- MOSEI / regression --------------------
        if getattr(self.args, "task", None) == "regression":
            mse_sum = 0.0
            mae_sum = 0.0
            n = 0

            with torch.no_grad():
                for batch in test_loader:
                    # mosei_collate_fn
                    text_cls = batch.get("text_cls", None)
                    audio = batch.get("audio", None)
                    vision = batch.get("vision", None)
                    audio_mask = batch.get("audio_mask", None)
                    vision_mask = batch.get("vision_mask", None)
                    y = batch.get("labels", None)

                    if y is None:
                        raise KeyError("Regression task expects batch['labels'].")
                    if text_cls is None:
                        raise KeyError("Regression task (cached text) expects batch['text_cls'].")

                    y = y.to(self.device)
                    text_cls = text_cls.to(self.device)

                    if audio is not None:
                        audio = audio.to(self.device)
                    if vision is not None:
                        vision = vision.to(self.device)
                    if audio_mask is not None:
                        audio_mask = audio_mask.to(self.device)
                    if vision_mask is not None:
                        vision_mask = vision_mask.to(self.device)

                    #  text_cls， text_inputs
                    pred = self.global_model(
                        text_cls=text_cls,
                        text_inputs=None,
                        audio=audio, vision=vision,
                        audio_mask=audio_mask, vision_mask=vision_mask
                    )
                    if pred.dim() == 2 and pred.size(-1) == 1:
                        pred = pred.squeeze(-1)

                    diff = pred - y
                    mse_sum += (diff * diff).sum().item()
                    mae_sum += diff.abs().sum().item()
                    n += y.size(0)

            mse = mse_sum / max(n, 1)
            mae = mae_sum / max(n, 1)

            metrics = {
                "loss": mse,
                "mae": mae,
                "metric": -mse,
            }
            return metrics

        # --------------------  --------------------
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        image_id_list = []
        img_proj_list = []
        txt_proj_list = []

        with torch.no_grad():
            for batch in test_loader:
                img_feat = batch['image'].to(self.device)
                txt_feat = batch['text'].to(self.device)

                target = batch['label'].to(self.device) if self.args.task == 'classification' else None

                output, mu_joint, log_var_joint, modality_outputs = self.global_model(
                    img_feat=img_feat,
                    txt_feat=txt_feat,
                    sample=False
                )

                if self.args.task == 'retrieval':
                    img_proj = self.global_model.task_head(modality_outputs['image']['mu'])
                    txt_proj = self.global_model.task_head(modality_outputs['text']['mu'])

                    image_ids = self._extract_image_ids(batch)
                    if len(image_ids) != img_proj.size(0):
                        raise ValueError(
                            f"image_ids length({len(image_ids)}) differ with batch({img_proj.size(0)})"
                        )

                    image_id_list.extend(image_ids)
                    img_proj_list.append(img_proj.detach().cpu())
                    txt_proj_list.append(txt_proj.detach().cpu())
                else:
                    img_proj, txt_proj = None, None

                loss, _ = criterion(
                    output, target, mu_joint, log_var_joint, modality_outputs,
                    img_proj=img_proj,
                    txt_proj=txt_proj
                )

                total_loss += float(loss.item()) * img_feat.size(0)

                if self.args.task == 'classification':
                    pred = output.argmax(dim=1)
                    total_correct += (pred == target).sum().item()

                total_samples += img_feat.size(0)

        metrics = {'loss': total_loss / max(total_samples, 1)}

        if self.args.task == 'classification':
            metrics['accuracy'] = total_correct / max(total_samples, 1)
            return metrics

        if len(image_id_list) == 0:
            if self.logger:
                self.logger.warning("evaluate: ")
            return metrics

        all_img_proj = torch.cat(img_proj_list, dim=0)
        all_txt_proj = torch.cat(txt_proj_list, dim=0)
        retrieval_metrics = self.compute_retrieval_metrics_flickr30k(
            all_img_proj, all_txt_proj, image_id_list
        )
        metrics.update(retrieval_metrics)
        return metrics

    @torch.no_grad()
    def compute_retrieval_metrics_flickr30k(self, img_features_rep, txt_features, image_ids, ks=(1, 5, 10)):
        """

        """
        if len(image_ids) != txt_features.size(0) or txt_features.size(0) != img_features_rep.size(0):
            raise ValueError(
                f"len: len(image_ids)={len(image_ids)}, "
                f"img={img_features_rep.size(0)}, txt={txt_features.size(0)}"
            )

        device = txt_features.device  #
        img_features_rep = F.normalize(img_features_rep, dim=-1)
        txt_features = F.normalize(txt_features, dim=-1)

        #
        image_id_to_img_index = {}
        uniq_img_feats = []
        image_id_to_text_indices = {}

        for t_idx, img_id in enumerate(image_ids):
            if img_id not in image_id_to_text_indices:
                image_id_to_text_indices[img_id] = []
            image_id_to_text_indices[img_id].append(t_idx)

            if img_id not in image_id_to_img_index:
                image_id_to_img_index[img_id] = len(uniq_img_feats)
                uniq_img_feats.append(img_features_rep[t_idx])

        uniq_img_feats = torch.stack(uniq_img_feats, dim=0).to(device)   # [N, D]
        txt_features = txt_features.to(device)                            # [M, D]

        N = uniq_img_feats.size(0)
        M = txt_features.size(0)

        img_index_to_pos_texts = [None] * N
        for img_id, img_idx in image_id_to_img_index.items():
            img_index_to_pos_texts[img_idx] = image_id_to_text_indices[img_id]

        text_to_pos_img = [None] * M
        for t_idx, img_id in enumerate(image_ids):
            text_to_pos_img[t_idx] = image_id_to_img_index[img_id]

        # 2)
        sim = uniq_img_feats @ txt_features.T  # cosine

        metrics = {}

        # ---------- i2t ----------
        ranks_i2t = np.empty(N, dtype=np.int64)
        sorted_txt = torch.argsort(sim, dim=1, descending=True)  # [N, M]

        for i in range(N):
            pos = set(img_index_to_pos_texts[i])
            order = sorted_txt[i].tolist()
            r = None
            for k, t_idx in enumerate(order):
                if t_idx in pos:
                    r = k
                    break
            ranks_i2t[i] = r if r is not None else (M + 1)

        for k in ks:
            metrics[f"i2t_R@{k}"] = float((ranks_i2t < k).mean())
        metrics["i2t_MRR"] = float((1.0 / (ranks_i2t + 1)).mean())

        # ---------- t2i ----------
        ranks_t2i = np.empty(M, dtype=np.int64)
        sorted_img = torch.argsort(sim.T, dim=1, descending=True)  # [M, N]

        for t in range(M):
            pos_img = text_to_pos_img[t]
            order = sorted_img[t].tolist()
            r = None
            for k, img_idx in enumerate(order):
                if img_idx == pos_img:
                    r = k
                    break
            ranks_t2i[t] = r if r is not None else (N + 1)

        for k in ks:
            metrics[f"t2i_R@{k}"] = float((ranks_t2i < k).mean())
        metrics["t2i_MRR"] = float((1.0 / (ranks_t2i + 1)).mean())

        return metrics