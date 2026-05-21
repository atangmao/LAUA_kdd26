# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

from trainers.mosei_base_trainer import MOSEIBaseTrainer


def nt_xent_loss(z1, z2, temperature=0.07):
    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)              # [2B,D]
    logits = (z @ z.t()) / temperature          # [2B,2B]
    neg = torch.finfo(logits.dtype).min
    logits.fill_diagonal_(neg)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device)
    ], dim=0)
    return F.cross_entropy(logits, labels)


def aug_seq(x, dropout_p=0.2, noise_std=0.0):
    x = F.dropout(x, p=dropout_p, training=True)
    if noise_std and noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x


def _slice_any(v, sl):
    if v is None:
        return None
    if torch.is_tensor(v):
        return v[sl]
    if isinstance(v, (list, tuple)):
        return v[sl]
    # 标量/其他：原样返回（一般不该出现在 batch 维度字段里）
    return v


def _microbatches_from_batch(batch, micro_bs):
    # 用 labels 的 batch 维度当作 B
    y = batch["labels"]
    B = y.size(0) if torch.is_tensor(y) else len(y)
    for s in range(0, B, micro_bs):
        sl = slice(s, min(B, s + micro_bs))
        mb = {}
        for k, v in batch.items():
            mb[k] = _slice_any(v, sl)
        yield mb


class MOSEIFedMMSSLTrainer(MOSEIBaseTrainer):
    """
    MOSEI FedMM-SSL (unimodal: MSE + InfoNCE, multimodal: MSE)
    """

    def train_round(self):
        gp = self.server.get_global_params()
        for c in self.clients:
            c.set_model_params(gp)

        # --- hyperparams ---
        T = float(getattr(self.args, "temperature", 0.07))
        p = float(getattr(self.args, "ssl_dropout_p", 0.2))
        noise = float(getattr(self.args, "ssl_noise_std", 0.0))
        lam = float(getattr(self.args, "ssl_weight", 1.0))
        reg_w = float(getattr(self.args, "ssl_reg_weight", 1.0))

        # AMP + micro-batch
        use_amp = bool(getattr(self.args, "amp", True))
        micro_bs = int(getattr(self.args, "micro_bs", 16))  #
        #
        #
        mse = torch.nn.MSELoss()

        losses, uncs = [], []

        for c in self.clients:
            ct = getattr(c, "client_type", "")

            #
            if ct == "mosei_all":
                for _ in range(self.args.local_epochs):
                    loss, unc = c.local_train(criterion=None)
                losses.append(loss)
                uncs.append(unc)
                continue

            #
            c.model.train().to(self.device)
            opt = c.optimizer
            scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))

            total_loss, total_unc, nb = 0.0, 0.0, 0

            for _ in range(self.args.local_epochs):
                for batch in c.train_loader:
                    #
                    y_full = batch["labels"]
                    B_full = y_full.size(0)

                    opt.zero_grad(set_to_none=True)

                    #
                    batch_loss_sum = 0.0
                    batch_ssl_sum = 0.0
                    batch_n = 0

                    for mb in _microbatches_from_batch(batch, micro_bs=micro_bs):
                        texts = mb.get("texts", None)
                        audio = mb.get("audio", None)
                        vision = mb.get("vision", None)
                        audio_mask = mb.get("audio_mask", None)
                        vision_mask = mb.get("vision_mask", None)

                        y = mb["labels"].to(self.device)
                        if y.dim() == 2 and y.size(-1) == 1:
                            y = y.squeeze(-1)

                        #
                        text_inputs = None
                        if texts is not None and hasattr(c.model, "encode_text"):
                            text_inputs = c.model.encode_text(texts, device=self.device)
                        else:
                            text_inputs = texts

                        if audio is not None: audio = audio.to(self.device)
                        if vision is not None: vision = vision.to(self.device)
                        if audio_mask is not None: audio_mask = audio_mask.to(self.device)
                        if vision_mask is not None: vision_mask = vision_mask.to(self.device)

                        #
                        micro_scale = float(y.size(0)) / float(B_full)

                        with torch.cuda.amp.autocast(enabled=(use_amp and torch.cuda.is_available())):
                            # --- regression ---
                            pred = c.model(
                                text_inputs=text_inputs,
                                audio=audio, vision=vision,
                                audio_mask=audio_mask, vision_mask=vision_mask
                            )
                            if pred.dim() == 2 and pred.size(-1) == 1:
                                pred = pred.squeeze(-1)
                            loss_reg = mse(pred, y)

                            # --- contrastive (2 views) ---
                            if ct == "mosei_text":
                                #  forward
                                e1 = c.model.encode_modalities(text_inputs=text_inputs)["text"]
                                e2 = c.model.encode_modalities(text_inputs=text_inputs)["text"]
                            elif ct == "mosei_audio":
                                a1 = aug_seq(audio, dropout_p=p, noise_std=noise)
                                a2 = aug_seq(audio, dropout_p=p, noise_std=noise)
                                e1 = c.model.encode_modalities(audio=a1, audio_mask=audio_mask)["audio"]
                                e2 = c.model.encode_modalities(audio=a2, audio_mask=audio_mask)["audio"]
                            elif ct == "mosei_vision":
                                v1 = aug_seq(vision, dropout_p=p, noise_std=noise)
                                v2 = aug_seq(vision, dropout_p=p, noise_std=noise)
                                e1 = c.model.encode_modalities(vision=v1, vision_mask=vision_mask)["vision"]
                                e2 = c.model.encode_modalities(vision=v2, vision_mask=vision_mask)["vision"]
                            else:
                                e1 = e2 = None

                            loss_ssl = nt_xent_loss(e1, e2, temperature=T) if e1 is not None else (loss_reg * 0.0)

                            loss = (reg_w * loss_reg + lam * loss_ssl) * micro_scale

                        scaler.scale(loss).backward()

                        #
                        batch_loss_sum += float((reg_w * loss_reg + lam * loss_ssl).detach().item()) * y.size(0)
                        batch_ssl_sum += float(loss_ssl.detach().item()) * y.size(0)
                        batch_n += int(y.size(0))

                        #
                        del pred, loss_reg, loss_ssl, loss, text_inputs, audio, vision, audio_mask, vision_mask


                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(c.model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()

                    total_loss += batch_loss_sum / max(1, batch_n)
                    total_unc += batch_ssl_sum / max(1, batch_n)
                    nb += 1

                    #
                    # torch.cuda.empty_cache()

            avg_loss = total_loss / max(1, nb)
            avg_unc = total_unc / max(1, nb)

            c.accumulated_uncertainty = avg_unc
            losses.append(avg_loss)
            uncs.append(avg_unc)

        self.server.fedavg_aggregate(self.clients)

        return {
            "loss": sum(losses) / max(1, len(losses)),
            "uncertainty": sum(uncs) / max(1, len(uncs)),
        }