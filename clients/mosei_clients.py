# -*- coding: utf-8 -*-
import torch
from clients.base_client import BaseClient


class _MOSEIBaseClient(BaseClient):
    """

    """

    def __init__(self, client_id, client_type, model, train_dataset, val_dataset=None,
                 args=None, logger=None, collate_fn=None):
        super().__init__(
            client_id=client_id,
            client_type=client_type,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            logger=logger,
            collate_fn=collate_fn
        )
        self.accumulated_uncertainty = 0.0

    def get_uncertainty(self):
        return self.accumulated_uncertainty


class MOSEIAllClient(_MOSEIBaseClient):
    def __init__(self, client_id, model, train_dataset, val_dataset=None, args=None, logger=None, collate_fn=None):
        super().__init__(client_id, "mosei_all", model, train_dataset, val_dataset, args, logger, collate_fn)

    def local_train(self, criterion=None):
        self.model.train()
        mse = torch.nn.MSELoss()
        total_loss, n = 0.0, 0

        for batch in self.train_loader:
            if "text_cls" not in batch or batch["text_cls"] is None or not torch.is_tensor(batch["text_cls"]):
                raise RuntimeError("BAD text_cls before model call")

            text_cls = batch["text_cls"].to(self.device)
            y = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(text_cls=text_cls, audio=None, vision=None, audio_mask=None, vision_mask=None)
            loss = mse(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            n += bs

        return total_loss / max(n, 1), 0.0

    def local_eval(self, criterion=None):
        if self.val_loader is None:
            return None, None
        self.model.eval()
        mse = torch.nn.MSELoss(reduction="sum")
        mae_sum, mse_sum, n = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                text_cls = batch["text_cls"].to(self.device)
                audio = batch["audio"].to(self.device)
                vision = batch["vision"].to(self.device)
                audio_mask = batch["audio_mask"].to(self.device)
                vision_mask = batch["vision_mask"].to(self.device)
                y = batch["labels"].to(self.device)

                pred = self.model(
                    text_cls=text_cls,
                    audio=audio, vision=vision,
                    audio_mask=audio_mask, vision_mask=vision_mask
                )
                mse_sum += mse(pred, y).item()
                mae_sum += (pred - y).abs().sum().item()
                n += y.size(0)

        return mse_sum / max(n, 1), mae_sum / max(n, 1)


class MOSEITextClient(_MOSEIBaseClient):
    def __init__(self, client_id, model, train_dataset, val_dataset=None, args=None, logger=None, collate_fn=None):
        super().__init__(client_id, "mosei_text", model, train_dataset, val_dataset, args, logger, collate_fn)

    def local_train(self, criterion=None):
        import torch

        # print(f"[HIT] MOSEITextClient.local_train file={__file__} client_id={self.client_id}")
        self.model.train()
        mse = torch.nn.MSELoss()
        total_loss, n = 0.0, 0

        for i, batch in enumerate(self.train_loader):
            # print("[DEBUG] batch keys =", list(batch.keys()))
            # print("[DEBUG] batch['text_cls'] type =", type(batch.get("text_cls", None)))
            # print("[DEBUG] batch['text_cls'] is None? =", batch.get("text_cls", None) is None)
            # if torch.is_tensor(batch.get("text_cls", None)):
            #     print("[DEBUG] text_cls shape =", tuple(batch["text_cls"].shape), "dtype=", batch["text_cls"].dtype)


            if ("text_cls" not in batch) or (batch["text_cls"] is None) or (not torch.is_tensor(batch["text_cls"])):
                raise RuntimeError("BAD text_cls before model call")

            text_cls = batch["text_cls"].to(self.device)
            y = batch["labels"].to(self.device)


            # import inspect
            # print("[DEBUG] model.forward signature =", inspect.signature(self.model.forward))

            pred = self.model(text_cls=text_cls, audio=None, vision=None, audio_mask=None, vision_mask=None)
            loss = mse(pred, y)


            break

        return 0.0, 0.0

    def local_eval(self, criterion=None):
        if self.val_loader is None:
            return None, None
        self.model.eval()
        mse = torch.nn.MSELoss(reduction="sum")
        mae_sum, mse_sum, n = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                text_cls = batch["text_cls"].to(self.device)
                y = batch["labels"].to(self.device)
                pred = self.model(text_cls=text_cls, audio=None, vision=None, audio_mask=None, vision_mask=None)
                mse_sum += mse(pred, y).item()
                mae_sum += (pred - y).abs().sum().item()
                n += y.size(0)

        return mse_sum / max(n, 1), mae_sum / max(n, 1)


class MOSEIAudioClient(_MOSEIBaseClient):
    def __init__(self, client_id, model, train_dataset, val_dataset=None, args=None, logger=None, collate_fn=None):
        super().__init__(client_id, "mosei_audio", model, train_dataset, val_dataset, args, logger, collate_fn)

    def local_train(self, criterion=None):
        self.model.train()
        mse = torch.nn.MSELoss()
        total_loss, n = 0.0, 0

        for batch in self.train_loader:
            audio = batch["audio"].to(self.device)
            audio_mask = batch["audio_mask"].to(self.device)
            y = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(text_cls=None, text_inputs=None, audio=audio, vision=None, audio_mask=audio_mask, vision_mask=None)
            loss = mse(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            n += bs

        return total_loss / max(n, 1), 0.0

    def local_eval(self, criterion=None):
        if self.val_loader is None:
            return None, None
        self.model.eval()
        mse = torch.nn.MSELoss(reduction="sum")
        mae_sum, mse_sum, n = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                audio = batch["audio"].to(self.device)
                audio_mask = batch["audio_mask"].to(self.device)
                y = batch["labels"].to(self.device)
                pred = self.model(text_cls=None, text_inputs=None, audio=audio, vision=None, audio_mask=audio_mask, vision_mask=None)
                mse_sum += mse(pred, y).item()
                mae_sum += (pred - y).abs().sum().item()
                n += y.size(0)

        return mse_sum / max(n, 1), mae_sum / max(n, 1)


class MOSEIVisionClient(_MOSEIBaseClient):
    def __init__(self, client_id, model, train_dataset, val_dataset=None, args=None, logger=None, collate_fn=None):
        super().__init__(client_id, "mosei_vision", model, train_dataset, val_dataset, args, logger, collate_fn)

    def local_train(self, criterion=None):
        self.model.train()
        mse = torch.nn.MSELoss()
        total_loss, n = 0.0, 0

        for batch in self.train_loader:
            vision = batch["vision"].to(self.device)
            vision_mask = batch["vision_mask"].to(self.device)
            y = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(text_cls=None, text_inputs=None, audio=None, vision=vision, audio_mask=None, vision_mask=vision_mask)
            loss = mse(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            n += bs

        return total_loss / max(n, 1), 0.0

    def local_eval(self, criterion=None):
        if self.val_loader is None:
            return None, None
        self.model.eval()
        mse = torch.nn.MSELoss(reduction="sum")
        mae_sum, mse_sum, n = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                vision = batch["vision"].to(self.device)
                vision_mask = batch["vision_mask"].to(self.device)
                y = batch["labels"].to(self.device)
                pred = self.model(text_cls=None, text_inputs=None, audio=None, vision=vision, audio_mask=None, vision_mask=vision_mask)
                mse_sum += mse(pred, y).item()
                mae_sum += (pred - y).abs().sum().item()
                n += y.size(0)

        return mse_sum / max(n, 1), mae_sum / max(n, 1)