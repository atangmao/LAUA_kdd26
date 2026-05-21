# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from transformers import AutoModel, AutoTokenizer

def masked_mean(x, mask):
    """
    x: (B,T,D)
    mask: (B,T) bool, True=valid
    return: (B,D)
    """
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(x.device).float().unsqueeze(-1)  # (B,T,1)
    s = (x * m).sum(dim=1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return s / denom


class MOSEIRegressor(nn.Module):
    """
    Forward:
      pred = model(
        text_inputs= dict(...) or list[str] or None,
        audio=(B,Ta,35) or None,
        vision=(B,Tv,74) or None,
        audio_mask=(B,Ta) bool or None,
        vision_mask=(B,Tv) bool or None
      ) -> (B,)
    """
    def __init__(
        self,
        bert_name="/home/weiyi/Desktop/FL_POE/bert-base-uncased",
        audio_dim=35,
        vision_dim=74,
        hidden_dim=256,
        dropout=0.1,
        freeze_bert=False,
    ):
        super().__init__()
        if AutoModel is None:
            raise ImportError("Please install transformers: pip install transformers")

        self.text_encoder = AutoModel.from_pretrained(bert_name, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name, local_files_only=True)

        if freeze_bert:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)

        text_dim = self.text_encoder.config.hidden_size

        self.audio_mlp = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.vision_mlp = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    @torch.no_grad()
    def encode_text(self, texts, device):
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, text_inputs=None, text_cls=None, audio=None, vision=None, audio_mask=None, vision_mask=None):
        # --- text ---
        text_feat = None
        if text_inputs is not None:
            if isinstance(text_inputs, (list, tuple)):
                enc = self.tokenizer(
                    list(text_inputs),
                    padding=True, truncation=True, max_length=128,
                    return_tensors="pt"
                )
                enc = {k: v.to(next(self.parameters()).device) for k, v in enc.items()}
                out = self.text_encoder(**enc)
                cls = out.last_hidden_state[:, 0]  # (B, H)
                text_feat = self.text_mlp(cls)
            else:
                out = self.text_encoder(**text_inputs)
                cls = out.last_hidden_state[:, 0]
                text_feat = self.text_mlp(cls)
        elif text_cls is not None:
            #
            text_feat = self.text_mlp(text_cls)

        # --- audio ---
        if audio is None:
            audio_feat = None
        else:
            a = masked_mean(audio, audio_mask)  # (B,35)
            audio_feat = self.audio_mlp(a)

        # --- vision ---
        if vision is None:
            vision_feat = None
        else:
            v = masked_mean(vision, vision_mask)  # (B,74)
            vision_feat = self.vision_mlp(v)

        #
        dev = next(self.parameters()).device
        if text_feat is not None:
            B = text_feat.size(0)
        elif audio_feat is not None:
            B = audio_feat.size(0)
        elif vision_feat is not None:
            B = vision_feat.size(0)
        else:
            raise ValueError("All modalities are None.")

        Ht = self.text_mlp[-2].out_features
        Ha = self.audio_mlp[-2].out_features
        Hv = self.vision_mlp[-2].out_features

        if text_feat is None:
            text_feat = torch.zeros(B, Ht, device=dev)
        if audio_feat is None:
            audio_feat = torch.zeros(B, Ha, device=dev)
        if vision_feat is None:
            vision_feat = torch.zeros(B, Hv, device=dev)

        x = torch.cat([text_feat, audio_feat, vision_feat], dim=-1)
        y = self.fusion(x).squeeze(-1)  # (B,)
        return y

    def encode_modalities(self, text_inputs=None, text_cls=None, audio=None, vision=None, audio_mask=None,
                          vision_mask=None):
        # --- text ---
        text_feat = None
        if text_inputs is not None:
            out = self.text_encoder(**text_inputs)
            cls = out.last_hidden_state[:, 0]
            text_feat = self.text_mlp(cls)
        elif text_cls is not None:
            text_feat = self.text_mlp(text_cls)

        # --- audio ---
        audio_feat = None if audio is None else self.audio_mlp(masked_mean(audio, audio_mask))

        # --- vision ---
        vision_feat = None if vision is None else self.vision_mlp(masked_mean(vision, vision_mask))

        dev = next(self.parameters()).device
        H = self.audio_mlp[-2].out_features

        if text_feat is not None:
            B = text_feat.size(0)
        elif audio_feat is not None:
            B = audio_feat.size(0)
        elif vision_feat is not None:
            B = vision_feat.size(0)
        else:
            raise ValueError("All modalities are None.")

        if text_feat is None:  text_feat = torch.zeros(B, H, device=dev)
        if audio_feat is None: audio_feat = torch.zeros(B, H, device=dev)
        if vision_feat is None: vision_feat = torch.zeros(B, H, device=dev)

        fused = torch.cat([text_feat, audio_feat, vision_feat], dim=-1)
        return {"text": text_feat, "audio": audio_feat, "vision": vision_feat, "fused": fused}