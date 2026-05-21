# -*- coding: utf-8 -*-
"""
MOSEI dataset loader for pre-extracted high-level features (.pkl)


- ：dict[vid][seg_id] = (text:str, audio:np.ndarray(T,35), vision:np.ndarray(T,74), label:np.ndarray)
- ：dict[vid][seg_id] = (text:str, audio:np.ndarray(T,35), vision:np.ndarray(T,74), label:np.ndarray, text_cls:np.ndarray(768,))


：
- MOSEIPKLDataset
- mosei_collate_fn
- build_mosei_splits
"""

import os
import pickle
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MOSEIPKLDataset(Dataset):
    def __init__(self, pkl_path: str, require_text_cls: bool = True):
        super().__init__()
        self.pkl_path = pkl_path
        self.require_text_cls = require_text_cls

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"MOSEI pkl not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            data_dict = pickle.load(f)

        self.samples: List[Dict[str, Any]] = []
        for vid, seg_dict in data_dict.items():
            for seg_id, tpl in seg_dict.items():
                if not isinstance(tpl, (tuple, list)):
                    raise ValueError(f"Unexpected sample type: {type(tpl)} at {vid}/{seg_id}")

                if len(tpl) == 4:
                    text, audio, vision, label = tpl
                    text_cls = None
                elif len(tpl) == 5:
                    text, audio, vision, label, text_cls = tpl
                else:
                    raise ValueError(f"Unexpected tpl len={len(tpl)} at {vid}/{seg_id}")

                if self.require_text_cls and text_cls is None:
                    raise ValueError(
                        f"require_text_cls=True but text_cls is None at {vid}/{seg_id}. "
                        f"Please use *_cached.pkl (5-tuple) or run the caching script."
                    )

                audio = np.asarray(audio, dtype=np.float32)    # (T,35)
                vision = np.asarray(vision, dtype=np.float32)  # (T,74)
                y = float(np.asarray(label).reshape(-1)[0])

                if text_cls is not None:
                    text_cls = np.asarray(text_cls, dtype=np.float32).reshape(-1)
                    if text_cls.shape[0] != 768:
                        raise ValueError(f"text_cls dim must be 768, got {text_cls.shape} at {vid}/{seg_id}")

                self.samples.append({
                    "vid": vid,
                    "seg_id": seg_id,
                    "text": text if text is not None else "",
                    "audio": audio,
                    "vision": vision,
                    "label": y,
                    "text_cls": text_cls,  # np.ndarray(768,) or None
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "vid": s["vid"],
            "seg_id": s["seg_id"],
            "text": s["text"],  #
            "text_cls": (
                torch.from_numpy(s["text_cls"]).float()
                if s.get("text_cls", None) is not None
                else None
            ),  # (768,) or None
            "audio": torch.from_numpy(s["audio"]).float(),   # (T,35)
            "vision": torch.from_numpy(s["vision"]).float(), # (T,74)
            "label": torch.tensor(s["label"], dtype=torch.float32),  # scalar
            "sample_idx": idx,
        }


def mosei_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads audio/vision sequences and returns masks.

    """
    vids = [b.get("vid", "") for b in batch]
    seg_ids = [b.get("seg_id", "") for b in batch]
    texts = [b.get("text", "") or "" for b in batch]
    sample_idx = torch.tensor([int(b.get("sample_idx", -1)) for b in batch], dtype=torch.long)

    labels = torch.stack([b["label"] for b in batch], dim=0)  # (B,)

    # --- text_cls ---
    text_cls_list = [b.get("text_cls", None) for b in batch]
    if any(x is None for x in text_cls_list):
        raise ValueError("Found None text_cls in batch. Please ensure you load *_cached.pkl (5-tuple).")
    text_cls = torch.stack(text_cls_list, dim=0)  # (B,768)

    # --- audio/vision ---
    audio_seqs = [b["audio"] for b in batch]    # list[(T,35)]
    vision_seqs = [b["vision"] for b in batch]  # list[(T,74)]

    audio_lens = torch.tensor([x.shape[0] for x in audio_seqs], dtype=torch.long)
    vision_lens = torch.tensor([x.shape[0] for x in vision_seqs], dtype=torch.long)

    audio_pad = pad_sequence(audio_seqs, batch_first=True, padding_value=0.0)    # (B,Ta,35)
    vision_pad = pad_sequence(vision_seqs, batch_first=True, padding_value=0.0)  # (B,Tv,74)

    audio_mask = (torch.arange(audio_pad.size(1))[None, :] < audio_lens[:, None])     # (B,Ta) bool
    vision_mask = (torch.arange(vision_pad.size(1))[None, :] < vision_lens[:, None])  # (B,Tv) bool

    return {
        "vids": vids,
        "seg_ids": seg_ids,
        "texts": texts,               #
        "text_cls": text_cls,         # (B,768)
        "audio": audio_pad,           # (B,Ta,35)
        "vision": vision_pad,         # (B,Tv,74)
        "audio_lens": audio_lens,
        "vision_lens": vision_lens,
        "audio_mask": audio_mask,
        "vision_mask": vision_mask,
        "labels": labels,             # (B,)
        "sample_idx": sample_idx,
    }


def build_mosei_splits(data_root: str, use_cached_text_cls: bool = True):
    """
    Returns: train_ds, val_ds, test_ds

    """
    if use_cached_text_cls:
        train_p = os.path.join(data_root, "train_cached.pkl")
        dev_p = os.path.join(data_root, "dev_cached.pkl")
        test_p = os.path.join(data_root, "test_cached.pkl")
    else:
        train_p = os.path.join(data_root, "train.pkl")
        dev_p = os.path.join(data_root, "dev.pkl")
        test_p = os.path.join(data_root, "test.pkl")

    train_ds = MOSEIPKLDataset(train_p, require_text_cls=use_cached_text_cls)
    val_ds = MOSEIPKLDataset(dev_p, require_text_cls=use_cached_text_cls)
    test_ds = MOSEIPKLDataset(test_p, require_text_cls=use_cached_text_cls)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    base = "/home/userdata/data/cmumosei"
    tr, va, te = build_mosei_splits(base, use_cached_text_cls=True)
    b = mosei_collate_fn([tr[0], tr[1]])
    print("text_cls:", b["text_cls"].shape)
    print("audio:", b["audio"].shape, "vision:", b["vision"].shape, "labels:", b["labels"].shape)