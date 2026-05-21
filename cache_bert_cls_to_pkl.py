# -*- coding: utf-8 -*-
"""

"""

import os
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

BERT_PATH = "/home/weiyi/Desktop/FL_POE/bert-base-uncased"  # your own path
DATA_ROOT = "/home/userdata/data/cmumosei"                  # your mosei pkl path
SPLITS = ["train", "dev", "test"]
MAX_LENGTH = 128
BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_cls_batch(model, tokenizer, texts):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    out = model(**enc)
    cls = out.last_hidden_state[:, 0]  # (B,768)
    return cls.detach().cpu().numpy().astype(np.float32)


def process_one_split(in_path, out_path, model, tokenizer):
    with open(in_path, "rb") as f:
        data_dict = pickle.load(f)

    #
    keys = []
    texts = []
    for vid, seg_dict in data_dict.items():
        for seg_id, tpl in seg_dict.items():
            #
            if isinstance(tpl, (list, tuple)) and len(tpl) == 5:
                continue

            text = tpl[0] if tpl[0] is not None else ""
            keys.append((vid, seg_id))
            texts.append(text)

    print(f"[{os.path.basename(in_path)}] need cache segments: {len(texts)}")

    #
    cls_list = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        cls = extract_cls_batch(model, tokenizer, batch_texts)
        cls_list.append(cls)
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  cached {min(i + len(batch_texts), len(texts))}/{len(texts)}")

    if len(cls_list) > 0:
        cls_all = np.concatenate(cls_list, axis=0)  # (N,768)
    else:
        cls_all = np.zeros((0, 768), dtype=np.float32)

    # 回写
    for idx, (vid, seg_id) in enumerate(keys):
        tpl = data_dict[vid][seg_id]
        if len(tpl) == 5:
            continue
        text, audio, vision, label = tpl
        data_dict[vid][seg_id] = (text, audio, vision, label, cls_all[idx])

    with open(out_path, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"saved -> {out_path}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(BERT_PATH, local_files_only=True).to(DEVICE)
    model.eval()

    for sp in SPLITS:
        in_path = os.path.join(DATA_ROOT, f"{sp}.pkl")
        out_path = os.path.join(DATA_ROOT, f"{sp}_cached.pkl")
        process_one_split(in_path, out_path, model, tokenizer)


if __name__ == "__main__":
    main()