# LAUA: Uncertainty-aware Multimodal Federated Learning KDD '26

Federated multimodal learning experiments on:
- **Flickr30K** (image-text)
- **MSCOCO 2017 Captions** (image-text)
- **CMU-MOSEI** (text/audio/vision, regression)

## Required workflow (do this in order)
1) Prepare dataset folders (`--data_root`)  
2) **Cache features / preprocess first**
   - Flickr30K / MSCOCO: cache CLIP features to `.pt`
   - MOSEI: generate `*_cached.pkl` with `text_cls` (768-dim BERT [CLS])
3) Run `main.py` to train/evaluate

---

## 1. Repository layout (based on your uploaded files)

Put your files under the project root like this:

```text
project_root/
├── main.py
├── extract_flicker30k_clip_cache.py
├── extract_MSCOCO_clip_cache.py
├── cache_bert_cls_to_pkl.py
├── data_cmumosei.py
├── clients/
│   ├── __init__.py                 # create an empty file if missing
│   ├── base_client.py
│   ├── multimodal_client.py
│   ├── unimodal_client.py
│   └── mosei_clients.py
└── data/
    ├── __init__.py
    ├── collate.py
    ├── clip_extractor.py
    ├── flickr_dataset.py
    ├── coco_dataset.py
    └── mosei_dataset.py            # REQUIRED if you run MOSEI (imported by data/__init__.py)
.......
```

Also required by `main.py` (must exist in your repo, because `main.py` imports them):
- `utils/` (e.g., `utils/logger.py`, `utils/helpers.py`)
- `trainers/` (e.g., `trainers/fedssl_moe_trainer.py`, `trainers/mosei_fedssl_moe_trainer.py`)

---

## 2. Installation

### 2.1 Python
Python 3.9+ recommended.

### 2.2 Dependencies (minimum)
- `torch`
- `numpy`, `pandas`
- `Pillow`, `tqdm`
- `transformers` (MOSEI only)
- `openai-clip` (CLIP)

Example install:
```bash
pip install numpy pandas pillow tqdm
pip install transformers
pip install git+https://github.com/openai/CLIP.git
```

Install PyTorch from: https://pytorch.org/get-started/locally/

---

## 3. Must-do step: cache features first, then run `main.py`

For **Flickr30K / MSCOCO**, there are two modes:

- **Recommended (cache mode)**: dataset loads cached `.pt` CLIP features → much faster
- **Not recommended (online mode)**: `data/collate.py` encodes CLIP during training → slow and GPU-heavy

### 3.1 Choose a cache directory
Recommended to create one in the project root:
```bash
mkdir -p ./clip_cache
```

When running `main.py`, always pass:
```bash
--clip_cache_dir ./clip_cache
```

---

## 4. Dataset formats (`--data_root`)

### 4.1 Flickr30K (`--dataset flickr30k`)
`--data_root /path/to/flickr30k_root` must contain:

```text
/path/to/flickr30k_root/
├── flickr30k-images/               # or flickr30k_images/
│   └── *.jpg
└── results.csv                     # read with sep="|", must include columns: image_name, comment
```

### 4.2 MSCOCO 2017 Captions (`--dataset mscoco`)
`--data_root /path/to/coco_root` must contain:

```text
/path/to/coco_root/
├── train2017/                      # may be train2017/train2017/ (supported)
├── val2017/                        # may be val2017/val2017/ (supported)
└── annotations_trainval2017/
    └── annotations/
        ├── captions_train2017.json
        └── captions_val2017.json
```

### 4.3 CMU-MOSEI (`--dataset mosei`)
MOSEI training expects cached PKL files under `--data_root`:

```text
/path/to/mosei_root/
├── train_cached.pkl
├── dev_cached.pkl
└── test_cached.pkl
```

Cached PKL format (per segment):
- input: `(text, audio, vision, label)`
- output: `(text, audio, vision, label, text_cls)` where `text_cls` is `float32` with shape `(768,)`

---

## 5. Step 1: feature caching / preprocessing (run BEFORE `main.py`)

### 5.1 Cache Flickr30K CLIP features
This script saves into `--cached_dir`:
```bash
python extract_flicker30k_clip_cache.py \
  --data_root /path/to/flickr30k_root \
  --cached_dir ./clip_cache \
  --model ViT-B/32
```

Expected outputs in `./clip_cache`:
- `flickr30k_clip_image_features_by_image_id.pt`
- `flickr30k_clip_text_features_by_sample_idx.pt`

### 5.2 Cache MSCOCO CLIP features (train/val)
**Important**: `extract_MSCOCO_clip_cache.py` saves cache files into `--root_dir` (not a separate cache folder).

Option A (run twice):
```bash
python extract_MSCOCO_clip_cache.py --root_dir /path/to/coco_root --split train --model ViT-B/32
python extract_MSCOCO_clip_cache.py --root_dir /path/to/coco_root --split val   --model ViT-B/32
```

Option B (all in one run):
```bash
python extract_MSCOCO_clip_cache.py --root_dir /path/to/coco_root --split all --model ViT-B/32
```

Expected outputs in `/path/to/coco_root`:
- `mscoco2017_train_clip_image_features_by_image_id.pt`
- `mscoco2017_train_clip_text_features_by_sample_idx.pt`
- `mscoco2017_val_clip_image_features_by_image_id.pt`
- `mscoco2017_val_clip_text_features_by_sample_idx.pt`

Recommended setting when training MSCOCO:
- `--data_root /path/to/coco_root`
- `--clip_cache_dir /path/to/coco_root`

### 5.3 MOSEI: generate `*_cached.pkl` (adds `text_cls`)
Edit these constants inside `cache_bert_cls_to_pkl.py`:
- `BERT_PATH` = your offline/local `bert-base-uncased` directory
- `DATA_ROOT` = directory containing `train.pkl`, `dev.pkl`, `test.pkl`

Then run:
```bash
python cache_bert_cls_to_pkl.py
```

Outputs in `DATA_ROOT`:
- `train_cached.pkl`
- `dev_cached.pkl`
- `test_cached.pkl`

(Optional) inspect PKL structure:
```bash
python data_cmumosei.py
```

---

## 6. Step 2: run `main.py`

`main.py` has dataset-specific default paths like `/home/XXXX/...`. To avoid wrong paths, **pass `--data_root` (and `--clip_cache_dir` for Flickr/COCO) explicitly**.

### 6.1 Flickr30K (recommended: cached)
```bash
python main.py \
  --dataset flickr30k \
  --data_root /path/to/flickr30k_root \
  --clip_cache_dir ./clip_cache \
  --clip_model ViT-B/32 \
  --device cuda \
  --gpu_id 0 \
  --num_rounds 100 \
  --local_epochs 1 \
  --batch_size 128 \
  --lr 1e-4 \
  --weight_decay 1e-4
```

### 6.2 MSCOCO (recommended: cached)
Because MSCOCO cache is saved into `--root_dir`, set `--clip_cache_dir` to the same directory:
```bash
python main.py \
  --dataset mscoco \
  --data_root /path/to/coco_root \
  --clip_cache_dir /path/to/coco_root \
  --clip_model ViT-B/32 \
  --device cuda \
  --gpu_id 0 \
  --num_rounds 100 \
  --local_epochs 1 \
  --batch_size 128 \
  --lr 1e-4 \
  --weight_decay 1e-4
```

### 6.3 MOSEI
```bash
python main.py \
  --dataset mosei \
  --data_root /path/to/mosei_root \
  --device cuda \
  --gpu_id 0 \
  --num_rounds 100 \
  --local_epochs 1 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4
```

---

## 7. Outputs

Training outputs are written to:
- `--output_dir` (default: `./outputs`)
- each run creates `./outputs/<exp_name>/`

Typical files:
- `train.log`
- `config.yaml`
- `results.txt`
- checkpoints (if enabled by trainer)

---
