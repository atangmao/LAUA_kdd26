import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from PIL import Image
from torchvision import transforms


class MSCOCODataset(Dataset):
    """
    MSCOCO 2017 captions dataset loader (Flickr30K-style).

    - One caption = one sample (N = #captions)
    - Image features cached by image_id (dict)
    - Text features cached by sample_idx (Tensor [N, D])
    """

    def __init__(self, root_dir, split="train", transform=None, max_words=30, cached_feature_dir=None):
        """
        Args:
            root_dir (str): e.g. "/home/userdata/data/MSCOCO"
            split (str): "train" | "val" | "all"
            transform: PIL -> Tensor transform (only used when not using cache)
            max_words: kept for interface consistency (not enforced here)
            cached_feature_dir (str|None): if provided, try load cached CLIP features
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_words = max_words
        self.cached_feature_dir = cached_feature_dir

        self.default_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # your actual structure: train2017/train2017, val2017/val2017
        self.train_image_dir = self._resolve_image_dir("train2017")
        self.val_image_dir = self._resolve_image_dir("val2017")

        # annotations path (your structure)
        self.ann_dir = os.path.join(self.root_dir, "annotations_trainval2017", "annotations")
        self.train_ann = os.path.join(self.ann_dir, "captions_train2017.json")
        self.val_ann = os.path.join(self.ann_dir, "captions_val2017.json")

        self.data = []
        self._load_data()

        # cache
        self.use_cached_features = False
        self._image_features_by_id = None  # dict: image_id(int) -> Tensor [D]
        self._text_features = None         # Tensor [N, D], aligned with dataset idx

        if self.cached_feature_dir is not None:
            self._try_load_cached_features()

    def _resolve_image_dir(self, split_dir_name: str):
        """
        Try both:
          root/split_dir_name
          root/split_dir_name/split_dir_name
        because user mentioned train2017 has another train2017 inside.
        """
        p1 = os.path.join(self.root_dir, split_dir_name)
        p2 = os.path.join(self.root_dir, split_dir_name, split_dir_name)
        if os.path.isdir(p2):
            return p2
        return p1

    def _load_coco_caption_file(self, ann_path: str, image_dir: str):
        with open(ann_path, "r", encoding="utf-8") as f:
            j = json.load(f)

        id2file = {int(img["id"]): img["file_name"] for img in j.get("images", [])}

        out = []
        for ann in j.get("annotations", []):
            image_id = int(ann["image_id"])
            caption = ann.get("caption", "") or ""
            file_name = id2file.get(image_id)
            if file_name is None:
                continue

            # In case file_name contains "train2017/xxx.jpg", normalize:
            file_name = file_name.replace("\\", "/")
            if "/" in file_name:
                # join from root_dir to avoid double train2017/train2017
                image_path = os.path.join(self.root_dir, file_name)
            else:
                image_path = os.path.join(image_dir, file_name)

            out.append({
                "image_id": image_id,
                "image_path": image_path,
                "text": caption,
            })
        return out

    def _load_data(self):
        if self.split not in ["train", "val", "all"]:
            raise ValueError(f"split must be 'train'|'val'|'all', got {self.split}")

        if not os.path.isdir(self.ann_dir):
            raise FileNotFoundError(f"Annotation dir not found: {self.ann_dir}")

        if self.split in ["train", "all"]:
            if not os.path.exists(self.train_ann):
                raise FileNotFoundError(f"Not found: {self.train_ann}")
            self.data.extend(self._load_coco_caption_file(self.train_ann, self.train_image_dir))

        if self.split in ["val", "all"]:
            if not os.path.exists(self.val_ann):
                raise FileNotFoundError(f"Not found: {self.val_ann}")
            self.data.extend(self._load_coco_caption_file(self.val_ann, self.val_image_dir))

        print(f"[MSCOCODataset] split={self.split}, samples={len(self.data)}")

    def _try_load_cached_features(self):
        os.makedirs(self.cached_feature_dir, exist_ok=True)
        img_path = os.path.join(self.cached_feature_dir, f"mscoco2017_{self.split}_clip_image_features_by_image_id.pt")
        txt_path = os.path.join(self.cached_feature_dir, f"mscoco2017_{self.split}_clip_text_features_by_sample_idx.pt")

        if not (os.path.exists(img_path) and os.path.exists(txt_path)):
            print(f"[MSCOCODataset] Cached features not found in {self.cached_feature_dir}")
            return

        image_features_by_id = torch.load(img_path, map_location="cpu")
        text_features = torch.load(txt_path, map_location="cpu")

        if not isinstance(image_features_by_id, dict):
            print("[MSCOCODataset] Bad cache: image_features_by_id is not a dict")
            return
        if not torch.is_tensor(text_features):
            print("[MSCOCODataset] Bad cache: text_features is not a Tensor")
            return

        # must match current dataset length (train/val/all)
        if text_features.size(0) != len(self.data):
            print(f"[MSCOCODataset] Bad cache: text_features len {text_features.size(0)} "
                  f"!= dataset size {len(self.data)} (split={self.split}). "
                  f"Re-extract cache for this split, or use split='all'.")
            return

        self._image_features_by_id = image_features_by_id
        self._text_features = text_features
        self.use_cached_features = True
        print(f"[MSCOCODataset] Using cached CLIP features from {self.cached_feature_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.use_cached_features:
            image_id = int(item["image_id"])
            img_feat = self._image_features_by_id[image_id]  # [D] CPU
            txt_feat = self._text_features[idx]              # [D] CPU
            return {
                "image": img_feat,
                "text": txt_feat,
                "id": image_id,
                "sample_idx": idx,
            }

        # raw mode
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = self.default_transform(image)
        except Exception as e:
            print(f"[MSCOCODataset] Error loading image {item['image_path']}: {e}")
            image = torch.zeros(3, 224, 224)

        return {
            "image": image,
            "text": item["text"],
            "id": int(item["image_id"]),
            "sample_idx": idx,
        }


class COCOSubset(Subset):
    """
    Subset wrapper that supports modality masking:
      modality in {'both','image','text'} (some code uses 'image_only'/'text_only' too)

    Key fix:
      When base dataset returns cached features (tensors), masked modality MUST be a zero tensor,
      NOT ""/None, so that cached collate_fn can torch.stack().
    """
    def __init__(self, dataset, indices, modality='both'):
        super().__init__(dataset, indices)
        self.modality = modality

    def __getitem__(self, idx):
        item = self.dataset[self.indices[idx]]

        # Normalize modality labels
        m = self.modality
        if m == 'image_only':
            m = 'image'
        if m == 'text_only':
            m = 'text'

        # detect "cached feature mode" by actual returned types
        img_is_tensor = torch.is_tensor(item.get('image', None))
        txt_is_tensor = torch.is_tensor(item.get('text', None))
        cached = img_is_tensor and txt_is_tensor

        if m == 'both':
            return item

        item = dict(item)  # avoid in-place modification side effects

        if m == 'image':
            # mask text
            if cached:
                item['text'] = torch.zeros_like(item['text'])
            else:
                # raw mode: keep text as string (collate_fn should handle raw mode)
                item['text'] = ""
            return item

        if m == 'text':
            # mask image
            if cached:
                item['image'] = torch.zeros_like(item['image'])
            else:
                # raw mode: MSCOCODataset returns image tensor already (after transform)
                if torch.is_tensor(item.get('image', None)):
                    item['image'] = torch.zeros_like(item['image'])
                else:
                    item['image'] = torch.zeros(3, 224, 224)
            return item

        # unknown modality -> return as is
        return item