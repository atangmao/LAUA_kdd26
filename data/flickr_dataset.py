import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms


class Flickr30KDataset(Dataset):
    """Flickr30K """

    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        max_words=30,
        cached_feature_dir=None,
        strict=True,   #
        verbose=True,  #
    ):
        """
        Args:
            root_dir (str):
            split (str):
            transform (callable, optional):
            max_words (int):
            cached_feature_dir (str|None):
            strict (bool):
            verbose (bool):
        """
        self.root_dir = os.path.abspath(os.path.expanduser(str(root_dir)))
        self.split = split
        self.transform = transform
        self.max_words = max_words
        self.cached_feature_dir = cached_feature_dir
        self.strict = strict
        self.verbose = verbose

        #
        cand1 = os.path.join(self.root_dir, "flickr30k-images")
        cand2 = os.path.join(self.root_dir, "flickr30k_images")

        if os.path.exists(cand1):
            self.image_dir = cand1
        elif os.path.exists(cand2):
            self.image_dir = cand2
        else:
            #
            self.image_dir = cand1

        self.caption_path = os.path.join(self.root_dir, "results.csv")

        self.data = []
        self.default_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

        if self.verbose:
            print(
                f"[Flickr30KDataset] root_dir={self.root_dir}\n"
                f"[Flickr30KDataset] image_dir={self.image_dir} (exists={os.path.exists(self.image_dir)})\n"
                f"[Flickr30KDataset] caption_path={self.caption_path} (exists={os.path.exists(self.caption_path)})\n"
                f"[Flickr30KDataset] cached_feature_dir={self.cached_feature_dir}"
            )

        self._load_data()

        # ----    ----
        self.use_cached_features = False
        self._image_features_by_id = None   # dict: image_id(int) -> Tensor [D]
        self._text_features = None          # Tensor [N, D]，

        if self.cached_feature_dir is not None:
            self._try_load_cached_features()

    def _try_load_cached_features(self):
        os.makedirs(self.cached_feature_dir, exist_ok=True)
        img_path = os.path.join(
            self.cached_feature_dir, "flickr30k_clip_image_features_by_image_id.pt"
        )
        txt_path = os.path.join(
            self.cached_feature_dir, "flickr30k_clip_text_features_by_sample_idx.pt"
        )

        if not (os.path.exists(img_path) and os.path.exists(txt_path)):
            print(f"[Flickr30KDataset] Cached features not found in {self.cached_feature_dir}")
            return

        image_features_by_id = torch.load(img_path, map_location="cpu")
        text_features = torch.load(txt_path, map_location="cpu")

        if not isinstance(image_features_by_id, dict):
            print("[Flickr30KDataset] Bad cache: image_features_by_id is not a dict")
            return
        if not torch.is_tensor(text_features):
            print("[Flickr30KDataset] Bad cache: text_features is not a Tensor")
            return
        if text_features.size(0) != len(self.data):
            print(
                f"[Flickr30KDataset] Bad cache: text_features len {text_features.size(0)} "
                f"!= dataset size {len(self.data)}. "
                f"Likely your data_root points to a different dataset version or loading failed earlier."
            )
            return

        self._image_features_by_id = image_features_by_id
        self._text_features = text_features
        self.use_cached_features = True
        print(f"[Flickr30KDataset] Using cached CLIP features from {self.cached_feature_dir}")

    def _load_data(self):
        """   """
        if not os.path.exists(self.image_dir):
            msg = f"[Flickr30KDataset] Image directory not found: {self.image_dir}"
            if self.strict:
                raise FileNotFoundError(msg)
            print("Error:", msg)
            return

        all_images = sorted(os.listdir(self.image_dir))
        if len(all_images) == 0:
            msg = f"[Flickr30KDataset] Image directory is empty: {self.image_dir}"
            if self.strict:
                raise RuntimeError(msg)
            print("Error:", msg)
            return

        image_map = {img: i for i, img in enumerate(all_images)}

        if os.path.exists(self.caption_path):
            try:
                df = pd.read_csv(self.caption_path, sep="|")
                df.columns = [c.strip() for c in df.columns]

                #
                if "image_name" not in df.columns or "comment" not in df.columns:
                    raise ValueError(f"results.csv columns unexpected: {df.columns}")

                for _, row in df.iterrows():
                    img_name = str(row["image_name"]).strip()
                    caption = str(row["comment"]).strip()

                    if img_name in image_map:
                        self.data.append(
                            {
                                "image_path": os.path.join(self.image_dir, img_name),
                                "text": caption,
                                "image_id": image_map[img_name],
                            }
                        )
            except Exception as e:
                print(f"[Flickr30KDataset] Warning: Failed to load captions from {self.caption_path}: {e}")
                self._load_images_only(all_images)
        else:
            print("[Flickr30KDataset] Warning: results.csv not found, loading images only")
            self._load_images_only(all_images)

        if self.verbose:
            print(f"[Flickr30KDataset] Loaded samples: {len(self.data)}")

        if len(self.data) == 0:
            msg = (
                f"[Flickr30KDataset] Loaded 0 samples. "
                f"Check image_dir and results.csv parsing."
            )
            if self.strict:
                raise RuntimeError(msg)
            print("Error:", msg)

    def _load_images_only(self, all_images):
        for img_name in all_images:
            self.data.append(
                {
                    "image_path": os.path.join(self.image_dir, img_name),
                    "text": "",
                    "image_id": -1,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # ----    ----
        if self.use_cached_features:
            image_id = int(item["image_id"])
            if self._image_features_by_id is None or self._text_features is None:
                raise RuntimeError("[Flickr30KDataset] use_cached_features=True but cache tensors are None")
            if image_id not in self._image_features_by_id:
                raise KeyError(f"[Flickr30KDataset] image_id {image_id} not in cached image_features_by_id")
            img_feat = self._image_features_by_id[image_id]  # [D] CPU
            txt_feat = self._text_features[idx]              # [D] CPU
            return {"image": img_feat, "text": txt_feat, "id": image_id, "sample_idx": idx}

        # ----   ----
        try:
            image = Image.open(item["image_path"]).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = self.default_transform(image)
        except Exception as e:
            print(f"[Flickr30KDataset] Error loading image {item['image_path']}: {e}")
            image = torch.zeros(3, 224, 224)

        return {
            "image": image,
            "text": item["text"],
            "id": int(item["image_id"]),
            "sample_idx": idx,
        }