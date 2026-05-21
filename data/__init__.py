from .flickr_dataset import Flickr30KDataset
from .coco_dataset import MSCOCODataset, COCOSubset
from torch.utils.data import Dataset, Subset
import numpy as np
import torch
import clip

from data.collate import flickr_collate_fn, flickr_collate_cached_fn


def get_dataset(args):
    cache_dir = getattr(args, "clip_cache_dir", "/home/userdata/data/flicker30k_clip/")

    if args.dataset == "flickr30k":
        #
        _, preprocess = clip.load(args.clip_model, device=args.device, jit=False)

        full_dataset = Flickr30KDataset(
            root_dir=args.data_root,
            transform=preprocess,
            cached_feature_dir=cache_dir
        )

        collate_fn = flickr_collate_cached_fn if getattr(full_dataset, "use_cached_features", False) else flickr_collate_fn

        num_samples = len(full_dataset)
        indices = list(range(num_samples))
        train_split = int(np.floor(args.train_ratio * num_samples))
        val_split = int(np.floor(args.val_ratio * num_samples))

        train_indices = indices[:train_split]
        val_indices = indices[train_split: train_split + val_split]
        test_indices = indices[train_split + val_split:]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
        return train_dataset, val_dataset, test_dataset, collate_fn

    elif args.dataset == "mscoco":
        _, preprocess = clip.load(args.clip_model, device=args.device, jit=False)

        train_dataset = MSCOCODataset(
            root_dir=args.data_root,
            split="train",
            transform=preprocess,
            cached_feature_dir=cache_dir
        )
        val_dataset = MSCOCODataset(
            root_dir=args.data_root,
            split="val",
            transform=preprocess,
            cached_feature_dir=cache_dir
        )
        test_dataset = val_dataset  #

        collate_fn = flickr_collate_cached_fn if getattr(train_dataset, "use_cached_features", False) else flickr_collate_fn
        return train_dataset, val_dataset, test_dataset, collate_fn

    elif args.dataset == "mosei":
        from data.mosei_dataset import build_mosei_splits, mosei_collate_fn
        train_dataset, val_dataset, test_dataset = build_mosei_splits(
            args.data_root, use_cached_text_cls=True
        )
        return train_dataset, val_dataset, test_dataset, mosei_collate_fn

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


class FederatedDataSplitter:
    """

    """

    def __init__(self, num_multimodal_clients=0, num_image_clients=0,
                 num_text_clients=0, num_audio_clients=0, num_vision_clients=0,
                 partition='iid', alpha=0.5, seed=42, mm_ratio=0.6):
        self.num_multimodal_clients = num_multimodal_clients
        self.num_image_clients = num_image_clients
        self.num_text_clients = num_text_clients
        self.num_audio_clients = num_audio_clients
        self.num_vision_clients = num_vision_clients

        self.partition = partition
        self.alpha = alpha
        self.seed = seed
        self.mm_ratio = mm_ratio

        self.num_clients = (num_multimodal_clients + num_image_clients + num_text_clients
                            + num_audio_clients + num_vision_clients)

        np.random.seed(seed)

    def split(self, dataset, labels=None):
        num_items = len(dataset)
        indices = np.random.permutation(num_items)

        if self.num_clients <= 0:
            raise ValueError("num_clients must be > 0")

        if self.partition == 'iid':
            batch_idxs = np.array_split(indices, self.num_clients)

        elif self.partition == 'dirichlet':
            batch_idxs = []
            start = 0

            num_mm_clients = self.num_multimodal_clients
            mm_total = int(num_items * self.mm_ratio)
            mm_each = (mm_total // num_mm_clients) if num_mm_clients > 0 else 0

            for _ in range(num_mm_clients):
                end = start + mm_each
                batch_idxs.append(indices[start:end])
                start = end

            remaining = num_items - start
            num_rest_clients = self.num_clients - num_mm_clients

            if num_rest_clients > 0:
                if remaining < num_rest_clients:
                    raise RuntimeError(
                        f"Not enough remaining samples ({remaining}) to give each unimodal client >=1 "
                        f"(unimodal_clients={num_rest_clients}). Reduce mm_ratio or reduce #clients."
                    )

                counts = np.ones(num_rest_clients, dtype=int)
                remaining2 = remaining - num_rest_clients

                if remaining2 > 0:
                    proportions = np.random.dirichlet([self.alpha] * num_rest_clients)
                    extra = (proportions * remaining2).astype(int)
                    counts += extra

                diff = remaining - int(counts.sum())
                if diff > 0:
                    order = np.argsort(-counts)
                    for i in range(diff):
                        counts[order[i % num_rest_clients]] += 1
                elif diff < 0:
                    need = -diff
                    order = np.argsort(-counts)
                    j = 0
                    while need > 0:
                        idx = order[j % num_rest_clients]
                        if counts[idx] > 1:
                            counts[idx] -= 1
                            need -= 1
                        j += 1
                        if j > 10_000_000:
                            raise RuntimeError("Failed to adjust counts; unexpected state.")

                if (counts < 1).any():
                    raise RuntimeError(f"Invalid counts after adjustment: {counts}")
                if int(counts.sum()) != remaining:
                    raise RuntimeError(f"Counts do not sum to remaining: sum={counts.sum()}, remaining={remaining}")

                for c in counts.tolist():
                    end = start + c
                    batch_idxs.append(indices[start:end])
                    start = end
        else:
            raise ValueError(f"Partition {self.partition} not implemented yet.")

        client_splits = {}
        client_id = 0


        for _ in range(self.num_multimodal_clients):
            client_splits[client_id] = {'indices': batch_idxs[client_id].tolist(), 'modality': 'both'}
            client_id += 1

        for _ in range(self.num_image_clients):
            client_splits[client_id] = {'indices': batch_idxs[client_id].tolist(), 'modality': 'image'}
            client_id += 1

        for _ in range(self.num_text_clients):
            client_splits[client_id] = {'indices': batch_idxs[client_id].tolist(), 'modality': 'text'}
            client_id += 1

        for _ in range(self.num_audio_clients):
            client_splits[client_id] = {'indices': batch_idxs[client_id].tolist(), 'modality': 'audio'}
            client_id += 1

        for _ in range(self.num_vision_clients):
            client_splits[client_id] = {'indices': batch_idxs[client_id].tolist(), 'modality': 'vision'}
            client_id += 1

        return client_splits

    def get_client_info(self, client_splits):
        info = {
            'num_clients': self.num_clients,
            'num_multimodal': self.num_multimodal_clients,
            'num_image_only': self.num_image_clients,
            'num_text_only': self.num_text_clients,
            'num_audio_only': self.num_audio_clients,
            'num_vision_only': self.num_vision_clients,
            'partition': self.partition,
            'clients': []
        }
        for client_id, data_info in client_splits.items():
            info['clients'].append({
                'id': client_id,
                'modality': data_info['modality'],
                'num_samples': len(data_info['indices'])
            })
        return info


class Flickr30KSubset(Dataset):
    def __init__(self, dataset, indices, modality='both'):
        self.dataset = dataset
        self.indices = indices
        self.modality = modality

    def _is_cached_item(self, item):
        img = item.get('image', None)
        txt = item.get('text', None)
        return torch.is_tensor(img) and torch.is_tensor(txt)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        item = self.dataset[base_idx]
        cached = self._is_cached_item(item)

        real_sample_idx = int(item.get('sample_idx', base_idx))

        if self.modality == 'image':
            if cached:
                return {
                    'image': item['image'],
                    'text': torch.zeros_like(item['text']),
                    'id': item.get('id', -1),
                    'sample_idx': real_sample_idx,
                }
            else:
                return {
                    'image': item['image'],
                    'text': '',
                    'id': item.get('id', -1),
                    'sample_idx': real_sample_idx,
                }

        elif self.modality == 'text':
            if cached:
                return {
                    'image': torch.zeros_like(item['image']),
                    'text': item['text'],
                    'id': item.get('id', -1),
                    'sample_idx': real_sample_idx,
                }
            else:
                return {
                    'image': torch.zeros_like(item['image']),
                    'text': item['text'],
                    'id': item.get('id', -1),
                    'sample_idx': real_sample_idx,
                }
        else:
            item = dict(item)
            item['sample_idx'] = real_sample_idx
            return item

    def __len__(self):
        return len(self.indices)