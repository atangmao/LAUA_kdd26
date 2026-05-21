# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import json


class CLIPFeatureExtractor:
    """"""

    def __init__(self, model_name='ViT-B/32', device='cuda', cache_dir='./cache'):
        """

        """
        self.device = device
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        #
        print(f'Loading CLIP model: {model_name}')
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        #
        self.embed_dim = self.model.visual.output_dim
        print(f'CLIP embedding dimension: {self.embed_dim}')

    @torch.no_grad()
    def extract_image_features(self, image_paths, batch_size=64):
        """

        Args:
            image_paths:
            batch_size:

        Returns:
            features: [N, embed_dim]
        """
        all_features = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc='Extracting image features'):
            batch_paths = image_paths[i:i + batch_size]
            images = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image = self.preprocess(image)
                    images.append(image)
                except Exception as e:
                    print(f'Error loading image {path}: {e}')

                    images.append(torch.zeros(3, 224, 224))

            images = torch.stack(images).to(self.device)
            features = self.model.encode_image(images)
            features = features.float().cpu()
            all_features.append(features)

        return torch.cat(all_features, dim=0)

    @torch.no_grad()
    def extract_text_features(self, texts, batch_size=64):
        """

        Args:
            texts:
            batch_size:

        Returns:
            features: [N, embed_dim]
        """
        all_features = []

        for i in tqdm(range(0, len(texts), batch_size), desc='Extracting text features'):
            batch_texts = texts[i:i + batch_size]

            #
            tokens = clip.tokenize(batch_texts, truncate=True).to(self.device)
            features = self.model.encode_text(tokens)
            features = features.float().cpu()
            all_features.append(features)

        return torch.cat(all_features, dim=0)

    def extract_and_cache(self, dataset_name, image_paths, texts, sample_ids):
        """

        Args:
            dataset_name:
            image_paths:
            texts:
            sample_ids:

        Returns:
            cache_path:
        """
        cache_path = os.path.join(self.cache_dir, f'{dataset_name}_clip_features.pt')

        if os.path.exists(cache_path):
            print(f'Loading cached features from {cache_path}')
            return cache_path

        print(f'Extracting features for {dataset_name}...')

        #
        image_features = self.extract_image_features(image_paths)
        text_features = self.extract_text_features(texts)

        #
        cache_data = {
            'image_features': image_features,
            'text_features': text_features,
            'sample_ids': sample_ids,
            'embed_dim': self.embed_dim
        }

        torch.save(cache_data, cache_path)
        print(f'Features cached to {cache_path}')

        return cache_path

    @staticmethod
    def load_cached_features(cache_path):
        """

        Args:
            cache_path:

        Returns:
            cache_data:
        """
        return torch.load(cache_path)
