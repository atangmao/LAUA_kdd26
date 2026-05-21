# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from collections import defaultdict


class FederatedDataSplitter:
    """  """

    def __init__(self, num_multimodal_clients, num_image_clients, num_text_clients,
                 partition='iid', alpha=0.5, seed=42, mm_ratio=0.1):
        """
        Args:
            num_multimodal_clients:  (K)
            num_image_clients:  (L)
            num_text_clients:  (M)
            partition:
            alpha:
            seed:
        """
        self.num_multimodal = num_multimodal_clients
        self.num_image = num_image_clients
        self.num_text = num_text_clients
        self.total_clients = num_multimodal_clients + num_image_clients + num_text_clients

        self.partition = partition
        self.alpha = alpha
        self.seed = seed
        self.mm_ratio = mm_ratio  #

        np.random.seed(seed)

    def split(self, dataset, labels=None):
        """

        Args:
            dataset:
            labels:

        Returns:
            client_indices: dict，
        """
        n_samples = len(dataset)
        indices = np.arange(n_samples)

        if self.partition == 'iid':
            client_indices = self._iid_split(indices)
        elif self.partition == 'noniid':
            if labels is None:
                raise ValueError('Non-IID need labels')
            client_indices = self._noniid_split(indices, labels)
        elif self.partition == 'dirichlet':
            if labels is None:
                raise ValueError('Dirichlet need labels')
            client_indices = self._dirichlet_split(indices, labels)
        else:
            raise ValueError(f'Unknown partition: {self.partition}')

        return client_indices

    def _iid_split(self, indices):
        """IID"""
        np.random.shuffle(indices)

        client_indices = {}

        #
        n_samples = len(indices)
        samples_per_client = n_samples // self.total_clients

        current_idx = 0
        client_id = 0

        #
        for i in range(self.num_multimodal):
            start = current_idx
            end = start + samples_per_client
            client_indices[client_id] = {
                'indices': indices[start:end].tolist(),
                'modality': 'both'
            }
            current_idx = end
            client_id += 1

        #
        for i in range(self.num_image):
            start = current_idx
            end = start + samples_per_client
            client_indices[client_id] = {
                'indices': indices[start:end].tolist(),
                'modality': 'image'
            }
            current_idx = end
            client_id += 1

        #
        for i in range(self.num_text):
            start = current_idx
            end = start + samples_per_client
            #
            if i == self.num_text - 1:
                end = n_samples
            client_indices[client_id] = {
                'indices': indices[start:end].tolist(),
                'modality': 'text'
            }
            current_idx = end
            client_id += 1

        return client_indices

    def _noniid_split(self, indices, labels):
        """Non-IID"""
        labels = np.array(labels)
        n_classes = len(np.unique(labels))

        #
        class_indices = defaultdict(list)
        for idx in indices:
            class_indices[labels[idx]].append(idx)

        #
        classes_per_client = max(2, n_classes // self.total_clients)

        client_indices = {}
        client_id = 0
        class_ids = list(class_indices.keys())
        np.random.shuffle(class_ids)

        class_ptr = 0

        def get_client_data(n_classes_for_client, modality):
            nonlocal class_ptr
            client_data = []
            for _ in range(n_classes_for_client):
                if class_ptr >= len(class_ids):
                    class_ptr = 0
                    np.random.shuffle(class_ids)
                c = class_ids[class_ptr]
                #
                c_indices = class_indices[c]
                np.random.shuffle(c_indices)
                n_take = len(c_indices) // self.total_clients
                client_data.extend(c_indices[:n_take])
                class_ptr += 1
            return {'indices': client_data, 'modality': modality}

        #
        for i in range(self.num_multimodal):
            client_indices[client_id] = get_client_data(classes_per_client, 'both')
            client_id += 1

        for i in range(self.num_image):
            client_indices[client_id] = get_client_data(classes_per_client, 'image')
            client_id += 1

        for i in range(self.num_text):
            client_indices[client_id] = get_client_data(classes_per_client, 'text')
            client_id += 1

        return client_indices

    def _dirichlet_split(self, indices, labels):
        """Dirichlet

        """

        labels = np.array(labels)
        n_samples = len(indices)

        if self.total_clients <= 0:
            raise ValueError("total_clients must be > 0")
        if n_samples == 0:
            raise ValueError("Empty dataset: n_samples=0")
        if n_samples < self.total_clients:
            raise ValueError(
                f"Not enough samples ({n_samples}) to give each client at least 1 sample "
                f"(clients={self.total_clients}). Reduce #clients or use larger dataset."
            )

        #
        class_indices = defaultdict(list)
        for idx in indices:
            class_indices[labels[idx]].append(idx)
        class_ids = list(class_indices.keys())

        #
        client_indices = {i: {'indices': [], 'modality': None} for i in range(self.total_clients)}
        print("[Splitter] counts:", {cid: len(d["indices"]) for cid, d in client_indices.items()})

        #
        for i in range(self.num_multimodal):
            client_indices[i]['modality'] = 'both'
        for i in range(self.num_multimodal, self.num_multimodal + self.num_image):
            client_indices[i]['modality'] = 'image'
        for i in range(self.num_multimodal + self.num_image, self.total_clients):
            client_indices[i]['modality'] = 'text'

        #
        for c in class_ids:
            c_indices = np.array(class_indices[c], dtype=int)
            if c_indices.size == 0:
                continue
            np.random.shuffle(c_indices)

            proportions = np.random.dirichlet([self.alpha] * self.total_clients)
            proportions = proportions / proportions.sum()

            cut_points = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]
            splits = np.split(c_indices, cut_points)

            for client_id, split in enumerate(splits):
                if split.size > 0:
                    client_indices[client_id]['indices'].extend(split.tolist())

        # ---- ----
        empty_clients = [cid for cid, d in client_indices.items() if len(d['indices']) == 0]
        if empty_clients:
            # donors：
            donors = sorted(
                [cid for cid, d in client_indices.items() if len(d['indices']) > 1],
                key=lambda cid: len(client_indices[cid]['indices']),
                reverse=True
            )

            if not donors:
                #
                raise RuntimeError(
                    "Dirichlet split produced empty clients but no donor client has >1 samples. "
                    "Try larger alpha or reduce #clients."
                )

            donor_ptr = 0
            for ecid in empty_clients:
                #
                while donor_ptr < len(donors) and len(client_indices[donors[donor_ptr]]['indices']) <= 1:
                    donor_ptr += 1
                if donor_ptr >= len(donors):
                    #
                    donors = sorted(
                        [cid for cid, d in client_indices.items() if len(d['indices']) > 1],
                        key=lambda cid: len(client_indices[cid]['indices']),
                        reverse=True
                    )
                    donor_ptr = 0
                if not donors or len(client_indices[donors[donor_ptr]]['indices']) <= 1:
                    raise RuntimeError(
                        "Unable to fix empty clients (insufficient donor samples). "
                        "Try larger alpha or reduce #clients."
                    )

                dcid = donors[donor_ptr]
                moved = client_indices[dcid]['indices'].pop()  # 挪 1 个样本
                client_indices[ecid]['indices'].append(moved)

        #
        for cid in range(self.total_clients):
            if len(client_indices[cid]['indices']) == 0:
                raise RuntimeError(f"Client {cid} still has 0 samples after fix. This should not happen.")

        return client_indices

    def get_client_info(self, client_indices):
        """

        Args:
            client_indices:

        Returns:
            info:
        """
        info = {
            'num_clients': len(client_indices),
            'num_multimodal': self.num_multimodal,
            'num_image_only': self.num_image,
            'num_text_only': self.num_text,
            'partition': self.partition,
            'clients': []
        }

        for client_id, data in client_indices.items():
            info['clients'].append({
                'id': client_id,
                'modality': data['modality'],
                'num_samples': len(data['indices'])
            })

        return info