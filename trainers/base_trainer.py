# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

import os
import math
import torch
from abc import ABC, abstractmethod
import time
import json


class BaseTrainer(ABC):
    """   """

    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.device = args.device

        #
        self.current_round = 0

        # best_metric
        self.best_metric = -math.inf
        self.best_round = -1
        self.bad_rounds = 0

        # history：
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metric': [],
            'eval_rounds': [],        #
            'val_metrics_raw': [],    # NEW：evaluate
        }

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def train_round(self):
        pass

    @abstractmethod
    def evaluate(self, test=False):
        """"""
        pass

    def _get_main_metric(self, val_metrics: dict) -> float:
        """"""
        if self.args.task == 'classification':
            return float(val_metrics.get('accuracy', 0.0))
        elif self.args.task == 'regression':
            return float(val_metrics.get('metric', -math.inf))  # 你在
        else:
            #
            i2t = float(val_metrics.get('i2t_R@1', 0.0))
            t2i = float(val_metrics.get('t2i_R@1', 0.0))
            return 0.5 * (i2t + t2i)

    def train(self):
        self.setup()

        patience = int(getattr(self.args, 'patience', 8))
        min_delta = float(getattr(self.args, 'min_delta', 1e-3))

        self.bad_rounds = 0
        self.best_round = -1
        if self.best_metric is None:
            self.best_metric = -math.inf

        # timing
        self._round_times_sec = []        #
        self._round_wall_times_sec = []   # NEW：
        self._cum_wall_times_sec = []     # NEW：

        t0 = time.time()
        cum = 0.0

        for round_idx in range(self.args.num_rounds):
            self.current_round = round_idx

            r0 = time.time()  # round wall start

            # -------- train_round timing (old behavior) --------
            tr0 = time.time()
            train_metrics = self.train_round()
            self._round_times_sec.append(time.time() - tr0)

            self.history['train_loss'].append(train_metrics.get('loss', 0.0))

            stop_now = False

            if (round_idx + 1) % self.args.eval_interval == 0:
                val_metrics = self.evaluate(test=False)
                self.history['val_loss'].append(val_metrics.get('loss', 0.0))

                metric = self._get_main_metric(val_metrics)
                self.history['val_metric'].append(metric)

                # NEW：
                self.history['eval_rounds'].append(int(round_idx + 1))
                self.history['val_metrics_raw'].append(val_metrics)

                improved = (metric > self.best_metric + min_delta)

                if improved:
                    self.best_metric = metric
                    self.best_round = round_idx
                    self.bad_rounds = 0
                    # self.save_checkpoint('best.pt')
                else:
                    self.bad_rounds += 1

                self.log(
                    f'Round {round_idx + 1}/{self.args.num_rounds} - '
                    f'Train Loss: {train_metrics.get("loss", 0):.4f} - '
                    f'Val Metric: {metric:.4f} - '
                    f'Best: {self.best_metric:.4f} - '
                    f'BadRounds: {self.bad_rounds}/{patience}'
                    + (' (improved)' if improved else '')
                )

                if self.bad_rounds >= patience:
                    self.log(
                        f'Early stopping at round {round_idx + 1} '
                        f'(best round: {self.best_round + 1}, best metric: {self.best_metric:.4f}).'
                    )
                    stop_now = True

            if (round_idx + 1) % self.args.save_interval == 0:
                self.save_checkpoint(f'round_{round_idx + 1}.pt')

            # -------- round wall timing (NEW) --------
            rw = time.time() - r0
            self._round_wall_times_sec.append(float(rw))
            cum += float(rw)
            self._cum_wall_times_sec.append(float(cum))

            if stop_now:
                break

        total_time = time.time() - t0
        executed_rounds = int(self.current_round + 1)

        # 写
        os.makedirs(self.args.exp_dir, exist_ok=True)
        timing = {
            "method": str(getattr(self.args, "method", "unknown")),
            "planned_rounds": int(self.args.num_rounds),
            "executed_rounds": executed_rounds,
            "best_round": int(self.best_round + 1) if self.best_round >= 0 else -1,
            "local_epochs": int(getattr(self.args, "local_epochs", 1)),
            "total_time_sec": float(total_time),

            #
            "round_times_sec": [float(x) for x in self._round_times_sec],

            # NEW：wall-clock（train + eval + save）
            "round_wall_times_sec": [float(x) for x in self._round_wall_times_sec],
            "cum_wall_times_sec": [float(x) for x in self._cum_wall_times_sec],
        }
        with open(os.path.join(self.args.exp_dir, "timing.json"), "w", encoding="utf-8") as f:
            json.dump(timing, f, ensure_ascii=False, indent=2)

        self.log('Training completed. Running final evaluation...')
        final_metrics = self.evaluate(test=True)
        self.log(f'Final Test Metrics: {final_metrics}')

        return self.history

    def save_checkpoint(self, filename):
        checkpoint_path = os.path.join(self.args.exp_dir, filename)
        checkpoint = {
            'round': self.current_round,
            'model_state_dict': self.get_model_state(),
            'best_metric': self.best_metric,
            'best_round': self.best_round,
            'bad_rounds': self.bad_rounds,
            'history': self.history,
            'args': self.args
        }
        torch.save(checkpoint, checkpoint_path)
        self.log(f'Checkpoint saved to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.current_round = checkpoint.get('round', 0)
        self.set_model_state(checkpoint['model_state_dict'])
        self.best_metric = checkpoint.get('best_metric', -math.inf)
        self.best_round = checkpoint.get('best_round', -1)
        self.bad_rounds = checkpoint.get('bad_rounds', 0)
        self.history = checkpoint.get('history', self.history)
        self.log(f'Checkpoint loaded from {checkpoint_path}')

    @abstractmethod
    def get_model_state(self):
        pass

    @abstractmethod
    def set_model_state(self, state_dict):
        pass

    def log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)