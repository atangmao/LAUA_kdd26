# server/bandit.py
import math
from collections import defaultdict

class MultiArmedBanditSampler:
    def __init__(self, args):
        self.gamma = getattr(args, "bandit_gamma", 0.9)
        self.C = getattr(args, "clients_per_round", 10)
        self.L = defaultdict(float)  # discounted cumulative loss
        self.I = defaultdict(float)  # discounted selection count
        self.P = defaultdict(float)  # discounted block seen count

    def update(self, selected_clients, client_losses):
        g = self.gamma
        for cid in list(self.L.keys()):
            self.L[cid] *= g
            self.I[cid] *= g
        for blk in list(self.P.keys()):
            self.P[blk] *= g

        for c in selected_clients:
            cid = c.client_id
            self.L[cid] += float(client_losses.get(cid, 0.0))
            self.I[cid] += 1.0
            for blk in getattr(c, "blocks", []):
                self.P[blk] += 1.0

    def score(self, client):
        cid = client.client_id
        I = self.I[cid]
        L = self.L[cid]
        denom = I + sum(self.P[blk] for blk in getattr(client, "blocks", []))
        denom = max(denom, 1e-6)
        exploit = L / max(I, 1e-6)
        explore = math.sqrt(1.0 / denom)
        return exploit + explore

    def select(self, all_clients):
        scored = sorted(all_clients, key=self.score, reverse=True)
        return scored[: min(self.C, len(scored))]