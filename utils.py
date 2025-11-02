import numpy as np
import torch
import random

class GAEBuffer:
    def __init__(self, gamma, lam):
        self.g = gamma; self.l = lam
        self.data = {k: [] for k in ['obs','act','rew','val','logp','ret','adv']}

    def store(self, **kw):
        for k,v in kw.items(): self.data[k].append(v)

    def finish(self):
        rews, vals = np.array(self.data['rew']), np.array(self.data['val'])
        T = len(rews); adv = np.zeros(T)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nextv = 0.0 if t == T-1 else vals[t+1]
            delta = rews[t] + self.g * nextv - vals[t]
            lastgaelam = delta + self.g * self.l * lastgaelam
            adv[t] = lastgaelam
        ret = adv + vals
        self.data['adv'], self.data['ret'] = adv, ret
        return self.data


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)