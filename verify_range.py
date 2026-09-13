from collections import Counter
import importlib, numpy as np
from config import cfg

gen = importlib.import_module(cfg.emitter)
data = gen.create_emitter_data(500, 1, 0.0, cfg.noise_level, np.random.default_rng(1))

print(Counter((bool(l["order_fixed"]), bool(l["length_fixed"])) for _, l in data))
print(Counter(len([x for x in l["lengths"] if x is not None]) for _, l in data))
print(Counter(len(set(np.asarray(l["levels"]).ravel().tolist())) for _, l in data))