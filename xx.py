import numpy as np
from config import cfg
from data import create_emitter_data

for dr in (0.0, 0.1):
    seqs, lab = create_emitter_data(1, cfg.samples_per_emitter, dr,
                                    cfg.noise_level, np.random.default_rng(0))[0]
    print(f"drop={dr}: seqs är {type(seqs).__name__}, shape {np.shape(seqs)}")
    for i, s in enumerate(seqs):
        print(f"   signal {i}: {type(s).__name__}, shape {np.shape(s)}")