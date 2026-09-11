import numpy as np
from config import cfg
from data import create_emitter_data

for n in (1, cfg.samples_per_emitter):
    seqs, lab = create_emitter_data(1, n, 0.0, cfg.noise_level, np.random.default_rng(0))[0]
    print(f"n_signals={n}: typ {type(seqs).__name__}, "
          f"{'ndim ' + str(np.ndim(seqs)) if isinstance(seqs, np.ndarray) else 'len ' + str(len(seqs))}, "
          f"första elementet har ndim {np.ndim(seqs[0])}")



import numpy as np
from config import cfg
from data import create_emitter_data

seqs, lab = create_emitter_data(1, 2, 0.0, cfg.noise_level, np.random.default_rng(0))[0]
s = seqs[0]
print("typ:", type(s))
print("repr:", repr(s)[:300])
if hasattr(s, "keys"):
    print("nycklar:", list(s.keys()))
elif hasattr(s, "__dict__"):
    print("attribut:", list(vars(s))[:20])
print("\nlabel:", {k: (type(v).__name__, np.shape(v)) for k, v in lab.items()})