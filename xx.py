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



import numpy as np
from config import cfg
from data import create_emitter_data

d = create_emitter_data(2, 3, 0.0, cfg.noise_level, np.random.default_rng(0))   # 2 emittrar, 3 signaler

print("returen        :", type(d).__name__, "len", len(d))
e = d[0]
print("d[0]           :", type(e).__name__, "len", len(e) if hasattr(e, "__len__") else "-")
for i, part in enumerate(e):
    print(f"  d[0][{i}]      : {type(part).__name__}, shape {np.shape(part)}")
    if isinstance(part, (list, tuple, np.ndarray)) and len(part):
        print(f"                 första elementet: {type(part[0]).__name__}, "
              f"shape {np.shape(part[0])}, värde {np.ravel(part[0])[:3]}")