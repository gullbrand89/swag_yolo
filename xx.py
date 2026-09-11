import numpy as np
from vocab import in_bin_of, in_cont_of
from data import make_channels
from config import cfg

p = np.concatenate([np.geomspace(1, 999, 500), [1000, 1500, 5000]])
ch = make_channels(p)
assert (ch["bins"] == [in_bin_of(x) for x in p]).all(), "data.py och vocab.py är osams"
assert np.allclose(ch["cont"], [in_cont_of(x) for x in p], atol=1e-6)
print("binningen är konsekvent")
print(f"overflow: {(ch['bins'] == cfg.in_bins - 1).mean():.2%} av testpulserna")