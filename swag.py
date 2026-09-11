# vocab.py -- ersätt den linjära binningen
def _log_span():
    return math.log(cfg.pri_min), math.log(cfg.pri_max)

def bin_of(pri):
    lo, hi = _log_span()
    x = (math.log(min(max(pri, cfg.pri_min), cfg.pri_max)) - lo) / (hi - lo)
    return max(0, min(cfg.n_bins - 1, int(x * (cfg.n_bins - 1))))

def pri_of_bin(b):
    lo, hi = _log_span()
    return math.exp(lo + (b + 0.5) / (cfg.n_bins - 1) * (hi - lo))

def binwidth_at(pri):
    """Binbredden i µs VID en given PRI -- skalar med pri."""
    lo, hi = _log_span()
    return np.asarray(pri) * (hi - lo) / (cfg.n_bins - 1)

def rel_binwidth():
    """Binbredden relativt -- konstant, till skillnad från den absoluta."""
    lo, hi = _log_span()
    return (hi - lo) / (cfg.n_bins - 1)


from vocab import bin_of, pri_of_bin
from config import cfg

assert all(bin_of(pri_of_bin(b)) == b for b in range(cfg.n_bins)), "rundturen brister"

d = 0.05 / 31                                    # din generators minsta relativa avstånd
bad = [p for p in (50, 100, 250, 500, 1000, 2000)
       if bin_of(p) == bin_of(p * (1 + d))]
print("kollisioner:", bad or "inga")