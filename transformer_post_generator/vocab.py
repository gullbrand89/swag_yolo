"""
Vokabulär och binning, byggda från cfg.

Numeriska token är uppdelade i TVÅ rymder:

    B0..B<n_bins-1>   nivåer, bins i emitterrymden
    D0..D<max_dur>    dwelltider i pulser

De delade tidigare en enda N-rymd, vilket innebar att samma inbäddning fick bära två
orelaterade betydelser och att ordinal utjämning tillämpades likadant på båda. En bin
är en mätning med ändlig noggrannhet; en dwelltid är ett exakt antal. Med skilda
rymder kan de få var sitt sigma, och utdatarymden vid varje position innehåller bara
lagliga värden.

Antalstoken efter LEVELS är borttaget -- se labels.py.
"""
import math

import torch

from .config import cfg

SPECIAL = ["<pad>", "<bos>", "<eos>"]
STRUCT = ["LEVELS", "ORDER", "FIXED", "RANDOM", "RANGE", "DWELL", "INF", "END"]
LEVEL_NAMES = [f"L{i}" for i in range(cfg.max_levels)]
BIN = [f"B{i}" for i in range(cfg.n_bins)]
DUR = [f"D{i}" for i in range(cfg.max_dur + 1)]
VOCAB = SPECIAL + STRUCT + LEVEL_NAMES + BIN + DUR
TOK2ID = {t: i for i, t in enumerate(VOCAB)}

PAD, BOS, EOS = TOK2ID["<pad>"], TOK2ID["<bos>"], TOK2ID["<eos>"]
BIN_START, DUR_START = TOK2ID["B0"], TOK2ID["D0"]

IS_BIN = torch.zeros(len(VOCAB), dtype=torch.bool)
IS_DUR = torch.zeros(len(VOCAB), dtype=torch.bool)
IS_BIN[BIN_START:BIN_START + len(BIN)] = True
IS_DUR[DUR_START:DUR_START + len(DUR)] = True
IS_NUM = IS_BIN | IS_DUR

# Utjämningen får inte läcka över gränsen mellan rymderna: massa som hamnar på ett
# D-token när facitet är ett B-token vore inte "nästan rätt", det vore fel sorts svar.
RANGE_LO = torch.zeros(len(VOCAB), dtype=torch.long)
RANGE_HI = torch.zeros(len(VOCAB), dtype=torch.long)
SIGMA = torch.ones(len(VOCAB))
for _start, _n, _sig in ((BIN_START, len(BIN), cfg.sigma_bin),
                         (DUR_START, len(DUR), cfg.sigma_dur)):
    RANGE_LO[_start:_start + _n] = _start
    RANGE_HI[_start:_start + _n] = _start + _n - 1
    SIGMA[_start:_start + _n] = _sig

_wmin = math.ceil(3 * max(cfg.sigma_bin, cfg.sigma_dur))
assert cfg.smooth_width >= _wmin, (
    f"smooth_width={cfg.smooth_width} kapar en fördelning med sigma upp till "
    f"{max(cfg.sigma_bin, cfg.sigma_dur)}. Sätt smooth_width >= {_wmin}.")
    _w_in = (cfg.in_max - cfg.in_min) / (cfg.in_bins - 2)
_w_out = (cfg.pri_max - cfg.pri_min) / (cfg.n_bins - 1)
assert cfg.in_min == cfg.pri_min and abs(_w_in - _w_out) < 1e-9, (
    f"input-bin {_w_in:.6f} µs != facit-bin {_w_out:.6f} µs: binsen glider isär "
    f"och nivåtoken blir tvetydiga. Sätt in_max = in_min + "
    f"(in_bins-2)/(n_bins-1) * (pri_max-pri_min).")



# ---------------- facit (linjär, emitterrymd)
def bin_of(pri):
    b = int((pri - cfg.pri_min) / (cfg.pri_max - cfg.pri_min) * (cfg.n_bins - 1))
    return max(0, min(cfg.n_bins - 1, b))


def pri_of_bin(b):
    return cfg.pri_min + (b + 0.5) / (cfg.n_bins - 1) * (cfg.pri_max - cfg.pri_min)


# ---------------- input (linjär, observationsrymd; sista bin = overflow)
def in_bin_of(pri):
    if pri >= cfg.in_max:
        return cfg.in_bins - 1
    x = (max(pri, cfg.in_min) - cfg.in_min) / (cfg.in_max - cfg.in_min)
    return max(0, min(cfg.in_bins - 2, int(x * (cfg.in_bins - 2))))


def in_cont_of(pri):
    x = (min(max(pri, cfg.in_min), cfg.in_max) - cfg.in_min) / (cfg.in_max - cfg.in_min)
    return x * 2 - 1


# ---------------- id <-> token
def safe_ids(tokens):
    for t in tokens:
        if t not in TOK2ID:
            raise KeyError(f"okänt token {t!r} i facit: {' '.join(map(str, tokens))}")
    return [TOK2ID[t] for t in tokens]


def ids_to_tokens(ids):
    out = []
    for i in ids.tolist():
        if i == EOS: break
        if i in (BOS, PAD): continue
        out.append(VOCAB[i])
    return out
