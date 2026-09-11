"""Vokabulär och binning, byggda från cfg."""
import math
import torch
from config import cfg

SPECIAL = ["<pad>", "<bos>", "<eos>"]
STRUCT = ["LEVELS", "ORDER", "FIXED", "RANDOM", "RANGE", "DWELL", "INF", "END"]
LEVEL_NAMES = [f"L{i}" for i in range(cfg.max_levels)]
NUM = [f"N{i}" for i in range(cfg.max_int + 1)]
VOCAB = SPECIAL + STRUCT + LEVEL_NAMES + NUM
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, EOS = TOK2ID["<pad>"], TOK2ID["<bos>"], TOK2ID["<eos>"]
NUM_START = TOK2ID["N0"]
IS_NUM = torch.zeros(len(VOCAB), dtype=torch.bool)
IS_NUM[NUM_START:NUM_START + len(NUM)] = True

assert cfg.max_int >= cfg.n_bins - 1, "max_int måste täcka n_bins-1"


# ---------------- facit (linjär, emitterrymd)
def bin_of(pri):
    b = int((pri - cfg.pri_min) / (cfg.pri_max - cfg.pri_min) * (cfg.n_bins - 1))
    return max(0, min(cfg.n_bins - 1, b))

def pri_of_bin(b):
    return cfg.pri_min + (b + 0.5) / (cfg.n_bins - 1) * (cfg.pri_max - cfg.pri_min)


# ---------------- input (log, observationsrymd; sista bin = overflow)
def in_bin_of(pri):
    if pri >= cfg.in_max:
        return cfg.in_bins - 1
    x = (max(pri, cfg.in_min) - cfg.in_min) / (cfg.in_max - cfg.in_min)
    return max(0, min(cfg.in_bins - 2, int(x * (cfg.in_bins - 2))))

def in_cont_of(pri):
    x = (min(max(pri, cfg.in_min), cfg.in_max) - cfg.in_min) / (cfg.in_max - cfg.in_min)
    return x * 2 - 1


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
