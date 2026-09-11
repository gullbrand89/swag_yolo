"""
Skiljer de två förklaringarna till en stigande num-loss åt.

    python sharpness.py runs/20260911_120000/model.pt

Mäter, över alla N-token i en evalbatch, hur mycket sannolikhet modellen lägger på
facitbinnen och dess grannar, och jämför med målfördelningen som ordinal smoothing
faktiskt optimerar mot.

Tolkning
--------
  modell[0] >> mål[0]  och vingarna << målet
        -> modellen är SKARPARE än målet. Lossen stiger av straffet, inte av fel.
           Kolla att argmax-träffen samtidigt är hög. Åtgärd: dämpa logit-skalan
           (uniform smoothing på strukturtoken) eller sänk smooth_width.

  massa inom +-w mycket under 1, och medel |fel| i bins växer
        -> modellen har blivit SÄMRE, fördelningen sprids ut. Åtgärd: sänk lr,
           förläng warmup, leta efter ett hopp i train.csv.

Kör den på checkpointen från steg 1000 och den från steg 3000 och jämför raderna.
"""
import math
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import cfg
from data import collate, make_pairs
from model import build_model
from vocab import IS_NUM, NUM, NUM_START, PAD

N_EMITTERS = 64


def main(ckpt):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    pairs = make_pairs(np.random.default_rng(cfg.eval_seed), N_EMITTERS, 0.0)
    src, tgt_in, tgt_out = next(iter(DataLoader(pairs, batch_size=len(pairs),
                                                shuffle=False, collate_fn=collate)))
    src = {k: v.to(dev) for k, v in src.items()}
    with torch.no_grad():
        prob = model(src, tgt_in.to(dev)).softmax(-1).float().cpu()

    isnum = IS_NUM[tgt_out] & (tgt_out != PAD)
    idx = tgt_out[isnum]                     # (N,) facit-id
    p = prob[isnum]                          # (N, V)
    n = idx.numel()

    w = cfg.smooth_width
    offs = list(range(-w, w + 1))
    lo, hi = NUM_START, NUM_START + len(NUM) - 1

    model_prof, in_range = [], []
    for off in offs:
        j = idx + off
        ok = (j >= lo) & (j <= hi)
        g = p.gather(1, j.clamp(lo, hi)[:, None]).squeeze(1)
        model_prof.append(float((g * ok).sum() / ok.sum()))
        in_range.append(int(ok.sum()))

    q = np.array([math.exp(-0.5 * off ** 2) for off in offs])
    q /= q.sum()

    pred = p.argmax(1)
    err = (pred - idx).abs().float()
    is_num_pred = IS_NUM[pred].float()

    print(f"checkpoint      : {ckpt}")
    print(f"N-token         : {n}   smooth_width = {w}   n_bins = {cfg.n_bins}")
    print()
    print("offset          :", "  ".join(f"{o:>6d}" for o in offs))
    print("mål   q         :", "  ".join(f"{v:6.3f}" for v in q))
    print("modell  p       :", "  ".join(f"{v:6.3f}" for v in model_prof))
    print()
    print(f"massa inom ±{w}   : mål {q.sum():.3f}   modell {sum(model_prof):.3f}")
    print(f"argmax exakt    : {(err == 0).mean():.3f}")
    print(f"argmax inom ±{w}  : {(err <= w).float().mean():.3f}")
    print(f"medel |fel| bins: {err.mean():.2f}   median {err.median():.0f}   "
          f"p95 {err.quantile(0.95):.0f}")
    print(f"argmax är N-tok : {is_num_pred.mean():.3f}")
    print(f"entropi (nat)   : modell {-(p * p.clamp_min(1e-12).log()).sum(1).mean():.3f}"
          f"   mål {-(q * np.log(q)).sum():.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("användning: python sharpness.py runs/<körning>/model.pt")
    main(sys.argv[1])
