"""
Delar upp num-lossen på de tre sorters N-token facitet faktiskt innehåller, och
testar om nivåfelen är förväxlingar mellan nivåer hos samma emitter.

    python num_breakdown.py runs/20260911_120000/model.pt

    antal    N<n> efter LEVELS      -- antalet nivåer, 1..max_levels
    nivåbin  N<bin> i definitionen  -- nivåerna, 0..n_bins-1
    längd    N<k> i DWELL-blocket   -- dwelltider i pulser

Alla tre får i dag samma ordinala smoothing och samma vikt i lossen, trots att de
lever i helt olika skalor.

Kolumner
--------
  n          antal token i gruppen
  loss       medel-korsentropi mot det smoothade målet (uppdelningen av loss_num)
  massa±w    sannolikhet modellen lägger inom smooth_width från facit (mål: 1.0)
  exakt      andel där argmax träffar facit
  inom±w     andel där argmax ligger inom smooth_width
  |fel|      medelavstånd i enheter (bins / nivåer / pulser)
  median     samma sak, median -- skiljer sig |fel| mycket åt är felen en svans
  p95        95:e percentilen

Förväxlingstestet
-----------------
För varje nivåbin där argmax missar med mer än w: landar gissningen inom w från
NÅGON ANNAN sann nivå hos samma emitter? Hög andel = modellen kan uppsättningen
men inte vilken nivå som hör till vilket fack. Låg andel = den gissar på värden
som inte finns i signalen alls, och då är det upplösning eller instabilitet.
"""
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import cfg
from data import collate, make_pairs
from loss import ordinal_targets
from vocab import IS_NUM, NUM_START, PAD, VOCAB

N_EMITTERS = 64


def n_tokens_of_row(ids):
    """-> lista med (position, grupp, index) för varje N-token, enligt labels.to_tokens."""
    toks = [VOCAB[i] for i in ids.tolist() if i != PAD]
    out, i = [], 0
    if not toks or toks[0] != "LEVELS":
        return out
    out.append((1, "antal", 0))
    try:
        n = int(toks[1][1:])
    except (ValueError, IndexError):
        return out
    i = 2
    for k in range(n):                       # L<k>  N<bin>
        if i + 1 >= len(toks):
            return out
        out.append((i + 1, "nivåbin", k))
        i += 2
    if i >= len(toks) or toks[i] != "ORDER":
        return out
    i += 2                                   # ORDER + FIXED/RANDOM
    while i < len(toks) and toks[i].startswith("L") and toks[i][1:].isdigit():
        i += 1                               # nivåordningen, inga N-token
    if i >= len(toks) or toks[i] != "DWELL":
        return out
    i += 2                                   # DWELL + FIXED/RANGE
    k = 0
    while i < len(toks) and toks[i] != "END":
        if toks[i] != "INF":
            out.append((i, "längd", k))
        i += 1
        k += 1
    return out


def _target_entropy(w):
    q = np.array([np.exp(-0.5 * o ** 2) for o in range(-w, w + 1)])
    q /= q.sum()
    return float(-(q * np.log(q)).sum())


def main(ckpt):
    from model import build_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    pairs = make_pairs(np.random.default_rng(cfg.eval_seed), N_EMITTERS, 0.0)
    src, tgt_in, tgt_out = next(iter(DataLoader(pairs, batch_size=len(pairs),
                                                shuffle=False, collate_fn=collate)))
    src = {k: v.to(dev) for k, v in src.items()}
    with torch.no_grad():
        logits = model(src, tgt_in.to(dev)).float()
        logp = F.log_softmax(logits, -1)
        per_tok = -(ordinal_targets(tgt_out.to(dev)) * logp).sum(-1).cpu()
        prob = logp.exp().cpu()

    w = cfg.smooth_width
    bin_us = (cfg.pri_max - cfg.pri_min) / (cfg.n_bins - 1)
    groups = defaultdict(lambda: defaultdict(list))
    rows = []                                # (sanna nivåbins, [(k, sann, gissad)])

    for r in range(tgt_out.size(0)):
        recs = n_tokens_of_row(tgt_out[r])
        lvl = []
        for pos, grp, idx in recs:
            if pos >= tgt_out.size(1) or not IS_NUM[tgt_out[r, pos]]:
                continue
            t = int(tgt_out[r, pos])
            p = prob[r, pos]
            a = int(p.argmax())
            lo, hi = max(NUM_START, t - w), min(len(VOCAB) - 1, t + w)
            g = groups[grp]
            g["loss"].append(float(per_tok[r, pos]))
            g["mass"].append(float(p[lo:hi + 1].sum()))
            g["err"].append(abs(a - t))
            g["idx"].append(idx)
            if grp == "nivåbin":
                lvl.append((idx, t, a))
        if lvl:
            rows.append(([t for _, t, _ in lvl], lvl))

    print(f"checkpoint : {ckpt}")
    print(f"n_bins {cfg.n_bins}  ({bin_us:.4f} µs/bin)   smooth_width {w}   "
          f"±{w} bins = ±{w * bin_us:.3f} µs")
    print(f"golv för num vid w={w} : {_target_entropy(w):.3f}")
    print()
    head = (f"{'grupp':<10}{'n':>7}{'loss':>9}{'massa±w':>10}{'exakt':>8}"
            f"{'inom±w':>9}{'|fel|':>9}{'median':>8}{'p95':>7}")
    print(head)
    print("-" * len(head))
    for grp in ("antal", "nivåbin", "längd"):
        g = groups.get(grp)
        if not g:
            continue
        err = np.array(g["err"])
        print(f"{grp:<10}{len(err):>7}{np.mean(g['loss']):>9.3f}{np.mean(g['mass']):>10.3f}"
              f"{(err == 0).mean():>8.3f}{(err <= w).mean():>9.3f}"
              f"{err.mean():>9.2f}{np.median(err):>8.0f}{np.percentile(err, 95):>7.0f}")

    g = groups.get("nivåbin")
    if g:
        print("\nnivåbin per nivåindex (L0 = lägsta bin):")
        idx = np.array(g["idx"]); err = np.array(g["err"]); ls = np.array(g["loss"])
        for k in sorted(set(idx.tolist()))[:8]:
            m = idx == k
            print(f"  L{k:<3} n={m.sum():>5}  loss {ls[m].mean():>6.3f}  "
                  f"exakt {(err[m] == 0).mean():.3f}  |fel| {err[m].mean():>7.1f} bins "
                  f"= {err[m].mean() * bin_us:.2f} µs")

        # ---- förväxlingstest
        miss = other = far = 0
        for true_bins, lvl in rows:
            for k, t, a in lvl:
                if abs(a - t) <= w:
                    continue
                miss += 1
                if any(abs(a - u) <= w for u in true_bins if u != t):
                    other += 1
                else:
                    far += 1
        if miss:
            print(f"\nförväxlingstest, {miss} nivåbins utanför ±{w}:")
            print(f"  landar på en ANNAN sann nivå hos samma emitter : "
                  f"{other:>5}  ({other / miss:.1%})")
            print(f"  landar på ett värde som inte finns i facitet   : "
                  f"{far:>5}  ({far / miss:.1%})")

    g = groups.get("längd")
    if g:
        print("\nlängd per position i DWELL-blocket:")
        idx = np.array(g["idx"]); err = np.array(g["err"]); ls = np.array(g["loss"])
        for k in sorted(set(idx.tolist()))[:6]:
            m = idx == k
            print(f"  #{k:<3} n={m.sum():>5}  loss {ls[m].mean():>6.3f}  "
                  f"exakt {(err[m] == 0).mean():.3f}  |fel| {err[m].mean():>6.1f} pulser")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("användning: python num_breakdown.py runs/<körning>/model.pt")
    main(sys.argv[1])
