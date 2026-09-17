"""
Delar upp num-lossen på nivåbins och dwelltider.

    python num_breakdown.py runs/20260916_101955/model.pt

Uppdaterad för den delade tokenrymden. Tidigare delade antal, nivåer och längder
en enda N-rymd, och det här skriptet fick gå igenom grammatiken för att gissa
vilken roll varje token hade utifrån sin POSITION. Nu står rollen i token självt:
B är en nivåbin, D är en dwelltid. Hela den gissningen är borta, och därmed också
möjligheten att den gissade fel.

Antalstoken finns inte längre -- antalet nivåer läses av genom att räkna par, och
är därför inte något modellen kan ha fel på.

Kolumner
--------
  n          antal token i gruppen
  loss       medel-korsentropi mot det smoothade målet
  golv       målfördelningens entropi för gruppens sigma -- lossen kan inte gå under
  massa±w    sannolikhet modellen lägger inom smooth_width från facit (mål: 1.0)
  exakt      andel där argmax träffar facit
  inom±w     andel där argmax ligger inom smooth_width
  |fel|      medelavstånd i gruppens egen enhet (bins respektive pulser)
  median     skiljer sig |fel| mycket från medianen är felen en svans, inte en
             allmän oskärpa
  p95        svansens storlek
"""
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformer_post_generator.config import cfg
from transformer_post_generator.data import collate, make_pairs
from transformer_post_generator.loss import ordinal_targets
from transformer_post_generator.vocab import IS_BIN, IS_DUR, PAD, RANGE_HI, RANGE_LO

N_EMITTERS = 64


def roller(ids):
    """
    -> [(position, grupp, index_i_sin_grupp)] för varje numeriskt token.

    Ingen grammatik behövs: prefixet avgör rollen. Indexet räknas i den ordning
    token dyker upp, så nivåbin 0 är den lägsta nivån och dwelltid 0 är den
    första i DWELL-blocket.
    """
    out, b_i, d_i = [], 0, 0
    for pos, t in enumerate(ids.tolist()):
        if t == PAD:
            continue
        if IS_BIN[t]:
            out.append((pos, "nivåbin", b_i)); b_i += 1
        elif IS_DUR[t]:
            out.append((pos, "dwelltid", d_i)); d_i += 1
    return out


def golv(sigma, w):
    """Målfördelningens entropi -- den nedre gränsen för korsentropin."""
    s = max(float(sigma), 1e-6)
    q = np.array([np.exp(-0.5 * (o / s) ** 2) for o in range(-w, w + 1)])
    q = q / q.sum()
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


def main(ckpt):
    from transformer_post_generator.model import build_model
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
    grupper = defaultdict(lambda: defaultdict(list))
    rader = []

    for r in range(tgt_out.size(0)):
        lvl = []
        for pos, grp, idx in roller(tgt_out[r]):
            t = int(tgt_out[r, pos])
            p = prob[r, pos]
            a = int(p.argmax())
            lo, hi = int(RANGE_LO[t]), int(RANGE_HI[t])
            g = grupper[grp]
            g["loss"].append(float(per_tok[r, pos]))
            g["mass"].append(float(p[max(lo, t - w):min(hi, t + w) + 1].sum()))
            # fel räknas bara inom samma rymd; ett D-svar på en B-fråga är inte
            # "långt ifrån", det är fel sorts svar och redovisas separat
            g["ratt_rymd"].append(lo == int(RANGE_LO[a]) and (IS_BIN[a] or IS_DUR[a]))
            g["err"].append(abs(a - t))
            g["idx"].append(idx)
            if grp == "nivåbin":
                lvl.append((idx, t, a))
        if lvl:
            rader.append(([t for _, t, _ in lvl], lvl))

    print(f"checkpoint : {ckpt}")
    print(f"nivåbins {cfg.n_bins} ({bin_us:.4f} µs/bin, sigma {cfg.sigma_bin})   "
          f"dwelltider 0-{cfg.max_dur} (pulser, sigma {cfg.sigma_dur})   "
          f"smooth_width {w}")
    print()
    head = (f"{'grupp':<10}{'n':>7}{'loss':>9}{'golv':>8}{'massa±w':>10}{'exakt':>8}"
            f"{'inom±w':>9}{'|fel|':>9}{'median':>8}{'p95':>7}{'rätt rymd':>11}")
    print(head)
    print("-" * len(head))
    for grp, sig in (("nivåbin", cfg.sigma_bin), ("dwelltid", cfg.sigma_dur)):
        g = grupper.get(grp)
        if not g:
            continue
        err = np.array(g["err"])
        print(f"{grp:<10}{len(err):>7}{np.mean(g['loss']):>9.3f}{golv(sig, w):>8.3f}"
              f"{np.mean(g['mass']):>10.3f}{(err == 0).mean():>8.3f}"
              f"{(err <= w).mean():>9.3f}{err.mean():>9.2f}{np.median(err):>8.0f}"
              f"{np.percentile(err, 95):>7.0f}{np.mean(g['ratt_rymd']):>11.3f}")

    g = grupper.get("nivåbin")
    if g:
        print("\nnivåbin per nivåindex (0 = lägsta bin):")
        idx = np.array(g["idx"]); err = np.array(g["err"]); ls = np.array(g["loss"])
        for k in sorted(set(idx.tolist()))[:8]:
            m = idx == k
            print(f"  L{k:<3} n={m.sum():>5}  loss {ls[m].mean():>6.3f}  "
                  f"exakt {(err[m] == 0).mean():.3f}  |fel| {err[m].mean():>7.1f} bins "
                  f"= {err[m].mean() * bin_us:.2f} µs")

        miss = other = far = 0
        for sanna, lvl in rader:
            for k, t, a in lvl:
                if abs(a - t) <= w:
                    continue
                miss += 1
                if any(abs(a - u) <= w for u in sanna if u != t):
                    other += 1
                else:
                    far += 1
        if miss:
            print(f"\nförväxlingstest, {miss} nivåbins utanför ±{w}:")
            print(f"  landar på en ANNAN sann nivå hos samma emitter : "
                  f"{other:>5}  ({other / miss:.1%})")
            print(f"  landar på ett värde som inte finns i facitet   : "
                  f"{far:>5}  ({far / miss:.1%})")

    g = grupper.get("dwelltid")
    if g:
        print("\ndwelltid per position i DWELL-blocket:")
        idx = np.array(g["idx"]); err = np.array(g["err"]); ls = np.array(g["loss"])
        for k in sorted(set(idx.tolist()))[:6]:
            m = idx == k
            print(f"  #{k:<3} n={m.sum():>5}  loss {ls[m].mean():>6.3f}  "
                  f"exakt {(err[m] == 0).mean():.3f}  |fel| {err[m].mean():>6.1f} pulser")
        print("  (vid DWELL RANGE är #0 intervallets min och #1 dess max -- maxvärdet"
              "\n   realiseras sällan i fönstret och är därför svårare per konstruktion)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("användning: python num_breakdown.py runs/<körning>/model.pt")
    main(sys.argv[1])
