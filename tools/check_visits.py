"""
Räcker observationsfönstret för att se nivåcykeln?

    python check_visits.py

check_dwell hoppar över signaler med färre än fyra besök. Det här skriptet svarar på
varför de är få: är emittern statisk, eller är dwelltiderna för långa för fönstret?

Skillnaden avgör om max_length ska kortas eller förlängas.

Nyckeltal
---------
  runs per signal        antal besök (sammanhängande körningar på samma nivå)
  runs / nivåer          hur många gånger hela nivåuppsättningen hinner visas.
                         < 1 betyder att minst en nivå aldrig ses som ett eget besök,
                         < 2 att cykelns ordning inte går att fastställa.
  dwelltid               längden på kompletta besök, i pulser
  krävt fönster          2 * antal nivåer * mediandwell -- ungefär vad som behövs för
                         att se cykeln två varv
"""

import numpy as np

from transformer_post_generator import emitter_module
from transformer_post_generator.config import cfg
from transformer_post_generator.vocab import bin_of

N_EMITTERS = 200


def as_signals(seqs):
    if isinstance(seqs, (list, tuple)):
        return [np.asarray(s, dtype=float).ravel() for s in seqs]
    a = np.asarray(seqs, dtype=float)
    return [a] if a.ndim == 1 else [row for row in a]


def runs_of(pri, tol):
    b = [bin_of(p) for p in pri]
    out, start = [], 0
    for i in range(1, len(b)):
        if abs(b[i] - b[i - 1]) > tol:
            out.append((b[start], i - start))
            start = i
    out.append((b[start], len(b) - start))
    return out


def pct(x, q):
    return np.percentile(x, q) if len(x) else float("nan")


def main():
    gen = emitter_module()
    data = gen.create_emitter_data(N_EMITTERS, cfg.samples_per_emitter, 0.0,
                                   cfg.noise_level, np.random.default_rng(cfg.eval_seed))

    n_runs, n_lvls, ratio, dwell, npulse = [], [], [], [], []
    few_static, few_slow = 0, 0
    need = []

    for seqs, label in data:
        k = len({bin_of(v) for v in np.asarray(label["levels"]).ravel()})
        for s in as_signals(seqs):
            r = runs_of(s, cfg.run_tol_bins)
            n_runs.append(len(r)); n_lvls.append(k); npulse.append(len(s))
            ratio.append(len(r) / max(1, k))
            mid = [L for _, L in r[1:-1]]
            dwell.extend(mid)
            if len(r) < 4:
                if k == 1:
                    few_static += 1
                else:
                    few_slow += 1
            d = np.median(mid) if mid else (len(s) / max(1, len(r)))
            need.append(2 * k * d)

    n_runs = np.array(n_runs); n_lvls = np.array(n_lvls)
    ratio = np.array(ratio); dwell = np.array(dwell); need = np.array(need)
    n = len(n_runs)

    print(f"{n} signaler, median {int(np.median(npulse))} pulser per signal, "
          f"run_tol_bins {cfg.run_tol_bins}\n")

    print("VARFÖR FÅ BESÖK")
    print(f"  signaler med < 4 runs              : {few_static + few_slow:>5}"
          f"  ({(few_static + few_slow) / n:.1%})")
    print(f"    därav statiska (1 nivå i facit)  : {few_static:>5}  ({few_static / n:.1%})"
          f"   <- legitimt")
    print(f"    därav flera nivåer, långa dwell  : {few_slow:>5}  ({few_slow / n:.1%})"
          f"   <- fönstret för kort")
    print()
    print("BESÖK")
    print(f"  runs per signal   : median {np.median(n_runs):.0f}   "
          f"p10 {pct(n_runs, 10):.0f}   p90 {pct(n_runs, 90):.0f}")
    print(f"  nivåer i facit    : median {np.median(n_lvls):.0f}")
    print(f"  runs / nivåer     : median {np.median(ratio):.1f}   "
          f"p10 {pct(ratio, 10):.1f}")
    for lo, hi, txt in [(0, 0.999, "< 1  någon nivå ses aldrig som eget besök"),
                        (1, 1.999, "1-2  varje nivå ses, men cykeln kan inte fastställas"),
                        (2, 3.999, "2-4  cykeln syns nätt och jämnt"),
                        (4, 10 ** 9, "4+   gott om varv")]:
        m = (ratio >= lo) & (ratio <= hi)
        print(f"      {txt:<48} {m.sum():>5}  ({m.mean():.1%})")
    print()
    print("DWELLTID (kompletta besök, pulser)")
    print(f"  median {np.median(dwell):.0f}   p90 {pct(dwell, 90):.0f}   "
          f"p99 {pct(dwell, 99):.0f}   max {dwell.max():.0f}")
    print()
    print("KRÄVT FÖNSTER för två varv av cykeln (pulser)")
    print(f"  median {np.median(need):.0f}   p90 {pct(need, 90):.0f}   "
          f"p99 {pct(need, 99):.0f}")
    for L in (256, 512, 1024, 2048, 4096):
        print(f"    {L:>5} pulser räcker för {np.mean(need <= L):.1%} av signalerna")


if __name__ == "__main__":
    main()
