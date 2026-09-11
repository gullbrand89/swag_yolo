"""
Hur mycket av facitet syns faktiskt i signalen?

    python label_coverage.py

Ingen modell inblandad. För varje emitter jämförs de nivåer facitet påstår med de
nivåer som går att se i den observerade pulsföljden. Är täckningen låg är en del av
facitet obestämbart per konstruktion, och ingen mängd träning kan laga det -- då är
det facitet eller fönsterlängden som ska ändras, inte modellen.

Körs med drop_rate = 0 så att inga bortfall slår ihop två PRI till en summa.

Utskrift
--------
  nivåer i facit        fördelning av len(label["levels"])
  täckta nivåer         hur många av dem som har minst en observerad puls inom ±tol bins
  täckningsgrad         täckta / totalt, per emitter
  helt täckta           andel emittrar där ALLA facitnivåer syns
  obesvarade kluster    observerade nivåer som INTE finns i facit (motsatt fel)
  besök per nivå        hur många pulser som faller på varje facitnivå -- nivåer med
                        0 eller 1 besök är i praktiken ogissningsbara
"""
import importlib
from collections import Counter

import numpy as np

from config import cfg
from vocab import bin_of

TOL_BINS = 4          # hur nära en observerad puls måste ligga för att räknas som besök
N_EMITTERS = 300


def as_signals(seqs):
    if isinstance(seqs, (list, tuple)):
        return [np.asarray(s, dtype=float).ravel() for s in seqs]
    a = np.asarray(seqs, dtype=float)
    return [a] if a.ndim == 1 else [row for row in a]


def main():
    gen = importlib.import_module(cfg.emitter)
    data = gen.create_emitter_data(N_EMITTERS, cfg.samples_per_emitter, 0.0,
                                   cfg.noise_level, np.random.default_rng(cfg.eval_seed))

    n_lvls, n_cov, cov_frac, extra, visits = [], [], [], [], []
    full = 0
    n_sig = 0

    for seqs, label in data:
        lvl_bins = sorted({bin_of(v) for v in np.asarray(label["levels"]).ravel()})
        for s in as_signals(seqs):
            n_sig += 1
            obs = np.array([bin_of(p) for p in s])
            counts = [int(np.sum(np.abs(obs - b) <= TOL_BINS)) for b in lvl_bins]
            c = sum(1 for x in counts if x > 0)
            n_lvls.append(len(lvl_bins))
            n_cov.append(c)
            cov_frac.append(c / max(1, len(lvl_bins)))
            visits.extend(counts)
            full += int(c == len(lvl_bins))
            # observerade bins som inte hör till någon facitnivå
            far = sum(1 for o in np.unique(obs)
                      if not any(abs(int(o) - b) <= TOL_BINS for b in lvl_bins))
            extra.append(far)

    n_lvls = np.array(n_lvls); n_cov = np.array(n_cov)
    cov_frac = np.array(cov_frac); visits = np.array(visits); extra = np.array(extra)

    print(f"{n_sig} signaler från {N_EMITTERS} emittrar, drop = 0, "
          f"tolerans ±{TOL_BINS} bins (±{TOL_BINS * (cfg.pri_max - cfg.pri_min) / (cfg.n_bins - 1):.2f} µs)")
    print(f"pulser per signal : median {int(np.median([len(s) for seqs, _ in data for s in as_signals(seqs)]))}")
    print()
    print(f"nivåer i facit    : median {np.median(n_lvls):.0f}   "
          f"medel {n_lvls.mean():.1f}   min {n_lvls.min()}   max {n_lvls.max()}")
    print(f"täckta nivåer     : median {np.median(n_cov):.0f}   medel {n_cov.mean():.1f}")
    print(f"täckningsgrad     : medel {cov_frac.mean():.3f}   "
          f"p10 {np.percentile(cov_frac, 10):.3f}   p50 {np.percentile(cov_frac, 50):.3f}")
    print(f"helt täckta       : {full}/{n_sig} = {full / n_sig:.1%}")
    print(f"obesvarade kluster: medel {extra.mean():.1f} observerade bins per signal "
          f"utan motsvarighet i facit")
    print()
    print("besök per facitnivå (antal pulser inom toleransen):")
    for lo, hi, txt in [(0, 0, "0 besök  (osynlig)"), (1, 1, "1 besök"),
                        (2, 4, "2-4 besök"), (5, 19, "5-19 besök"), (20, 10**9, "20+ besök")]:
        m = (visits >= lo) & (visits <= hi)
        print(f"  {txt:<20} {m.sum():>7}  ({m.mean():.1%})")
    print()
    print("fördelning av antalet nivåer i facit:")
    for k, v in sorted(Counter(n_lvls.tolist()).items()):
        print(f"  {k:>3} nivåer : {v:>5}  ({v / len(n_lvls):.1%})")


if __name__ == "__main__":
    main()
