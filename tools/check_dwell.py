"""
Stämmer DWELL-blocket i facitet med vad signalen faktiskt visar?

    python check_dwell.py

to_tokens avgör FIXED/RANGE enbart på flaggan length_fixed och tittar aldrig på
värdena, och roundtrip_ok bevarar flaggan. Alltså kan facitet vara fel utan att något
i testsviten märker det. Det här skriptet segmenterar signalen i besök, mäter de
faktiska dwelltiderna och jämför med vad facitet påstår.

Kontrollerna
------------
  FIXED men operiodisk    facitet påstår en cykel som inte finns i signalen
  RANGE men periodisk     det ÄR en cykel, du kastar bort den
  cykeln matchar inte     FIXED, periodisk, men fel värden (eller fel rotation)
  max utanför intervallet  RANGE där observerade längder ligger utanför [lo, hi]
  intervallet för vitt     RANGE där lo/hi är generatorns sanna gränser i stället för
                           de observerade -- då är maxvärdet obestämbart för modellen

Körs med drop = 0, annars slås besök ihop av bortfall.
"""
from collections import Counter

import numpy as np

from transformer_post_generator import emitter_module
from transformer_post_generator.config import cfg
from transformer_post_generator.labels import _min_period
from transformer_post_generator.vocab import bin_of

N_EMITTERS = 200
MAX_PERIOD = 8          # längsta längdcykel vi letar efter
SHOW = 5                # antal exempel att skriva ut per feltyp


def as_signals(seqs):
    if isinstance(seqs, (list, tuple)):
        return [np.asarray(s, dtype=float).ravel() for s in seqs]
    a = np.asarray(seqs, dtype=float)
    return [a] if a.ndim == 1 else [row for row in a]


def runs_of(pri, tol):
    """-> [(bin, längd)] för varje sammanhängande besök på samma nivå."""
    b = [bin_of(p) for p in pri]
    out, start = [], 0
    for i in range(1, len(b)):
        if abs(b[i] - b[i - 1]) > tol:
            out.append((b[start], i - start))
            start = i
    out.append((b[start], len(b) - start))
    return out


def period_of(xs, max_p=MAX_PERIOD):
    """Minsta p <= max_p så att xs är en exakt upprepning av sina p första. Annars None."""
    n = len(xs)
    for p in range(1, min(max_p, n) + 1):
        if all(xs[i] == xs[i % p] for i in range(n)):
            return p
    return None


def main():
    gen = emitter_module()
    data = gen.create_emitter_data(N_EMITTERS, cfg.samples_per_emitter, 0.0,
                                   cfg.noise_level, np.random.default_rng(cfg.eval_seed))

    tally = Counter()
    examples = {k: [] for k in ("fixed_operiodisk", "range_periodisk", "cykel_fel",
                                "utanför", "för_vitt", "för_få_besök")}
    n = 0

    for e, (seqs, label) in enumerate(data):
        fixed = bool(label["length_fixed"])
        lab = [x for x in label["lengths"] if x is not None]
        for s in as_signals(seqs):
            n += 1
            runs = runs_of(s, cfg.run_tol_bins)
            if len(runs) < 4:                      # första och sista är avkapade
                tally["för_få_besök"] += 1
                examples["för_få_besök"].append((e, len(runs)))
                continue
            obs = [L for _, L in runs[1:-1]]
            p = period_of(obs)

            if fixed:
                if p is None:
                    tally["fixed_operiodisk"] += 1
                    examples["fixed_operiodisk"].append((e, lab, obs[:10]))
                else:
                    cyc = obs[:p]
                    want = _min_period(lab)
                    rot = any(cyc[i:] + cyc[:i] == list(want)
                              for i in range(len(cyc))) if len(cyc) == len(want) else False
                    if not rot:
                        tally["cykel_fel"] += 1
                        examples["cykel_fel"].append((e, list(want), cyc))
                    else:
                        tally["ok_fixed"] += 1
            else:
                if not lab:
                    tally["range_tom"] += 1
                    continue
                lo, hi = min(lab), max(lab)
                olo, ohi = min(obs), max(obs)
                if p is not None and len(obs) >= 3 * p:
                    tally["range_periodisk"] += 1
                    examples["range_periodisk"].append((e, p, obs[:10]))
                elif olo < lo or ohi > hi:
                    tally["utanför"] += 1
                    examples["utanför"].append((e, (lo, hi), (olo, ohi)))
                elif hi - ohi > max(2, 0.25 * (hi - lo)):
                    tally["för_vitt"] += 1
                    examples["för_vitt"].append((e, (lo, hi), (olo, ohi)))
                else:
                    tally["ok_range"] += 1

    print(f"{n} signaler från {N_EMITTERS} emittrar, drop = 0, "
          f"run_tol_bins {cfg.run_tol_bins}\n")
    order = ["ok_fixed", "ok_range", "fixed_operiodisk", "cykel_fel",
             "range_periodisk", "utanför", "för_vitt", "range_tom", "för_få_besök"]
    for k in order:
        if tally[k]:
            print(f"  {k:<18} {tally[k]:>6}  ({tally[k] / n:.1%})")

    def dump(key, header, fmt):
        if not examples.get(key):
            return
        print(f"\n{header}")
        for row in examples[key][:SHOW]:
            print("   " + fmt(row))

    dump("fixed_operiodisk", "FIXED men ingen cykel i signalen (facit / observerat):",
         lambda r: f"emitter {r[0]:>3}  facit {r[1]}  observerat {r[2]}")
    dump("cykel_fel", "FIXED och periodisk, men cykeln stämmer inte:",
         lambda r: f"emitter {r[0]:>3}  facit {r[1]}  observerat {r[2]}")
    dump("range_periodisk", "RANGE fast längderna är en cykel:",
         lambda r: f"emitter {r[0]:>3}  period {r[1]}  observerat {r[2]}")
    dump("utanför", "RANGE där observerade längder ligger utanför [lo, hi]:",
         lambda r: f"emitter {r[0]:>3}  facit {r[1]}  observerat {r[2]}")
    dump("för_vitt", "RANGE där facitets max aldrig realiseras (obestämbart):",
         lambda r: f"emitter {r[0]:>3}  facit {r[1]}  observerat {r[2]}")


if __name__ == "__main__":
    main()
