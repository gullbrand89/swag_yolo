"""
Går det ÖVERHUVUDTAGET att skilja ORDER FIXED från ORDER RANDOM i signalen?

    python -m tools.check_order
    python -m tools.check_order --variants stagger jitter --n 400
    python -m tools.check_order --pulses 1024      # vad ett dubbelt fönster skulle ge
    python -m tools.check_order --drop 0.20

Varför
------
Modellen ligger på klassprioren för FIXED/RANDOM även på stagger och jitter, där den
har hundratals kompletta cykler att jämföra. Två förklaringar återstår: antingen bär
facitet ingen information (bugg), eller så bär det information som modellen inte lär
sig (inlärningsproblem). Det här skriptet avgör vilket, utan att träna någonting.

Metoden
-------
Pulserna kollapsas till BESÖK. För varje kandidatperiod p röstar besök i, i+p, i+2p...
fram vilken nivå som hör till cykelposition i mod p, och konfidensen är andelen besök
som stämmer med den framröstade cykeln. En fast cykel ger konfidens nära 1 vid rätt
period. En slumpad ordning ger ingen period alls och stannar nära 1/k.

Två fällor som skriptet undviker
--------------------------------
  * Med p nära antalet besök får varje cykelposition bara en röst, och då blir
    konfidensen trivialt 1.0 även för rent brus. Därför krävs minst MIN_ROSTER
    röster per position.
  * Kanterna: första och sista besöket är avklippta av observationsfönstret och
    räknas inte.

Läsning av utskriften
---------------------
  konf@sann      konfidens vid facitets egen period. Är den inte nära 1.0 för de
                 fasta varianterna stämmer inte facitet med signalen -- det är en
                 bugg, och då är det meningslöst att träna vidare på den etiketten.
  konf@bäst      högsta konfidens över alla testade perioder. Det är vad en
                 detektor utan facit kan få fram.
  TAK            träffsäkerheten hos den bästa möjliga tröskeln på konf@bäst.
                 Det är den övre gränsen för order_type_ok. Ligger taket högt
                 medan modellen ligger på prioren är det ett inlärningsproblem.
"""
import argparse
from collections import Counter

import numpy as np

from transformer_post_generator.all_emitters import GEN, VARIANTS, create_emitter_data
from transformer_post_generator.verify import _quantize, _visits
from transformer_post_generator.vocab import bin_of

MIN_ROSTER = 3          # minsta antal röster per cykelposition
TOL_BINS = 1


def besok(pri, levels, trim_edges=True):
    """Pulser -> följd av nivåindex, ett per besök."""
    lvl_uniq = sorted(set(bin_of(v) for v in levels))
    obs = np.array([bin_of(p) for p in _quantize(pri)])
    seq, runs, _ = _visits(obs, lvl_uniq, TOL_BINS)
    keep = list(zip(seq, runs))
    if trim_edges and len(keep) > 2:
        keep = keep[1:-1]
    return [v for v, _ in keep if v >= 0], len(lvl_uniq)


def konfidens(obs, p):
    """Andel besök som stämmer med den majoritetsröstade cykeln av längd p."""
    if p < 1 or len(obs) < p * MIN_ROSTER:
        return None
    rost = [Counter() for _ in range(p)]
    for i, v in enumerate(obs):
        rost[i % p][v] += 1
    cykel = [c.most_common(1)[0][0] for c in rost]
    return sum(1 for i, v in enumerate(obs) if v == cykel[i % p]) / len(obs)


def _min_period(x):
    n = len(x)
    for p in range(1, n + 1):
        if n % p == 0 and all(x[i] == x[i % p] for i in range(n)):
            return x[:p]
    return list(x)


def analysera(pri, lab):
    """-> dict, eller None om det inte finns nog med besök för att säga något."""
    obs, k = besok(pri, lab["levels"])
    if len(obs) < MIN_ROSTER:
        return None

    # facitets egen period, i antal besök
    lvl_uniq = sorted(set(bin_of(v) for v in lab["levels"]))
    idx = [lvl_uniq.index(bin_of(v)) for v in lab["levels"]]
    sann_p = len(_min_period(idx))

    p_max = min(2 * k + 2, len(obs) // MIN_ROSTER)
    kandidater = {p: konfidens(obs, p) for p in range(1, p_max + 1)}
    kandidater = {p: c for p, c in kandidater.items() if c is not None}
    if not kandidater:
        return dict(k=k, besok=len(obs), sann_p=sann_p, nog=False,
                    fixed=lab["order_fixed"], variant=lab["variant"])

    p_bast = max(kandidater, key=kandidater.get)
    return dict(k=k, besok=len(obs), sann_p=sann_p, nog=True,
                fixed=lab["order_fixed"], variant=lab["variant"],
                konf_sann=konfidens(obs, sann_p), konf_bast=kandidater[p_bast],
                p_bast=p_bast, slump=1.0 / max(k, 1))


def tak(rader):
    """Bästa möjliga tröskel på konf_bast -> (träffsäkerhet, tröskel)."""
    s = [r for r in rader if r["nog"]]
    if not s or len({r["fixed"] for r in s}) < 2:
        return None, None
    v = sorted({r["konf_bast"] for r in s})
    bast = (0.0, 0.0)
    for t in v:
        # >= t  ->  FIXED
        acc = sum((r["konf_bast"] >= t) == r["fixed"] for r in s) / len(s)
        if acc > bast[0]:
            bast = (acc, t)
    return bast


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="emittrar per variant")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pulses", type=int, default=None,
                    help="tillfälligt annat observationsfönster, för att se vad "
                         "ett längre fönster skulle ge innan du tränar om")
    a = ap.parse_args()

    if a.pulses:
        GEN.n_pulses = a.pulses

    namn = a.variants or [v for v in VARIANTS if GEN.weights[v] > 0]
    print(f"n_pulses {GEN.n_pulses}   lim_states {GEN.lim_states}   "
          f"lim_ds_k {GEN.lim_ds_k}   lim_val_q {GEN.lim_val_q}   "
          f"drop {a.drop}   {a.n} emittrar/variant")
    print(f"minst {MIN_ROSTER} röster per cykelposition\n")

    head = (f"{'variant':<18}{'facit':>8}{'n':>5}{'besök':>7}{'k':>5}{'sann p':>8}"
            f"{'p testbar':>11}{'konf@sann':>11}{'konf@bäst':>11}{'slump':>8}")
    print(head)
    print("-" * len(head))

    alla = []
    for v in namn:
        rng = np.random.default_rng([a.seed, VARIANTS.index(v)])
        data = create_emitter_data(a.n, 1, a.drop, None, rng, only=[v])
        rader = [r for r in (analysera(s[0], lab) for s, lab in data) if r]
        if not rader:
            print(f"{v:<18}{'-':>8}{0:>5}   (för få besök för att säga något)")
            continue
        alla += rader
        ok = [r for r in rader if r["nog"]]
        fix = "FIXED" if rader[0]["fixed"] else "RANDOM"
        print(f"{v:<18}{fix:>8}{len(rader):>5}{med([r['besok'] for r in rader]):>7.0f}"
              f"{med([r['k'] for r in rader]):>5.0f}{med([r['sann_p'] for r in rader]):>8.0f}"
              f"{sum(r.get('konf_sann') is not None for r in ok) / max(len(ok), 1):>11.1%}"
              f"{med([r['konf_sann'] for r in ok]):>11.3f}"
              f"{med([r['konf_bast'] for r in ok]):>11.3f}"
              f"{med([r['slump'] for r in ok]):>8.3f}")

    acc, t = tak(alla)
    ok = [r for r in alla if r["nog"]]
    print()
    if acc is None:
        print("Bara en klass i urvalet -- kör med både fasta och slumpade varianter.")
        return
    andel_fixed = sum(r["fixed"] for r in ok) / len(ok)
    print(f"TAK för order_type_ok: {acc:.3f}  (tröskel konf@bäst >= {t:.3f}, n={len(ok)})")
    print(f"  majoritetsgissning:  {max(andel_fixed, 1 - andel_fixed):.3f}")
    print(f"  modellen i 12000-stegskörningen: 0.677")
    print()
    print("Tröskeln är vald på samma data som den mäts på, så talet är optimistiskt.")
    print("Det är en ÖVRE gräns -- ligger den nära majoritetsgissningen bär signalen")
    print("ingen information och modellen kan inte klandras. Ligger den klart över är")
    print("informationen där och modellen missar den.")


if __name__ == "__main__":
    main()
