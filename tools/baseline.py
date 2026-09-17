"""
Klassisk baslinje: samma facit, utan modell.

    python -m tools.baseline
    python -m tools.baseline --n 200 --drops 0.0 0.1

Varför
------
Nästan varje fält i posten går att härleda direkt ur pulsföljden med en algoritm,
och koden för det fanns redan spridd i verify.py och tools/. Den här filen limmar
ihop den till en skattare med samma utdata som modellen, mätt med samma compare().

Utan den siffran går det inte att svara på varför uppgiften behöver en transformer.
Med den blir svaret en tabell: där baslinjen är bättre är modellen onödig, och där
den är sämre -- bortfall, brus, emittertyper ingen skrivit en regel för -- ligger
hela motiveringen.

Vad varje fält kommer ifrån
---------------------------
  LEVELS      histogram över binnade PRI, bins med minst min_pulses träffar
              (samma kod som nivåkontrollen i verify_label)
  besök       verify._visits: pulser -> (nivåindex per besök, runlängd per besök)
  ORDER-typ   majoritetsröstning över kandidatperioder, som tools/check_order.
              Högsta konfidens över alla perioder jämförs med en tröskel.
  ORDER-följd cykeln som rösterna ger, roterad till minsta rotation -- samma
              kanonisering som labels.to_tokens
  DWELL-typ   är runlängderna periodiska (FIXED) eller utspridda (RANGE)
  DWELL-värde minsta perioden av runlängderna, respektive (min, max)

Att ge baslinjen dess bästa chans
---------------------------------
En jämförelse där baslinjen är illa inställd är värdelös. Enda robusthetsratten här
är --min-pulser, och den är svept. En STARKARE klassisk baslinje skulle upptäcka
sammanslagningar explicit -- en puls vars PRI ligger nära dubbla grannarnas är
troligen två pulser med en tappad emellan, inte en ny nivå. Det är inte gjort här,
och det är den ärliga reservationen att skriva ut i rapporten.

Tröskeln
--------
FIXED/RANDOM avgörs av en tröskel på konfidensen. Den är inte fri: kör
tools/check_order, som räknar fram den bästa möjliga tröskeln och taket den ger.
Vid n_pulses = 512 låg den på 0.71 och taket på 0.883. Ändrar du fönstret eller
generatorns parametrar måste den mätas om.
"""
import argparse
from collections import Counter

import numpy as np

from transformer_post_generator.config import cfg
from transformer_post_generator.labels import to_tokens
from transformer_post_generator.runlog import compare
from transformer_post_generator.verify import _quantize, _visits
from transformer_post_generator.vocab import bin_of, pri_of_bin
from transformer_post_generator import emitter_module

KONF_TROSKEL = 0.71        # se docstring -- mät om med tools/check_order
MIN_ROSTER = 3             # minsta antal röster per cykelposition
MIN_PULSER = 8             # bins med färre pulser räknas som artefakter. Svept
                           # över 1/4/8/16: 8 är bästa kompromissen. Lägre släpper
                           # igenom sammanslagna pulser som falska nivåer vid
                           # bortfall, högre börjar kasta bort äkta korta besök.
TOL_BINS = 1


def _min_period(x):
    n = len(x)
    for p in range(1, n + 1):
        if n % p == 0 and all(x[i] == x[i % p] for i in range(n)):
            return list(x[:p])
    return list(x)


def _konfidens(obs, p):
    """Andel besök som stämmer med den majoritetsröstade cykeln av längd p."""
    if p < 1 or len(obs) < p * MIN_ROSTER:
        return None, None
    rost = [Counter() for _ in range(p)]
    for i, v in enumerate(obs):
        rost[i % p][v] += 1
    cykel = [c.most_common(1)[0][0] for c in rost]
    traff = sum(1 for i, v in enumerate(obs) if v == cykel[i % p])
    return traff / len(obs), cykel


def estimate(pri, trosk=KONF_TROSKEL, min_pulser=MIN_PULSER):
    """-> facit-dict i samma form som generatorns etikett, eller None."""
    obs_bins = np.array([bin_of(p) for p in _quantize(pri)])

    # ---- 1. nivåer: histogram, svaga bins bort
    uniq, counts = np.unique(obs_bins, return_counts=True)
    lvl_bins = sorted(int(b) for b in uniq[counts >= min_pulser])
    if not lvl_bins:
        return None
    levels_us = [pri_of_bin(b) for b in lvl_bins]

    # ---- 2. besök: pulser -> (nivåindex, runlängd)
    seq, runs, _ = _visits(obs_bins, lvl_bins, TOL_BINS)
    par = list(zip(seq, runs))
    if len(par) > 2:
        par = par[1:-1]                       # kanterna är avhuggna dwells
    par = [(v, r) for v, r in par if v >= 0]
    if not par:
        return dict(levels=levels_us, lengths=[None], order_fixed=True,
                    length_fixed=True)
    besok = [v for v, _ in par]
    langder = [r for _, r in par]

    # en enda nivå: statisk, aldrig något byte
    if len(lvl_bins) == 1:
        return dict(levels=levels_us, lengths=[None], order_fixed=True,
                    length_fixed=True)

    # ---- 3. ORDER: bästa perioden över alla kandidater
    p_max = min(2 * len(lvl_bins) + 2, len(besok) // MIN_ROSTER)
    bast_k, bast_p, bast_c = 0.0, None, None
    for p in range(1, p_max + 1):
        k, cyk = _konfidens(besok, p)
        if k is not None and k > bast_k:
            bast_k, bast_p, bast_c = k, p, cyk
    order_fixed = bast_c is not None and bast_k >= trosk

    if order_fixed:
        # minsta rotation, samma regel som to_tokens
        r = min(range(len(bast_c)), key=lambda j: bast_c[j:] + bast_c[:j])
        cykel = bast_c[r:] + bast_c[:r]
        cykel_us = [pri_of_bin(lvl_bins[i]) for i in cykel]
    else:
        cykel_us = levels_us

    # ---- 4. DWELL: periodiska runlängder -> FIXED, annars ett intervall
    lang_min = _min_period(langder)
    periodisk = len(lang_min) <= max(len(lvl_bins), 1) and len(lang_min) < len(langder)
    if len(set(langder)) == 1:
        return dict(levels=cykel_us, lengths=[int(langder[0])],
                    order_fixed=order_fixed, length_fixed=True)
    if periodisk:
        return dict(levels=cykel_us, lengths=[int(x) for x in lang_min],
                    order_fixed=order_fixed, length_fixed=True)
    return dict(levels=cykel_us, lengths=[int(min(langder)), int(max(langder))],
                order_fixed=order_fixed, length_fixed=False)


def _tokens(d):
    return to_tokens(d["levels"], d["lengths"], d["order_fixed"], d["length_fixed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=cfg.eval_n_emitters)
    ap.add_argument("--drops", type=float, nargs="*", default=list(cfg.eval_p_drops))
    ap.add_argument("--trosk", type=float, default=KONF_TROSKEL)
    ap.add_argument("--min-pulser", type=int, default=MIN_PULSER,
                    help="bins med färre pulser räknas som artefakter")
    a = ap.parse_args()

    gen = emitter_module()
    kolumner = ["parsed", "exact", "level_recall", "level_precision", "n_levels_ok",
                "n_lengths_ok", "order_type_ok", "length_type_ok"]
    print(f"klassisk baslinje, {a.n} emittrar, tröskel {a.trosk}, min_pulser {a.min_pulser}\n")
    head = f"{'set':<12}" + "".join(f"{k:>17}" for k in kolumner)
    print(head); print("-" * len(head))

    for p in a.drops:
        # samma frö och samma anrop som data.make_eval_sets -> samma emittrar
        rng = np.random.default_rng(cfg.eval_seed)
        data = gen.create_emitter_data(a.n, cfg.eval_samples_per_emitter, p,
                                       cfg.noise_level, rng)
        rader = []
        for seqs, lab in data:
            sant = to_tokens(lab["levels"], lab["lengths"],
                             lab["order_fixed"], lab["length_fixed"])
            for s in (seqs if isinstance(seqs, (list, tuple)) else [seqs]):
                d = estimate(np.asarray(s, dtype=float), a.trosk, a.min_pulser)
                try:
                    gissning = _tokens(d) if d else []
                except Exception:
                    gissning = []            # otolkbar skattning = misslyckad post
                rader.append(compare(gissning, sant))
        m = {k: float(np.mean([r[k] for r in rader if k in r])) for k in kolumner}
        print(f"{'drop_%.2f' % p:<12}" + "".join(f"{m[k]:>17.3f}" for k in kolumner))

    print("\nSätt tabellen bredvid eval.json från träningen. Där baslinjen vinner är")
    print("modellen onödig; där den förlorar ligger motiveringen till den.")


if __name__ == "__main__":
    main()
