"""
Mät binningsparametrarna ur datan i stället för att härleda dem.

    python measure_levels.py                    # 5000 emittrar ur cfg.emitter
    python measure_levels.py --n 20000
    python measure_levels.py --variants         # uppdelat per mönstervariant

Svarar på tre frågor:

  1. Täcker cfg.pri_min / cfg.pri_max datan? Nivåer utanför spannet KLIPPS tyst in i
     kant-binen, så två emittrar med olika nivåer får identiska tokens. Det syns inte
     i någon loss och förstör facitet för allt ovanför gränsen.

  2. Hur tätt ligger nivåerna? Inte minimum -- percentilerna. Minimum och median
     skiljer typiskt två tiopotenser, och att designa för minimum betyder att betala
     för den värsta enskilda emittern i hela korpusen.

  3. Hur många bins krävs, och är logaritmisk eller linjär binning billigare? Det
     beror på om avstånden är relativa (log vinner) eller absoluta (linjär vinner),
     och det är en egenskap hos datan, inte ett designval.

Du behöver inte veta hur generatorn fungerar -- bara vad den producerar.
"""
import argparse
import importlib
import math

import numpy as np

from config import cfg

# Samma modul som data.py använder, men utan att dra in torch för en numpy-analys.
create_emitter_data = importlib.import_module(cfg.emitter).create_emitter_data


def separations(level_sets):
    """-> (alla nivåer, minsta parvisa avstånd per emitter: absolut, relativt)"""
    sets = [np.unique(np.asarray(s, dtype=float)) for s in level_sets if len(s)]
    if not sets:
        raise ValueError("inga nivåer")
    absu, rel = [], []
    for s in sets:
        if s.size < 2:
            continue                       # en nivå: inget avstånd att mäta
        d = np.diff(s)
        i = int(d.argmin())
        absu.append(d[i])
        rel.append(d[i] / s[i])            # relativt den lägre av de två
    return np.concatenate(sets), np.array(absu), np.array(rel), len(sets)


def report(level_sets, coverage=0.99, margin=2.0, cfg_range=None,
           d_model=None, batch=None, tgt_len=45):
    """
    level_sets : lista av nivålistor i µs, en per emitter
    coverage   : andel emittrar vars nivåpar måste gå att skilja åt
    margin     : säkerhetsfaktor på binbredden (2 = binen halva minsta avståndet)
    cfg_range  : (pri_min, pri_max) att kontrollera klippning mot; None = cfg:s egna
    """
    d_model = cfg.d_model if d_model is None else d_model
    batch = cfg.batch if batch is None else batch
    lo_cfg, hi_cfg = cfg_range or (cfg.pri_min, cfg.pri_max)

    allv, absu, rel, n_em = separations(level_sets)
    lo, hi = allv.min(), allv.max()

    print(f"nivåer      {allv.size} st i {n_em} emittrar, {lo:.4f} – {hi:.4f} µs")
    print(f"            {len(absu)} emittrar har minst två nivåer\n")

    # ---- 1. klippning
    under, over = (allv < lo_cfg).mean(), (allv > hi_cfg).mean()
    flag = "OK" if under + over == 0 else "KLIPPS"
    print(f"{flag:<7} cfg.pri_min = {lo_cfg:g}, cfg.pri_max = {hi_cfg:g}")
    if under + over:
        print(f"        {under:.2%} av nivåerna under, {over:.2%} över spannet.")
        print(f"        De mappas alla till kant-binen och blir omöjliga att skilja åt.")
    print()

    # ---- 2. avstånden
    print(f"{'minsta parvisa avstånd':<26} {'absolut (µs)':>14} {'relativt (%)':>14}")
    for name, p in (("minimum", 0), ("1:a percentilen", 1),
                    ("5:e percentilen", 5), ("median", 50)):
        print(f"  {name:<24} {np.percentile(absu, p):>14.4f} "
              f"{np.percentile(rel, p) * 100:>14.4f}")

    # ---- 3. bins
    q = (1.0 - coverage) * 100
    d_rel, d_abs = np.percentile(rel, q), np.percentile(absu, q)
    n_log = math.ceil(math.log(hi / lo) / (d_rel / margin)) + 1
    n_lin = math.ceil((hi - lo) / (d_abs / margin)) + 1
    best, n = ("logaritmisk", n_log) if n_log <= n_lin else ("linjär", n_lin)

    print(f"\nför att skilja nivåerna åt hos {coverage:.0%} av emittrarna, "
          f"med marginal {margin:g}x:")
    print(f"  logaritmisk binning   n_bins = {n_log}")
    print(f"  linjär binning        n_bins = {n_lin}")
    print(f"  -> {best} är billigare här ({n} mot {max(n_log, n_lin)} bins)")
    print(f"     de {1-coverage:.0%} som inte ryms redovisas separat, "
          f"som 'under upplösningsgränsen'")

    V = 3 + 8 + cfg.max_levels + n
    print(f"\nkostnad vid n_bins = {n}:")
    print(f"  vokabulär             {V} tokens")
    print(f"  ut- och inbäddningslager  {2 * d_model * V / 1e6:.2f} M parametrar")
    print(f"  ordinal_targets       {batch * tgt_len * V * 4 / 1e6:.0f} MB per steg "
          f"(B={batch}, T={tgt_len})")

    print(f"\nconfig.py:")
    print(f"  pri_min: float = {lo * 0.98:.4g}")
    print(f"  pri_max: float = {hi * 1.02:.4g}")
    print(f"  n_bins:  int   = {n}")
    print(f"  max_int: int   = {n - 1}        # höj om någon dwell-längd är större")
    if best == "logaritmisk" and n_log < n_lin:
        print(f"\n  Avstånden är relativa, inte absoluta -- byt bin_of/pri_of_bin i")
        print(f"  vocab.py till logaritmisk binning. Är kvoten nära 1 spelar det")
        print(f"  ingen roll och du kan låta den vara linjär.")

    return dict(pri_min=lo, pri_max=hi, n_bins=n, kind=best,
                d_rel=d_rel, d_abs=d_abs, clipped=under + over)


def bands(level_sets, n_bands=5, coverage=0.99):
    """
    Är avstånden RELATIVA eller ABSOLUTA? Det avgör vilken binning som är rätt, och
    det syns bara om man delar upp på PRI -- ett aggregat över hela korpusen döljer det.

      relativa kolumnen ungefär konstant  -> logaritmisk binning
      absoluta kolumnen ungefär konstant  -> linjär binning

    Är ingen av dem konstant måste n_bins sättas efter det värsta bandet, och då
    hjälper ingen av binningarna särskilt mycket.
    """
    sets = [np.unique(np.asarray(s, dtype=float)) for s in level_sets if len(s) > 1]
    if not sets:
        return
    pri = np.concatenate([s[:-1] for s in sets])        # den lägre nivån i varje par
    sep = np.concatenate([np.diff(s) for s in sets])
    q = (1.0 - coverage) * 100

    edges = np.geomspace(pri.min(), pri.max() * 1.001, n_bands + 1)
    print(f"\navstånd per PRI-band ({coverage:.0%}-percentil inom bandet):")
    print(f"  {'band (µs)':<22} {'n':>7} {'absolut (µs)':>14} {'relativt (%)':>14}")
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pri >= a) & (pri < b)
        if m.sum() < 20:
            continue
        A = np.percentile(sep[m], q)
        R = np.percentile(sep[m] / pri[m], q) * 100
        rows.append((A, R))
        print(f"  {a:>9.3g} – {b:<9.3g} {m.sum():>7} {A:>14.4f} {R:>14.4f}")

    if len(rows) < 2:
        return
    A = np.array([r[0] for r in rows]); R = np.array([r[1] for r in rows])
    spread = lambda x: x.max() / max(x.min(), 1e-12)
    sa, sr = spread(A), spread(R)
    print(f"\n  spridning över banden:  absolut {sa:.1f}x   relativt {sr:.1f}x")
    if sr < sa / 2:
        print("  -> avstånden är RELATIVA. Logaritmisk binning.")
    elif sa < sr / 2:
        print("  -> avstånden är ABSOLUTA. Linjär binning.")
    else:
        print("  -> varken eller. Sätt n_bins efter det värsta bandet; ingen av")
        print("     binningarna ger dig något gratis.")


def levels_of(data):
    return [lab["levels"] for _, lab in data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="antal emittrar att mäta på")
    ap.add_argument("--coverage", type=float, default=0.99)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bands", type=int, default=5,
                    help="antal PRI-band i relativ/absolut-testet")
    ap.add_argument("--variants", action="store_true",
                    help="dela upp per mönstervariant (kräver all_emitters)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"generator: {cfg.emitter}   {args.n} emittrar\n")

    # n_signals=1: nivåerna sitter i etiketten, signalerna behövs inte
    data = create_emitter_data(args.n, 1, 0.0, cfg.noise_level, rng)
    report(levels_of(data), args.coverage, args.margin)
    bands(levels_of(data), args.bands, args.coverage)

    if args.variants:
        try:
            from all_emitters import GEN, VARIANTS
        except ImportError:
            print("\n(--variants kräver all_emitters)")
            return
        n_per = max(200, args.n // max(1, len(VARIANTS)))
        for v in VARIANTS:
            if GEN.weights[v] <= 0:
                continue
            d = create_emitter_data(n_per, 1, 0.0, cfg.noise_level,
                                    np.random.default_rng(args.seed), only=[v])
            try:
                _, absu, rel, _ = separations(levels_of(d))
            except ValueError:
                continue
            if absu.size == 0:
                print(f"\n{v:<20} en nivå per emitter, inget avstånd att mäta")
                continue
            print(f"\n{v:<20} minsta avstånd: "
                  f"min {absu.min():.4f} µs / {rel.min()*100:.3f} %,   "
                  f"p1 {np.percentile(absu,1):.4f} µs / {np.percentile(rel,1)*100:.3f} %")


if __name__ == "__main__":
    main()
