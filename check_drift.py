"""
Driver dataströmmen över tid?

    python check_drift.py                 # 2000 anrop i en process
    python check_drift.py --workers 4     # härmar DataLoaderns fyra processer

En stigande träningsloss på FÄRSK data har i princip bara en förklaring: att datan
blir svårare ju längre körningen pågår. Det kräver ingen träning för att mäta -- det
räcker att anropa generatorn lika många gånger som träningen gör och se om
statistiken förändras.

Anropsmönstret är exakt StreamDataset.__getitem__: make_pairs(rng) med en emitter
per anrop, samma rng återanvänd, p_drop dragen ur uniform(0, cfg.p_drop).

Misstanken som testas: modulnivå-tillstånd i create_emitter_data -- en cache, en
räknare, en lista som växer -- som gör att emittrarna ändrar karaktär efter några
tusen anrop. Fyra worker-processer bygger upp sådant tillstånd var för sig.

Utskriften delar anropen i tio block och visar medelvärdet per block. Sista raden
jämför första och sista blocket i enheter av blockets egen spridning: ligger något
över ~3 har det drivit på riktigt och är inte brus.
"""
import argparse

import numpy as np

from config import cfg
from data import make_pairs
from vocab import ids_to_tokens  # noqa: F401  (håller importvägen densamma som träningen)

FIELDS = ("nivåer", "tokens", "pulser", "medel-PRI", "max-PRI",
          "distinkta_bins", "overflow%", "ORDER_RANDOM%", "DWELL_RANGE%")


def sample(rng):
    """Ett anrop, precis som StreamDataset.__getitem__ -> rader med mätvärden."""
    out = []
    for ch, tok in make_pairs(rng):
        b = np.asarray(ch["bins"])
        p = np.asarray(ch["pri"], dtype=float)
        try:
            n_lvl = int(tok[1][1:])
        except (IndexError, ValueError):
            n_lvl = np.nan
        out.append((
            n_lvl,
            len(tok),
            len(b),
            float(p.mean()),
            float(p.max()),
            len(np.unique(b)),
            100.0 * float((b == cfg.in_bins - 1).mean()),
            100.0 * float("RANDOM" in tok),
            100.0 * float("RANGE" in tok),
        ))
    return out


def run(n_calls, seed):
    rng = np.random.default_rng([cfg.seed, seed])
    rows = []
    for i in range(n_calls):
        rows.extend(sample(rng))
        if (i + 1) % max(1, n_calls // 10) == 0:
            print(f"  {i + 1}/{n_calls} anrop", end="\r", flush=True)
    return np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=2000,
                    help="anrop per worker (träningen gör ~steg * batch/samples)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    print(f"{a.workers} ström(mar) x {a.calls} anrop, p_drop ~ "
          f"uniform(0, {cfg.p_drop}) per anrop\n")
    blocks = []
    for w in range(a.workers):
        r = run(a.calls, w)
        print(f"  ström {w}: {len(r)} sekvenser            ")
        blocks.append(r)
    data = np.concatenate(blocks)

    # dela varje ström i tio block och slå ihop blockvis, så att tidsordningen bevaras
    k = 10
    per = len(blocks[0]) // k
    grid = np.array([[b[i * per:(i + 1) * per].mean(0) for i in range(k)]
                     for b in blocks]).mean(0)

    w = max(len(f) for f in FIELDS) + 2
    print("\nblockmedel (tidsordning vänster -> höger)\n")
    print(" " * w + "".join(f"{i + 1:>9}" for i in range(k)))
    for j, f in enumerate(FIELDS):
        print(f"{f:<{w}}" + "".join(f"{grid[i, j]:9.2f}" for i in range(k)))

    print("\nförsta blocket mot sista, i enheter av spridningen mellan blocken")
    print(f"{'fält':<{w}}{'först':>10}{'sist':>10}{'diff':>10}{'z':>8}")
    flagged = []
    for j, f in enumerate(FIELDS):
        col = grid[:, j]
        s = col.std(ddof=1)
        z = 0.0 if s == 0 else (col[-1] - col[0]) / s
        print(f"{f:<{w}}{col[0]:10.2f}{col[-1]:10.2f}{col[-1] - col[0]:+10.2f}{z:8.1f}")
        if abs(z) >= 3:
            flagged.append(f)

    print()
    if flagged:
        print("DRIVER: " + ", ".join(flagged))
        print("Strömmen är inte stationär. Leta efter tillstånd på modulnivå i")
        print(f"{cfg.emitter} -- en cache, en räknare, en lista som växer.")
    else:
        print("Ingen drift i statistiken. Strömmen är stationär, och den stigande")
        print("lossen kommer inte av att datan blir svårare. Då återstår workers.")


if __name__ == "__main__":
    main()
