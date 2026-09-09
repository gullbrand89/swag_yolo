"""
Verifierar att compare() och pipelinen är konsekventa.

    python check_metrics.py runs/<tidsstampel>

Kontroller:
  1. compare(t, t) ska ge exact=1 och level_recall=1 för varje facit i preds-filen
  2. exakta prediktioner (P == T) ska ge recall/precision 1 -- annars bugg i compare
  3. fördelning av mått uppdelat på exakta vs icke-exakta exempel
  4. rundtur to_tokens(parse(t)) == t på facit från filen
"""
import sys
from collections import Counter
from pathlib import Path
import numpy as np

from labels import parse, to_tokens
from runlog import compare


def read_preds(path):
    """-> lista av (true_tokens, pred_tokens)"""
    out, t = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("T "):
            t = line[2:].split()
        elif line.startswith("P ") and t is not None:
            out.append((t, line[2:].split())); t = None
    return out


def main(run_dir):
    files = sorted(Path(run_dir).glob("preds_*.txt"))
    if not files:
        sys.exit(f"inga preds_*.txt i {run_dir}")
    pairs = read_preds(files[0])
    print(f"{files[0].name}: {len(pairs)} exempel\n")

    # --- 1. facit mot sig självt
    bad_self = []
    for t, _ in pairs:
        m = compare(t, t)
        if m.get("exact") != 1.0 or m.get("level_recall") != 1.0 or m.get("level_precision") != 1.0:
            bad_self.append((t, m))
    print(f"1) compare(t, t) fel på {len(bad_self)}/{len(pairs)}")
    for t, m in bad_self[:3]:
        print("   facit:", " ".join(t))
        print("   mått :", {k: round(v, 3) for k, v in m.items()})

    # --- 2. exakta prediktioner
    exact_pairs = [(t, p) for t, p in pairs if t == p]
    bad_exact = []
    for t, p in exact_pairs:
        m = compare(p, t)
        if m.get("level_recall") != 1.0 or m.get("level_precision") != 1.0:
            bad_exact.append((t, m))
    print(f"\n2) {len(exact_pairs)}/{len(pairs)} prediktioner är exakta; "
          f"av dem har {len(bad_exact)} recall/precision < 1")
    for t, m in bad_exact[:3]:
        print("   facit:", " ".join(t))
        print("   mått :", {k: round(v, 3) for k, v in m.items()})
        d = parse(t)
        print("   nivåer (µs):", [round(x, 2) for x in d["levels"]])

    # --- 3. mått uppdelat
    print("\n3) medelvärden")
    for name, subset in (("exakta", exact_pairs),
                         ("icke-exakta", [(t, p) for t, p in pairs if t != p]),
                         ("alla", pairs)):
        if not subset: continue
        rows = [compare(p, t) for t, p in subset]
        keys = sorted(set().union(*(r.keys() for r in rows)))
        vals = {k: np.mean([r[k] for r in rows if k in r]) for k in keys}
        print(f"   {name:12s} n={len(subset):5d}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in vals.items()))

    # --- 4. rundtur
    bad_rt = []
    for t, _ in pairs:
        try:
            d = parse(t)
            src = d["order"] if d["order_fixed"] else d["levels"]
            if to_tokens(src, d["lengths"], d["order_fixed"], d["length_fixed"]) != t:
                bad_rt.append(t)
        except Exception as e:
            bad_rt.append(t + [f"<{e}>"])
    print(f"\n4) rundtur misslyckas på {len(bad_rt)}/{len(pairs)}")
    for t in bad_rt[:3]:
        print("   ", " ".join(map(str, t)))

    # --- extra: hur ser facit ut? (grov typindelning)
    kinds = Counter()
    for t, _ in pairs:
        d = parse(t)
        n = len(d["levels"])
        kinds[(n == 1, d["order_fixed"], d["length_fixed"])] += 1
    print("\n   facitfördelning (en_nivå, order_fixed, length_fixed):")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"     {k}: {v}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")
