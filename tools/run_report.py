"""
Läser en körning och visar träningskurvan och eval-måtten bredvid varandra.

    python run_report.py                      # senaste körningen under cfg.run_root
    python run_report.py runs/20260913_101955

Frågan den svarar på: när loss_num steg -- blev modellen SÄMRE, eller blev den bara
mer bestämd än det utjämnade målet tillåter? De ser likadana ut i lossen och
motsatta i eval-måtten, och de kräver motsatta åtgärder.
"""
import csv
import json
import sys
from pathlib import Path

EVAL_KEYS = ("exact", "level_recall", "n_levels_ok", "order_type_ok",
             "length_type_ok", "n_lengths_ok")


def newest(root):
    dirs = sorted((p for p in Path(root).iterdir() if p.is_dir()), key=lambda p: p.name)
    if not dirs:
        sys.exit(f"inga körningar under {root}")
    return dirs[-1]


def main(run):
    run = Path(run)
    rows = list(csv.DictReader(open(run / "train.csv", encoding="utf-8")))
    if not rows:
        sys.exit("tom train.csv")
    num = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")

    print(f"körning: {run}\n")
    cols = [c for c in ("loss", "loss_num", "loss_order", "loss_grammar", "lr")
            if c in rows[0]]
    print("TRÄNING")
    print("  " + "step".rjust(6) + "".join(c.rjust(13) for c in cols))
    step = max(1, len(rows) // 20)
    for r in rows[::step] + ([rows[-1]] if len(rows) % step else []):
        print("  " + r["step"].rjust(6)
              + "".join(f"{num(r, c):13.4f}" if c != "lr" else f"{num(r, c):13.2e}"
                        for c in cols))

    # var bottnade varje fält, och hur mycket steg det efteråt?
    print("\n  botten -> slut")
    for c in cols:
        if c == "lr":
            continue
        v = [num(r, c) for r in rows]
        i = min(range(len(v)), key=lambda k: v[k])
        d = v[-1] - v[i]
        flag = "  <-- steg" if d > 0.05 else ""
        print(f"    {c:<14} min {v[i]:.4f} vid steg {rows[i]['step']:>6}"
              f"   slut {v[-1]:.4f}   {d:+.4f}{flag}")

    p = run / "eval.json"
    if not p.exists():
        print("\ningen eval.json -- körningen hann inte till första evalen")
        return
    evals = json.load(open(p, encoding="utf-8"))
    sets = sorted({k.split("/")[0] for e in evals for k in e if "/" in k})

    print("\nEVAL")
    for s in sets:
        keys = [k for k in EVAL_KEYS if f"{s}/{k}" in evals[0]]
        if not keys:
            continue
        print(f"\n  {s}")
        print("  " + "step".rjust(6) + "".join(k.rjust(16) for k in keys))
        for e in evals:
            print("  " + str(e["step"]).rjust(6)
                  + "".join(f"{e[f'{s}/{k}']:16.3f}" for k in keys))

    # slutsatsen
    v = [num(r, "loss_num") for r in rows] if "loss_num" in rows[0] else []
    if v and len(evals) >= 2:
        i = min(range(len(v)), key=lambda k: v[k])
        rose = v[-1] - v[i]
        s0 = sets[0]
        key = "level_recall" if f"{s0}/level_recall" in evals[0] else "exact"
        a, b = evals[0][f"{s0}/{key}"], evals[-1][f"{s0}/{key}"]
        print(f"\nSLUTSATS")
        print(f"  loss_num steg {rose:+.3f} efter sin botten")
        print(f"  {s0}/{key} gick {a:.3f} -> {b:.3f} ({b - a:+.3f})")
        if rose > 0.05 and b >= a - 0.005:
            print("  -> modellen blev INTE sämre. Lossen mäter kalibrering, inte")
            print("     träffsäkerhet. Åtgärden ligger i målfördelningen, inte i")
            print("     optimeraren: smalare smoothing eller ett regressionshuvud.")
        elif rose > 0.05:
            print("  -> modellen blev faktiskt sämre medan lr sjönk. Det är varken")
            print("     instabilitet eller kalibrering -- leta i datan.")
        else:
            print("  -> loss_num steg inte nämnvärt i den här körningen.")


if __name__ == "__main__":
    from transformer_post_generator.config import cfg
    main(sys.argv[1] if len(sys.argv) > 1 else newest(cfg.run_root))
