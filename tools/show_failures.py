"""
Skriver ut de facit som inte stämmer med sin signal, i detalj.

    python show_failures.py                      # blandning, alla varianter
    python show_failures.py ds_revisit           # bara en variant
    python show_failures.py ds_revisit 5         # och visa 5 exempel

För varje avvikelse visas:
  * facitet som tokens och i klartext
  * facitets cykler (nivåer och längder) efter reduktion och rotation
  * de observerade besöken som (nivå, längd)-par
  * en radvis jämförelse mellan observerat och förväntat, med avvikelser markerade
"""
import sys
import numpy as np

from transformer_post_generator.config import cfg
from transformer_post_generator.vocab import bin_of, pri_of_bin
from transformer_post_generator.labels import to_tokens, parse
from transformer_post_generator.verify import (verify_label, _as_label, _quantize, _visits,
                    _min_period, _best_rotation, _print_report)

from transformer_post_generator.data import create_emitter_data          # följer cfg.emitter

try:
    from transformer_post_generator.all_emitters import VARIANTS            # bara all_emitters har varianter
except ImportError:
    VARIANTS = None


# ------------------------------------------------------------------ detaljer
def visit_pairs(pri, lab, tol_bins=1, trim_edges=True):
    """-> (par per besök, nivåbins) ur signalen"""
    lvl_uniq = sorted(set(bin_of(v) for v in lab["levels"]))
    obs_bins = np.array([bin_of(p) for p in _quantize(pri)])
    seq, runs, unmatched = _visits(obs_bins, lvl_uniq, tol_bins)
    keep = list(zip(seq, runs))
    if trim_edges and len(keep) > 2:
        keep = keep[1:-1]
    return [(v, r) for v, r in keep if v >= 0], lvl_uniq, unmatched


def show_one(pri, label, tol_bins=1, n_visits=16):
    lab = _as_label(label)
    tok = to_tokens(lab["levels"], lab["lengths"], lab["order_fixed"], lab["length_fixed"])

    print("=" * 78)
    if isinstance(label, dict) and "variant" in label:
        print("variant:", label["variant"])
    print("facit  :", " ".join(tok))

    d = parse(tok)
    lvl_uniq = sorted(set(bin_of(v) for v in lab["levels"]))
    names = {b: f"L{i}" for i, b in enumerate(lvl_uniq)}
    print("nivåer :", ", ".join(f"{names[b]}={pri_of_bin(b):.4g} (bin {b})" for b in lvl_uniq))

    idx = {b: i for i, b in enumerate(lvl_uniq)}
    lab_cycle = _min_period([idx[bin_of(v)] for v in lab["levels"]])
    lab_lens = _min_period([x for x in lab["lengths"] if x is not None])
    print(f"cykler : nivåer {lab_cycle} (period {len(lab_cycle)}), "
          f"längder {lab_lens} (period {len(lab_lens)})")
    print(f"flaggor: order_fixed={lab['order_fixed']}  length_fixed={lab['length_fixed']}")

    pairs, _, unmatched = visit_pairs(pri, lab, tol_bins)
    print(f"\nobserverade besök ({len(pairs)} st, {unmatched} pulser utan nivåmatchning):")
    print("  nivå   ", " ".join(f"{v:>4}" for v, _ in pairs[:n_visits]))
    print("  längd  ", " ".join(f"{r:>4}" for _, r in pairs[:n_visits]))

    # förväntat, med bästa rotation
    if lab["order_fixed"] and lab["length_fixed"] and lab_lens:
        if len(lab_cycle) == len(lab_lens):
            cyc = list(zip(lab_cycle, lab_lens))
            r, frac = _best_rotation(pairs, cyc)
            rot = cyc[r:] + cyc[:r]
            exp = [rot[i % len(rot)] for i in range(len(pairs))]
            print(f"\nförväntat (parvis, rotation {r}, i takt {frac:.0%}):")
            print("  nivå   ", " ".join(f"{v:>4}" for v, _ in exp[:n_visits]))
            print("  längd  ", " ".join(f"{l:>4}" for _, l in exp[:n_visits]))
            mark = ["   ^" if pairs[i] != exp[i] else "    " for i in range(min(n_visits, len(pairs)))]
            print("  avvik  ", " ".join(mark))
        else:
            ro, fo = _best_rotation([v for v, _ in pairs], lab_cycle)
            rl, fl = _best_rotation([r for _, r in pairs], lab_lens)
            eo = [(lab_cycle[ro:] + lab_cycle[:ro])[i % len(lab_cycle)] for i in range(len(pairs))]
            el = [(lab_lens[rl:] + lab_lens[:rl])[i % len(lab_lens)] for i in range(len(pairs))]
            print(f"\nförväntat (separat: ordning {fo:.0%}, längder {fl:.0%}):")
            print("  nivå   ", " ".join(f"{v:>4}" for v in eo[:n_visits]))
            print("  längd  ", " ".join(f"{l:>4}" for l in el[:n_visits]))

    print()
    verify_label(pri, label, tol_bins=tol_bins, verbose=True)


# ------------------------------------------------------------------ main
def main(variant=None, n_show=3, n_emitters=300, n_signals=1, seed=0, tol_bins=1):
    rng = np.random.default_rng(seed)
    kw = dict(only=[variant]) if (variant and VARIANTS) else {}
    data = create_emitter_data(n_emitters, n_signals, 0.0, None, rng, **kw)

    failed, total = [], 0
    for seqs, lab in data:
        for s in seqs:
            total += 1
            rep = verify_label(s, lab, tol_bins=tol_bins, verbose=False)
            if not rep["ok"]:
                failed.append((s, lab, rep))

    print(f"{total - len(failed)}/{total} stämmer  ({len(failed)} avvikelser)\n")
    if not failed:
        return

    # gruppera avvikelserna
    from collections import Counter
    c = Counter()
    for _, lab, rep in failed:
        bits = []
        if not rep["levels"]["ok"]: bits.append("nivåer")
        if not rep["order"]["ok"]: bits.append("ordning")
        if not rep["lengths"]["ok"]: bits.append("längder")
        if not rep["unmatched_pulses"]["ok"]: bits.append("omatchade pulser")
        key = (lab.get("variant", "?"), "+".join(bits) or "?")
        c[key] += 1
    print("avvikelser per variant och fält:")
    for (v, b), n in c.most_common():
        print(f"  {n:>5}  {v:<18} {b}")
    print()

    for s, lab, _ in failed[:n_show]:
        show_one(s, lab, tol_bins)
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    variant = args[0] if args and not args[0].isdigit() else None
    n_show = int(args[-1]) if args and args[-1].isdigit() else 3
    main(variant, n_show)
