"""
Lokaliserar exakt var den parvisa matchningen brister.

    python diag_rotation.py                      # blandning
    python diag_rotation.py ds_fix_fix_same      # en variant
    python diag_rotation.py ds_fix_fix_same 5    # och fem exempel

För varje misslyckat exempel svarar skriptet på tre frågor:
  1. Är facitet en rotation av generatorns par?      (facitfel eller ej)
  2. Var sitter avvikelserna: i slutet, i början, eller utspridda?
  3. Skiljer de sig i nivå, i längd, eller i båda?
"""
import sys
from collections import Counter

import numpy as np

from .vocab import bin_of
from .labels import to_tokens, parse
from .verify import (verify_label, _as_label, _quantize, _visits,
                    _min_period, _best_rotation)

from data import create_emitter_data          # följer cfg.emitter

try:
    from all_emitters import VARIANTS            # bara all_emitters har varianter
except ImportError:
    VARIANTS = None


def observed_pairs(pri, lab, tol_bins=1, trim_edges=True):
    lvl_uniq = sorted(set(bin_of(v) for v in lab["levels"]))
    obs_bins = np.array([bin_of(p) for p in _quantize(pri)])
    seq, runs, unmatched = _visits(obs_bins, lvl_uniq, tol_bins)
    keep = list(zip(seq, runs))
    trimmed = 0
    if trim_edges and len(keep) > 2:
        keep = keep[1:-1]; trimmed = 2
    return [(v, r) for v, r in keep if v >= 0], lvl_uniq, unmatched, trimmed


def analyse(pri, label, tol_bins=1):
    lab = _as_label(label)
    variant = label.get("variant", "?") if isinstance(label, dict) else "?"

    # --- facit vs generator
    uniq = sorted(set(bin_of(v) for v in lab["levels"]))
    idx = {b: i for i, b in enumerate(uniq)}
    gen_pairs = list(zip([idx[bin_of(v)] for v in lab["levels"]],
                         [x for x in lab["lengths"]]))
    tok = to_tokens(lab["levels"], lab["lengths"], lab["order_fixed"], lab["length_fixed"])
    d = parse(tok)
    lab_pairs = (list(zip([idx[bin_of(v)] for v in d["order"]], d["lengths"]))
                 if d["order_fixed"] and d["length_fixed"] else None)

    is_rot = None
    if lab_pairs and len(lab_pairs) == len(gen_pairs):
        is_rot = any(lab_pairs == gen_pairs[i:] + gen_pairs[:i] for i in range(len(gen_pairs)))

    # --- observerat
    pairs, lvl_uniq, unmatched, trimmed = observed_pairs(pri, lab, tol_bins)
    cyc_lv = _min_period([idx[bin_of(v)] for v in lab["levels"]])
    cyc_ln = _min_period([x for x in lab["lengths"] if x is not None])
    paired = len(cyc_lv) == len(cyc_ln) and lab["order_fixed"] and lab["length_fixed"]

    print("=" * 78)
    print(f"variant {variant}   facit: {' '.join(tok)}")
    print(f"facit är rotation av generatorns par: {is_rot}")
    print(f"nivåcykel period {len(cyc_lv)}, längdcykel period {len(cyc_ln)}, "
          f"parvis matchning: {paired}")
    print(f"{len(pairs)} besök, {len(pairs) % max(1, len(cyc_lv))} över ett helt varv, "
          f"{unmatched} omatchade pulser, {trimmed} trimmade besök")

    if not paired:
        ro, fo = _best_rotation([v for v, _ in pairs], cyc_lv)
        rl, fl = _best_rotation([r for _, r in pairs], cyc_ln)
        print(f"separat: ordning {fo:.1%} (rot {ro}), längder {fl:.1%} (rot {rl})")
        return

    cyc = list(zip(cyc_lv, cyc_ln))
    r, frac = _best_rotation(pairs, cyc)
    rot = cyc[r:] + cyc[:r]
    exp = [rot[i % len(rot)] for i in range(len(pairs))]
    fel = [i for i in range(len(pairs)) if pairs[i] != exp[i]]

    print(f"parvis i takt {frac:.1%} med rotation {r}; {len(fel)} avvikelser")

    if not fel:
        return

    # var sitter felen?
    n, last_varv = len(pairs), len(pairs) - len(pairs) % len(cyc)
    i_slut = sum(1 for i in fel if i >= last_varv)
    i_start = sum(1 for i in fel if i < len(cyc))
    print(f"  varav {i_start} i första varvet, {i_slut} i den ofullständiga svansen "
          f"(index >= {last_varv}), {len(fel) - i_start - i_slut} däremellan")
    print(f"  felindex: {fel[:25]}{' ...' if len(fel) > 25 else ''}")

    # vad skiljer?
    kind = Counter()
    for i in fel:
        (ov, orl), (ev, erl) = pairs[i], exp[i]
        if ov != ev and orl != erl: kind["nivå + längd"] += 1
        elif ov != ev:              kind["bara nivå"] += 1
        else:                       kind[f"bara längd (diff {orl - erl:+d})"] += 1
    print("  avvikelsetyp:", dict(kind))

    # visa några
    print("  i     observerat   förväntat")
    for i in fel[:8]:
        print(f"  {i:>4}  {str(pairs[i]):>12}   {str(exp[i]):>10}")

    # hypotestest: räcker det med tolerans på längden?
    tol_hits = sum(1 for i in range(len(pairs))
                   if pairs[i][0] == exp[i][0] and abs(pairs[i][1] - exp[i][1]) <= 1)
    print(f"  med längdtolerans ±1: {tol_hits/len(pairs):.1%} i takt")

    # hypotestest: räcker det att jämföra hela varv?
    if last_varv:
        hits = sum(1 for i in range(last_varv) if pairs[i] == exp[i])
        print(f"  bara hela varv (0..{last_varv}): {hits/last_varv:.1%} i takt")


def main(variant=None, n_show=3, n_emitters=300, seed=0):
    rng = np.random.default_rng(seed)
    kw = dict(only=[variant]) if (variant and VARIANTS) else {}
    data = create_emitter_data(n_emitters, 1, 0.0, None, rng, **kw)

    bad = []
    for seqs, lab in data:
        for s in seqs:
            if not verify_label(s, lab, verbose=False)["ok"]:
                bad.append((s, lab))
    print(f"{len(data) - len(bad)}/{len(data)} stämmer, {len(bad)} avvikelser\n")

    for pri, lab in bad[:n_show]:
        analyse(pri, lab)
        print()


if __name__ == "__main__":
    a = sys.argv[1:]
    v = a[0] if a and not a[0].isdigit() else None
    n = int(a[-1]) if a and a[-1].isdigit() else 3
    main(v, n)
