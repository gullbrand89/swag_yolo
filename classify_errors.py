"""
Delar upp felen i preds-filerna efter vad som faktiskt skiljer.

    python classify_failures.py runs/<tidsstämpel>
    python classify_failures.py runs/<tidsstämpel> --show 5

`exact` är en konjunktion: antal nivåer, varje nivås bin, ordningstyp, ordningen,
längdtyp och längdcykeln måste alla stämma token för token. Ett fel på en enda bin
-- 1,76 µs av ett spann på 900, alltså 0,2 % -- räknas som total miss, trots att
level_recall med sin ±2 µs-tolerans förlåter exakt det.

Skriptet skiljer på fel som betyder något och fel som inte gör det:

  exakt            posten är identisk
  bin-bagatell     allt stämmer utom att en eller flera nivåer ligger inom
                   tol_bins från facit
  oidentifierbart  FIXED mot RANDOM på en emitter med högst två nivåer. Generatorn
                   upprepar aldrig samma nivå två gånger i rad, så med två nivåer
                   ÄR en slumpad ordning exakt alternerande -- den är inte skiljbar
                   från en fast cykel, och signalen innehåller ingen information
                   som kunde avgöra saken
  intervalldelmängd  DWELL RANGE där det förutsagda intervallet ligger innanför
                   facitets. Observationsfönstret innehöll troligen inte
                   ytterlighetsvärdena, så det är den korrekta slutsatsen av det
                   som faktiskt syntes
  äkta fel         allt annat: fel antal nivåer, fel ordning, fel längdcykel

De tre mittersta kategorierna är fall där facitet inte går att härleda ur signalen,
inte fall där modellen har fel. Rapportera gärna både `exact` och `exact + de tre` --
det första är den strikta gränsen, det andra det praktiska taket.
"""
import sys
from collections import Counter
from pathlib import Path

from config import cfg
from labels import parse
from vocab import bin_of


def read_preds(path):
    """-> lista av (true_tokens, pred_tokens)"""
    out, t = [], None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("T "):
            t = line[2:].split()
        elif line.startswith("P ") and t is not None:
            out.append((t, line[2:].split()))
            t = None
    return out


def _post(tokens):
    """
    Posten utan ett eventuellt scratchpad-block. Definierad här i stället för
    importerad, så att skriptet fungerar oavsett om scratchpaden finns i projektet.
    Utan VISITS-prefix är den en ren genomsläppning.
    """
    tokens = list(tokens)
    if not tokens or tokens[0] != "VISITS":
        return tokens
    try:
        return tokens[tokens.index("ENDVISITS") + 1:]
    except ValueError:                 # VISITS utan ENDVISITS, trasig utdata
        return tokens


def classify(true_tok, pred_tok, tol_bins=1):
    """-> (kategori, detalj)"""
    t_tok, p_tok = _post(true_tok), _post(pred_tok)
    if p_tok == t_tok:
        return "exakt", ""

    try:
        t, p = parse(t_tok), parse(p_tok)
    except Exception as e:
        return "äkta fel", f"parse: {e}"

    # ---- strukturen måste stämma innan något kan kallas bagatell
    t_lv = t["order"] if t["order_fixed"] else t["levels"]
    p_lv = p["order"] if p["order_fixed"] else p["levels"]
    n_t = len({bin_of(v) for v in t_lv})
    n_p = len({bin_of(v) for v in p_lv})
    if n_t != n_p:
        return "äkta fel", f"{n_p} nivåer mot {n_t}"

    # ---- oidentifierbart: FIXED mot RANDOM med högst två nivåer
    if p["order_fixed"] != t["order_fixed"]:
        if n_t <= 2:
            return "oidentifierbart", f"{n_t} nivåer, FIXED/RANDOM ej skiljbara"
        return "äkta fel", "fel ordningstyp"

    if p["length_fixed"] != t["length_fixed"]:
        return "äkta fel", "fel längdtyp"

    # ---- längderna
    tl = [x for x in t["lengths"] if x is not None]
    pl = [x for x in p["lengths"] if x is not None]
    if not t["length_fixed"] and len(tl) == 2 and len(pl) == 2:
        # RANGE: lika intervall är inget fel, då sitter skillnaden i nivåerna och
        # vi ska falla igenom till bin-kontrollen nedan
        lo_t, hi_t, lo_p, hi_p = min(tl), max(tl), min(pl), max(pl)
        if (lo_p, hi_p) != (lo_t, hi_t):
            if lo_t <= lo_p and hi_p <= hi_t:
                return "intervalldelmängd", f"[{lo_p}, {hi_p}] i [{lo_t}, {hi_t}]"
            return "äkta fel", f"intervall [{lo_p}, {hi_p}] mot [{lo_t}, {hi_t}]"
    elif p["lengths"] != t["lengths"]:
        return "äkta fel", f"längdcykel {p['lengths']} mot {t['lengths']}"

    # ---- ordningen: samma följd av nivåINDEX (sorterat på bin)
    if t["order_fixed"]:
        ti = {b: i for i, b in enumerate(sorted({bin_of(v) for v in t_lv}))}
        pi = {b: i for i, b in enumerate(sorted({bin_of(v) for v in p_lv}))}
        if [ti[bin_of(v)] for v in t["order"]] != [pi[bin_of(v)] for v in p["order"]]:
            return "äkta fel", "fel ordning"

    # ---- kvar: bara nivåvärdena skiljer. Hur mycket?
    tb = sorted({bin_of(v) for v in t_lv})
    pb = sorted({bin_of(v) for v in p_lv})
    diffs = [abs(a - b) for a, b in zip(pb, tb)]
    if max(diffs) <= tol_bins:
        w = (cfg.pri_max - cfg.pri_min) / (cfg.n_bins - 1)
        return "bin-bagatell", f"max {max(diffs)} bin ({max(diffs)*w:.2f} µs)"
    return "äkta fel", f"nivå {max(diffs)} bins fel"


def main(run_dir, n_show=3, tol_bins=1):
    files = sorted(Path(run_dir).glob("preds_*.txt"))
    if not files:
        sys.exit(f"inga preds_*.txt i {run_dir}")

    for f in files:
        pairs = read_preds(f)
        if not pairs:
            continue
        rows = [(t, p) + classify(t, p, tol_bins) for t, p in pairs]
        c = Counter(r[2] for r in rows)
        n = len(rows)

        order = ["exakt", "bin-bagatell", "oidentifierbart", "intervalldelmängd", "äkta fel"]
        print(f"\n{f.name}   {n} exempel")
        cum = 0
        for k in order:
            if not c[k]:
                continue
            cum += c[k]
            tag = "" if k == "äkta fel" else f"   (kumulativt {cum/n:.1%})"
            print(f"   {k:<18} {c[k]:>5}  {c[k]/n:>6.1%}{tag}")

        strict = c["exakt"] / n
        practical = (n - c["äkta fel"]) / n
        print(f"   {'-'*46}")
        print(f"   exact (strikt)     {strict:>6.1%}")
        print(f"   praktiskt tak      {practical:>6.1%}   "
              f"= allt utom äkta fel")

        # ---- vanligaste äkta felen
        def kind_of(d):
            for key in ("nivåer", "bins fel", "intervall", "längdcykel",
                        "ordningstyp", "fel ordning", "längdtyp", "parse"):
                if key in d:
                    return key
            return d
        real = Counter(kind_of(r[3]) for r in rows if r[2] == "äkta fel")
        if real:
            print("   vanligaste äkta fel:")
            for d, k in real.most_common(5):
                print(f"     {k:>5}  {d}")

        # ---- exempel
        shown = 0
        for t, p, kind, detail in rows:
            if kind == "äkta fel" and shown < n_show:
                print(f"\n   [{detail}]")
                print("   FACIT:", " ".join(_post(t)))
                print("   PRED :", " ".join(_post(p)))
                shown += 1


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    n = 3
    if "--show" in sys.argv:
        n = int(sys.argv[sys.argv.index("--show") + 1])
    main(a[0] if a else "runs", n)