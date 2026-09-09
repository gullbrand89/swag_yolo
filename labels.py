"""
dict <-> tokens.

Format (nivå- och längdcykel som separata block, oberoende perioder):
  LEVELS N<n>  L0 N<bin>  L1 N<bin> ...
  ORDER  FIXED L0 L2 L1   |  ORDER RANDOM
  DWELL  FIXED N10 N5     |  DWELL RANGE N5 N12   (min, max)
  END
Längd: N<k> pulser eller INF (static).
"""
from config import cfg
from vocab import LEVEL_NAMES, TOK2ID, bin_of, pri_of_bin


def _min_period(x):
    """[10,10,10] -> [10]  och  [5,8,5,8] -> [5,8]. Oförändrad om ingen upprepning."""
    x = list(x)
    n = len(x)
    for p in range(1, n + 1):
        if n % p == 0 and all(x[i] == x[i % p] for i in range(n)):
            return x[:p]
    return x


def _rot(x):
    r = min(range(len(x)), key=lambda k: x[k:] + x[:k])
    return x[r:] + x[:r]

def _L(n):
    return "INF" if n is None else f"N{int(n)}"


def to_tokens(levels, lengths, order_fixed, length_fixed, compress_lengths=None):
    """
    levels : nivåcykeln i µs (kan innehålla upprepningar)
    lengths: längdcykeln i pulser (None = INF). Behöver INTE ha samma period som levels.
    compress_lengths : reducera längdcykeln till sin minsta period, så att
                       [10,10,10] blir [10] och [5,8,5,8] blir [5,8].
                       None -> använd cfg.compress_lengths (default True).

    Separata block:
      ORDER FIXED L0 L1 L2   DWELL FIXED N10 N5      (två cykler, ev. olika period)
      ORDER FIXED L0 L1 L2   DWELL RANGE N5 N12      (slumpad längd, min och max)
      ORDER RANDOM           DWELL FIXED N10 N5 N8
      ORDER RANDOM           DWELL RANGE N5 N12

    Kanonisering:
      * nivåer namnges L0, L1, ... efter bin-värde i stigande ordning
      * varje cykel reduceras till sin minsta period
      * varje cykel roteras till sin minsta rotation
      * har cyklerna samma period roteras de TILLSAMMANS, så att kopplingen
        nivå <-> längd bevaras
    """
    if compress_lengths is None:
        compress_lengths = getattr(cfg, "compress_lengths", True)

    levels = list(levels)
    lengths = [None if x is None else int(x) for x in lengths]

    # ---- nivådefinitioner
    bins = [bin_of(v) for v in levels]
    uniq = sorted(set(bins))
    assert len(uniq) <= cfg.max_levels, f"{len(uniq)} nivåer, max {cfg.max_levels}"
    # Två skilda nivåvärden får inte hamna i samma bin -- då blir facitet oobserverbart.
    # (Upprepningar i cykeln är däremot tillåtna, jfr ds_revisit.)
    assert len(uniq) == len({round(float(v), 9) for v in levels}), \
        f"bin-kollision mellan nivåer: {sorted(set(levels))} -> bins {uniq}"
    name = {b: LEVEL_NAMES[i] for i, b in enumerate(uniq)}
    seq = [name[b] for b in bins]

    t = ["LEVELS", f"N{len(uniq)}"]
    for b in uniq:
        t += [name[b], f"N{b}"]

    # ---- reducera cyklerna till minsta period
    seq_min = _min_period(seq)
    len_min = _min_period(lengths) if compress_lengths else list(lengths)

    # ---- ordning + längder
    if order_fixed:
        if length_fixed and len(len_min) == len(seq_min):
            # samma period: rotera ihop så att kopplingen bevaras
            pairs = _rot(list(zip(seq_min, len_min)))
            seq_out = [p[0] for p in pairs]
            len_out = [p[1] for p in pairs]
        else:
            seq_out, len_out = _rot(seq_min), None
        t += ["ORDER", "FIXED"] + seq_out
    else:
        seq_out, len_out = None, None
        t += ["ORDER", "RANDOM"]

    if length_fixed:
        if len_out is None:
            len_out = _rot(len_min)
        t += ["DWELL", "FIXED"] + [_L(n) for n in len_out]
    else:
        vals = [n for n in lengths if n is not None]
        lo, hi = (min(vals), max(vals)) if vals else (0, 0)
        t += ["DWELL", "RANGE", f"N{lo}", f"N{hi}"]

    t += ["END"]

    bad = [x for x in t if x not in TOK2ID]
    assert not bad, f"tokens saknas i vokabulär: {bad}"
    assert len(t) + 2 <= cfg.max_tgt, f"facit {len(t)+2} tokens > max_tgt {cfg.max_tgt}"
    return t


def parse(tokens):
    it = iter(tokens)
    def nxt():
        try: return next(it)
        except StopIteration: raise ValueError("oväntat slut")
    def num(tok):
        if not (tok.startswith("N") and tok[1:].isdigit()): raise ValueError(f"väntade tal, fick {tok}")
        return int(tok[1:])
    def length(tok): return None if tok == "INF" else num(tok)
    def is_level(tok): return tok.startswith("L") and tok[1:].isdigit()

    if nxt() != "LEVELS": raise ValueError("saknar LEVELS")
    n = num(nxt()); name2pri = {}
    for _ in range(n):
        lvl = nxt()
        if not is_level(lvl): raise ValueError(f"väntade nivånamn, fick {lvl}")
        name2pri[lvl] = pri_of_bin(num(nxt()))

    if nxt() != "ORDER": raise ValueError("saknar ORDER")
    ot = nxt(); order, lengths = [], []
    order_fixed = length_fixed = None
    tok = nxt()
    if ot == "FIXED":
        order_fixed = True
        while is_level(tok):
            if tok not in name2pri: raise ValueError(f"odefinierad nivå {tok}")
            order.append(name2pri[tok]); tok = nxt()
        if not order: raise ValueError("tom ordning")
    elif ot == "RANDOM":
        order_fixed = False
    else:
        raise ValueError(f"okänd ORDER-typ {ot}")

    if tok != "DWELL": raise ValueError(f"väntade DWELL, fick {tok}")
    dt = nxt()
    if dt not in ("FIXED", "RANGE"): raise ValueError(f"okänd DWELL-typ {dt}")
    length_fixed = dt == "FIXED"; tok = nxt()
    while tok != "END":
        lengths.append(length(tok)); tok = nxt()
    if not lengths: raise ValueError("tomt DWELL-block")
    if not length_fixed and len(lengths) != 2:
        raise ValueError(f"DWELL RANGE kräver min och max, fick {len(lengths)}")

    if tok != "END": raise ValueError(f"väntade END, fick {tok}")
    if length_fixed is None: raise ValueError("saknar längdinformation")
    return dict(levels=sorted(name2pri.values()), order=order, lengths=lengths,
                order_fixed=order_fixed, length_fixed=length_fixed)


def roundtrip_ok(levels, lengths, order_fixed, length_fixed):
    t = to_tokens(levels, lengths, order_fixed, length_fixed)
    d = parse(t)
    src = d["order"] if d["order_fixed"] else d["levels"]
    return to_tokens(src, d["lengths"], d["order_fixed"], d["length_fixed"]) == t
