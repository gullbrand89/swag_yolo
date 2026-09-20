"""
Formatet per tillstånd: posten bär hela emittermodellen själv.

    LEVELS L0 B482 L1 B613 L2 B1210     nivåordlistan, stigande bins (som förut)
    S L2 D3  S L0 D1  S L2 D3  S L1 D7  ett tillstånd i taget: komponent + dwell
    SEEN D2                              pulser tillstånd 0 hunnit visa i fönstret
    END

Tillstånd `S <komponent> <dwell>`:
    komponent  L2          nivå 2 ur LEVELS
               *           vilken nivå som helst (slumpad ordning)
    dwell      D7          fast, 7 pulser
               RANGE D5 D12  likformig 5..12, dras om per besök
               INF         lämnas aldrig (statisk); då ingen SEEN

Ankring vid fönstrets SLUT, skrivet BAKÅT i tiden. Tillstånd 0 pågår när fönstret
tar slut, tillstånd 1 är det som var före, osv.: tillstånd j = besöket j steg före
slutet (mod P). Det är samma riktning som besökskanalen räknar (cfg.visit_from_end),
så att skriva tillstånd j är en uppslagning av besök j från slutet -- ingen räkning,
ingen kunskap om P. Tack vare periodiciteten är följden framåt den omvända: nästa
tillstånd efter 0 är P-1, sedan P-2 ... (tools.post_till_modell vänder den). SEEN
säger hur långt tillstånd 0 kommit, så återstående dwell är D_0 - SEEN.

Kanonisering:
  1. följden av (komponent, dwell) reduceras till sin minsta period och skrivs en gång
  2. nollbrett RANGE => fast dwell
  3. en unik nivå => S L0 INF: att emittern "byter" mellan lika värden är inte
     observerbart
  4. två nivåer => aldrig *: följden alternerar och är identisk med en fast tvåcykel
  5. slumpad ordning med >= 3 nivåer => *, med dwellcykeln ankrad vid slutet

Utan torch. Token S, * och SEEN saknas i vocab.py tills formatet kopplas in;
_giltig() kontrollerar mot VOCAB + EXTRA_TOKENS.
"""
from math import gcd

from .config import cfg
from .vocab import TOK2ID, LEVEL_NAMES, bin_of, pri_of_bin

EXTRA_TOKENS = () if "S" in TOK2ID else ("S", "*", "SEEN")   # tills formatet är påslaget


def _lcm(a, b):
    return a * b // gcd(a, b)


def _min_period(seq):
    n = len(seq)
    for P in range(1, n + 1):
        if n % P == 0 and all(seq[i] == seq[i % P] for i in range(n)):
            return P
    return n


def _dwell_tokens(d):
    if d[0] == "D":
        return [f"D{d[1]}"]
    if d[0] == "R":
        return ["RANGE", f"D{d[1]}", f"D{d[2]}"]
    return ["INF"]


def to_tokens(levels, lengths, order_fixed, length_fixed, fas):
    """
    levels, lengths, order_fixed, length_fixed : generatorns etikett (som förut)
    fas : label["fas"][i] från all_emitters._make_signal -- läget vid fönstrets slut
    -> tokenlista
    """
    levels = list(levels)
    lengths = [None if x is None else int(x) for x in lengths]
    is_inf = any(x is None for x in lengths)

    # nollbrett intervall är en fast dwell
    if not length_fixed and lengths and None not in lengths and min(lengths) == max(lengths):
        length_fixed, lengths = True, [lengths[0]]

    bins = [bin_of(v) for v in levels]
    uniq = sorted(set(bins))
    assert len(uniq) <= cfg.max_levels, f"{len(uniq)} nivåer, max {cfg.max_levels}"
    assert len(uniq) == len({round(float(v), 9) for v in levels}), \
        f"bin-kollision mellan nivåer: {sorted(set(levels))} -> {uniq}"
    name = {b: LEVEL_NAMES[i] for i, b in enumerate(uniq)}
    t = ["LEVELS"]
    for b in uniq:
        t += [name[b], f"B{b}"]

    nu, nc, nl = len(uniq), len(levels), len(lengths)
    i_last = int(fas["i_last"])

    if is_inf or nu == 1:
        # statisk, eller "byten" mellan lika värden som inte syns: ett tillstånd
        t += ["S", name[uniq[0]], "INF", "END"]
        _giltig(t)
        return t

    # ---- dwell per besöksindex i (räknat från fönstrets start)
    if length_fixed:
        lph = int(fas["phase"]) if order_fixed else int(fas["lphase"])
        dw = lambda i: ("D", lengths[(lph + i) % nl])
        per_dw = _min_period(lengths)
    else:
        a, b = sorted(lengths)
        dw = lambda i: ("R", a, b)
        per_dw = 1

    # ---- komponent per besöksindex
    if order_fixed:
        ph = int(fas["phase"])
        comp = lambda i: name[bins[(ph + i) % nc]]
        per_c = nc
    elif nu == 2:
        # alternerar; sista besökets nivå är känd
        b_last = bin_of(fas["last_level"])
        other = uniq[0] if uniq[1] == b_last else uniq[1]
        comp = lambda i: name[b_last] if (i - i_last) % 2 == 0 else name[other]
        per_c = 2
    else:
        comp = lambda i: "*"
        per_c = 1

    L = _lcm(per_c, per_dw)
    seq = [(comp(i), dw(i)) for i in range(i_last, i_last - L, -1)]     # bakåt i tiden
    P = _min_period(seq)
    for c, d in seq[:P]:
        t += ["S", c] + _dwell_tokens(d)
    t += ["SEEN", f"D{int(fas['seen'])}", "END"]
    _giltig(t)
    return t


def _giltig(t):
    bad = [x for x in t if x not in TOK2ID and x not in EXTRA_TOKENS]
    assert not bad, f"tokens saknas i vokabulär: {bad}"
    assert len(t) + 2 <= cfg.max_tgt, f"facit {len(t) + 2} tokens > max_tgt {cfg.max_tgt}"


def parse(tokens):
    """
    -> dict(levels=[µs per nivå, index = L-nummer], bins=[...],
            states=[(komponent, dwell), ...]   komponent: nivåindex eller None (*)
                                                dwell: ("D", k) | ("R", a, b) | ("INF",)
            seen=int eller None)
    Kastar ValueError på en post som inte följer grammatiken.
    """
    t = list(tokens)
    if not t or t[0] != "LEVELS":
        raise ValueError("börjar inte med LEVELS")
    i, bins = 1, []
    while i < len(t) and t[i] != "S":
        if not (t[i].startswith("L") and i + 1 < len(t) and t[i + 1].startswith("B")):
            raise ValueError(f"trasigt nivåblock vid {i}: {t[i:i+2]}")
        if t[i] != LEVEL_NAMES[len(bins)]:
            raise ValueError(f"nivånamn i fel ordning: {t[i]}")
        bins.append(int(t[i + 1][1:])); i += 2
    if not bins or any(b2 <= b1 for b1, b2 in zip(bins, bins[1:])):
        raise ValueError("nivåbins inte strikt stigande")
    states, seen = [], None
    while i < len(t) and t[i] == "S":
        c = t[i + 1] if i + 1 < len(t) else None
        if c == "*":
            comp = None
        elif c in LEVEL_NAMES[:len(bins)]:
            comp = LEVEL_NAMES.index(c)
        else:
            raise ValueError(f"ogiltig komponent {c}")
        i += 2
        if i >= len(t):
            raise ValueError("tillstånd utan dwell")
        if t[i] == "INF":
            d, i = ("INF",), i + 1
        elif t[i] == "RANGE":
            if i + 2 >= len(t) or not (t[i+1].startswith("D") and t[i+2].startswith("D")):
                raise ValueError("RANGE utan två D-token")
            a, b = int(t[i + 1][1:]), int(t[i + 2][1:])
            if a >= b:
                raise ValueError(f"RANGE D{a} D{b} är inte ett intervall")
            d, i = ("R", a, b), i + 3
        elif t[i].startswith("D"):
            d, i = ("D", int(t[i][1:])), i + 1
        else:
            raise ValueError(f"ogiltig dwell {t[i]}")
        states.append((comp, d))
    if not states:
        raise ValueError("inga tillstånd")
    if i < len(t) and t[i] == "SEEN":
        if i + 1 >= len(t) or not t[i + 1].startswith("D"):
            raise ValueError("SEEN utan D-token")
        seen, i = int(t[i + 1][1:]), i + 2
    if i >= len(t) or t[i] != "END" or i != len(t) - 1:
        raise ValueError("saknar END, eller token efter END")
    if states[0][1][0] == "INF" and (len(states) != 1 or seen is not None):
        raise ValueError("INF måste vara ensamt tillstånd utan SEEN")
    if states[0][1][0] != "INF" and seen is None:
        raise ValueError("SEEN saknas")
    return dict(bins=bins, levels=[pri_of_bin(b) for b in bins], states=states, seen=seen)


def tokens_of(d):
    """parse() -> tokens, för rundturstest (parse(to_tokens(x)) -> tokens_of == to_tokens(x))."""
    t = ["LEVELS"]
    for k, b in enumerate(d["bins"]):
        t += [LEVEL_NAMES[k], f"B{b}"]
    for comp, dw in d["states"]:
        t += ["S", "*" if comp is None else LEVEL_NAMES[comp]] + _dwell_tokens(dw)
    if d["seen"] is not None:
        t += ["SEEN", f"D{d['seen']}"]
    return t + ["END"]


def fas_for_prefix(pri_clean, n_obs, fas_full):
    """
    Läget vid slutet av ett AVHUGGET fönster pri_clean[:n_obs], ur den rena signalen
    och emitterns faser (fas_full från generatorn, vars i_last/seen gäller hela
    signalen). Till prediktion: modellen ser n_obs pulser, facit måste ankras där.
    Bara ren signal: besöken räknas ur pulserna med run_tol_bins, och ett tappat
    besök skulle förskjuta i_last.
    """
    import numpy as np
    pri = np.asarray(pri_clean, dtype=float)[:n_obs]
    b = [bin_of(x) for x in pri]
    n_vis, langd = 0, 0
    for i, x in enumerate(b):
        if i == 0 or abs(x - b[i - 1]) > cfg.run_tol_bins:
            n_vis += 1; langd = 1
        else:
            langd += 1
    return dict(i_last=n_vis - 1, phase=fas_full["phase"], lphase=fas_full["lphase"],
                last_level=float(pri[-1]), seen=int(langd))


def label_to_tokens(label, i=0):
    return to_tokens(label["levels"], label["lengths"], label["order_fixed"],
                     label["length_fixed"], label["fas"][i])
