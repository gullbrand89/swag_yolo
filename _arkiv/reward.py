"""
Verifierbar belöning för RL-träning.

Idén, samma som DeepSeek använder för matematik och kod: belöna inte likheten med
ett facit, belöna att svaret *går att verifiera*. Här finns verifieraren redan --
verify.py mäter en bibliotekspost mot den signal den påstår sig beskriva. reward()
gör samma sak, men graderat i [0, 1] i stället för binärt, och vektoriserat så att
den orkar köras några tusen gånger per träningssteg.

Konsekvensen är att belöningen INTE behöver facitet. Den mäter modellens utdata mot
pulsföljden, vilket betyder att samma träningsloop går att köra på inspelad signal
utan sanning. Facitet används bara till utvärdering (runlog.compare), aldrig till
gradienten.

    from reward import reward, reward_report
    r = reward(pred_tokens, pri)            # float i [0, 1]
    print(reward_report(pred_tokens, pri))  # delkomponenterna, för felsökning


Belöningshackning
-----------------
En verifierare är alltid ett spel, och två drag är uppenbara här. Båda är
avsiktligt stängda:

  ORDER RANDOM   slipper undan ordningskontrollen -- verify.py godkänner den
                 villkorslöst. Här straffas den i stället mot hur periodisk
                 den observerade besöksföljden faktiskt är: påstår modellen
                 slump i en signal som uppenbart cyklar får den noll.

  DWELL RANGE N0 N511  täcker varje tänkbar dwell-längd. Här multipliceras
                 täckningen med hur snävt intervallet är jämfört med det
                 observerade, så ett brett intervall är värt nästan ingenting.

Övriga degenererade utdata fångas av F1:t på nivåerna: för få nivåer sänker
recall, för många sänker precision.
"""
import numpy as np

from config import cfg
from vocab import bin_of, pri_of_bin
from labels import parse, to_tokens, _min_period


# ------------------------------------------------------------------ vikter
class RW:
    """Delbelöningarnas vikter. Summerar till 1,0 så att reward() ligger i [0, 1]."""
    parse     = 0.10      # facitet går att tolka alls (motsvarar DeepSeeks formatbelöning)
    canonical = 0.05      # ... och är på kanonisk form: to_tokens(parse(t)) == t
    levels    = 0.35      # nivåerna matchar signalens toppar (F1)
    coverage  = 0.15      # andel pulser som matchar någon förutsagd nivå
    order     = 0.175     # besöksföljden stämmer med den påstådda ordningen
    lengths   = 0.175     # dwell-längderna stämmer med den påstådda längdcykeln

    tol_bins   = 1        # tolerans vid nivåmatchning
    min_pulses = 3        # bins med färre pulser räknas som artefakter, inte nivåer
    trim_edges = True     # första och sista dwellen är avklippta av fönstret
    max_period = 8        # längsta cykel som prövas när RANDOM ska motbevisas
    max_merge  = 3        # högsta antal ihopslagna intervall som tolkas som bortfall

    @classmethod
    def total(cls):
        return cls.parse + cls.canonical + cls.levels + cls.coverage + cls.order + cls.lengths


# ------------------------------------------------------------------ signalhjälp
def _quantize(pri):
    """Ta bort flyttalsbrus utan att röra verkliga skillnader."""
    w = (cfg.pri_max - cfg.pri_min) / cfg.n_bins
    return np.round(np.asarray(pri, dtype=float) / w * 100) * w / 100


def obs_bins_of(pri):
    return np.array([bin_of(p) for p in _quantize(pri)], dtype=np.int64)


def strong_bins(obs, min_pulses=None):
    """De bins i signalen som bär tillräckligt många pulser för att vara nivåer."""
    mp = RW.min_pulses if min_pulses is None else min_pulses
    if obs.size == 0:
        return np.zeros(0, dtype=np.int64)
    uniq, cnt = np.unique(obs, return_counts=True)
    return uniq[cnt >= mp]


def binwidth():
    return (cfg.pri_max - cfg.pri_min) / (cfg.n_bins - 1)


def visits(pri, lvl_us, tol_bins=None, max_merge=None):
    """
    Kollapsa pulser till besök. -> (nivåindex per besök, längd per besök, omatchad andel)

    Ett bortfall slår ihop två intervall till ett som är ungefär dubbelt så långt.
    Utan hänsyn till det bryts besöket i två och en falsk puls skjuts in emellan,
    vilket raderar både ordnings- och längdinformationen: vid 20 % bortfall låg
    ordningsbelöningen på ren slumpnivå. Därför matchas varje puls mot m * nivå för
    m = 1..max_merge, och en träff på m räknas som m pulser på den nivån.

    Toleransen växer med m eftersom felet gör det. En liten straffterm gör att m=1
    vinner vid lika avstånd, så en nivå på 600 µs inte tolkas som två på 300.
    """
    tol_bins = RW.tol_bins if tol_bins is None else tol_bins
    max_merge = RW.max_merge if max_merge is None else max_merge

    P = np.asarray(pri, dtype=float)
    L = np.asarray(lvl_us, dtype=float)
    if L.size == 0 or P.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), 1.0

    tol_us = (tol_bins + 0.5) * binwidth()
    ms = np.arange(1, max_merge + 1, dtype=float)

    d = np.abs(P[:, None, None] - ms[None, :, None] * L[None, None, :])   # (T, M, K)
    key = np.where(d <= tol_us * ms[None, :, None],
                   d + 0.5 * tol_us * (ms[None, :, None] - 1.0), np.inf)
    flat = key.reshape(P.size, -1)                    # m-major: m=1 först, så lika avstånd
    best = flat.argmin(1)                             # bryts till förmån för m=1
    hit = np.isfinite(flat[np.arange(P.size), best])

    idx = np.where(hit, best % L.size, -1)
    mult = np.where(hit, best // L.size + 1, 1)

    chg = np.empty(idx.size, dtype=bool)
    chg[0] = True
    chg[1:] = idx[1:] != idx[:-1]
    starts = np.flatnonzero(chg)
    runs = np.add.reduceat(mult, starts)              # summera pulser, inte intervall
    return idx[starts], runs.astype(np.int64), float(1.0 - hit.mean())


def _best_fit(obs, cycle):
    """Bästa andel positioner som stämmer, över alla rotationer av cycle."""
    obs = np.asarray(obs)
    cyc = np.asarray(list(cycle))
    if obs.size == 0 or cyc.size == 0:
        return 0.0
    p, n = cyc.size, obs.size
    pos = np.arange(n) % p
    best = 0.0
    for r in range(p):
        rot = np.roll(cyc, -r)
        best = max(best, float((obs == rot[pos]).mean()))
    return best


def _periodicity(obs, max_p=None):
    """
    Hur väl den observerade besöksföljden förklaras av NÅGON fast cykel, med
    majoritetsröstning per cykelposition. 1,0 = perfekt periodisk.
    Används för att motbevisa ett påstående om ORDER RANDOM.
    """
    max_p = RW.max_period if max_p is None else max_p
    obs = np.asarray(obs)
    if obs.size < 3:
        return 0.0
    best = 0.0
    for p in range(1, min(max_p, obs.size // 3) + 1):
        cyc = np.empty(p, dtype=obs.dtype)
        for k in range(p):
            vals, cnt = np.unique(obs[k::p], return_counts=True)
            cyc[k] = vals[cnt.argmax()]
        best = max(best, float((obs == cyc[np.arange(obs.size) % p]).mean()))
    return best


_NULL_PER = {}

def _null_periodicity(k, n):
    """
    Hur periodisk en ÄKTA slumpmässig besöksföljd över k nivåer ser ut, med samma
    majoritetsröstning. Nödvändigt eftersom generatorn aldrig upprepar samma nivå
    två gånger i rad: med två nivåer blir slumpen exakt alternerande, alltså
    perfekt "periodisk", och ORDER RANDOM är då inte skiljbart från ORDER FIXED.
    Utan den här baslinjen straffar belöningen korrekta RANDOM-facit.

    Cachead per (k, längdintervall), så nollhypotesen räknas en gång.
    """
    k, n = int(k), int(n)
    if k <= 1:
        return 1.0
    key = (k, min(n, 1024) // 16)
    if key not in _NULL_PER:
        rng = np.random.default_rng(9973 + 101 * k + key[1])
        vals = []
        for _ in range(8):
            s = np.empty(max(n, 3), dtype=np.int64)
            s[0] = rng.integers(k)
            for i in range(1, s.size):
                x = int(rng.integers(k))
                while x == s[i - 1]:
                    x = int(rng.integers(k))
                s[i] = x
            vals.append(_periodicity(s))
        _NULL_PER[key] = float(np.mean(vals))
    return _NULL_PER[key]


def _randomness(obs):
    """
    1,0 = besöksföljden är lika oregelbunden som ren slump, 0,0 = den är en ren cykel.
    Mäts som överskottet över nollhypotesen, inte som periodiciteten rakt av.
    """
    obs = np.asarray(obs)
    if obs.size < 6:
        return 1.0                                   # för få besök för att motbevisa slump
    k = int(np.unique(obs).size)
    base = _null_periodicity(k, obs.size)
    if base >= 0.999:                                # t.ex. två nivåer: oskiljbart
        return 1.0
    excess = (_periodicity(obs) - base) / (1.0 - base)
    return float(1.0 - min(1.0, max(0.0, excess)))


def _f1(pred_bins, true_bins, tol=None):
    """Girig 1-1-matchning inom tolerans. -> (f1, precision, recall)"""
    tol = RW.tol_bins if tol is None else tol
    pred, true = list(pred_bins), list(true_bins)
    if not pred and not true:
        return 1.0, 1.0, 1.0
    if not pred or not true:
        return 0.0, 0.0, 0.0
    rem, hit = list(pred), 0
    for t in true:
        if not rem:
            break
        k = min(range(len(rem)), key=lambda i: abs(rem[i] - t))
        if abs(rem[k] - t) <= tol:
            hit += 1
            rem.pop(k)
    prec, rec = hit / len(pred), hit / len(true)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec


# ------------------------------------------------------------------ belöning
def reward_report(tokens, pri):
    """
    -> dict med `total` plus varje delkomponent. reward() är det här minus dicten.
    Kastar aldrig: en obegriplig utdata ger total 0,0 och parsed False.
    """
    out = dict(total=0.0, parsed=False, canonical=False, n_levels=0,
               levels=0.0, precision=0.0, recall=0.0, coverage=0.0,
               order=0.0, lengths=0.0, n_visits=0, order_type="", length_type="")

    tokens = list(tokens)
    try:
        d = parse(tokens)
    except Exception:
        return out
    out["parsed"] = True
    out["order_type"] = "FIXED" if d["order_fixed"] else "RANDOM"

    # ---- kanonisk form: samma post ska inte kunna skrivas på två sätt
    try:
        src = d["order"] if d["order_fixed"] else d["levels"]
        out["canonical"] = to_tokens(src, d["lengths"], d["order_fixed"],
                                     d["length_fixed"]) == tokens
    except Exception:
        out["canonical"] = False

    cycle_us = d["order"] if d["order_fixed"] else d["levels"]
    pred_uniq = sorted({bin_of(v) for v in cycle_us})
    out["n_levels"] = len(pred_uniq)

    obs = obs_bins_of(pri)
    if obs.size == 0 or not pred_uniq:
        out["total"] = RW.parse + RW.canonical * out["canonical"]
        return out

    # ---- 1. nivåer mot signalens egna toppar
    out["levels"], out["precision"], out["recall"] = _f1(pred_uniq, strong_bins(obs).tolist())

    # ---- 2. täckning (mot nivåerna i µs, så att ihopslagna intervall kan räknas)
    pred_us = [pri_of_bin(b) for b in pred_uniq]
    seq, runs, unmatched = visits(_quantize(pri), pred_us)
    out["coverage"] = 1.0 - unmatched

    keep = list(zip(seq.tolist(), runs.tolist()))
    if RW.trim_edges and len(keep) > 2:
        keep = keep[1:-1]                       # avklippta dwells i kanterna
    pairs = [(v, r) for v, r in keep if v >= 0]
    seq_t = np.array([v for v, _ in pairs], dtype=np.int64)
    runs_t = np.array([r for _, r in pairs], dtype=np.int64)
    out["n_visits"] = int(seq_t.size)

    # ---- 3. ordning
    if d["order_fixed"]:
        idx = {b: i for i, b in enumerate(pred_uniq)}
        cyc = _min_period([idx[bin_of(v)] for v in d["order"]])
        out["order"] = _best_fit(seq_t, cyc)
    else:
        # påstår modellen slump måste signalen faktiskt sakna periodicitet,
        # mätt mot vad ren slump över lika många nivåer ser ut som
        out["order"] = _randomness(seq_t)

    # ---- 4. längder
    lens = list(d["lengths"])
    if any(x is None for x in lens):
        out["length_type"] = "INF"
        out["lengths"] = 1.0 if len(set(seq_t.tolist())) <= 1 else 0.0
    elif d["length_fixed"]:
        out["length_type"] = "FIXED"
        out["lengths"] = _best_fit(runs_t, _min_period([int(x) for x in lens]))
    else:
        out["length_type"] = "RANGE"
        lo, hi = min(lens), max(lens)
        if runs_t.size == 0:
            out["lengths"] = 0.0
        else:
            inside = float(((runs_t >= lo) & (runs_t <= hi)).mean())
            obs_span = int(runs_t.max() - runs_t.min()) + 1
            tight = min(1.0, obs_span / max(1, int(hi) - int(lo) + 1))
            out["lengths"] = inside * tight     # brett intervall är nästan värdelöst

    # Ordning och längder skalas med täckningen. Annars räcker det att påstå EN nivå
    # för att båda ska bli triviellt uppfyllda -- en cykel med period 1 stämmer alltid,
    # och INF stämmer alltid när bara en nivå besöks. Den utdatan får nu bara betalt
    # för den andel av signalen den faktiskt förklarar.
    out["total"] = (RW.parse
                    + RW.canonical * float(out["canonical"])
                    + RW.levels * out["levels"]
                    + RW.coverage * out["coverage"]
                    + RW.order * out["order"] * out["coverage"]
                    + RW.lengths * out["lengths"] * out["coverage"])
    return out


def reward(tokens, pri):
    """-> float i [0, 1]. Kastar aldrig."""
    try:
        return float(reward_report(tokens, pri)["total"])
    except Exception:
        return 0.0


def rewards(token_lists, pris):
    """Belöning för en hel rollout-batch. pris[i] hör till token_lists[i]."""
    return np.array([reward(t, p) for t, p in zip(token_lists, pris)], dtype=np.float32)


# ------------------------------------------------------------------ egentest
if __name__ == "__main__":
    import time
    from all_emitters import create_emitter_data, VARIANTS

    def label_to_tokens(l):
        return to_tokens(l["levels"], l["lengths"], l["order_fixed"], l["length_fixed"])

    rng = np.random.default_rng(0)

    print("1) sant facit ska ligga nära 1,0\n")
    print(f"   {'variant':<18} {'total':>6} {'niv':>6} {'täck':>6} {'ordn':>6} {'längd':>6}")
    for v in VARIANTS:
        seqs, lab = create_emitter_data(1, 1, 0.0, None, np.random.default_rng(1), only=[v])[0]
        r = reward_report(label_to_tokens(lab), seqs[0])
        print(f"   {v:<18} {r['total']:>6.3f} {r['levels']:>6.3f} {r['coverage']:>6.3f} "
              f"{r['order']:>6.3f} {r['lengths']:>6.3f}")

    print("\n2) degenererade utdata ska ligga lågt\n")
    seqs, lab = create_emitter_data(1, 1, 0.0, None, np.random.default_rng(3),
                                    only=["ds_fix_fix_same"])[0]
    pri, true_tok = seqs[0], label_to_tokens(lab)
    d = parse(true_tok)
    b = sorted({bin_of(x) for x in d["levels"]})

    cases = {
        "sant facit": true_tok,
        "en enda nivå, INF": ["LEVELS", "N1", "L0", f"N{b[0]}", "ORDER", "FIXED", "L0",
                              "DWELL", "FIXED", "INF", "END"],
        "rätt nivåer, ORDER RANDOM": (["LEVELS", f"N{len(b)}"]
                                      + [t for i, x in enumerate(b) for t in (f"L{i}", f"N{x}")]
                                      + ["ORDER", "RANDOM", "DWELL", "RANGE", "N0", "N511", "END"]),
        "rätt nivåer, maximalt RANGE": (["LEVELS", f"N{len(b)}"]
                                        + [t for i, x in enumerate(b) for t in (f"L{i}", f"N{x}")]
                                        + ["ORDER", "FIXED"] + [f"L{i}" for i in range(len(b))]
                                        + ["DWELL", "RANGE", "N0", "N511", "END"]),
        "20 nivåer, alla fel": (["LEVELS", "N20"]
                                + [t for i in range(20) for t in (f"L{i}", f"N{7 + 25 * i}")]
                                + ["ORDER", "RANDOM", "DWELL", "FIXED", "N5", "END"]),
        "trasiga tokens": ["LEVELS", "ORDER", "END"],
    }
    for name, tok in cases.items():
        r = reward_report(tok, pri)
        print(f"   {name:<28} {r['total']:>6.3f}   parsed={str(r['parsed']):<5} "
              f"niv={r['levels']:.2f} täck={r['coverage']:.2f} "
              f"ordn={r['order']:.2f} längd={r['lengths']:.2f}")

    print("\n3) hastighet")
    data = create_emitter_data(64, 1, 0.0, None, rng)
    toks = [label_to_tokens(l) for _, l in data]
    pris = [s[0] for s, _ in data]
    t = time.perf_counter()
    for _ in range(8):
        rewards(toks, pris)
    dt = (time.perf_counter() - t) / (8 * 64)
    print(f"   {dt*1000:.3f} ms per belöning  ->  {dt*128*1000:.0f} ms för en rollout på 128")
