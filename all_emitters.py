"""
Samlad generator för alla mönstervarianter.

Standardgeneratorn. Väljs med config.py:

    cfg.emitter = "all_emitters"

Samma signatur och returformat som din riktiga generator:

    create_emitter_data(n_emitters, n_signals, drop_rate, noise_level, rng)
        -> [(pri_sequences, label_dict), ...]

label_dict innehåller levels, lengths, order_fixed, length_fixed — plus "variant",
som INTE går in i facitet men följer med för gruppering i utvärderingen.

Varianterna väljs enligt vikterna i GEN.weights. Sätt en vikt till 0 för att
stänga av en variant, eller använd only=[...] i anropet för att köra en enda.

    create_emitter_data(100, 4, 0.0, None, rng, only=["stagger"])

För utvärdering finns make_variant_eval_sets(), som ger ett evalset per variant
och bortfallsnivå, så att du kan se vilken variant som brister.
"""
import numpy as np

from config import cfg


# ------------------------------------------------------------------ inställningar
class GEN:
    n_pulses = 512
    min_gap_bins = 4               # minsta avstånd mellan nivåer, i facit-bins
    level_span = (2, 6)            # antal nivåer för flernivåmönster
    length_span = (4, 16)          # spann att dra dwell-längder ur
    jitter_us = 0.0

    weights = {
        "static":            1.0,   # en nivå, INF
        "jitter":            1.0,   # flera nivåer, slumpvis vald varje puls
        "stagger":           1.0,   # fast cykel, en puls per nivå
        "ds_fix_fix_same":   2.0,   # fast ordning, fast längdcykel, samma period
        "ds_fix_fix_diff":   1.0,   # ... men olika period
        "ds_fix_range":      2.0,   # fast ordning, längd ur intervall
        "ds_rand_fix":       1.5,   # slumpad ordning, fast längdcykel
        "ds_rand_range":     1.5,   # slumpad ordning, längd ur intervall
        "ds_revisit":        1.0,   # fast ordning med återbesök (L0 L1 L2 L0 L3)
    }


VARIANTS = list(GEN.weights)


# ------------------------------------------------------------------ hjälp
def _binwidth():
    return (cfg.pri_max - cfg.pri_min) / cfg.n_bins

def _draw_levels(rng, n):
    """n nivåer med garanterat avstånd, så att de aldrig delar bin."""
    gap = GEN.min_gap_bins * _binwidth()
    for _ in range(200):
        v = np.sort(rng.uniform(cfg.pri_min + gap, cfg.pri_max - gap, n))
        if n == 1 or np.min(np.diff(v)) >= gap:
            return v
    v = np.linspace(cfg.pri_min + gap, cfg.pri_max - gap, n)
    return v + rng.uniform(-gap / 4, gap / 4, n)

def _draw_lengths(rng, k, distinct=None):
    """k längder ur spannet; distinct = antal olika värden att välja bland."""
    lo, hi = GEN.length_span
    pool = np.arange(lo, hi + 1)
    d = min(distinct or k, pool.size)
    vals = rng.choice(pool, size=d, replace=False)
    return [int(vals[int(rng.integers(d))]) for _ in range(k)]

def _n_levels(rng, lo=None):
    a, b = GEN.level_span
    return int(rng.integers(max(a, lo or a), b + 1))


# ------------------------------------------------------------------ varianter
def _make_emitter(variant, rng):
    """-> (cycle_us, lengths, order_fixed, length_fixed)"""
    if variant == "static":
        lv = _draw_levels(rng, 1)
        return lv, [None], True, True

    if variant == "jitter":
        n = _n_levels(rng, 2)
        lv = _draw_levels(rng, n)
        return lv, [1], False, True                    # slumpvis nivå, en puls var

    if variant == "stagger":
        n = _n_levels(rng, 2)
        lv = _draw_levels(rng, n)
        return lv[rng.permutation(n)], [1], True, True

    n = _n_levels(rng, 2)
    lv = _draw_levels(rng, n)
    cycle = lv[rng.permutation(n)]

    if variant == "ds_fix_fix_same":
        return cycle, _draw_lengths(rng, n, distinct=min(n, 3)), True, True

    if variant == "ds_fix_fix_diff":
        k = n
        while k == n or k < 2:                          # period skild från nivåcykeln
            k = int(rng.integers(2, max(3, n + 3)))
        return cycle, _draw_lengths(rng, k, distinct=min(k, 3)), True, True

    if variant == "ds_fix_range":
        return cycle, _range_pair(rng), True, False

    if variant == "ds_rand_fix":
        k = int(rng.integers(1, 4))
        return cycle, _draw_lengths(rng, k, distinct=min(k, 3)), False, True

    if variant == "ds_rand_range":
        return cycle, _range_pair(rng), False, False

    if variant == "ds_revisit":
        order = _revisit_order(rng, n)
        return lv[order], _draw_lengths(rng, len(order), distinct=min(3, len(order))), True, True

    raise ValueError(f"okänd variant {variant}")


def _revisit_order(rng, n):
    """
    Permutation av 0..n-1 där en nivå återkommer en gång (L0 L1 L2 L0 L3).
    Samma nivå får ALDRIG stå två gånger i rad, och cykeln är cirkulär -- sista
    och första elementet räknas som grannar. Annars slås de två besöken ihop i
    signalen och dwell-längden blir summan, vilket gör facitet oobserverbart.
    """
    if n < 3:
        return list(rng.permutation(n))
    for _ in range(50):
        order = list(rng.permutation(n))
        dup = int(order[int(rng.integers(n))])          # vilken nivå som återkommer
        cand = [i for i in range(1, len(order) + 1)
                if order[i - 1] != dup and order[i % len(order)] != dup]
        if cand:
            order.insert(int(rng.choice(cand)), dup)
            return order
    return list(rng.permutation(n))                      # fallback: inget återbesök


def _range_pair(rng):
    lo, hi = GEN.length_span
    a = int(rng.integers(lo, hi))
    b = int(rng.integers(a + 1, hi + 1))
    return [a, b]


# ------------------------------------------------------------------ utrullning
def _make_signal(cycle, lengths, order_fixed, length_fixed, rng, drop_rate, drop_rng=None):
    """
    rng      : signalens ström (startfas, slumpad ordning, slumpade längder)
    drop_rng : bortfallets ström. Egen ström, annars förskjuts alla efterföljande
               signaler så fort drop_rate > 0 och evalseten slutar vara jämförbara.
    """
    drop_rng = rng if drop_rng is None else drop_rng
    n_pulses = GEN.n_pulses
    is_inf = any(x is None for x in lengths)

    if is_inf:
        pri = np.repeat(cycle[0], n_pulses).astype(float)
    else:
        n_visits = int(np.ceil(n_pulses / max(1.0, float(np.mean(lengths))))) + len(cycle) + 4

        phase = int(rng.integers(0, len(cycle)))
        if order_fixed:
            reps = int(np.ceil(n_visits / len(cycle))) + 1
            lv = np.tile(cycle, reps)[phase:phase + n_visits]
        else:
            idx = rng.integers(0, len(cycle), n_visits)
            for i in range(1, n_visits):                # aldrig samma nivå två gånger i rad
                while len(cycle) > 1 and idx[i] == idx[i - 1]:
                    idx[i] = rng.integers(0, len(cycle))
            lv = cycle[idx]

        if length_fixed:
            # Nivå- och längdcykeln rullas med SAMMA löpande index. Vid fast ordning
            # betyder besök i alltså cycle[(phase+i) % nc] tillsammans med
            # lengths[(phase+i) % nl] -- exakt den parning labels.to_tokens
            # kanoniserar, och den verify.py:s parvisa kontroll letar efter.
            # Med oberoende startfas här stämmer facitet bara i 1 av nc fall.
            lphase = phase if order_fixed else int(rng.integers(0, len(lengths)))
            reps = int(np.ceil((n_visits + lphase) / len(lengths))) + 1
            ln = np.tile(np.array(lengths, dtype=int), reps)[lphase:lphase + n_visits]
        else:
            ln = rng.integers(lengths[0], lengths[1] + 1, n_visits)

        pri = np.repeat(lv, ln)[:n_pulses].astype(float)

    if GEN.jitter_us > 0:
        pri = pri + rng.uniform(-GEN.jitter_us, GEN.jitter_us, pri.size)

    if drop_rate > 0:
        toa = np.concatenate([[0.0], np.cumsum(pri)])
        keep = drop_rng.random(toa.size) > drop_rate
        keep[0] = True
        pri = np.diff(toa[keep])

    return pri


# ------------------------------------------------------------------ API
def create_emitter_data(n_emitters, n_signals, drop_rate=0.0, noise_level=None,
                        rng=None, only=None):
    """only : lista med variantnamn, eller None för viktad blandning."""
    rng = rng or np.random.default_rng()
    # Tre strömmar: emittrar, signaler och bortfall oberoende av varandra. Då ger
    # samma seed samma emittrar OCH samma signaler för alla drop_rate, så att
    # bortfallskurvan mäter bortfall och ingenting annat.
    em_rng, sig_rng, drop_rng = rng.spawn(3)

    names = only or VARIANTS
    w = np.array([GEN.weights[v] for v in names], dtype=float)
    if w.sum() <= 0:
        raise ValueError("alla vikter är noll")
    w = w / w.sum()

    out = []
    for _ in range(n_emitters):
        variant = names[int(em_rng.choice(len(names), p=w))] if len(names) > 1 else names[0]
        cycle, lengths, order_fixed, length_fixed = _make_emitter(variant, em_rng)

        label = dict(levels=cycle.tolist(), lengths=list(lengths),
                     order_fixed=bool(order_fixed), length_fixed=bool(length_fixed),
                     variant=variant)
        seqs = [_make_signal(cycle, lengths, order_fixed, length_fixed,
                             sig_rng, drop_rate, drop_rng)
                for _ in range(n_signals)]
        out.append((seqs, label))
    return out


def make_variant_eval_sets(n_per_variant=200, n_signals=1, p_drops=(0.0,), seed=None):
    """
    -> {"stagger/drop_0.00": [(channels, tokens), ...], ...}
    Ett evalset per variant och bortfallsnivå, med samma emittrar i alla nivåer.
    Kräver data.make_channels och data.label_to_tokens.
    """
    from data import make_channels, label_to_tokens
    seed = cfg.eval_seed if seed is None else seed

    sets = {}
    for v in VARIANTS:
        if GEN.weights[v] <= 0:
            continue
        for p in p_drops:
            rng = np.random.default_rng([seed, VARIANTS.index(v)])
            data = create_emitter_data(n_per_variant, n_signals, p, None, rng, only=[v])
            pairs = []
            for seqs, lab in data:
                tok = label_to_tokens(lab)
                pairs.extend((make_channels(s), tok) for s in seqs)
            sets[f"{v}/drop_{p:.2f}"] = pairs
    return sets


# ------------------------------------------------------------------ snabbtest
if __name__ == "__main__":
    import time
    from collections import Counter
    from labels import to_tokens, parse
    from verify import verify_stats

    rng = np.random.default_rng(0)

    t = time.perf_counter()
    data = create_emitter_data(400, 2, 0.0, None, rng)
    dt = time.perf_counter() - t
    n_sig = sum(len(s) for s, _ in data)
    print(f"{n_sig} signaler på {dt*1000:.0f} ms  ->  {dt/n_sig*1000:.3f} ms per signal")
    print("fördelning:", Counter(l["variant"] for _, l in data).most_common(), "\n")

    # ett exempelfacit per variant
    for v in VARIANTS:
        if GEN.weights[v] <= 0:
            continue
        seqs, lab = create_emitter_data(1, 1, 0.0, None,
                                        np.random.default_rng(1), only=[v])[0]
        tok = to_tokens(lab["levels"], lab["lengths"], lab["order_fixed"], lab["length_fixed"])
        assert parse(tok)
        print(f"{v:18s} {' '.join(tok)}")

    print()
    verify_stats(data)
