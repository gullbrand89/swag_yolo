"""
Snabb dwell & switch-generator för avgränsade experiment på LÄNGDER.

Väljs med config.py:

    cfg.emitter = "dwell_emitter"

Samma signatur och returformat som din riktiga generator:

    create_emitter_data(n_emitters, n_signals, drop_rate, noise_level, rng)
        -> [(pri_sequences, label_dict), ...]

Rattarna i DWELL nedan är avsedda att vridas EN I TAGET, så att varje körning
svarar på en fråga. Föreslagen ordning:

    1. dwell_mode="const", dwell_const=10        alla dwells lika, fast
    2. dwell_mode="cycle", n_lengths=2           två längder i fast cykel
    3. dwell_mode="cycle", n_lengths=3           tre längder
    4. dwell_mode="range"                        slumpad längd ur intervall
    5. same_period=False                         nivå- och längdcykel olika period
    6. order_random=True                         slumpad nivåordning

Vektoriserad utrullning, ~0,2 ms per signal.
"""
import numpy as np

from config import cfg


# ------------------------------------------------------------------ inställningar
class DWELL:
    n_levels = (3, 4)          # antal nivåer, inklusive gränser
    n_pulses = 512             # pulser per signal
    min_gap_bins = 4           # minsta avstånd mellan nivåer, i facit-bins

    dwell_mode = "cycle"       # "const" | "cycle" | "range"
    dwell_const = 10           # för "const"
    n_lengths = 2              # för "cycle": antal längder i cykeln
    length_span = (4, 16)      # spann att dra längder ur ("cycle" och "range")

    same_period = True         # "cycle": längdcykeln lika lång som nivåcykeln
    order_random = False       # slumpad nivåordning i stället för fast cykel
    allow_revisit = False      # nivå får återkomma i cykeln (L0 L1 L2 L0 L3)
    jitter_us = 0.0


# ------------------------------------------------------------------ hjälp
def _binwidth():
    return (cfg.pri_max - cfg.pri_min) / cfg.n_bins

def _draw_levels(rng, n):
    gap = DWELL.min_gap_bins * _binwidth()
    for _ in range(200):
        v = np.sort(rng.uniform(cfg.pri_min + gap, cfg.pri_max - gap, n))
        if n == 1 or np.min(np.diff(v)) >= gap:
            return v
    v = np.linspace(cfg.pri_min + gap, cfg.pri_max - gap, n)
    return v + rng.uniform(-gap / 4, gap / 4, n)

def _draw_lengths(rng, n):
    """n distinkta längder ur spannet."""
    lo, hi = DWELL.length_span
    pool = np.arange(lo, hi + 1)
    n = min(n, pool.size)
    return np.sort(rng.choice(pool, size=n, replace=False)).tolist()


# ------------------------------------------------------------------ generator
def create_emitter_data(n_emitters, n_signals, drop_rate=0.0, noise_level=None, rng=None):
    rng = rng or np.random.default_rng()
    em_rng, sig_rng, drop_rng = rng.spawn(3)
    out = []

    for _ in range(n_emitters):
        n = int(em_rng.integers(DWELL.n_levels[0], DWELL.n_levels[1] + 1))
        levels = _draw_levels(em_rng, n)

        # ---- nivåcykel
        order_idx = em_rng.permutation(n).tolist()
        if DWELL.allow_revisit and n >= 3:
            pos = int(em_rng.integers(2, len(order_idx) + 1))
            order_idx.insert(pos, order_idx[0])
        cycle = levels[order_idx]

        # ---- längder
        if DWELL.dwell_mode == "const":
            lengths = [DWELL.dwell_const] * len(cycle)
            length_fixed = True
        elif DWELL.dwell_mode == "cycle":
            k = len(cycle) if DWELL.same_period else DWELL.n_lengths
            vals = _draw_lengths(em_rng, min(DWELL.n_lengths, k))
            # fyll ut till periodlängd k genom att upprepa/slumpa ur vals
            lengths = [int(vals[int(em_rng.integers(len(vals)))]) for _ in range(k)]
            length_fixed = True
        elif DWELL.dwell_mode == "range":
            lo, hi = DWELL.length_span
            a = int(em_rng.integers(lo, hi))
            b = int(em_rng.integers(a + 1, hi + 1))
            lengths = [a, b]                       # min, max
            length_fixed = False
        else:
            raise ValueError(DWELL.dwell_mode)

        label = dict(levels=cycle.tolist(), lengths=lengths,
                     order_fixed=not DWELL.order_random, length_fixed=length_fixed)

        seqs = [_make_signal(cycle, lengths, length_fixed, sig_rng, drop_rate, drop_rng)
                for _ in range(n_signals)]
        out.append((seqs, label))

    return out


def _make_signal(cycle, lengths, length_fixed, rng, drop_rate, drop_rng=None):
    """Rulla ut cyklerna med slumpad startfas. Vektoriserat per besök."""
    drop_rng = rng if drop_rng is None else drop_rng
    n_pulses = DWELL.n_pulses
    lo, hi = (lengths[0], lengths[1]) if not length_fixed else (None, None)

    # hur många besök behövs? ta i överkant
    mean_len = np.mean(lengths) if length_fixed else (lo + hi) / 2
    n_visits = int(np.ceil(n_pulses / max(1, mean_len))) + len(cycle) + 4

    # nivåföljd
    phase = int(rng.integers(0, len(cycle)))
    if DWELL.order_random:
        idx = rng.integers(0, len(cycle), n_visits)
        # undvik samma nivå två gånger i rad
        for i in range(1, n_visits):
            while idx[i] == idx[i - 1] and len(cycle) > 1:
                idx[i] = rng.integers(0, len(cycle))
        lv = cycle[idx]
    else:
        lv = np.tile(cycle, int(np.ceil(n_visits / len(cycle))) + 1)[phase:phase + n_visits]

    # längdföljd -- samma löpande index som nivåcykeln vid fast ordning, se all_emitters
    if length_fixed:
        lphase = int(rng.integers(0, len(lengths))) if DWELL.order_random else phase
        ln = np.tile(np.array(lengths), int(np.ceil((n_visits + lphase) / len(lengths))) + 1
                     )[lphase:lphase + n_visits]
    else:
        ln = rng.integers(lo, hi + 1, n_visits)

    pri = np.repeat(lv, ln)[:n_pulses]

    if DWELL.jitter_us > 0:
        pri = pri + rng.uniform(-DWELL.jitter_us, DWELL.jitter_us, pri.size)

    if drop_rate > 0:
        toa = np.concatenate([[0.0], np.cumsum(pri)])
        keep = drop_rng.random(toa.size) > drop_rate
        keep[0] = True
        pri = np.diff(toa[keep])

    return pri


# ------------------------------------------------------------------ snabbtest
if __name__ == "__main__":
    import time
    from labels import to_tokens, parse
    from verify import verify_stats

    rng = np.random.default_rng(0)

    t = time.perf_counter()
    data = create_emitter_data(200, 4, 0.0, None, rng)
    dt = time.perf_counter() - t
    n_sig = sum(len(s) for s, _ in data)
    print(f"{n_sig} signaler på {dt*1000:.0f} ms  ->  {dt/n_sig*1000:.3f} ms per signal\n")

    for seqs, lab in data[:3]:
        tok = to_tokens(lab["levels"], lab["lengths"], lab["order_fixed"], lab["length_fixed"])
        print(" ".join(tok))
    print()

    verify_stats(data)
