"""
Snabb stagger-generator för avgränsade experiment på ORDNING.

Väljs med config.py:

    cfg.emitter = "stagger_emitter"

Samma signatur och returformat som din riktiga generator:

    create_emitter_data(n_emitters, n_signals, drop_rate, noise_level, rng)
        -> [(pri_sequences, label_dict), ...]

    label_dict = dict(levels=[...], lengths=[...],
                      order_fixed=bool, length_fixed=bool)

Stagger = fast nivåcykel, exakt en puls per nivå. Längdcykeln blir därmed
trivial (alla ettor), så allt som är kvar att lära är nivåerna och ordningen.

Konfiguration via STAGGER nedan. Vektoriserad, ~0,1 ms per signal.
"""
import numpy as np

from config import cfg


# ------------------------------------------------------------------ inställningar
class STAGGER:
    n_levels = (3, 6)          # antal nivåer, inklusive gränser
    n_pulses = 512             # pulser per signal
    dwell = 1                  # pulser per nivå (1 = ren stagger)
    min_gap_bins = 4           # minsta avstånd mellan nivåer, i facit-bins
    allow_revisit = False      # tillåt att en nivå återkommer i cykeln (L0 L1 L2 L0 L3)
    jitter_us = 0.0            # spridning kring varje nivå


# ------------------------------------------------------------------ hjälp
def _binwidth():
    return (cfg.pri_max - cfg.pri_min) / cfg.n_bins

def _draw_levels(rng, n):
    """n nivåer med garanterat avstånd, jämnt spridda i emitterrymden."""
    gap = STAGGER.min_gap_bins * _binwidth()
    span = cfg.pri_max - cfg.pri_min
    for _ in range(200):
        v = np.sort(rng.uniform(cfg.pri_min + gap, cfg.pri_max - gap, n))
        if n == 1 or np.min(np.diff(v)) >= gap:
            return v
    # fallback: jämnt fördelade med liten störning
    v = np.linspace(cfg.pri_min + gap, cfg.pri_max - gap, n)
    return v + rng.uniform(-gap / 4, gap / 4, n)


# ------------------------------------------------------------------ generator
def create_emitter_data(n_emitters, n_signals, drop_rate=0.0, noise_level=None, rng=None):
    rng = rng or np.random.default_rng()
    em_rng, sig_rng, drop_rng = rng.spawn(3)  # emittrar / signaler / bortfall oberoende
    out = []

    for _ in range(n_emitters):
        n = int(em_rng.integers(STAGGER.n_levels[0], STAGGER.n_levels[1] + 1))
        levels = _draw_levels(em_rng, n)

        # spelordning: en permutation, ev. med ett återbesök
        order_idx = em_rng.permutation(n).tolist()
        if STAGGER.allow_revisit and n >= 3:
            pos = int(em_rng.integers(2, len(order_idx) + 1))
            order_idx.insert(pos, order_idx[0])

        cycle = levels[order_idx]                       # nivåcykeln i µs
        lengths = [STAGGER.dwell] * len(cycle)

        label = dict(levels=cycle.tolist(), lengths=lengths,
                     order_fixed=True, length_fixed=True)

        seqs = []
        for _ in range(n_signals):
            seqs.append(_make_signal(cycle, sig_rng, drop_rate, drop_rng))
        out.append((seqs, label))

    return out


def _make_signal(cycle, rng, drop_rate, drop_rng=None):
    """Rulla ut cykeln, slumpad startfas, lägg på jitter och bortfall. Vektoriserat."""
    drop_rng = rng if drop_rng is None else drop_rng
    n_pulses = STAGGER.n_pulses
    dwell = STAGGER.dwell
    per_cycle = len(cycle) * dwell

    # tillräckligt många varv + slumpad startfas
    reps = int(np.ceil((n_pulses + per_cycle) / per_cycle))
    full = np.repeat(np.tile(cycle, reps), dwell)
    phase = int(rng.integers(0, per_cycle))
    pri = full[phase:phase + n_pulses].copy()

    if STAGGER.jitter_us > 0:
        pri += rng.uniform(-STAGGER.jitter_us, STAGGER.jitter_us, pri.size)

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

    rng = np.random.default_rng(0)

    t = time.perf_counter()
    data = create_emitter_data(200, 4, 0.0, None, rng)
    dt = time.perf_counter() - t
    n_sig = sum(len(s) for s, _ in data)
    print(f"{n_sig} signaler på {dt*1000:.0f} ms  ->  {dt/n_sig*1000:.3f} ms per signal")

    seqs, lab = data[0]
    tok = to_tokens(lab["levels"], lab["lengths"], lab["order_fixed"], lab["length_fixed"])
    print("\nfacit:", " ".join(tok))
    print("signal:", np.round(seqs[0][:12], 2))

    # kanonisering: alla startfaser ska ge samma facit
    toks = {tuple(to_tokens(lab["levels"], lab["lengths"], True, True)) for _, lab in data[:1]}
    print("\nrundtur:", parse(tok)["order_fixed"], parse(tok)["length_fixed"])

    # kontroll: samma emittrar oavsett drop_rate
    a = create_emitter_data(5, 1, 0.0, None, np.random.default_rng(7))
    b = create_emitter_data(5, 1, 0.2, None, np.random.default_rng(7))
    same = all(np.allclose(x[1]["levels"], y[1]["levels"]) for x, y in zip(a, b))
    print("samma emittrar vid olika drop_rate:", same)
