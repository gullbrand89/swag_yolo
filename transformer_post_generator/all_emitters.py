"""
Emittergenerator som följer den riktiga datans parameterkedja.

Per emitter dras först en PRI-rymd, sedan ett rutnät i den, och sist ett mönster
på rutnätet:

    pri_mean ~ sample_x(pri_lim, pri_mode)          clippad medel-PRI
    J        ~ U(lim_jitt)                          jitterförhållande
    m        = 2*pri_mean / (J + 1/J)               korrigering, se _emitter_grid
    val_lim  = (m/J, m*J)                           rutnätets ändpunkter
    val_Q    ~ randint(lim_val_quantization)        antal lägen i rutnätet

Nivåerna är alltså LOKALA kring emitterns eget medel -- en emitters egna värden
spänner J^2, alltså på sin höjd en faktor 100, inte hela korpusens spann. Den
tidigare generatorn spred nivåerna över hela [pri_min, pri_max], vilket gjorde
uppgiften svårare på ett sätt som inte motsvarar verkligheten.

Mönstret på rutnätet dras med sample_xp: en slumpvandring där stegen aldrig är
noll, så att två lika nivåer aldrig står bredvid varandra, och där cykeln
behandlas som CIRKULÄR. Det sista är inte kosmetik -- mönstret rullas ut med
np.tile, så sista och första elementet blir grannar vid sömmen mellan varv, och
är de lika smälter de ihop till en dubbelt så lång dwell.

Tre oberoende flaggor styr dwell-and-switch:

    val_switch   determined -> ORDER FIXED    rand -> ORDER RANDOM
    Np_switch    determined -> DWELL FIXED    rand -> DWELL RANGE
    Np_per_dwell equal      -> en dwell för alla tillstånd
                 different  -> en dwell per tillstånd, i besöksordning

Np_per_dwell biter bara när Np_switch är determined; med rand dras dwelltiden om
vid varje besök och en per-tillstånd-dwell finns inte.
"""
import numpy as np

from .config import cfg
from .vocab import bin_of


# ------------------------------------------------------------------ inställningar
class GEN:
    n_pulses = 512

    # emitterns PRI-rymd
    pri_lim = (50.0, 250.0)        # medel-PRI dras här; kedjan vidgar till ~[1, 495]
    pri_mode = "lin"               # "lin" | "log" -- spelar liten roll över 5:1
    lim_jitt = (1.05, 10.0)        # J, jitterförhållande

    # mönstret
    lim_val_q = (2, 32)            # antal lägen i rutnätet
    lim_stag = (2, 32)             # stagger: mönsterlängd i pulser
    lim_states = (2, 16)           # dwell & switch: antal tillstånd
    lim_ds_k = (1, 32)             # dwell & switch: pulser per dwell (determined)
    lim_ds_range = (2, 16)         # ... och intervallets ändpunkter (rand)

    p_equal_dwell = 0.5            # Np_per_dwell: equal mot different

    jitter_us = 0.0
    min_gap_bins = 1               # varning om rutnätet är finare än facit-binen

    weights = {
        "static":        1.0,      # en nivå, INF
        "jitter":        1.0,      # slumpvis läge varje puls
        "stagger":       1.0,      # fast mönster, en puls per element
        "ds_det_det":    1.0,      # fast ordning, fast dwell
        "ds_det_rand":   1.0,      # fast ordning, dwell ur intervall
        "ds_rand_det":   1.0,      # slumpad ordning, fast dwell
        "ds_rand_rand":  1.0,      # slumpad ordning, dwell ur intervall
    }


VARIANTS = list(GEN.weights)


# ------------------------------------------------------------------ dragning
def sample_x(rng, lims, mode, Ns=1):
    """Motsvarar den riktiga generatorns sample_x."""
    if mode == "lin":
        x = rng.uniform(lims[0], lims[1], Ns)
    elif mode == "log":
        x = np.exp(rng.uniform(np.log(lims[0]), np.log(lims[1]), Ns))
    elif mode == "int":
        x = rng.integers(lims[0], lims[1] + 1, Ns)
    else:
        raise ValueError(f"okänt läge {mode}")
    return x[0] if Ns == 1 else x


def sample_xp(rng, low, high, Ns, q_levels):
    """
    Ns värden ur rutnätet linspace(low, high, q_levels), som en slumpvandring.

    Stegen dras ur 1..q-1, vilket gör nästa index likformigt över de q-1 andra
    värdena och därmed omöjliggör grannupprepning inne i mönstret. Cykeln är
    dessutom cirkulär: idx[0] != idx[-1] krävs, eftersom np.tile ställer dem
    bredvid varandra vid varvsbytet. Det nya sista värdet måste skilja sig från
    BÅDE idx[-2] (som står före) och idx[0] (som står efter vid varvet).

    -> (värden i µs, rutnätsindex)
    """
    if q_levels < 2:
        raise ValueError("q_levels måste vara minst 2")
    vals = np.linspace(low, high, q_levels)
    if Ns < 2:
        i = int(rng.integers(0, q_levels))
        return vals[[i]], np.array([i])
    if q_levels == 2 and Ns % 2:
        Ns += 1                                  # cirkulär udda cykel finns inte på 2 lägen
    for _ in range(100):
        steg = np.concatenate((rng.integers(0, q_levels, 1),
                               rng.integers(1, q_levels, Ns - 1)))
        idx = np.mod(np.cumsum(steg), q_levels).astype(int)
        if idx[0] != idx[-1]:
            return vals[idx], idx
        giltiga = np.setdiff1d(np.arange(q_levels), [idx[0], idx[-2]])
        if giltiga.size:
            idx[-1] = int(rng.choice(giltiga))
            return vals[idx], idx
    raise RuntimeError("hittade ingen giltig cykel")


def _emitter_grid(rng):
    """
    -> (low, high, q_levels) för en emitter.

    Korrigeringen m = 2*pri_mean/(J + 1/J) finns för att [m/J, m*J] annars har
    mittpunkten m*(J + 1/J)/2. Utan den skulle en emitter med J = 10 få en
    medel-PRI fem gånger den avsedda. Med den blir (low + high)/2 == pri_mean.

    q_levels sänks vid behov så att två rutnätslägen aldrig hamnar i samma
    facit-bin. Ett smalt spann med många lägen ger annars ett rutnätssteg under
    binbredden, nivåerna smälter ihop vid binningen och facitet blir oobserverbart
    -- to_tokens fångar det med en assert, men först efter att emittern skapats,
    alltså mitt i en DataLoader-arbetare. Det inträffar sällan (runt 0.01 % av
    emittrarna), men vid hundratusentals dragningar per körning betyder sällan
    flera gånger. Att sänka q i stället för att kasta emittern behåller dess
    PRI-rymd och ändrar bara hur fint den är kvantiserad.
    """
    for _ in range(100):
        pri_mean = float(sample_x(rng, GEN.pri_lim, GEN.pri_mode))
        J = float(sample_x(rng, GEN.lim_jitt, "lin"))
        m = 2 * pri_mean / (J + 1 / J)
        low, high = m / J, m * J
        q = int(sample_x(rng, GEN.lim_val_q, "int"))

        # geometriskt tak: steget måste vara minst min_gap_bins binbredder
        bw = (cfg.pri_max - cfg.pri_min) / (cfg.n_bins - 1)
        q = max(2, min(q, int(1 + (high - low) / (GEN.min_gap_bins * bw))))

        # och kontrollera mot den faktiska binningen -- bin_of trunkerar, så
        # geometrin räcker inte som garanti
        while q > 2 and len({bin_of(v) for v in np.linspace(low, high, q)}) < q:
            q -= 1
        if len({bin_of(v) for v in np.linspace(low, high, q)}) == q:
            return low, high, q
        # hela spannet ryms i en bin: emittern går inte att observera, dra om
    raise RuntimeError("hittade ingen emitter vars nivåer går att särskilja")


# ------------------------------------------------------------------ varianter
def _make_emitter(variant, rng):
    """-> (cycle_us, lengths, order_fixed, length_fixed)"""
    low, high, q = _emitter_grid(rng)

    if variant == "static":
        lv, _ = sample_xp(rng, low, high, 1, q)
        return lv, [None], True, True

    if variant == "jitter":
        # slumpvis läge varje puls: hela rutnätet är nivåer, ordningen slumpad
        lv = np.linspace(low, high, q)
        return lv, [1], False, True

    if variant == "stagger":
        ns = int(sample_x(rng, GEN.lim_stag, "int"))
        lv, _ = sample_xp(rng, low, high, ns, q)
        return lv, [1], True, True

    if not variant.startswith("ds_"):
        raise ValueError(f"okänd variant {variant}")

    _, val_switch, np_switch = variant.split("_")
    ns = int(sample_x(rng, GEN.lim_states, "int"))
    lv, _ = sample_xp(rng, low, high, ns, q)
    order_fixed = val_switch == "det"

    if np_switch == "rand":
        # dwelltiden dras om vid VARJE besök ur [a, b] -- ingen cykel, ett intervall
        a, b = sorted(int(x) for x in sample_x(rng, GEN.lim_ds_range, "int", 2))
        return lv, [a, b], order_fixed, False

    if rng.random() < GEN.p_equal_dwell:
        k = int(sample_x(rng, GEN.lim_ds_k, "int"))
        return lv, [k], order_fixed, True

    # en dwell per tillstånd, i besöksordning -- kopplingen nivå <-> längd är äkta
    # och den gemensamma rotationen i to_tokens är rätt kanonisering
    k = sample_x(rng, GEN.lim_ds_k, "int", ns)
    return lv, [int(x) for x in k], order_fixed, True


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
            # Vandringen går över de DISTINKTA nivåerna, inte över cykelns
            # positioner. Cykeln kan innehålla samma värde på flera platser, och
            # en spärr mot samma INDEX två gånger i rad hindrar då inte att två
            # olika platser med samma värde hamnar efter varandra. De två besöken
            # smälter ihop i signalen och dwellen blir summan -- exakt det
            # check_dwell rapporterade som "utanför" och som dubblade längder.
            uniq = np.unique(cycle)
            nu = len(uniq)
            sel = np.empty(n_visits, dtype=int)
            sel[0] = int(rng.integers(0, nu))
            for i in range(1, nu > 1 and n_visits or 1):
                sel[i] = (sel[i - 1] + int(rng.integers(1, nu))) % nu
            if nu == 1:
                sel[:] = 0
            lv = uniq[sel]

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
    if drop_rate is None:
        drop_rate = 0
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
    from .data import make_channels, label_to_tokens
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
