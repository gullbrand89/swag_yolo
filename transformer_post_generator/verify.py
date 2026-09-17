"""
Verifiering av facit mot verklig signal.

    from verify import verify_label

    seqs, lab = create_emitter_data(1, 1, 0.0, None, rng)[0]
    verify_label(seqs[0], lab)                    # dict från generatorn
    verify_label(seqs[0], tokens)                 # eller token-listan

Kontrollerar, fält för fält:
  nivåer   – antal och värden (i bins)
  ordning  – besöksföljden mot facitets cykel, med bästa rotation
  längder  – dwell-längder mot facitets cykel (FIXED) eller mängd (RANDOM)

Tar hänsyn till två saker som annars ger falsklarm:
  * flyttalsbrus  – PRI avrundas till 1/100 binbredd före binning
  * observationsfönstret – första och sista dwellen är avklippta och räknas inte
"""
import numpy as np

from .config import cfg
from .vocab import bin_of, pri_of_bin
from .labels import parse


# ------------------------------------------------------------------ hjälp
def _binwidth():
    return (cfg.pri_max - cfg.pri_min) / cfg.n_bins

def _quantize(pri):
    """Ta bort flyttalsbrus utan att röra verkliga skillnader."""
    w = _binwidth()
    return np.round(np.asarray(pri, dtype=float) / w * 100) * w / 100

def _as_label(label):
    """dict från generatorn eller token-lista -> enhetlig dict."""
    if isinstance(label, dict):
        return dict(levels=list(label["levels"]), lengths=list(label["lengths"]),
                    order_fixed=bool(label["order_fixed"]),
                    length_fixed=bool(label["length_fixed"]))
    d = parse(list(label))
    return dict(levels=d["order"] if d["order_fixed"] else d["levels"],
                lengths=d["lengths"], order_fixed=d["order_fixed"],
                length_fixed=d["length_fixed"])

def _visits(pri_bins, lvl_bins, tol):
    """-> (nivåindex per besök, längd per besök, antal pulser utan nivåmatchning)"""
    idx, unmatched = [], 0
    for b in pri_bins:
        j = int(np.argmin([abs(b - lb) for lb in lvl_bins]))
        if abs(b - lvl_bins[j]) <= tol:
            idx.append(j)
        else:
            idx.append(-1); unmatched += 1
    seq, runs = [], []
    for v in idx:
        if seq and v == seq[-1]:
            runs[-1] += 1
        else:
            seq.append(v); runs.append(1)
    return seq, runs, unmatched

def _min_period(x):
    n = len(x)
    for p in range(1, n + 1):
        if n % p == 0 and all(x[i] == x[i % p] for i in range(n)):
            return x[:p]
    return list(x)

def _best_rotation(obs, cycle):
    """-> (rotation, andel positioner som stämmer)"""
    if not cycle or not obs:
        return 0, 0.0
    best, best_hits = 0, -1
    for r in range(len(cycle)):
        rot = cycle[r:] + cycle[:r]
        hits = sum(1 for i, v in enumerate(obs) if v == rot[i % len(rot)])
        if hits > best_hits:
            best, best_hits = r, hits
    return best, best_hits / len(obs)


# ------------------------------------------------------------------ verifiering
def verify_label(pri, label, tol_bins=1, min_pulses=1, trim_edges=True, verbose=True):
    """
    Returnerar en dict med en post per kontroll: {"ok": bool, ...detaljer}.

    min_pulses : bins med färre pulser än så räknas som artefakter, inte nivåer.
        Var 3 tidigare, vilket var säkert när varje dwell var minst 4 pulser lång.
        Med lim_ds_k = [1, 32] är en dwell på 1-2 pulser fullt laglig, och ett besök
        som kapas av fönsterkanten lämnar lika få. Tröskeln kastade då bort äkta
        nivåer och rapporterade facitet som fel. Höj den bara om du har brus eller
        en detektor som hittar på bins.
    trim_edges : hoppa över första och sista dwellen vid längdkontrollen.
    """
    lab = _as_label(label)
    pri = _quantize(pri)
    obs_bins = np.array([bin_of(p) for p in pri])

    lvl_cycle = lab["levels"]                       # spelordning (kan ha upprepningar)
    lvl_uniq = sorted(set(bin_of(v) for v in lvl_cycle))
    rep = {}

    # ---- 1. nivåer
    uniq, counts = np.unique(obs_bins, return_counts=True)
    strong = sorted(uniq[counts >= min_pulses].tolist())
    weak = sorted(uniq[counts < min_pulses].tolist())
    matched, extra = [], []
    for b in strong:
        if any(abs(b - lb) <= tol_bins for lb in lvl_uniq):
            matched.append(b)
        else:
            extra.append(b)
    missing = [lb for lb in lvl_uniq if not any(abs(b - lb) <= tol_bins for b in strong)]
    # Två helt olika saker, som inte får blandas ihop:
    #   extra   signalen visar en nivå som facitet inte har -> facitet är FEL
    #   missing facitet har en nivå som fönstret aldrig visar -> facitet är rätt,
    #           men den här observationen räcker inte för att bekräfta det
    # Bara det första gör facitet ogiltigt. Det andra är observerbarhetstaket, och
    # att räkna det som fel gör att testet mäter fönstrets längd i stället för
    # facitets riktighet.
    rep["levels"] = dict(
        ok=not extra,
        ok_complete=not extra and not missing,
        n_label=len(lvl_uniq), n_signal=len(strong),
        label_bins=lvl_uniq, signal_bins=strong,
        extra_in_signal=extra, missing_in_signal=missing,
        unobserved=missing,
        weak_bins=weak)

    # ---- besök
    seq, runs, unmatched = _visits(obs_bins, lvl_uniq, tol_bins)
    rep["unmatched_pulses"] = dict(ok=unmatched == 0, n=unmatched,
                                   frac=unmatched / max(1, len(pri)))

    # ---- besök trimmade för längdkontroll
    keep = list(zip(seq, runs))
    if trim_edges and len(keep) > 2:
        keep = keep[1:-1]                            # avklippta dwells i kanterna
    obs_pairs = [(v, r) for v, r in keep if v >= 0]
    obs_seq_t = [v for v, _ in obs_pairs]
    obs_runs = [r for _, r in obs_pairs]

    lab_lengths = [x for x in lab["lengths"] if x is not None]
    is_inf = any(x is None for x in lab["lengths"])
    idx = {b: i for i, b in enumerate(lvl_uniq)}
    lab_cycle = _min_period([idx[bin_of(v)] for v in lvl_cycle])
    lab_lens = _min_period(lab_lengths) if lab_lengths else []

    # ---- 2+3. parvis kontroll när ordning OCH längd är fasta och perioderna lika
    paired = (lab["order_fixed"] and lab["length_fixed"] and not is_inf
              and len(lab_cycle) == len(lab_lens) and obs_pairs)
    if paired:
        cyc = list(zip(lab_cycle, lab_lens))
        r, frac = _best_rotation(obs_pairs, cyc)
        rot = cyc[r:] + cyc[:r]
        ok = frac > 0.98
        rep["order"] = dict(ok=ok, type="FIXED", in_sync=frac, paired=True,
                            label_cycle=[c[0] for c in cyc], best_rotation=r,
                            aligned=[c[0] for c in rot], n_visits=len(obs_seq_t),
                            observed_head=obs_seq_t[:12])
        rep["lengths"] = dict(ok=ok, type="FIXED", in_sync=frac, paired=True,
                              label_cycle=[c[1] for c in cyc], best_rotation=r,
                              aligned=[c[1] for c in rot],
                              observed_head=obs_runs[:12])
    else:
        # ---- ordning separat
        if lab["order_fixed"]:
            obs_seq = [v for v in seq if v >= 0]
            r, frac = _best_rotation(obs_seq, lab_cycle)
            rep["order"] = dict(ok=frac > 0.98, type="FIXED", in_sync=frac, paired=False,
                                label_cycle=lab_cycle, best_rotation=r,
                                aligned=lab_cycle[r:] + lab_cycle[:r],
                                n_visits=len(obs_seq), observed_head=obs_seq[:12])
        else:
            obs_seq = [v for v in seq if v >= 0]
            repeats = sum(1 for a, b in zip(obs_seq, obs_seq[1:]) if a == b)
            rep["order"] = dict(ok=True, type="RANDOM", n_visits=len(obs_seq),
                                immediate_repeats=repeats, observed_head=obs_seq[:12])

        # ---- längder separat
        if is_inf:
            rep["lengths"] = dict(ok=len(set(seq)) <= 1, type="INF", n_visits=len(seq),
                                  note="static: förväntar ett enda besök")
        elif lab["length_fixed"]:
            r, frac = _best_rotation(obs_runs, lab_lens)
            rep["lengths"] = dict(ok=frac > 0.98, type="FIXED", in_sync=frac, paired=False,
                                  label_cycle=lab_lens, best_rotation=r,
                                  aligned=lab_lens[r:] + lab_lens[:r],
                                  observed_head=obs_runs[:12])
        else:
            lo, hi = min(lab_lengths), max(lab_lengths)
            bad = [r for r in obs_runs if not (lo <= r <= hi)]
            rep["lengths"] = dict(ok=not bad, type="RANGE", allowed=(lo, hi),
                                  outside_set=sorted(set(bad)),
                                  frac_ok=1 - len(bad) / max(1, len(obs_runs)),
                                  observed_head=obs_runs[:12])

    rep["ok"] = all(v["ok"] for k, v in rep.items() if isinstance(v, dict) and "ok" in v)

    if verbose:
        _print_report(rep, lab)
    return rep


def _print_report(rep, lab):
    def flag(b): return "OK  " if b else "FEL "

    L = rep["levels"]
    print(f"{flag(L['ok'])} nivåer   facit {L['n_label']} st {L['label_bins']}")
    print(f"          signal {L['n_signal']} st {L['signal_bins']}")
    if L["extra_in_signal"]:
        print(f"          i signalen men inte i facit: {L['extra_in_signal']}")
    if L["missing_in_signal"]:
        print(f"          i facit men inte i signalen: {L['missing_in_signal']}")
    if L["weak_bins"]:
        print(f"          svaga bins (< min_pulses, ignorerade): {L['weak_bins']}")

    U = rep["unmatched_pulses"]
    if U["n"]:
        print(f"{flag(U['ok'])} pulser   {U['n']} st ({U['frac']:.1%}) matchar ingen nivå")

    O = rep["order"]
    if O["type"] == "FIXED":
        how = "parvis med längderna" if O.get("paired") else "separat"
        print(f"{flag(O['ok'])} ordning  FAST ({how}), i takt {O['in_sync']:.1%} över "
              f"{O['n_visits']} besök")
        print(f"          facitcykel {O['label_cycle']} roterad {O['best_rotation']} "
              f"-> {O['aligned']}")
        print(f"          observerat {O['observed_head']}")
    else:
        print(f"{flag(True)} ordning  SLUMPAD, {O['n_visits']} besök, "
              f"{O['immediate_repeats']} direkta upprepningar")
        print(f"          observerat {O['observed_head']}")

    D = rep["lengths"]
    if D["type"] == "FIXED":
        print(f"{flag(D['ok'])} längder  FAST cykel, i takt {D['in_sync']:.1%}")
        print(f"          facitcykel {D['label_cycle']} -> {D['aligned']}")
        print(f"          observerat {D['observed_head']}")
    elif D["type"] == "RANDOM":
        print(f"{flag(D['ok'])} längder  SLUMPAD ur {D['allowed']}, "
              f"{D['frac_ok']:.1%} inom mängden")
        if D["outside_set"]:
            print(f"          utanför mängden: {D['outside_set']}")
        print(f"          observerat {D['observed_head']}")
    else:
        print(f"{flag(D['ok'])} längder  INF ({D['note']})")

    print(f"\n{'ALLT STÄMMER' if rep['ok'] else 'AVVIKELSER HITTADE'}")


# ------------------------------------------------------------------ batch
def verify_many(data, label_fn=None, tol_bins=1, show_failures=1, **kw):
    """
    data : lista av (pri_sequences, label) från create_emitter_data
    label_fn : valfri, label -> tokens (om du vill verifiera tokenversionen)
    Returnerar (antal ok, lista med (emitter_idx, signal_idx, rapport, flaggor)).
    """
    n_ok, failed = 0, []
    for i, (seqs, lab) in enumerate(data):
        target = label_fn(lab) if label_fn else lab
        flags = _flags(lab)
        for j, s in enumerate(seqs):
            rep = verify_label(s, target, tol_bins=tol_bins, verbose=False, **kw)
            if rep["ok"]:
                n_ok += 1
            else:
                failed.append((i, j, rep, flags))
    print(f"{n_ok}/{n_ok + len(failed)} signaler stämmer med sitt facit")
    for k in range(min(show_failures, len(failed))):
        i, j, rep, fl = failed[k]
        print(f"\navvikelse {k+1}, emitter {i} signal {j}  {fl}:")
        _print_report(rep, None)
    return n_ok, failed


# ------------------------------------------------------------------ statistik
def _flags(label):
    """Kort beskrivning av emittertypen, för gruppering."""
    lab = _as_label(label)
    lengths = [x for x in lab["lengths"] if x is not None]
    if not lengths:
        return "static/INF"
    n_lvl = len(set(bin_of(v) for v in lab["levels"]))
    o = "order_fixed" if lab["order_fixed"] else "order_random"
    l = "len_fixed" if lab["length_fixed"] else "len_random"
    extra = []
    if n_lvl == 1:
        extra.append("1 nivå")
    if len(set(lengths)) == 1:
        extra.append(f"alla dwell={lengths[0]}")
    if lab["order_fixed"] and len(lab["levels"]) != len(set(bin_of(v) for v in lab["levels"])):
        extra.append("återbesök")
    if lab["order_fixed"] and lab["length_fixed"] and \
       len(_min_period(lab["levels"])) != len(_min_period(lengths)):
        extra.append("olika period")
    s = f"{o}/{l}"
    return s + (" [" + ", ".join(extra) + "]" if extra else "")


def verify_stats(data, label_fn=None, tol_bins=1, group_by="flags", **kw):
    """
    Kör verify_label på allt och sammanställer statistik.

    group_by : "flags"   – gruppera på order_fixed/length_fixed + särdrag
               "n_levels" – gruppera på antal distinkta nivåer
               None       – bara totalen

    Skriver en tabell med andel OK per fält (nivåer, ordning, längder) per grupp,
    plus de vanligaste felmönstren.
    """
    from collections import defaultdict, Counter

    rows = []
    for i, (seqs, lab) in enumerate(data):
        target = label_fn(lab) if label_fn else lab
        parsed = _as_label(lab)
        if group_by == "n_levels":
            key = f"{len(set(bin_of(v) for v in parsed['levels']))} nivåer"
        elif group_by == "flags":
            key = _flags(lab)
        else:
            key = "alla"
        for j, s in enumerate(seqs):
            rep = verify_label(s, target, tol_bins=tol_bins, verbose=False, **kw)
            rows.append((key, rep, i, j))

    groups = defaultdict(list)
    for key, rep, i, j in rows:
        groups[key].append(rep)

    fields = ["levels", "order", "lengths", "unmatched_pulses"]
    head = f"{'grupp':<42} {'n':>6}  {'allt':>6}  " + "  ".join(f"{f[:9]:>9}" for f in fields)
    print(head); print("-" * len(head))
    for key in sorted(groups, key=lambda k: -len(groups[k])):
        reps = groups[key]
        n = len(reps)
        allt = sum(r["ok"] for r in reps) / n
        cells = []
        for f in fields:
            cells.append(sum(r[f]["ok"] for r in reps) / n)
        print(f"{key:<42} {n:>6}  {allt:>6.1%}  " + "  ".join(f"{c:>9.1%}" for c in cells))
    n = len(rows)
    allt = sum(r["ok"] for _, r, _, _ in rows) / n
    print("-" * len(head))
    print(f"{'TOTALT':<42} {n:>6}  {allt:>6.1%}  " +
          "  ".join(f"{sum(r[f]['ok'] for _, r, _, _ in rows)/n:>9.1%}" for f in fields))

    # ---- vanligaste felmönstren
    pat = Counter()
    for _, rep, _, _ in rows:
        if rep["ok"]:
            continue
        bits = []
        L = rep["levels"]
        if L["extra_in_signal"]: bits.append(f"{len(L['extra_in_signal'])} extra nivå(er) i signalen")
        if L["missing_in_signal"]: bits.append(f"{len(L['missing_in_signal'])} nivå(er) saknas i signalen")
        if not rep["unmatched_pulses"]["ok"]: bits.append("pulser utan nivåmatchning")
        if not rep["order"]["ok"]: bits.append(f"ordning ur takt ({rep['order'].get('in_sync', 0):.0%})")
        if not rep["lengths"]["ok"]:
            D = rep["lengths"]
            bits.append("längder ur takt" if D["type"] == "FIXED" else "längd utanför mängden")
        pat[" + ".join(bits) or "okänt"] += 1
    if pat:
        print("\nvanligaste felmönster:")
        for p, c in pat.most_common(10):
            print(f"  {c:>6}  {p}")

    # ---- kvantitativa detaljer på avvikelserna
    order_sync = [r["order"]["in_sync"] for _, r, _, _ in rows
                  if r["order"].get("type") == "FIXED" and not r["order"]["ok"]]
    len_sync = [r["lengths"]["in_sync"] for _, r, _, _ in rows
                if r["lengths"].get("type") == "FIXED" and not r["lengths"]["ok"]]
    if order_sync:
        print(f"\nordning ur takt: median {np.median(order_sync):.0%}, "
              f"min {min(order_sync):.0%}  (n={len(order_sync)})")
    if len_sync:
        print(f"längder ur takt: median {np.median(len_sync):.0%}, "
              f"min {min(len_sync):.0%}  (n={len(len_sync)})")

    return groups


# ------------------------------------------------------------------ härledning
def infer_order(pri, levels, period, tol_bins=1, trim_edges=True):
    """
    Härled nivåcykeln ur signalen när facitet tappat bort ordningen.

    pri     : PRI-sekvens (µs)
    levels  : nivåerna som finns (ordning spelar ingen roll, dubbletter ok)
    period  : cykelns längd i antal BESÖK
    ->  (cycle_us, confidence, counts)
        cycle_us   : nivåvärden i spelordning, längd = period
        confidence : andel besök som stämmer med den härledda cykeln
        counts     : röstfördelning per cykelposition, för inspektion

    Fungerar för godtyckliga dwell-längder: pulser kollapsas till besök först.
    Tål enstaka bortfall och att sekvensen börjar mitt i en cykel, eftersom
    varje cykelposition avgörs med majoritetsröstning.
    """
    from collections import Counter

    lvl_uniq = sorted(set(bin_of(v) for v in levels))
    obs_bins = np.array([bin_of(p) for p in _quantize(pri)])
    seq, runs, _ = _visits(obs_bins, lvl_uniq, tol_bins)

    keep = list(zip(seq, runs))
    if trim_edges and len(keep) > 2:
        keep = keep[1:-1]
    obs = [v for v, _ in keep if v >= 0]
    if len(obs) < period:
        raise ValueError(f"för få besök ({len(obs)}) för period {period}")

    counts = [Counter() for _ in range(period)]
    for i, v in enumerate(obs):
        counts[i % period][v] += 1
    cycle_idx = [c.most_common(1)[0][0] for c in counts]

    hits = sum(1 for i, v in enumerate(obs) if v == cycle_idx[i % period])
    conf = hits / len(obs)

    # kanonisk rotation, samma regel som to_tokens
    r = min(range(period), key=lambda k: cycle_idx[k:] + cycle_idx[:k])
    cycle_idx = cycle_idx[r:] + cycle_idx[:r]

    cycle_us = [pri_of_bin(lvl_uniq[i]) for i in cycle_idx]
    return cycle_us, conf, [dict(c) for c in counts]


def infer_order_report(pri, levels, period, tol_bins=1):
    """Som infer_order men skriver ut en läsbar sammanfattning."""
    cycle, conf, counts = infer_order(pri, levels, period, tol_bins)
    lvl_uniq = sorted(set(bin_of(v) for v in levels))
    names = {b: f"L{i}" for i, b in enumerate(lvl_uniq)}
    print(f"period {period}, konfidens {conf:.1%}")
    print("cykel:", " → ".join(names[bin_of(v)] for v in cycle))
    print("       " + " ".join(f"{v:.4g}" for v in cycle))
    ambiguous = [i for i, c in enumerate(counts) if len(c) > 1]
    if ambiguous:
        print(f"tvetydiga positioner (fler än en kandidat): {ambiguous}")
        for i in ambiguous[:5]:
            print(f"  pos {i}: " + ", ".join(f"{names[lvl_uniq[k]]}×{n}"
                                             for k, n in sorted(counts[i].items())))
    return cycle, conf
