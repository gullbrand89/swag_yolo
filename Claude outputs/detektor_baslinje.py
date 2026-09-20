"""
Baslinje utan transformer: din detektor av korrupta PRI-värden + konverteraren.

Frågan: räcker "ta bort de korrupta pulserna, ta histogrammet av resten" för att
få samma emittermodell som transformern ger? Mäts på SAMMA 1000 evalsignaler och
med SAMMA mått ("emittermodell identisk med facitpostens").

Tre sätt att köra, beroende på hur din detektor anropas:

  1. Två steg (detektorn körs hur du vill däremellan):
       python -m tools.detektor_baslinje dump                # -> runs/baslinje/signaler.npz
       ... kör din modell på varje signal, spara flaggorna ...
       python -m tools.detektor_baslinje eval --flaggor runs/baslinje/flaggor.npz

     signaler.npz: nycklarna s0, s1, ... (1-D float, PRI i µs) och ett fält "meta"
     (JSON: variant, drop per signal). flaggor.npz: samma nycklar, 1-D int 0/1 av
     samma längd, 1 = korrupt.

  2. Ett steg, med en funktion pri -> flaggor:
       python -m tools.detektor_baslinje eval --detektor mitt_paket.modul:funktion
     Funktionen tar en 1-D numpy-array (PRI i µs) och ger 0/1 av samma längd.

  3. Taket: generatorns sanning som detektor (varje ihopslaget intervall flaggat):
       python -m tools.detektor_baslinje eval --orakel

Med --run runs/<mapp> räknas transformerns kolumn ur samma körnings preds-filer,
så att tabellen blir en enda. Kräver inte torch.

Nivåmängden ur flaggorna: bins för oflaggade pulser, sammanhängande sträckor
(±run_tol_bins) blir en nivå, nivåvärdet är medel-PRI i sträckan. --min-pulser N
kräver minst N pulser per nivå (1 = ren detektor).
"""
import argparse
import importlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ data
def _evalsignaler():
    """Samma signaler som transformerns eval: cfg.eval_seed, eval_n_emitters, eval_p_drops."""
    from transformer_post_generator.config import cfg
    from transformer_post_generator.all_emitters import create_emitter_data
    from transformer_post_generator.data import label_to_tokens, as_signals
    ut = []
    for p in cfg.eval_p_drops:
        rng = np.random.default_rng(cfg.eval_seed)
        data = create_emitter_data(cfg.eval_n_emitters, cfg.eval_samples_per_emitter, p,
                                   cfg.noise_level, rng, with_aux=True)
        for seqs, lab in data:
            for i, (s, a) in enumerate(zip(as_signals(seqs), lab["aux"])):
                ut.append(dict(pri=np.asarray(s, dtype=float), drop=float(p),
                               variant=lab["variant"], tokens=label_to_tokens(lab, i),
                               sant_flagga=(np.asarray(a["merge"]) > 0).astype(np.int64)))
    return ut


def _spara(path, sig):
    arrs = {f"s{i}": r["pri"] for i, r in enumerate(sig)}
    meta = json.dumps([dict(variant=r["variant"], drop=r["drop"]) for r in sig])
    np.savez(path, meta=np.array(meta), **arrs)


def _lasa_flaggor(path, n):
    z = np.load(path, allow_pickle=False)
    ut = []
    for i in range(n):
        f = np.asarray(z[f"s{i}"]).astype(np.int64).ravel()
        ut.append(f)
    return ut


# ------------------------------------------------------------------ nivåmängd
def nivaer_ur_flaggor(pri, flagga, min_pulser=1):
    """-> lista med nivåvärden (µs), ur oflaggade pulser, klustrade i bin-rymden."""
    from transformer_post_generator.config import cfg
    from transformer_post_generator.vocab import bin_of
    pri = np.asarray(pri, dtype=float)
    ok = (np.asarray(flagga) == 0) & (pri < cfg.pri_max)
    if not ok.any():
        return []
    b = np.array([bin_of(v) for v in pri[ok]])
    v = pri[ok]
    ordn = np.argsort(b)
    b, v = b[ordn], v[ordn]
    nivaer, start = [], 0
    for i in range(1, len(b) + 1):
        if i == len(b) or b[i] - b[i - 1] > cfg.run_tol_bins:
            if i - start >= min_pulser:
                nivaer.append(float(v[start:i].mean()))
            start = i
    return nivaer


def _mu(r):
    return [None if m is None else round(m, 6) for m in r["Phi"]["mu"]]


# ------------------------------------------------------------------ mått
def _jamfor(sig, flaggor, min_pulser, namn):
    from tools.biblioteksformat import till_bibliotek
    st = {}
    for r, f in zip(sig, flaggor):
        d = r["drop"]
        s = st.setdefault(d, Counter())
        s["n"] += 1
        niv = nivaer_ur_flaggor(r["pri"], f, min_pulser)
        if not niv:
            s["tom"] += 1
            continue
        post = dict(levels=niv, lengths=[1], order_fixed=False, length_fixed=True)
        try:
            rb = till_bibliotek(r["pri"], post)
            rt = till_bibliotek(r["pri"], r["tokens"])
        except Exception:
            s["krasch"] += 1
            continue
        s["identisk"] += _mu(rb) == _mu(rt)
        s["antal_ratt"] += len(_mu(rb)) == len(_mu(rt))
        s["fler"] += len(_mu(rb)) > len(_mu(rt))
        s["farre"] += len(_mu(rb)) < len(_mu(rt))
        # flaggkvalitet mot sanningen
        sf = r["sant_flagga"]
        if len(sf) == len(f):
            pos = sf == 1
            if pos.any():
                s["_rec_n"] += int(pos.sum()); s["_rec"] += int((f[pos] == 1).sum())
            neg = ~pos
            s["_fp_n"] += int(neg.sum()); s["_fp"] += int((f[neg] == 1).sum())
    return {d: dict(namn=namn, n=s["n"],
                    identisk=s["identisk"] / s["n"], antal_ratt=s["antal_ratt"] / s["n"],
                    fler=s["fler"], farre=s["farre"], tom=s["tom"], krasch=s["krasch"],
                    recall=(s["_rec"] / s["_rec_n"]) if s["_rec_n"] else None,
                    falsklarm=(s["_fp"] / s["_fp_n"]) if s["_fp_n"] else None)
            for d, s in st.items()}


def _transformer(sig, run):
    """Transformerns kolumn ur preds_drop_<p>.txt i körningsmappen, samma mått."""
    from tools.biblioteksformat import till_bibliotek
    ut = {}
    for d in sorted({r["drop"] for r in sig}):
        fn = Path(run) / f"preds_drop_{d:.2f}.txt"
        if not fn.exists():
            continue
        blocks = open(fn).read().strip().split("\n\n")
        rows = [r for r in sig if r["drop"] == d]
        if len(blocks) != len(rows):
            print(f"  ! {fn.name}: {len(blocks)} poster, väntade {len(rows)} -- hoppar över")
            continue
        s = Counter()
        for r, blk in zip(rows, blocks):
            T, P = [l.split()[1:] for l in blk.split("\n")]
            s["n"] += 1
            try:
                rp = till_bibliotek(r["pri"], P); rt = till_bibliotek(r["pri"], T)
            except Exception:
                s["krasch"] += 1; continue
            s["identisk"] += _mu(rp) == _mu(rt)
            s["antal_ratt"] += len(_mu(rp)) == len(_mu(rt))
            s["fler"] += len(_mu(rp)) > len(_mu(rt)); s["farre"] += len(_mu(rp)) < len(_mu(rt))
        ut[d] = dict(namn="transformer", n=s["n"], identisk=s["identisk"] / s["n"],
                     antal_ratt=s["antal_ratt"] / s["n"], fler=s["fler"], farre=s["farre"],
                     tom=0, krasch=s["krasch"], recall=None, falsklarm=None)
    return ut


def _tabell(kolumner):
    drops = sorted({d for k in kolumner for d in k})
    namn = [next(iter(k.values()))["namn"] for k in kolumner if k]
    print(f"\n{'':<34}" + "".join(f"{n:>18}" for n in namn))
    for d in drops:
        print(f"drop {d:.2f}")
        for key, label in (("identisk", "  emittermodell identisk"), ("antal_ratt", "  antal nivåer rätt"),
                           ("fler", "  poster med FLER nivåer"), ("farre", "  poster med FÄRRE nivåer"),
                           ("recall", "  flaggade av de korrupta"), ("falsklarm", "  falsklarm på rena")):
            row = f"{label:<34}"
            for k in kolumner:
                v = k.get(d, {}).get(key)
                if v is None: row += f"{'–':>18}"
                elif isinstance(v, float): row += f"{v:>18.3f}"
                else: row += f"{v:>18d}"
            print(row)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--ut", default="runs/baslinje/signaler.npz")
    e = sub.add_parser("eval")
    e.add_argument("--flaggor", help="npz med flaggor per signal (från steg 1)")
    e.add_argument("--detektor", help="modul:funktion, pri -> 0/1-array")
    e.add_argument("--orakel", action="store_true", help="generatorns sanning som detektor (taket)")
    e.add_argument("--run", help="körningsmapp med preds_drop_*.txt för transformerns kolumn")
    e.add_argument("--min-pulser", type=int, default=1)
    e.add_argument("--ut", default="runs/baslinje/resultat.json")
    a = ap.parse_args()

    sig = _evalsignaler()
    print(f"{len(sig)} evalsignaler ({Counter(r['drop'] for r in sig)})")

    if a.cmd == "dump":
        Path(a.ut).parent.mkdir(parents=True, exist_ok=True)
        _spara(a.ut, sig)
        print(f"skrev {a.ut}: nycklar s0..s{len(sig)-1} + meta. Kör detektorn på varje och "
              f"spara flaggorna med samma nycklar, sedan: eval --flaggor <fil>")
        return

    kolumner = []
    if a.orakel:
        kolumner.append(_jamfor(sig, [r["sant_flagga"] for r in sig], a.min_pulser, "orakel-detektor"))
    if a.detektor:
        mod, fn = a.detektor.split(":")
        f = getattr(importlib.import_module(mod), fn)
        flaggor = [np.asarray(f(r["pri"])).astype(np.int64).ravel() for r in sig]
        kolumner.append(_jamfor(sig, flaggor, a.min_pulser, "din detektor"))
    if a.flaggor:
        flaggor = _lasa_flaggor(a.flaggor, len(sig))
        for r, f in zip(sig, flaggor):
            assert len(f) == len(r["pri"]), "flaggan har fel längd"
        kolumner.append(_jamfor(sig, flaggor, a.min_pulser, "din detektor"))
    if a.run:
        kolumner.append(_transformer(sig, a.run))
    if not kolumner:
        ap.error("ange --orakel, --detektor eller --flaggor (och gärna --run)")

    _tabell(kolumner)
    Path(a.ut).parent.mkdir(parents=True, exist_ok=True)
    with open(a.ut, "w") as f:
        json.dump([{str(d): v for d, v in k.items()} for k in kolumner], f, indent=1)
    print(f"\nskrev {a.ut}")


if __name__ == "__main__":
    main()
