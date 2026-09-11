"""
Överlever nivåerna vägen in i modellen?

    python input_fidelity.py

label_coverage visade att facitet går att se i signalen. Det här skriptet kollar nästa
led: vad som är kvar av nivåerna EFTER make_channels. Klipps pulserna mot in_min/in_max,
eller hamnar två skilda nivåer i samma input-bin, så är facitet obestämbart för modellen
hur bra signalen än är -- och inget i tests.py fångar det, eftersom det testet mäter
inputens upplösning och inte dess räckvidd.

Den avgörande raden är "nivåer som krockar i input-bin". Är den inte 0 är det där felet
sitter.
"""
import importlib

import numpy as np

from config import cfg
from data import make_channels
from vocab import bin_of, in_bin_of

N_EMITTERS = 200


def as_signals(seqs):
    if isinstance(seqs, (list, tuple)):
        return [np.asarray(s, dtype=float).ravel() for s in seqs]
    a = np.asarray(seqs, dtype=float)
    return [a] if a.ndim == 1 else [row for row in a]


def main():
    print("konfiguration")
    print(f"  facit  : pri_min {cfg.pri_min}  pri_max {cfg.pri_max}  n_bins {cfg.n_bins}"
          f"   -> {(cfg.pri_max - cfg.pri_min) / (cfg.n_bins - 1):.4f} µs/bin (linjär)")
    print(f"  input  : in_min  {cfg.in_min}  in_max  {cfg.in_max}  in_bins {cfg.in_bins}"
          f"   (logaritmisk)")
    print(f"  toa_scale {cfg.toa_scale}   run_tol_bins {cfg.run_tol_bins}")
    print()

    gen = importlib.import_module(cfg.emitter)
    data = gen.create_emitter_data(N_EMITTERS, cfg.samples_per_emitter, 0.0,
                                   cfg.noise_level, np.random.default_rng(cfg.eval_seed))

    lo_clip, hi_clip, n_pulse = 0, 0, 0
    collide_sig, n_sig = 0, 0
    collide_lvl, n_lvl = 0, 0
    distinct_in, distinct_lvl = [], []
    cont_min, cont_max = [], []
    toa_end, pri_mean = [], []
    sep_bins = []

    for seqs, label in data:
        lvl = sorted({float(v) for v in np.asarray(label["levels"]).ravel()})
        lvl_in = [in_bin_of(v) for v in lvl]
        c = len(lvl) - len(set(lvl_in))
        collide_lvl += c
        n_lvl += len(lvl)
        if len(lvl) > 1:
            d = np.diff(sorted(lvl_in))
            sep_bins.extend(d.tolist())

        for s in as_signals(seqs):
            n_sig += 1
            collide_sig += int(c > 0)
            ch = make_channels(s)
            b = ch["bins"]
            n_pulse += len(s)
            lo_clip += int(np.sum(s <= cfg.in_min))
            hi_clip += int(np.sum(s >= cfg.in_max))
            distinct_in.append(len(np.unique(b)))
            distinct_lvl.append(len(lvl))
            cont_min.append(float(ch["cont"].min())); cont_max.append(float(ch["cont"].max()))
            toa_end.append(float(ch["toa"][-1])); pri_mean.append(float(np.mean(s)))

    distinct_in = np.array(distinct_in); distinct_lvl = np.array(distinct_lvl)
    sep_bins = np.array(sep_bins) if sep_bins else np.array([0])

    print(f"{n_sig} signaler, {n_pulse} pulser")
    print()
    print("KLIPPNING")
    print(f"  pulser <= in_min ({cfg.in_min}) : {lo_clip:>8}  ({lo_clip / n_pulse:.2%})")
    print(f"  pulser >= in_max ({cfg.in_max}) : {hi_clip:>8}  ({hi_clip / n_pulse:.2%})")
    print(f"  cont-kanalens spann            : [{np.min(cont_min):+.3f}, {np.max(cont_max):+.3f}]"
          f"   (ska fylla ut [-1, +1])")
    print()
    print("UPPLÖSNING")
    print(f"  nivåer som krockar i input-bin : {collide_lvl:>8} av {n_lvl}"
          f"  ({collide_lvl / n_lvl:.2%})")
    print(f"  signaler med minst en krock    : {collide_sig:>8} av {n_sig}"
          f"  ({collide_sig / n_sig:.2%})")
    print(f"  avstånd mellan grannivåer i input-bins: median {np.median(sep_bins):.0f}"
          f"   p5 {np.percentile(sep_bins, 5):.0f}   min {sep_bins.min():.0f}")
    print(f"  distinkta input-bins per signal: median {np.median(distinct_in):.0f}"
          f"   mot {np.median(distinct_lvl):.0f} nivåer i facit")
    bad = int(np.sum(distinct_in < distinct_lvl))
    print(f"  signaler där inputen har FÄRRE distinkta bins än facit har nivåer:"
          f" {bad} ({bad / n_sig:.1%})")
    print()
    print("TIDSKANAL")
    print(f"  medel-PRI i korpusen           : {np.mean(pri_mean):.2f} µs"
          f"   (toa_scale bör ligga här)")
    print(f"  toa vid sista pulsen           : median {np.median(toa_end):.0f}"
          f"   max {np.max(toa_end):.0f}")


if __name__ == "__main__":
    main()
