"""
Snabb testsvit. Kör den innan varje längre träningskörning.

    python tests.py

Testerna är inte till för att bevisa att modellen fungerar utan för att fånga de
fel som annars kostar en hel körning innan de märks:

  0. generatorn håller kontraktet  -- oavsett vilken modul cfg.emitter pekar på
  1. facitet överlever rundturen to_tokens -> parse -> to_tokens
  2. facitet beskriver faktiskt sin egen signal (verify)
  3. binningen är konsekvent mellan vocab.py och data.py, och täcker datan
  4. samma seed ger samma emittrar OCH samma signaler för alla bortfallsnivåer
  5. en batch går genom modell, loss och avkodning och ger ändliga tal (kräver torch)

Steg 5 hoppas över om torch saknas, så sviten går att köra på en maskin utan GPU-stack.
"""
import sys

import numpy as np

from config import cfg
from data import create_emitter_data
from labels import roundtrip_ok, to_tokens
from verify import verify_label
from vocab import in_bin_of, in_cont_of

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FEL '} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def tok(lab):
    return to_tokens(lab["levels"], lab["lengths"], lab["order_fixed"], lab["length_fixed"])


def gen(n_emitters, n_signals=1, p_drop=0.0, seed=0):
    return create_emitter_data(n_emitters, n_signals, p_drop, cfg.noise_level,
                               np.random.default_rng(seed))


# ------------------------------------------------------------------ 0
def test_contract(n_signals=3):
    """
    Kontraktet är hela poängen med cfg.emitter: byter du generator ska ingenting
    annat behöva ändras. Ett kontrakt som bara står i en docstring är inget kontrakt.
    """
    print(f"\n0) generatorkontrakt ({cfg.emitter})")
    d = gen(2, n_signals)
    check("en post per emitter", len(d) == 2, f"{len(d)} poster")

    seqs, lab = d[0]
    check("en sekvens per signal", len(seqs) == n_signals,
          f"{len(seqs)} sekvenser, väntade {n_signals}")

    p = np.asarray(seqs[0], dtype=float)
    check("varje sekvens är 1-D med pulser", p.ndim == 1 and p.size >= 16,
          f"form {p.shape}")

    saknas = [k for k in ("levels", "lengths", "order_fixed", "length_fixed")
              if k not in lab]
    check("etiketten har alla fält", not saknas,
          f"saknar {saknas}" if saknas else f"{sorted(lab)}")


# ------------------------------------------------------------------ 1-2
def test_labels(n=400):
    print(f"\n1-2) facit ({n} emittrar x 2 signaler)")
    data = gen(n, 2)

    bad_rt = sum(not roundtrip_ok(l["levels"], l["lengths"], l["order_fixed"],
                                  l["length_fixed"]) for _, l in data)
    check("rundtur to_tokens -> parse -> to_tokens", bad_rt == 0, f"{bad_rt}/{n} fel")

    ok = sum(verify_label(s, l, verbose=False)["ok"] for seqs, l in data for s in seqs)
    tot = sum(len(s) for s, _ in data)
    check("facit beskriver sin signal", ok / tot >= 0.99, f"{ok}/{tot} = {ok/tot:.1%}")


# ------------------------------------------------------------------ 3
def test_binning(n=200):
    """
    Två fällor: att vocab.py och data.py har glidit isär (de har varsin kopia av
    inputbinningen), och att nivåer utanför pri_min/pri_max klipps tyst in i
    kant-binen så att olika emittrar får identiska tokens.
    """
    from data import make_channels
    print("\n3) binning")

    p = np.concatenate([np.geomspace(max(cfg.in_min, 1e-6), cfg.in_max * 0.999, 400),
                        [cfg.in_max, cfg.in_max * 2]])
    ch = make_channels(p)
    check("vocab.py och data.py binnar lika",
          bool((ch["bins"] == [in_bin_of(x) for x in p]).all()))
    check("vocab.py och data.py ger samma cont",
          bool(np.allclose(ch["cont"], [in_cont_of(x) for x in p], atol=1e-6)))

    lv = np.concatenate([np.unique(np.asarray(l["levels"], dtype=float).ravel())
                         for _, l in gen(n)])
    out = ((lv < cfg.pri_min) | (lv > cfg.pri_max)).mean()
    check("alla nivåer ryms i pri_min..pri_max", out == 0,
          f"{out:.2%} klipps, spann {lv.min():.3f}–{lv.max():.3f} µs")

    w_out = (cfg.pri_max - cfg.pri_min) / (cfg.n_bins - 1)
    w_in = (cfg.in_max - cfg.in_min) / (cfg.in_bins - 2)
    check("inputen är minst lika fin som utdatan", w_in <= w_out * 1.01,
          f"input {w_in:.4f} µs, utdata {w_out:.4f} µs")


# ------------------------------------------------------------------ 4
def test_determinism():
    print("\n4) reproducerbarhet över bortfallsnivåer")
    a = gen(6, 3, 0.00, seed=7)
    same_em = same_sig = True
    for p in (0.05, 0.20):
        b = gen(6, 3, p, seed=7)
        for (sa, la), (sb, lb) in zip(a, b):
            same_em &= bool(np.allclose(np.ravel(la["levels"]), np.ravel(lb["levels"])))
            for x, y in zip(sa, sb):
                # bortfall tar bort pulser; de som blir kvar måste vara en delmängd
                ta, tb = np.round(np.cumsum(x), 6), np.round(np.cumsum(y), 6)
                same_sig &= bool(np.isin(tb[:-1], ta).all())
    check("samma emittrar oavsett drop_rate", same_em)
    check("samma underliggande signal oavsett drop_rate", same_sig)


# ------------------------------------------------------------------ 5
def test_model():
    print("\n5) modell, loss och avkodning")
    try:
        import torch
    except ImportError:
        print("  --   torch saknas, hoppar över")
        return

    from data import StreamDataset, collate
    from loss import loss_by_field, loss_fn
    from model import build_model
    from vocab import EOS, PAD, ids_to_tokens

    ds = StreamDataset(4, 0)
    src, tgt_in, tgt_out = collate([ds[i] for i in range(2)])
    model = build_model()

    logits = model(src, tgt_in)
    check("forward ger rätt form", logits.shape[:2] == tgt_out.shape,
          f"{tuple(logits.shape)} mot {tuple(tgt_out.shape)}")

    l = loss_fn(logits, tgt_out)
    check("loss är ändlig", bool(torch.isfinite(l)), f"{l.item():.4f}")
    f = loss_by_field(logits, tgt_out)
    fields = f if isinstance(f, dict) else dict(zip(("num", "order", "grammar"), f))
    check("fältvis loss är ändlig", all(np.isfinite(list(fields.values()))),
          "  ".join(f"{k} {v:.3f}" for k, v in fields.items()))

    # vid slumpvikter ska lossen ligga nära ln(V) -- ligger den långt ifrån är
    # något fel på maskning eller målfördelning
    import math
    check("loss nära slumpnivån ln(V) vid init",
          abs(l.item() - math.log(logits.size(-1))) < 1.5,
          f"{l.item():.3f} mot ln({logits.size(-1)}) = {math.log(logits.size(-1)):.3f}")

    l.backward()
    g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    check("gradienten når vikterna", g > 0)

    model.eval()
    seq = model.greedy(src, max_new=32)
    check("greedy ger en rad per exempel", seq.size(0) == tgt_out.size(0),
          f"{seq.size(0)} rader")
    check("greedy börjar med BOS och innehåller inga PAD före EOS",
          all(_well_formed(seq[i], EOS, PAD) for i in range(seq.size(0))))
    check("utdatan går att läsa som tokens",
          all(isinstance(ids_to_tokens(seq[i].cpu()), list) for i in range(seq.size(0))))


def _well_formed(row, EOS, PAD):
    t = row[1:]
    e = (t == EOS).nonzero()
    cut = int(e[0, 0]) if e.numel() else t.numel()
    return not bool((t[:cut] == PAD).any())


# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"config: emitter={cfg.emitter}  model={cfg.model}  "
          f"n_bins={cfg.n_bins}  in_bins={cfg.in_bins}")
    test_contract()
    test_labels()
    test_binning()
    test_determinism()
    test_model()
    print("\n" + ("ALLT GRÖNT" if not FAIL else f"{len(FAIL)} FEL: " + ", ".join(FAIL)))
    sys.exit(1 if FAIL else 0)
