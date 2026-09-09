"""
Snabb testsvit. Kör den innan varje längre träningskörning.

    python tests.py

Testerna är inte till för att bevisa att modellen fungerar utan för att fånga de
fel som annars kostar en hel körning innan de märks:

  1. facitet överlever rundturen to_tokens -> parse -> to_tokens
  2. facitet beskriver faktiskt sin egen signal (verify)
  3. samma seed ger samma emittrar OCH samma signaler för alla bortfallsnivåer
  4. belöningen rankar ett korrekt facit över ett stört
  5. en batch går genom modell och loss och ger ett ändligt tal  (kräver torch)

Steg 5 hoppas över om torch saknas, så sviten går att köra på en maskin utan GPU-stack.
"""
import sys

import numpy as np

from config import cfg
from labels import parse, roundtrip_ok, to_tokens
from reward import reward
from verify import verify_label

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FEL '} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def tok(lab):
    return to_tokens(lab["levels"], lab["lengths"], lab["order_fixed"], lab["length_fixed"])


# ------------------------------------------------------------------ 1-2
def test_labels(n=400):
    from data import create_emitter_data
    print(f"\n1-2) facit ({cfg.emitter}, {n} emittrar x 2 signaler)")
    data = create_emitter_data(n, 2, 0.0, cfg.noise_level, np.random.default_rng(0))

    bad_rt = sum(not roundtrip_ok(l["levels"], l["lengths"], l["order_fixed"], l["length_fixed"])
                 for _, l in data)
    check("rundtur to_tokens -> parse -> to_tokens", bad_rt == 0, f"{bad_rt}/{n} fel")

    ok = sum(verify_label(s, l, verbose=False)["ok"] for seqs, l in data for s in seqs)
    tot = sum(len(s) for s, _ in data)
    check("facit beskriver sin signal", ok / tot >= 0.99, f"{ok}/{tot} = {ok/tot:.1%}")


# ------------------------------------------------------------------ 3
def test_determinism():
    from data import create_emitter_data
    print("\n3) reproducerbarhet över bortfallsnivåer")
    a = create_emitter_data(6, 3, 0.00, None, np.random.default_rng(7))
    same_em = same_sig = True
    for p in (0.05, 0.20):
        b = create_emitter_data(6, 3, p, None, np.random.default_rng(7))
        for (sa, la), (sb, lb) in zip(a, b):
            same_em &= bool(np.allclose(la["levels"], lb["levels"]))
            for x, y in zip(sa, sb):
                # bortfall tar bort pulser; TOA:erna som blir kvar måste vara en
                # delmängd av den rena signalens
                ta, tb = np.round(np.cumsum(x), 6), np.round(np.cumsum(y), 6)
                same_sig &= bool(np.isin(tb[:-1], ta).all())
    check("samma emittrar oavsett drop_rate", same_em)
    check("samma underliggande signal oavsett drop_rate", same_sig)


# ------------------------------------------------------------------ 4
def test_reward(n=200):
    from data import create_emitter_data
    from vocab import bin_of, pri_of_bin
    print("\n4) belöningen rankar rätt")
    rng = np.random.default_rng(0)

    for p in (0.0, 0.10, 0.20):
        data = create_emitter_data(n, 1, p, None, np.random.default_rng(0))
        true_r, win, tot = [], 0, 0
        for seqs, lab in data:
            t = tok(lab); s = seqs[0]
            r0 = reward(t, s)
            true_r.append(r0)
            d = parse(t)
            lv = list(d["order"] if d["order_fixed"] else d["levels"])
            i = int(rng.integers(len(lv)))
            lv[i] = pri_of_bin(min(cfg.n_bins - 1, bin_of(lv[i]) + 5))     # nivå 5 bins fel
            try:
                pt = to_tokens(lv, d["lengths"], d["order_fixed"], d["length_fixed"])
            except Exception:
                continue
            if pt == t:
                continue
            tot += 1
            win += r0 > reward(pt, s)
        mean = float(np.mean(true_r))
        floor = 0.99 if p == 0 else 0.70
        check(f"sant facit får hög belöning (drop {p:.2f})", mean >= floor, f"medel {mean:.3f}")
        check(f"sant facit slår stört facit (drop {p:.2f})", win / max(1, tot) >= 0.95,
              f"{win}/{tot} = {win/max(1,tot):.1%}")


# ------------------------------------------------------------------ 5
def test_model():
    print("\n5) modell, loss och sampling")
    try:
        import torch
    except ImportError:
        print("  --   torch saknas, hoppar över")
        return

    from data import StreamDataset, collate_rl
    from grpo import completion_mask
    from loss import loss_by_field, loss_fn
    from model import build_model
    from vocab import EOS

    ds = StreamDataset(4, 0)
    src, tgt_in, tgt_out, pris = collate_rl([ds[i] for i in range(2)])
    model = build_model()

    logits = model(src, tgt_in)
    check("forward ger rätt form", logits.shape[:2] == tgt_out.shape,
          f"{tuple(logits.shape)} mot {tuple(tgt_out.shape)}")

    l = loss_fn(logits, tgt_out)
    check("loss är ändlig", bool(torch.isfinite(l)), f"{l.item():.4f}")
    ln, lo, lg = loss_by_field(logits, tgt_out)
    check("fältvis loss är ändlig", all(np.isfinite([ln, lo, lg])),
          f"num {ln:.3f} order {lo:.3f} grammar {lg:.3f}")

    l.backward()
    g = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
    check("gradienten når vikterna", g > 0)

    model.eval()
    G = 3
    seq = model.generate(src, n=G, greedy=False, temperature=1.0, max_new=24)
    check("generate ger G sampel per exempel", seq.size(0) == tgt_out.size(0) * G,
          f"{seq.size(0)} rader")
    cm = completion_mask(seq)
    after_eos_ok = True
    for row in range(seq.size(0)):
        t = seq[row, 1:]
        e = (t == EOS).nonzero()
        if e.numel():
            after_eos_ok &= not bool(cm[row, e[0, 0] + 1:].any())
    check("completion_mask stänger av allt efter EOS", after_eos_ok)
    check("belöningen tar modellens utdata utan att krascha",
          all(np.isfinite(reward(_ids(seq[i]), pris[i // G])) for i in range(seq.size(0))))


def _ids(row):
    from vocab import ids_to_tokens
    return ids_to_tokens(row.cpu())


# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"config: emitter={cfg.emitter}  model={cfg.model}  n_bins={cfg.n_bins}")
    test_labels()
    test_determinism()
    test_reward()
    test_model()
    print("\n" + ("ALLT GRÖNT" if not FAIL else f"{len(FAIL)} FEL: " + ", ".join(FAIL)))
    sys.exit(1 if FAIL else 0)
