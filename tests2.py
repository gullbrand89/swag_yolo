"""
Snabb testsvit. Kör den innan varje längre träningskörning.

    python tests.py

Testerna är inte till för att bevisa att modellen fungerar utan för att fånga de
fel som annars kostar en hel körning innan de märks:

  0. generatorn håller kontraktet  -- oavsett vilken modul cfg.emitter pekar på
  1. facitet överlever rundturen to_tokens -> parse -> to_tokens
  2. facitet beskriver faktiskt sin egen signal (verify)
  3. en emitter har exakt ETT facit -- likvärdiga beskrivningar kanoniseras lika
     (nivåblocket har inget antalstoken: par läses tills ORDER dyker upp)
  4. binningen är konsekvent mellan vocab.py och data.py, och täcker datan
  5. samma seed ger samma emittrar OCH samma signaler för alla bortfallsnivåer
  6. en batch går genom modell, loss och avkodning och ger ändliga tal (kräver torch)

Steg 6 hoppas över om torch saknas, så sviten går att köra på en maskin utan GPU-stack.
"""
import sys

import numpy as np

from config import cfg
from data import create_emitter_data
from labels import roundtrip_ok, to_tokens
from verify import verify_label
from vocab import in_bin_of, in_cont_of, pri_of_bin

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FEL '} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def tok(lab):
    return to_tokens(lab["levels"], lab["lengths"], lab["order_fixed"], lab["length_fixed"])


def _n_levels(t):
    """Antal nivåer, nu när antalstoken är borta: (L, bin)-par mellan LEVELS och ORDER."""
    return (t.index("ORDER") - 1) // 2


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
def _levels(*fracs):
    """Nivåer på givna andelar av binrymden. Garanterat distinkta bins, oavsett config."""
    return [pri_of_bin(int((cfg.n_bins - 1) * f)) for f in fracs]


def test_canonical(n=300):
    """
    Rundturen i steg 1 kontrollerar bara att ETT facit överlever fram och tillbaka.
    Den upptäcker inte att två likvärdiga beskrivningar av SAMMA emitter ger olika
    tokensträngar -- och det är ett dyrare fel, för då ser modellen identiska
    signaler med olika mål och lossen får ett golv som ingen träning tar bort.
    """
    print("\n3) kanonisering: en emitter, ett facit")
    rng = np.random.default_rng(3)

    a = to_tokens(_levels(0.2, 0.5), [4, 4], True, False)   # nollbrett RANGE
    b = to_tokens(_levels(0.2, 0.5), [4], True, True)       # fast dwell
    check("RANGE med min == max blir FIXED", a == b,
          "" if a == b else f"\n        {' '.join(a)}\n        {' '.join(b)}")

    a = to_tokens(_levels(0.2, 0.5), [7, 7, 7], True, True)
    b = to_tokens(_levels(0.2, 0.5), [7], True, True)
    check("upprepad längdcykel kollapsar till minsta period", a == b,
          "" if a == b else f"\n        {' '.join(a)}\n        {' '.join(b)}")

    # rotation: samma cykel, annan startfas -> samma facit
    bad = None
    for _ in range(n):
        k = int(rng.integers(2, 6))
        bins = sorted(rng.choice(np.arange(1, cfg.n_bins - 1), size=k, replace=False))
        lv = [pri_of_bin(int(x)) for x in bins]
        dw = [int(x) for x in rng.integers(1, 30, size=k)]
        t0 = to_tokens(lv, dw, True, True)
        r = int(rng.integers(1, k))
        t1 = to_tokens(lv[r:] + lv[:r], dw[r:] + dw[:r], True, True)
        if t0 != t1 and bad is None:
            bad = (t0, t1)
    check("rotation av cykeln ger samma facit", bad is None,
          "" if bad is None else f"\n        {' '.join(bad[0])}\n        {' '.join(bad[1])}")

    # Med en eller två nivåer är ordningen inte observerbar: en omedelbar upprepning
    # smälter ihop med föregående besök, så följden alternerar alltid. RANDOM och
    # FIXED beskriver då samma emitter och måste ge samma facit.
    for k, lab in ((2, (0.3, 0.7)), (1, (0.4,))):
        dw = [6] if k == 2 else [None]
        a = to_tokens(_levels(*lab), dw, False, True)
        b = to_tokens(_levels(*lab), dw, True, True)
        check(f"{k} nivåer: ORDER RANDOM == ORDER FIXED", a == b,
              "" if a == b else f"\n        {' '.join(a)}\n        {' '.join(b)}")

    # ...men med tre nivåer ÄR ordningen observerbar och får inte slås ihop
    a = to_tokens(_levels(0.2, 0.5, 0.8), [4], False, True)
    b = to_tokens(_levels(0.2, 0.5, 0.8), [4], True, True)
    check("3 nivåer: RANDOM och FIXED hålls isär", a != b)

    # en påtvingad ordning vet inget om vilken längd som hör till vilken nivå, och
    # får inte hitta på en koppling som generatorn aldrig uppgav
    a = to_tokens(_levels(0.3, 0.7), [8, 5], False, True)
    b = to_tokens(_levels(0.3, 0.7), [8, 5], True, True)
    check("påtvingad ordning kopplar inte nivå till längd", a != b,
          f"\n        RANDOM {' '.join(a[a.index('DWELL'):])}"
          f"\n        FIXED  {' '.join(b[b.index('DWELL'):])}")

    # INF betyder "lämnar aldrig nivån" och går inte att uttrycka som ett intervall.
    # Ett RANGE med INF är självmotsägande och ska smälla, inte tyst tappa fältet.
    try:
        to_tokens(_levels(0.2, 0.5), [5, None], True, False)
        raised = False
    except AssertionError:
        raised = True
    check("RANGE med INF avvisas", raised)

    # och att regeln faktiskt slår igenom på riktig data
    data = gen(n)
    both = sum(1 for _, l in data
               if not l["length_fixed"]
               and [x for x in l["lengths"] if x is not None]
               and min(x for x in l["lengths"] if x is not None)
               == max(x for x in l["lengths"] if x is not None))
    kvar = sum(1 for _, l in data if "RANGE" in tok(l)
               and tok(l)[tok(l).index("RANGE") + 1] == tok(l)[tok(l).index("RANGE") + 2])
    check("inga nollbredda RANGE i facitet", kvar == 0,
          f"{kvar}/{len(data)} kvar, {both} kandidater i generatorns utdata")

    rnd = sum(1 for _, l in data if "RANDOM" in tok(l) and _n_levels(tok(l)) <= 2)
    check("inga ORDER RANDOM med färre än tre nivåer", rnd == 0,
          f"{rnd}/{len(data)} kvar")


# ------------------------------------------------------------------ 4
def test_binning(n=200):
    """
    Två fällor: att vocab.py och data.py har glidit isär (de har varsin kopia av
    inputbinningen), och att nivåer utanför pri_min/pri_max klipps tyst in i
    kant-binen så att olika emittrar får identiska tokens.
    """
    from data import make_channels
    print("\n4) binning")

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


# ------------------------------------------------------------------ 5
def _overlap_match(clean_pri, obs_pri, tol=1e-6):
    """
    Hur stor del av den glesade signalens pulser ligger på samma tidpunkt som i den
    rena? -> (andel träffar, antal jämförda, andel av obs bortom den renas slut)

    Bortfall tar bort pulser ur ett gemensamt pulståg, så de som blir kvar ska ha
    OFÖRÄNDRADE ankomsttider. Men "delmängd" håller inte rakt av, av tre skäl:

      * signalen trunkeras till samma ANTAL pulser, inte samma TID, så den glesade
        sträcker sig längre och slutet har inget att matcha mot -- därför jämförs
        bara det gemensamma tidsfönstret
      * faller ankarpulsen bort mäts allt från nästa puls och hela axeln förskjuts
        -- det ger nära 0 % träff och syns direkt i utskriften
      * float-summering i olika ordning skiljer i sista bitarna -- därför tolerans
        i stället för exakt likhet
    """
    ta = np.cumsum(np.asarray(clean_pri, dtype=float))
    tb = np.cumsum(np.asarray(obs_pri, dtype=float))
    if ta.size == 0 or tb.size == 0:
        return 0.0, 0, 1.0
    inside = tb <= ta[-1] + tol
    tb_in = tb[inside]
    if tb_in.size == 0:
        return 0.0, 0, 1.0
    i = np.clip(np.searchsorted(ta, tb_in), 1, ta.size - 1)
    d = np.minimum(np.abs(ta[i] - tb_in), np.abs(ta[i - 1] - tb_in))
    return float((d <= tol).mean()), int(tb_in.size), float(1.0 - inside.mean())


def test_determinism():
    """
    Evalmängderna måste vara jämförbara mellan bortfallsnivåer. Är de inte det går
    det inte att avgöra om ett sämre mått vid drop 0.20 beror på bortfallet eller
    på att det helt enkelt är andra emittrar.
    """
    print("\n5) reproducerbarhet över bortfallsnivåer")
    a = gen(6, 3, 0.00, seed=7)
    same_em = True
    worst, beyond = 1.0, 0.0
    for p in (0.05, 0.20):
        b = gen(6, 3, p, seed=7)
        for (sa, la), (sb, lb) in zip(a, b):
            same_em &= bool(np.allclose(np.ravel(la["levels"]), np.ravel(lb["levels"])))
            for x, y in zip(sa, sb):
                f, _, out = _overlap_match(x, y)
                worst = min(worst, f)
                beyond = max(beyond, out)
    check("samma emittrar oavsett drop_rate", same_em)
    check("samma underliggande signal oavsett drop_rate", worst >= 0.99,
          f"sämsta träff {worst:.1%} i gemensamt tidsfönster, "
          f"{beyond:.1%} av den glesade ligger bortom den rena")


# ------------------------------------------------------------------ 6
def test_model():
    print("\n6) modell, loss och avkodning")
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
    
b = open("runs/<tidsstämpel>/preds_drop_0.00.txt").read().strip().split("\n\n")
lvl = lambda s: s.split()[1:s.split().index("ORDER")]
ok = [lvl(x.split("\n")[0]) == lvl(x.split("\n")[1]) for x in b if x.count("ORDER") == 2]
print(sum(ok) / len(b))



# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"config: emitter={cfg.emitter}  model={cfg.model}  "
          f"n_bins={cfg.n_bins}  in_bins={cfg.in_bins}")
    test_contract()
    test_labels()
    test_canonical()
    test_binning()
    test_determinism()
    test_model()
    print("\n" + ("ALLT GRÖNT" if not FAIL else f"{len(FAIL)} FEL: " + ", ".join(FAIL)))
    sys.exit(1 if FAIL else 0)
