"""
Är modellen för skarp eller för platt jämfört med det mål den tränas mot?

    python sharpness.py runs/20260916_101955/model.pt

Uppdaterad för den delade tokenrymden: nivåbins och dwelltider har olika sigma
och redovisas därför var för sig. Ett gemensamt medelvärde över båda vore
meningslöst när de optimerar mot olika målfördelningar.

Vad frågan betyder
------------------
Korsentropi mot ett utjämnat mål mäter KALIBRERING, inte träffsäkerhet. Målet är
en fördelning, inte ett svar, och den lägsta förlusten fås när modellen förutsäger
exakt den fördelningen. Blir modellen mer bestämd än så straffas den obegränsat --
lägger den all massa på rätt bin blir korsentropin mot ett mål som vill ha 0.4 där
enorm, trots att argmax är perfekt.

Därför kan loss_num stiga medan modellen blir bättre, och därför räcker det inte
att titta på lossen.

Tolkning
--------
  modell[0] >> mål[0] och vingarna under målet
        -> SKARPARE än målet. Lossen stiger av straffet, inte av fel. Kolla att
           argmax-träffen samtidigt är hög. Åtgärd ligger i målfördelningen:
           mindre sigma, eller ett regressionshuvud utan golv.

  massa inom ±w klart under 1, |fel| stort
        -> PLATTARE än målet. Modellen hedgar. Då är det ett riktigt problem.

  massa hög men argmax fel
        -> modellen är säker på fel ställe. Ovanligt, och värt att titta närmare på.
"""
import math
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformer_post_generator.config import cfg
from transformer_post_generator.data import collate, make_pairs
from transformer_post_generator.vocab import IS_BIN, IS_DUR, PAD, RANGE_HI, RANGE_LO, SIGMA

N_EMITTERS = 64


def profil(prob, tgt, mask, w):
    """
    Medelsannolikhet på facit och dess grannar, för de token mask väljer ut.
    -> (offsets, modellens profil, målets profil, extra mått)
    """
    idx = tgt[mask]
    p = prob[mask]
    n = idx.numel()
    if n == 0:
        return None

    lo, hi = RANGE_LO[idx], RANGE_HI[idx]
    sig = float(SIGMA[idx[0]])                 # samma inom en rymd
    offs = list(range(-w, w + 1))

    modell = []
    for off in offs:
        j = idx + off
        ok = (j >= lo) & (j <= hi)
        g = p.gather(1, j.clamp_min(0).clamp_max(p.size(1) - 1)[:, None]).squeeze(1)
        modell.append(float((g * ok).sum() / ok.sum().clamp(min=1)))

    q = np.array([math.exp(-0.5 * (o / max(sig, 1e-6)) ** 2) for o in offs])
    q = q / q.sum()

    pred = p.argmax(1)
    samma_rymd = (RANGE_LO[pred] == lo)
    err = (pred - idx).abs().float()
    ent = float(-(p * p.clamp_min(1e-12).log()).sum(1).mean())
    qq = q[q > 0]

    return dict(n=n, sigma=sig, offs=offs, modell=modell, mal=q,
                massa=float(sum(modell)), mal_massa=float(q.sum()),
                exakt=float((err == 0).float().mean()),
                inom=float((err <= w).float().mean()),
                medel=float(err.mean()), median=float(err.median()),
                p95=float(err.quantile(0.95)),
                ratt_rymd=float(samma_rymd.float().mean()),
                entropi=ent, mal_entropi=float(-(qq * np.log(qq)).sum()))


def visa(namn, r, enhet):
    if r is None:
        print(f"\n{namn}: inga token i batchen")
        return
    print(f"\n{namn}   n = {r['n']}   sigma = {r['sigma']}")
    print("  offset :" + "".join(f"{o:>8d}" for o in r["offs"]))
    print("  mål    :" + "".join(f"{v:>8.3f}" for v in r["mal"]))
    print("  modell :" + "".join(f"{v:>8.3f}" for v in r["modell"]))
    print(f"  massa inom ±{cfg.smooth_width}: mål {r['mal_massa']:.3f}   "
          f"modell {r['massa']:.3f}")
    print(f"  argmax exakt {r['exakt']:.3f}   inom ±{cfg.smooth_width} {r['inom']:.3f}"
          f"   rätt tokenrymd {r['ratt_rymd']:.3f}")
    print(f"  |fel| medel {r['medel']:.2f} {enhet}   median {r['median']:.0f}   "
          f"p95 {r['p95']:.0f}")
    print(f"  entropi: modell {r['entropi']:.3f}   mål {r['mal_entropi']:.3f} nat"
          f"   -> {'PLATTARE' if r['entropi'] > r['mal_entropi'] else 'SKARPARE'} än målet")


def main(ckpt):
    from transformer_post_generator.model import build_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    pairs = make_pairs(np.random.default_rng(cfg.eval_seed), N_EMITTERS, 0.0)
    src, tgt_in, tgt_out = next(iter(DataLoader(pairs, batch_size=len(pairs),
                                                shuffle=False, collate_fn=collate)))
    src = {k: v.to(dev) for k, v in src.items()}
    with torch.no_grad():
        prob = model(src, tgt_in.to(dev)).softmax(-1).float().cpu()

    w = cfg.smooth_width
    giltig = tgt_out != PAD
    print(f"checkpoint : {ckpt}")
    print(f"smooth_width {w}   sigma_bin {cfg.sigma_bin}   sigma_dur {cfg.sigma_dur}")

    visa("NIVÅBINS", profil(prob, tgt_out, IS_BIN[tgt_out] & giltig, w), "bins")
    visa("DWELLTIDER", profil(prob, tgt_out, IS_DUR[tgt_out] & giltig, w), "pulser")

    print("\nJämför de två: har de olika karaktär är det ett tecken på att den"
          "\ndelade sigman var rätt beslut -- de behövde olika mål.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("användning: python sharpness.py runs/<körning>/model.pt")
    main(sys.argv[1])
