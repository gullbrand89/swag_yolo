"""
Samma modell, samma loss, samma optimerare -- men FAST data.

    python train_fixed.py                       # 200 emittrar, 2000 steg
    python train_fixed.py --steps 3000 --lr 1e-4 --emitters 200

Bisektion. Den vanliga träningen får en ny emitter vid varje __getitem__, så om
strömmen driver över tid eller generatorn har tillstånd som ändras mellan anrop syns
det som en stigande loss utan att något annat är fel. Här dras datan EN gång och
återanvänds, med en separat fast mängd som utvärdering.

Tolkning
--------
  train faller mot golvet, eval följer med
        -> modellen och lossen är friska. Felet ligger i dataströmmen.
  train faller, eval vänder uppåt
        -> normal överanpassning på 200 emittrar. Modellen KAN lära sig uppgiften,
           och då är det generaliseringen som är frågan, inte optimeringen.
  train stiger
        -> felet är modell- eller losssidan och har inget med datan att göra.

Evalen mäts med teacher forcing, precis som träningen, så talen är direkt jämförbara.
Greedy-avkodning läggs inte till här -- poängen är att isolera EN variabel.
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import cfg
from data import collate, make_pairs
from loss import loss_by_field, loss_fn
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--emitters", type=int, default=200)
    ap.add_argument("--batch", type=int, default=cfg.batch)
    ap.add_argument("--p_drop", type=float, default=0.0)
    a = ap.parse_args()

    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = make_pairs(np.random.default_rng(11), a.emitters, a.p_drop)
    ev = make_pairs(np.random.default_rng(12), a.emitters // 2, a.p_drop)
    print(f"fast data: {len(tr)} träningssekvenser, {len(ev)} evalsekvenser, "
          f"p_drop {a.p_drop}, {dev}")

    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, collate_fn=collate,
                    drop_last=True)
    el = DataLoader(ev, batch_size=a.batch, shuffle=False, collate_fn=collate)

    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=cfg.weight_decay)

    def lam(s):
        if s < cfg.warmup:
            return (s + 1) / cfg.warmup
        p = (s - cfg.warmup) / max(1, a.steps - cfg.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = n = 0.0
        for src, ti, to in el:
            src = {k: v.to(dev) for k, v in src.items()}
            l = loss_fn(model(src, ti.to(dev)), to.to(dev))
            tot += float(l); n += 1
        model.train()
        return tot / max(1, n)

    print(f"\n{'steg':>6}{'train':>9}{'num':>9}{'order':>9}{'grammar':>9}"
          f"{'eval':>9}{'lr':>10}{'s':>6}")
    step, t0 = 0, time.time()
    model.train()
    while step < a.steps:
        for src, ti, to in tl:
            src = {k: v.to(dev) for k, v in src.items()}
            ti, to = ti.to(dev), to.to(dev)
            logits = model(src, ti)
            loss = loss_fn(logits, to)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1

            if step % 100 == 0 or step >= a.steps:
                f = loss_by_field(logits, to)
                f = f if isinstance(f, dict) else dict(
                    zip(("num", "order", "grammar"), f))
                e = evaluate() if (step % 250 == 0 or step >= a.steps) else float("nan")
                print(f"{step:>6}{float(loss):>9.4f}"
                      + "".join(f"{v:>9.4f}" for v in f.values())
                      + f"{e:>9.4f}{sched.get_last_lr()[0]:>10.2e}"
                      + f"{time.time() - t0:>6.0f}")
            if step >= a.steps:
                break

    print("\nkolla: faller train hela vägen, eller vänder den uppåt som i "
          "den vanliga körningen?")


if __name__ == "__main__":
    main()
