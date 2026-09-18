"""
SFT: cross-entropy mot facit med teacher forcing.

    python train.py                    # använder config.cfg
    python train.py --steps 5000

Generator väljs med cfg.emitter, allt annat i config.py. Kör grpo.py efteråt för
RL-fasen; den startar från checkpointen den här skriver.

Med cfg.aux_weight > 0 optimeras postens loss plus en hjälp-loss på encodern (se
loss.aux_loss). I loggen är `loss` fortfarande postens loss, jämförbar med äldre
körningar; `loss_total` är det som faktiskt optimeras.
"""
import argparse
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import cfg
from .data import StreamDataset, collate, create_emitter_data, make_eval_sets
from .labels import roundtrip_ok
from .loss import aux_loss, loss_by_field, loss_fn
from .model import build_model
from .runlog import RunLog, evaluate


_AUX_COLS = ("loss_aux", "aux_new_acc", "aux_new_recall", "aux_pos_acc",
             "aux_merge_acc", "aux_merge_recall")


def sanity_checks(n=500):
    """Facitet måste överleva rundturen to_tokens -> parse -> to_tokens."""
    for _, f in create_emitter_data(n, 1, 0.0, cfg.noise_level, np.random.default_rng(1)):
        assert roundtrip_ok(f["levels"], f["lengths"], f["order_fixed"], f["length_fixed"]), f
    print(f"rundtur ok på {n} emittrar ({cfg.emitter})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=cfg.steps)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--batch", type=int, default=cfg.batch)
    args = ap.parse_args()

    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sanity_checks()

    run = RunLog(cfg.run_root, cfg.dict() | dict(mode="sft", steps=args.steps,
                                                 lr=args.lr, batch=args.batch))
    eval_sets = make_eval_sets()

    # StreamDataset ger en LISTA med samples_per_emitter par per index, så
    # batch_size måste divideras för att cfg.batch ska bli sekvenser och inte emittrar.
    loader = DataLoader(StreamDataset(args.steps * args.batch, cfg.seed),
                        batch_size=max(1, args.batch // cfg.samples_per_emitter),
                        shuffle=False, collate_fn=collate,
                        num_workers=cfg.num_workers, drop_last=True)

    model = build_model().to(device)
    print(f"{cfg.model}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"{device} -> {run.dir}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)

    def lr_lambda(s):
        if s < cfg.warmup:
            return (s + 1) / cfg.warmup
        p = (s - cfg.warmup) / max(1, args.steps - cfg.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    step, t0 = 0, time.time()
    model.train()
    for src, tgt_in, tgt_out in loader:
        src = {k: v.to(device) for k, v in src.items()}
        tgt_in, tgt_out = tgt_in.to(device), tgt_out.to(device)

        # loss_post är samma storhet som `loss` i tidigare körningar. Den loggas under
        # det namnet även nu, så att kurvorna går att lägga på varandra; det som
        # optimeras är loss_total.
        aux_stats = {}
        if cfg.aux_weight > 0:
            logits, aux = model(src, tgt_in, return_aux=True)
            loss_post = loss_fn(logits, tgt_out)
            loss_aux, aux_stats = aux_loss(aux, src)
            loss = loss_post + cfg.aux_weight * loss_aux
            aux_stats = dict(loss_aux=loss_aux.item(), **aux_stats)
        else:
            logits = model(src, tgt_in)
            loss = loss_post = loss_fn(logits, tgt_out)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = nn.utils.clip_grad_norm_(
            model.parameters(), cfg.clip if cfg.clip > 0 else float("inf"))
        opt.step(); sched.step(); step += 1

        # loss_by_field bygger en till tät (B, T, V)-tensor. Den är bara till för
        # loggen, så den körs inte varje steg.
        if step % cfg.log_every == 0 or step >= args.steps:
            ln, lo, lg = loss_by_field(logits, tgt_out)
            # grad_norm är normen FÖRE klippning. Ligger medianen långt över cfg.clip
            # är klippningen aktiv varje steg och sätter stegstorleken i stället för lr.
            with torch.no_grad():
                wn = sum(float(p.norm()) ** 2 for p in model.parameters()) ** 0.5
            # Kolumnerna måste vara desamma varje rad (csv.DictWriter låser dem vid
            # första raden), så recall-måtten fylls med nan när en batch råkar sakna
            # positiva exempel.
            aux_cols = {}
            if cfg.aux_weight > 0:
                aux_cols = {k: aux_stats.get(k, float("nan")) for k in _AUX_COLS}
            run.log(step=step, loss=loss_post.item(), loss_num=ln, loss_order=lo,
                    loss_grammar=lg, grad_norm=float(gn),
                    logit_max=float(logits.detach().abs().max()), weight_norm=wn,
                    lr=sched.get_last_lr()[0], loss_total=loss.item(), **aux_cols)
            aux_txt = ""
            if cfg.aux_weight > 0:
                aux_txt = (f"  aux {aux_cols['loss_aux']:.3f} "
                           f"(nytt {aux_cols['aux_new_recall']:.2f} "
                           f"pos {aux_cols['aux_pos_acc']:.2f} "
                           f"tapp {aux_cols['aux_merge_recall']:.2f})")
            print(f"step {step:6d}  loss {loss_post.item():.4f}  num {ln:.4f}  order {lo:.4f}  "
                  f"grammar {lg:.4f}{aux_txt}  |g| {float(gn):.2f}  {time.time()-t0:.0f}s")

        if step % cfg.eval_every == 0 or step >= args.steps:
            run.log_eval(step, evaluate(model, eval_sets, device, collate, run))
            run.save_model(model)
            model.train()

        if step >= args.steps:
            break

    print(f"klart, {step} steg på {time.time()-t0:.0f}s -> {run.dir}")


if __name__ == "__main__":
    main()
