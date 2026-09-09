"""
GRPO -- RL med verifierbar belöning, i DeepSeeks tappning.

    python grpo.py --init runs/20260908_101500/model.pt      # rekommenderat
    python grpo.py --steps 500 --group 16 --beta 0.0         # experiment

Vad det är
----------
Group Relative Policy Optimization, samma algoritm som DeepSeekMath och R1 använder.
Den skiljer sig från PPO på en punkt: ingen värdemodell. I stället samplas G svar på
samma indata, och gruppens medelbelöning är baslinjen. Fördelen för svar i blir

    A_i = (r_i - medel(r)) / std(r)

och den gäller för varje token i svaret. Det halverar minnesbehovet och tar bort en
modell som ändå är svår att träna.

Varför problemet passar
-----------------------
R1 fungerar för att matematik och kod har ett svar som går att KONTROLLERA utan att
be en modell om åsikt. Det gäller här också: en bibliotekspost är ett påstående om en
pulsföljd, och verify.py kontrollerar redan påståendet mot signalen. reward.py är den
kontrollen graderad till [0, 1].

Det ger tre saker som cross-entropy inte ger:

  1. Belöningen mäter posten, inte nästa token. Det angriper precis det LOSS.md §9
     beskriver: ett fel i antalet nivåer straffas en gång under teacher forcing men
     förstör hela posten vid avkodning. Här straffas det som det som det är -- hela
     rollouten får låg belöning.

  2. Belöningen behöver inte facit. Den mäter mot signalen. Samma loop går alltså att
     köra på inspelad data utan sanning, vilket är den intressanta delen.

  3. Modellen tränas på sin EGEN avkodning, inte på facitets prefix. Exposure bias
     försvinner därmed per konstruktion.

Ordning: SFT först (train.py), GRPO sedan. R1-Zero visar att ren RL från grunden går,
men bara när basmodellen redan kan formatet. Härifrån är sannolikheten att slumpvikter
producerar en parsbar post ungefär noll, så utan --init får du inget gradientflöde.

Läsa loggen
-----------
    reward        ska stiga. Gör den inte det: höj rl_temp eller sänk rl_beta.
    parsed        ska nå ~1,0 inom några hundra steg.
    exact         från utvärderingen, mot FACIT. Stiger reward men inte exact
                  hackar modellen verifieraren -- se avsnittet i reward.py.
    kl            driver iväg = modellen glider från SFT-lösningen. Höj rl_beta.
    uniq          antal olika svar i batchen. Faller den mot 1 har policyn
                  kollapsat; höj rl_temp eller sänk lr.
"""
import argparse
import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import cfg
from data import StreamDataset, collate, collate_rl, make_eval_sets
from model import build_model
from reward import reward_report
from runlog import RunLog, evaluate
from vocab import EOS, PAD, ids_to_tokens


# ------------------------------------------------------------------ hjälp
def completion_mask(seq):
    """
    True för de tokens i seq[:, 1:] som modellen faktiskt genererat: fram till och
    med första EOS. Allt efter är utfyllnad och ska varken belönas eller straffas.
    """
    tgt = seq[:, 1:]
    is_eos = (tgt == EOS).long()
    after = is_eos.cumsum(1) - is_eos          # 0 t.o.m. första EOS, >= 1 efter
    return (after == 0) & (tgt != PAD)


def token_logprobs(model, mem, mask, seq):
    """log p(seq[:, t] | seq[:, :t]) för t >= 1. -> (B, L-1)"""
    logits = model.decoder(mem, mask, seq[:, :-1]).float()
    logp = F.log_softmax(logits, -1)
    return logp.gather(-1, seq[:, 1:, None]).squeeze(-1)


def expand(src, model, G):
    """Encodera en gång, upprepa minnet G gånger. repeat_interleave är deriverbar,
    så gradienten från alla G rollouts summeras tillbaka in i encodern."""
    mem = model.encode(src)
    if G > 1:
        mem = mem.repeat_interleave(G, 0)
    mask = src["mask"].repeat_interleave(G, 0) if G > 1 else src["mask"]
    return mem, mask


# ------------------------------------------------------------------ träning
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=cfg.rl_init, help="SFT-checkpoint att starta från")
    ap.add_argument("--steps", type=int, default=cfg.rl_steps)
    ap.add_argument("--prompts", type=int, default=cfg.rl_prompts)
    ap.add_argument("--group", type=int, default=cfg.rl_group)
    ap.add_argument("--lr", type=float, default=cfg.rl_lr)
    ap.add_argument("--beta", type=float, default=cfg.rl_beta)
    ap.add_argument("--temp", type=float, default=cfg.rl_temp)
    args = ap.parse_args()

    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G, P = args.group, args.prompts

    model = build_model().to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"startar från {args.init}")
    else:
        print("VARNING: ingen --init. Slumpvikter producerar nästan aldrig en parsbar\n"
              "         post, så belöningen blir konstant 0 och gradienten försvinner.\n"
              "         Kör train.py först.")

    # Referensmodellen som KL:en mäts mot. Fryst kopia av startpunkten.
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # Hela RL-körningen går i eval-läge. Det stänger av dropout, vilket är avsiktligt:
    # med dropout på skiljer sig log-sannolikheterna mellan rollout och uppdatering,
    # och kvoten log p_ny - log p_gammal mäter då brus i stället för policyändring.
    # eval() stoppar inte gradienterna.
    model.eval()

    run = RunLog(cfg.rl_run_root, cfg.dict() | dict(
        mode="grpo", rl_init=args.init, rl_steps=args.steps, rl_prompts=P,
        rl_group=G, rl_lr=args.lr, rl_beta=args.beta, rl_temp=args.temp))
    print(f"{cfg.model}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, {device}")
    print(f"{P} signaler x {G} sampel = {P*G} rollouts per steg -> {run.dir}")

    eval_sets = make_eval_sets()
    loader = DataLoader(
        StreamDataset(args.steps * P, cfg.seed + 1),
        batch_size=max(1, P // cfg.samples_per_emitter),
        shuffle=False, collate_fn=collate_rl, num_workers=cfg.num_workers, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, cfg.rl_warmup)))

    step, t0 = 0, time.time()
    for src, _, _, pris in loader:
        src = {k: v.to(device) for k, v in src.items()}
        n_prompts = len(pris)

        # ---------------- 1. rollouts
        with torch.no_grad():
            seq = model.generate(src, n=G, greedy=False, temperature=args.temp,
                                 top_k=cfg.rl_top_k, max_new=cfg.rl_max_new)   # (n*G, L)
        cmask = completion_mask(seq)

        # ---------------- 2. belöning, mot SIGNALEN (facit rörs inte)
        toks = [ids_to_tokens(s) for s in seq.cpu()]
        reps = [reward_report(t, pris[i // G]) for i, t in enumerate(toks)]
        R = torch.tensor([r["total"] for r in reps], dtype=torch.float32,
                         device=device).view(n_prompts, G)

        # ---------------- 3. grupprelativ fördel
        adv = R - R.mean(1, keepdim=True)
        if cfg.rl_norm_adv:
            # DeepSeeks normalisering. Den skalar upp grupper där alla svar är nästan
            # lika bra, vilket ger lätta exempel för stort inflytande; sätt
            # cfg.rl_norm_adv = False för Dr. GRPO-varianten utan den biasen.
            adv = adv / (R.std(1, keepdim=True) + 1e-4)
        adv = adv.reshape(-1, 1)                                    # broadcast över tokens

        # ---------------- 4. referenser
        with torch.no_grad():
            mem_o, mask_g = expand(src, model, G)
            logp_old = token_logprobs(model, mem_o, mask_g, seq)
            mem_r, _ = expand(src, ref, G)
            logp_ref = token_logprobs(ref, mem_r, mask_g, seq)

        # ---------------- 5. uppdatering
        for _ in range(cfg.rl_inner_epochs):
            mem, mask_g = expand(src, model, G)
            logp = token_logprobs(model, mem, mask_g, seq)

            ratio = (logp - logp_old).exp()
            unclipped = ratio * adv
            clipped = ratio.clamp(1 - cfg.rl_clip, 1 + cfg.rl_clip) * adv

            # k3-skattaren: alltid >= 0, låg varians. Samma som DeepSeek använder.
            dlt = logp_ref - logp
            kl = dlt.exp() - dlt - 1.0

            per_tok = torch.min(unclipped, clipped) - args.beta * kl
            if cfg.rl_seq_mean:
                per_seq = (per_tok * cmask).sum(1) / cmask.sum(1).clamp(min=1)
                loss = -per_seq.mean()
            else:
                # medel över alla tokens i gruppen: inget längdberoende i vikten
                loss = -(per_tok * cmask).sum() / cmask.sum().clamp(min=1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        step += 1

        # ---------------- 6. logg
        n_tok = cmask.sum().clamp(min=1)
        stats = dict(
            step=step,
            reward=float(R.mean()),
            reward_std=float(R.std(1).mean()),           # spridning INOM grupperna
            reward_max=float(R.max(1).values.mean()),
            parsed=float(np.mean([r["parsed"] for r in reps])),
            canonical=float(np.mean([r["canonical"] for r in reps])),
            r_levels=float(np.mean([r["levels"] for r in reps])),
            r_coverage=float(np.mean([r["coverage"] for r in reps])),
            r_order=float(np.mean([r["order"] for r in reps])),
            r_lengths=float(np.mean([r["lengths"] for r in reps])),
            kl=float((kl * cmask).sum() / n_tok),
            gen_len=float(cmask.sum(1).float().mean()),
            uniq=len({" ".join(t) for t in toks}),
            loss=float(loss),
            grad_norm=float(gn),
            lr=sched.get_last_lr()[0],
        )
        run.log(**stats)

        if step % cfg.rl_log_every == 0:
            print(f"step {step:5d}  reward {stats['reward']:.3f}"
                  f" (bäst i grupp {stats['reward_max']:.3f}, spridning {stats['reward_std']:.3f})"
                  f"  parsed {stats['parsed']:.2f}  niv {stats['r_levels']:.2f}"
                  f"  ordn {stats['r_order']:.2f}  längd {stats['r_lengths']:.2f}"
                  f"  kl {stats['kl']:.4f}  uniq {stats['uniq']:>3}/{n_prompts*G}"
                  f"  {time.time()-t0:.0f}s")

        if step % cfg.rl_eval_every == 0 or step >= args.steps:
            # Utvärderingen är den enda platsen facit används. Stiger reward men inte
            # exact/level_recall har modellen hittat ett kryphål i verifieraren.
            run.log_eval(step, evaluate(model, eval_sets, device, collate, run))
            run.save_model(model)
            model.eval()

        if step >= args.steps:
            break

    print(f"klart, {step} steg på {time.time()-t0:.0f}s -> {run.dir}")


if __name__ == "__main__":
    main()
