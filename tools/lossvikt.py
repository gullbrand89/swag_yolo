"""
Hur fördelar sig lossens gradient över fälten?

    python -m tools.lossvikt                  # nuvarande cfg, plus alternativ
    python -m tools.lossvikt --n 2000
    python -m tools.lossvikt --satt 0.5 1.0 2.0    # struct, order, type

Varför det här verktyget finns
------------------------------
loss_fn viktar varje token efter vilket fält det tillhör, och normaliserar med
summan av vikterna. Vad en vikt BETYDER beror därför på hur många token fältet
har -- och det varierar med facitet. Ett typtoken är ett enda token i en post på
25, så vikt 1.0 på det är inte "lika viktigt som numeriken", det är en fyrtiondel
av den.

Det var så den ursprungliga uppställningen gick fel. num 1.0 och allt annat 0.5
lät balanserat, men gav de två typvalen 5,2 % av gradienten tillsammans -- 2,6 %
för ORDER:s FIXED/RANDOM ensamt -- och typvalen är de enda token där ett fel
nollar exact. loss_order stod stilla i 16 000 steg medan loss_num fortsatte falla.

Tabellen nedan är alltså den enda ärliga bilden av vad vikterna gör. Läs ANDELEN,
inte talet i cfg.
"""
import argparse

import numpy as np

from transformer_post_generator.all_emitters import create_emitter_data
from transformer_post_generator.config import cfg
from transformer_post_generator.labels import to_tokens
from transformer_post_generator.loss import FIELDS, field_masks
from transformer_post_generator.vocab import PAD, TOK2ID


def _batch(n, seed):
    """-> (B, T) med facit-id, PAD-utfyllt. Samma form som tgt_out i träningen."""
    data = create_emitter_data(n, 1, drop_rate=0.0, rng=np.random.default_rng(seed))
    seqs = [[TOK2ID[t] for t in to_tokens(l["levels"], l["lengths"],
                                          l["order_fixed"], l["length_fixed"])]
            for _, l in data]
    L = max(map(len, seqs))
    tgt = np.full((len(seqs), L), PAD, dtype=np.int64)
    for i, s in enumerate(seqs):
        tgt[i, :len(s)] = s
    return tgt, seqs


def budget(masker, vikter):
    """-> andel av total lossvikt per fält."""
    tot = sum(float(np.asarray(m).sum()) * v for m, v in zip(masker, vikter))
    return [float(np.asarray(m).sum()) * v / tot for m, v in zip(masker, vikter)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--satt", type=float, nargs=3, default=None,
                    metavar=("STRUCT", "ORDER", "TYPE"))
    a = ap.parse_args()

    import torch
    tgt_np, seqs = _batch(a.n, a.seed)
    tgt = torch.as_tensor(tgt_np)
    masker = field_masks(tgt)

    print(f"{a.n} facit, median {int(np.median([len(s) for s in seqs]))} token\n")
    print(f"{'fält':<10}{'token':>8}{'andel token':>14}")
    n_tot = sum(float(np.asarray(m).sum()) for m in masker)
    for namn, m in zip(FIELDS, masker):
        n = float(np.asarray(m).sum())
        print(f"{namn:<10}{int(n):>8}{n/n_tot:>14.1%}")

    nu = (cfg.struct_weight, cfg.order_weight, cfg.type_weight)
    kandidater = [("cfg nu", nu)]
    if a.satt:
        kandidater.append(("--satt", tuple(a.satt)))
    kandidater += [
        ("gammal (1 / 0.5)", (0.5, 0.5, 0.5)),
        ("försiktig",        (0.5, 1.0, 2.0)),
        ("mitten",           (0.5, 1.5, 3.0)),
        ("aggressiv",        (0.5, 2.0, 4.0)),
    ]

    print(f"\n{'vikter (struct/order/type)':<30}" +
          "".join(f"{f:>10}" for f in FIELDS))
    sedda = set()
    for namn, (sw, ow, tw) in kandidater:
        nyckel = (sw, ow, tw)
        if nyckel in sedda:
            continue
        sedda.add(nyckel)
        lika = [n for n, v in kandidater[1:] if tuple(v) == nyckel and n != namn]
        etikett = f"{namn}{' = ' + lika[0] if lika else ''}  {sw}/{ow}/{tw}"
        andelar = budget(masker, [1.0, tw, ow, sw])   # num, type, order, grammar
        # FIELDS-ordningen är num, type, order, grammar -- samma som masker
        print(f"{etikett:<30}" + "".join(f"{x:>10.1%}" for x in andelar))

    print("\nRaden 'gammal' är uppställningen som gav de frusna loss_order-kurvorna:")
    print("type 5,2 % för BÅDA typvalen, alltså 2,6 % för ORDER:s FIXED/RANDOM ensamt.")
    print("Sikta på att type och order tillsammans är jämförbara med num utan att")
    print("tränga undan den -- num fungerar redan (level_precision 0,94).")
    print()
    print("En varning om grammar: fältet är INTE homogent. Det innehåller både")
    print("LEVELS/END, som modellen lärt sig direkt, och beslutet att sluta räkna")
    print("upp nivåer (ORDER-tokenet i stället för ett nivånamn till). Det senare")
    print("är n_levels_ok, som ligger på 0,67. Sänk därför inte struct_weight för")
    print("att frigöra budget -- det billiga och det svåra sitter i samma vikt.")


if __name__ == "__main__":
    main()
