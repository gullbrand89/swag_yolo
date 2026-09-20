"""
Cross-entropy med ordinal smoothing på numeriska token och fältviktning,
plus hjälp-lossen på encodern (aux_loss).
"""
import torch
import torch.nn.functional as F

from .config import cfg
from .vocab import (IS_NUM, LEVEL_NAMES, PAD, RANGE_HI, RANGE_LO, SIGMA, TOK2ID,
                   VOCAB)

_ORDER, _DWELL, _END = TOK2ID["ORDER"], TOK2ID["DWELL"], TOK2ID["END"]
_TYPE = torch.tensor([TOK2ID["FIXED"], TOK2ID["RANDOM"], TOK2ID["RANGE"]])
# formatet per tillstånd: typvalen är RANGE/INF/* (dwellens sort, komponentens sort)
_TILLSTAND = cfg.post_format == "tillstand"
if _TILLSTAND:
    _S = TOK2ID["S"]
    _TYPE_T = torch.tensor([TOK2ID["RANGE"], TOK2ID["INF"], TOK2ID["*"]])
_LVL0, _LVL1 = TOK2ID[LEVEL_NAMES[0]], TOK2ID[LEVEL_NAMES[-1]]


def ordinal_targets(tgt):
    """
    Mjukt mål för numeriska token: en diskretiserad gaussian över grannvärden.

    Bredden tas per token ur vocab.SIGMA, så att nivåbins och dwelltider kan ha olika
    utjämning. Utsmetningen klamras till den egna rymden via RANGE_LO/RANGE_HI --
    massa som läckte över till den andra rymden vore inte "nästan rätt" utan fel
    sorts svar.

    Golvet för korsentropin är målfördelningens entropi, ungefär ln(sigma) + 1.42 för
    sigma >= 1. Sigma = 0 ger ett rent one-hot-mål och golvet noll.
    """
    B, T = tgt.shape
    V = len(VOCAB)
    w = cfg.smooth_width
    dev = tgt.device

    soft = torch.zeros(B, T, V, device=dev)
    soft.scatter_(2, tgt[..., None], 1.0)

    isnum = IS_NUM.to(dev)[tgt] & (tgt != PAD)
    if w > 0 and isnum.any():
        idx = tgt[isnum]
        lo, hi = RANGE_LO.to(dev)[idx], RANGE_HI.to(dev)[idx]
        sig = SIGMA.to(dev)[idx].clamp(min=1e-6)
        rows = torch.zeros(idx.numel(), V, device=dev)
        ar = torch.arange(idx.numel(), device=dev)
        for off in range(-w, w + 1):
            j = idx + off
            ok = (j >= lo) & (j <= hi)
            rows[ar[ok], j[ok]] += torch.exp(-0.5 * (off / sig[ok]) ** 2)
        soft[isnum] = rows / rows.sum(-1, keepdim=True)
    return soft


def field_masks(tgt):
    """
    Delar upp facit-positionerna i FYRA disjunkta fält som täcker allt utom PAD:
      num     : B- och D-token (nivåbins och dwelltider)
      type    : FIXED/RANDOM efter ORDER och FIXED/RANGE efter DWELL
      order   : nivånamnen inne i ORDER-blocket
      grammar : allt annat (LEVELS, ORDER, DWELL, END, INF, nivånamn i definitionerna)

    `type` bröts ut ur de två andra därför att typtokenen är de enda positioner
    där ETT fel gör hela posten fel. Ett bin som hamnar en bin bredvid kostar lite
    och syns i level_precision; ett FIXED som skulle ha varit RANDOM nollar exact.
    De hör inte hemma i samma vikt som allt annat.

    Tokenen FIXED, RANDOM och RANGE förekommer bara som typval, så masken behöver
    inte villkoras på var i posten de står. Den tidigare versionen villkorade på
    in_order för att hålla DWELL:s typval utanför `order`; nu ligger båda i `type`
    i stället, vilket är vad de är.

    OBS för loggen: `order` betyder inte samma sak som i körningar före den här
    ändringen -- typvalen räknas inte längre in. Kurvorna går inte att lägga
    ovanpå varandra rakt av.
    """
    valid = tgt != PAD
    dev = tgt.device
    isnum = IS_NUM.to(dev)[tgt] & valid
    is_lvl = (tgt >= _LVL0) & (tgt <= _LVL1)
    if _TILLSTAND:
        # formatet per tillstånd: num = B/D (bins, dwell, SEEN), type = RANGE/INF/*,
        # order = nivånamnen som komponenter (efter första S), grammar = resten
        # (LEVELS, nivånamnen i ordlistan, S, SEEN, END)
        in_states = torch.cumsum((tgt == _S).long(), 1) > 0
        typ = valid & torch.isin(tgt, _TYPE_T.to(dev))
        order = valid & in_states & is_lvl
        grammar = valid & ~isnum & ~typ & ~order
        return isnum, typ, order, grammar
    # inne i ORDER-blocket: efter ORDER, före DWELL/END (kumulativt per rad)
    after_order = torch.cumsum((tgt == _ORDER).long(), 1) > 0
    before_end = torch.cumsum(((tgt == _DWELL) | (tgt == _END)).long(), 1) == 0
    in_order = after_order & before_end

    typ = valid & torch.isin(tgt, _TYPE.to(dev))
    order = valid & in_order & is_lvl & ~typ
    grammar = valid & ~isnum & ~typ & ~order
    return isnum, typ, order, grammar


FIELDS = ("num", "type", "order", "grammar")


def token_weights(tgt):
    """
    Lossvikt per position. Noll på PAD, så vikten bär också masken.

    Vikterna speglar vad ett fel KOSTAR, inte hur många token fältet har:
      num      1.0                 ett bin fel är ett litet fel
      type     cfg.type_weight     ett typfel nollar exact
      order    cfg.order_weight    fel ordning nollar exact, men fältet är långt
      grammar  cfg.struct_weight   den är redan lärd (parsed ~ 1.0)

    Den gamla uppställningen var num 1.0 och allt annat struct_weight. Då fick de
    två typvalen tillsammans 2,6 % av gradienten medan de numeriska fick 51,5 %,
    och loss_order stod stilla i 16 000 steg medan loss_num fortsatte falla.
    Kör `python -m tools.lossvikt` för budgeten de nuvarande värdena ger.
    """
    isnum, typ, order, grammar = field_masks(tgt)
    w = torch.zeros(tgt.shape, dtype=torch.float, device=tgt.device)
    w[isnum] = 1.0
    w[typ] = cfg.type_weight
    w[order] = cfg.order_weight
    w[grammar] = cfg.struct_weight
    return w


def loss_fn(logits, tgt_out):
    logp = F.log_softmax(logits, -1)
    per_tok = -(ordinal_targets(tgt_out) * logp).sum(-1)
    w = token_weights(tgt_out)
    return (per_tok * w).sum() / w.sum().clamp(min=1e-6)


@torch.no_grad()
def loss_by_field(logits, tgt_out):
    """
    -> dict med medel-loss per fält, för loggning.

    Dict och inte tuple: fälten har blivit fler en gång och lär bli det igen, och
    en anropare som packar upp positionellt tappar då tyst det sista fältet.
    """
    logp = F.log_softmax(logits, -1)
    per_tok = -(ordinal_targets(tgt_out) * logp).sum(-1)
    return {namn: ((per_tok * m).sum() / m.sum().clamp(min=1)).item()
            for namn, m in zip(FIELDS, field_masks(tgt_out))}


@torch.no_grad()
def num_acc(logits, tgt_out, tol=0):
    """Andel numeriska token där argmax ligger inom `tol` steg från facit, inom
    samma tokenrymd. Det måttet -- inte lossen -- är framstegssignalen: med ett
    utjämnat mål straffas en modell som blir mer bestämd än målfördelningen även
    när argmax är helt rätt."""
    isnum, _, _, _ = field_masks(tgt_out)
    pred = logits.argmax(-1)
    dev = tgt_out.device
    same = (RANGE_LO.to(dev)[pred] == RANGE_LO.to(dev)[tgt_out]) & IS_NUM.to(dev)[pred]
    ok = isnum & same & ((pred - tgt_out).abs() <= tol)
    return (ok.sum() / isnum.sum().clamp(min=1)).item()


# ------------------------------------------------------------------ hjälp-loss
AUX_IGNORE = -100          # samma värde som i data.py och all_emitters.py
_AUX_HEADS = (("new", "aux_new"), ("pos", "aux_pos"), ("merge", "aux_merge"),
              ("cyc", "aux_cyc"))


def aux_loss(aux, src):
    """
    Hjälp-loss på encodern: vanlig cross-entropy per puls för vart och ett av huvudena
    i model.AuxHeads, mot målen som data._pack lagt i src. Positioner med AUX_IGNORE
    (utfyllnad, första pulsen för `new`, INF-emittrar för `pos`) räknas inte.

    -> (loss, stats)
       loss  : medel över de huvuden som hade minst ett giltigt mål. Multipliceras
               med cfg.aux_weight i train.py och läggs till postens loss.
       stats : dict för loggen
                 aux_new_acc     träffsäkerhet över alla pulser
                 aux_new_recall  andel av de VERKLIGA besöksstarterna som hittas.
                                 Det är den här som säger något: besöksstarter är
                                 sällsynta, så acc blir hög även om alla missas.
                 aux_pos_acc     exakt position i uppehållet
                 aux_merge_acc   träffsäkerhet över alla pulser
                 aux_merge_recall andel av de ihopslagna intervallen som känns igen
    """
    total, n, stats = 0.0, 0, {}
    for name, key in _AUX_HEADS:
        if name not in aux:
            continue                              # huvudet är avstängt i cfg
        tgt = src[key]
        valid = tgt != AUX_IGNORE
        if not bool(valid.any()):
            continue
        lg, t = aux[name][valid].float(), tgt[valid]
        l = F.cross_entropy(lg, t)
        total, n = total + l, n + 1
        with torch.no_grad():
            pred = lg.argmax(-1)
            stats[f"aux_{name}_acc"] = (pred == t).float().mean().item()
            if name in ("new", "merge"):
                posi = t > 0
                if bool(posi.any()):
                    stats[f"aux_{name}_recall"] = (pred[posi] == t[posi]).float().mean().item()
    # Räknehuvudet: ett mål per sekvens. Läggs till som ett eget led med egen vikt,
    # inte som ett fjärde medelvärde -- de tre per-puls-huvudena har hundratals mål
    # per sekvens och skulle annars dränka det.
    for name, key, vikt in (("count", "aux_count", cfg.aux_count_weight),
                            ("period", "aux_period", cfg.aux_period_weight)):
        if name not in aux:
            continue
        t = src[key]
        lg_all = aux[name]
        if lg_all.dim() == 3:
            # per-puls-huvud (period_per_pulse): sekvensens mål gäller varje giltig puls
            t = t[:, None].expand(-1, lg_all.size(1))
            valid = (t != AUX_IGNORE) & ~src["mask"]
        else:
            valid = t != AUX_IGNORE
        if bool(valid.any()):
            lg = lg_all[valid].float()
            lc = F.cross_entropy(lg, t[valid])
            total = total + vikt * lc
            n = max(n, 1)
            with torch.no_grad():
                pred = lg.argmax(-1)
                stats[f"aux_{name}_acc"] = (pred == t[valid]).float().mean().item()
                stats[f"aux_{name}_mae"] = (pred - t[valid]).abs().float().mean().item()
    if n == 0:
        return torch.zeros((), device=src["bins"].device), stats
    return total / n, stats
