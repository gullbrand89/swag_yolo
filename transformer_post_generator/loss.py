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


def loss_fn(logits, tgt_out):
    logp = F.log_softmax(logits, -1)
    per_tok = -(ordinal_targets(tgt_out) * logp).sum(-1)
    mask = tgt_out != PAD
    w = torch.where(IS_NUM.to(tgt_out.device)[tgt_out], 1.0, cfg.struct_weight)
    return (per_tok * w * mask).sum() / (w * mask).sum()


def field_masks(tgt):
    """
    Delar upp facit-positionerna i tre fält:
      num     : B- och D-token (nivåbins och dwelltider)
      order   : nivånamn inne i ORDER-blocket + FIXED/RANDOM-valen  (innehåll)
      grammar : allt annat (LEVELS, ORDER, DWELL, END, INF, nivånamn i definitionerna)
    """
    valid = tgt != PAD
    isnum = IS_NUM.to(tgt.device)[tgt] & valid
    is_lvl = (tgt >= _LVL0) & (tgt <= _LVL1)
    is_type = torch.isin(tgt, _TYPE.to(tgt.device))
    # inne i ORDER-blocket: efter ORDER, före DWELL/END (kumulativt per rad)
    after_order = torch.cumsum((tgt == _ORDER).long(), 1) > 0
    before_end = torch.cumsum(((tgt == _DWELL) | (tgt == _END)).long(), 1) == 0
    in_order = after_order & before_end
    # is_type villkoras på in_order, annars hamnar även DWELL:s FIXED/RANGE här och
    # `order` blandar ihop nivåordningen med om dwellen är fast eller slumpad.
    order = valid & in_order & (is_lvl | is_type)
    grammar = valid & ~isnum & ~order
    return isnum, order, grammar


@torch.no_grad()
def loss_by_field(logits, tgt_out):
    """-> (num, order, grammar) medel-loss per fält, för loggning."""
    logp = F.log_softmax(logits, -1)
    per_tok = -(ordinal_targets(tgt_out) * logp).sum(-1)
    out = []
    for m in field_masks(tgt_out):
        out.append(((per_tok * m).sum() / m.sum().clamp(min=1)).item())
    return tuple(out)


@torch.no_grad()
def num_acc(logits, tgt_out, tol=0):
    """Andel numeriska token där argmax ligger inom `tol` steg från facit, inom
    samma tokenrymd. Det måttet -- inte lossen -- är framstegssignalen: med ett
    utjämnat mål straffas en modell som blir mer bestämd än målfördelningen även
    när argmax är helt rätt."""
    isnum, _, _ = field_masks(tgt_out)
    pred = logits.argmax(-1)
    dev = tgt_out.device
    same = (RANGE_LO.to(dev)[pred] == RANGE_LO.to(dev)[tgt_out]) & IS_NUM.to(dev)[pred]
    ok = isnum & same & ((pred - tgt_out).abs() <= tol)
    return (ok.sum() / isnum.sum().clamp(min=1)).item()


# ------------------------------------------------------------------ hjälp-loss
AUX_IGNORE = -100          # samma värde som i data.py och all_emitters.py
_AUX_HEADS = (("new", "aux_new"), ("pos", "aux_pos"), ("merge", "aux_merge"))


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
    if n == 0:
        return torch.zeros((), device=src["bins"].device), stats
    return total / n, stats
