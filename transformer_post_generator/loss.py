"""Cross-entropy med ordinal smoothing på numeriska token och fältviktning."""
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
