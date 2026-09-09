"""Cross-entropy med ordinal smoothing på N-tokens och fältviktning."""
import math
import torch
import torch.nn.functional as F
from config import cfg
from vocab import VOCAB, TOK2ID, PAD, IS_NUM, NUM_START, NUM, LEVEL_NAMES

_ORDER, _DWELL, _END = TOK2ID["ORDER"], TOK2ID["DWELL"], TOK2ID["END"]
_TYPE = torch.tensor([TOK2ID["FIXED"], TOK2ID["RANDOM"], TOK2ID["RANGE"]])
_LVL0, _LVL1 = TOK2ID[LEVEL_NAMES[0]], TOK2ID[LEVEL_NAMES[-1]]


def ordinal_targets(tgt, sigma=1.0):
    B, T = tgt.shape; V = len(VOCAB); w = cfg.smooth_width
    soft = torch.zeros(B, T, V, device=tgt.device)
    soft.scatter_(2, tgt[..., None], 1.0)
    isnum = IS_NUM.to(tgt.device)[tgt] & (tgt != PAD)
    if w > 0 and isnum.any():
        idx = tgt[isnum]
        rows = torch.zeros(idx.numel(), V, device=tgt.device)
        ar = torch.arange(idx.numel(), device=tgt.device)
        for off in range(-w, w + 1):
            j = idx + off
            ok = (j >= NUM_START) & (j < NUM_START + len(NUM))
            rows[ar[ok], j[ok]] += math.exp(-0.5 * (off / sigma) ** 2)
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
      num     : N-tokens (nivåbins, längder, antal)
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
