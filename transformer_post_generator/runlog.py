"""Körningsmapp, loggar, fältvisa mått."""
import csv, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import cfg
from .vocab import ids_to_tokens
from .labels import parse


def _levels_pr(p_levels, t_levels, tol_us):
    tl, pl = list(t_levels), list(p_levels)
    hit = 0
    for x in tl:
        if not pl: break
        j = min(range(len(pl)), key=lambda k: abs(pl[k] - x))
        if abs(pl[j] - x) <= tol_us:
            hit += 1; pl.pop(j)
    return hit / max(1, len(p_levels)), hit / max(1, len(tl))


def compare_tillstand(pred, true, tol_us=2.0):
    """
    Mått på formatet per tillstånd. exact är här det som räknas: teckenidentisk post
    = identisk emittermodell (tools.post_till_modell). Resten lokaliserar felet:
      n_levels_ok, level_precision/recall   nivåordlistan
      n_states_ok      antal tillstånd (S) rätt
      states_ok        andel tillstånd (komponent OCH dwell) rätt, givet rätt antal
      comp_ok          andel komponenter rätt, givet rätt antal
      dwell_ok         andel dwell rätt, givet rätt antal
      seen_ok          SEEN rätt (1 om ingen SEEN i facit)
    """
    from .labels_tillstand import parse as parse_t
    m = dict(exact=float(pred == true), parsed=0.0)
    try:
        p, t = parse_t(pred), parse_t(true)
    except Exception:
        return m
    m["parsed"] = 1.0
    m["n_levels_ok"] = float(len(p["levels"]) == len(t["levels"]))
    m["level_precision"], m["level_recall"] = _levels_pr(p["levels"], t["levels"], tol_us)
    ps, ts = p["states"], t["states"]
    m["n_states_ok"] = float(len(ps) == len(ts))
    if len(ps) == len(ts):
        m["states_ok"] = float(np.mean([a == b for a, b in zip(ps, ts)]))
        m["comp_ok"] = float(np.mean([a[0] == b[0] for a, b in zip(ps, ts)]))
        m["dwell_ok"] = float(np.mean([a[1] == b[1] for a, b in zip(ps, ts)]))
    else:
        m["states_ok"] = m["comp_ok"] = m["dwell_ok"] = 0.0
    m["seen_ok"] = float(p["seen"] == t["seen"])
    return m


def compare(pred, true, tol_us=2.0):
    if cfg.post_format == "tillstand":
        return compare_tillstand(pred, true, tol_us)
    m = dict(exact=float(pred == true), parsed=0.0)
    try:
        p, t = parse(pred), parse(true)
    except Exception:
        return m
    m["parsed"] = 1.0
    m["n_levels_ok"] = float(len(p["levels"]) == len(t["levels"]))
    tl, pl = list(t["levels"]), list(p["levels"])
    hit = 0
    for x in tl:
        if not pl: break
        j = min(range(len(pl)), key=lambda k: abs(pl[k] - x))
        if abs(pl[j] - x) <= tol_us:
            hit += 1; pl.pop(j)
    m["level_recall"] = hit / max(1, len(tl))
    m["level_precision"] = hit / max(1, len(p["levels"]))
    m["order_type_ok"] = float(p["order_fixed"] == t["order_fixed"])
    m["length_type_ok"] = float(p["length_fixed"] == t["length_fixed"])
    tL = [x for x in t["lengths"] if x is not None]
    pL = [x for x in p["lengths"] if x is not None]
    m["n_lengths_ok"] = float(len(pL) == len(tL))
    if tL and len(pL) == len(tL):
        d = np.array(pL) - np.array(tL)
        m["len_bias"] = float(d.mean())
        m["len_within1"] = float((np.abs(d) <= 1).mean())
        m["len_big"] = float((np.abs(d) > 0.25 * np.maximum(1, np.array(tL))).mean())
    return m


@torch.no_grad()
def evaluate(model, eval_sets, device, collate, run=None, batch=256, max_new=None):
    model.eval(); results = {}
    for name, pairs in eval_sets.items():
        loader = DataLoader(pairs, batch_size=batch, shuffle=False, collate_fn=collate)
        rows, preds = [], []
        for src, _, tgt_out in loader:
            src = {k: v.to(device) for k, v in src.items()}
            out = model.greedy(src, max_new=max_new or cfg.eval_max_new).cpu()
            for i in range(out.size(0)):
                pt, tt = ids_to_tokens(out[i]), ids_to_tokens(tgt_out[i])
                rows.append(compare(pt, tt)); preds.append((tt, pt))
        keys = set().union(*(r.keys() for r in rows))
        results[name] = {k: float(np.mean([r[k] for r in rows if k in r])) for k in sorted(keys)}
        if run is not None:
            run.save_preds(name, preds)
    model.train()
    return results


class RunLog:
    def __init__(self, root, config):
        self.dir = Path(root) / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True)
        with open(self.dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
        self._f = open(self.dir / "train.csv", "w", newline="")
        self._w = None; self._evals = []

    def log(self, **kw):
        if self._w is None:
            self._w = csv.DictWriter(self._f, fieldnames=list(kw)); self._w.writeheader()
        self._w.writerow(kw); self._f.flush()

    def log_eval(self, step, metrics):
        self._evals.append(dict(step=step, **{f"{s}/{k}": v for s, m in metrics.items() for k, v in m.items()}))
        with open(self.dir / "eval.json", "w") as f:
            json.dump(self._evals, f, indent=1)
        print(f"--- eval @ {step}")
        for s, m in metrics.items():
            print(f"  {s}: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

    def save_preds(self, name, preds):
        with open(self.dir / f"preds_{name}.txt", "w") as f:
            for tt, pt in preds:
                f.write("T " + " ".join(tt) + "\nP " + " ".join(pt) + "\n\n")

    def save_model(self, model):
        torch.save(model.state_dict(), self.dir / "model.pt")
