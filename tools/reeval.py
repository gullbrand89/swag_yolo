"""
Kör om evalen på en sparad checkpoint, utan att träna.

    python -m tools.reeval runs/20260919_121852
    python -m tools.reeval runs/20260919_121852 --no-constrain     # som förut, för A/B
    python -m tools.reeval runs/20260919_121852 --tag test

Varför
------
Villkorad avkodning (cfg.constrain_levels) ändrar bara hur token VÄLJS, inte
vad modellen lärt sig. Effekten går alltså att mäta på en modell som redan
finns, på minuter i stället för en timme. Samma sak för varje annan ändring i
generate(): mät den här först, träna om bara om det behövs.

Konfigurationen läses ur körningens config.json och läggs på cfg INNAN paketet
importeras, så att vokabulär, kanaler och modellform blir de som checkpointen
tränades med. Flaggor som inte fanns när körningen gjordes (t.ex. use_recur i
äldre körningar) sätts av -- saknas nyckeln var featuren inte påslagen.

Resultat: eval_<tag>.json och preds_<tag>_<drop>.txt i körningsmappen. Den
ursprungliga eval.json rörs inte.
"""
import argparse
import json
from dataclasses import fields
from pathlib import Path


# Featureflaggor som påverkar modellens form. Saknas de i config.json fanns
# de inte när körningen gjordes, och då måste de vara AV för att model.pt ska
# gå att ladda.
_FORMFLAGGOR_AV_OM_SAKNAS = {"use_recur": False, "use_drop_flag": False,
                              "use_counter": True, "aux_count": False, "aux_cyc": False,
                              "use_visit": False, "use_seg_pos": False,
                              "period_per_pulse": False}


def _lagg_pa_cfg(cfg, run_cfg):
    namn = {f.name for f in fields(cfg)}
    satta, hoppade = [], []
    for k, v in run_cfg.items():
        if k not in namn:
            hoppade.append(k)                  # mode, steps, lr, batch: inte cfg-fält
            continue
        if isinstance(getattr(cfg, k), tuple) and isinstance(v, list):
            v = tuple(v)
        setattr(cfg, k, v); satta.append(k)
    for k, av in _FORMFLAGGOR_AV_OM_SAKNAS.items():
        if k not in run_cfg:
            setattr(cfg, k, av); satta.append(f"{k}={av} (saknades)")
    return satta, hoppade


class _Sparare:
    """Det evaluate() behöver av en RunLog: save_preds."""
    def __init__(self, katalog, tag):
        self.katalog, self.tag = katalog, tag
    def save_preds(self, name, preds):
        with open(self.katalog / f"preds_{self.tag}_{name}.txt", "w") as f:
            for tt, pt in preds:
                f.write("T " + " ".join(tt) + "\nP " + " ".join(pt) + "\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="körningsmapp med config.json och model.pt")
    ap.add_argument("--no-constrain", action="store_true",
                    help="stäng av villkorad avkodning (cfg.constrain_levels = False)")
    ap.add_argument("--count", dest="count", action="store_true", default=None,
                    help="slå på räknevillkoret (cfg.constrain_count)")
    ap.add_argument("--no-count", dest="count", action="store_false",
                    help="slå av räknevillkoret")
    ap.add_argument("--count-source", choices=("model", "cluster"), default=None,
                    help="varifrån K kommer: räknehuvudet (model) eller antal bin-kluster "
                         "(cluster; exakt på ren data, överskattar under bortfall). "
                         "Saknar checkpointen huvudet blir det cluster oavsett.")
    ap.add_argument("--period", dest="period", action="store_true", default=None,
                    help="slå på periodvillkoret i ORDER FIXED (cfg.constrain_period)")
    ap.add_argument("--no-period", dest="period", action="store_false",
                    help="slå av periodvillkoret")
    ap.add_argument("--tag", default=None,
                    help="namn på utfilerna; default 'reeval' eller 'reeval_nc'")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    katalog = Path(a.run)
    run_cfg = json.load(open(katalog / "config.json"))

    # cfg först, sedan resten av paketet -- vocab bygger sin tabell vid import
    from transformer_post_generator.config import cfg
    satta, hoppade = _lagg_pa_cfg(cfg, run_cfg)
    cfg.constrain_levels = not a.no_constrain
    if a.count is not None:
        cfg.constrain_count = a.count
    if a.count_source is not None:
        cfg.count_source = a.count_source
    if a.period is not None:
        cfg.constrain_period = a.period
    tag = a.tag or ("reeval_nc" if a.no_constrain
                    else (f"count_{cfg.count_source}" if cfg.constrain_count else "reeval"))

    import torch
    from transformer_post_generator.data import collate, make_eval_sets
    from transformer_post_generator.model import build_model
    from transformer_post_generator.runlog import evaluate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    state = torch.load(katalog / "model.pt", map_location=device)
    model.load_state_dict(state)
    print(f"{katalog}: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
          f"{device}, constrain_levels={cfg.constrain_levels}, "
          f"constrain_count={cfg.constrain_count} ({cfg.count_source}), "
          f"constrain_period={cfg.constrain_period}, aux_count={cfg.aux_count}, "
          f"aux_cyc={cfg.aux_cyc}, use_recur={cfg.use_recur}")
    if hoppade:
        print(f"  (ej cfg-fält, hoppade över: {', '.join(hoppade)})")

    metrics = evaluate(model, make_eval_sets(), device, collate,
                       run=_Sparare(katalog, tag), batch=a.batch)
    print(f"--- eval ({tag})")
    for s_, m in metrics.items():
        print(f"  {s_}: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    with open(katalog / f"eval_{tag}.json", "w") as f:
        json.dump(metrics, f, indent=1)
    print(f"\nskrev eval_{tag}.json och preds_{tag}_*.txt i {katalog}")


if __name__ == "__main__":
    main()
