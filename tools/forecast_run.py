"""
Prediktionsunderlag: modellen ser de första n_obs pulserna av en längre signal och
skriver sin post. Fortsättningen sparas som facit för prediktionen.

    python -m tools.forecast_run runs/20260920_092634
    python -m tools.forecast_run runs/20260920_092634 --n 150 --n-obs 512 --n-pred 128

Skriver <run>/forecast_<tag>.json med en post per signal:
    variant, pri (hela signalen, ren, n_obs + n_pred pulser), n_obs,
    tokens_pred (modellens post på pri[:n_obs]), tokens_true (facit för emittern,
    ankrat vid fönstret pri[:n_obs]).

Resten -- konvertering till emittermodell, utrullning av lambda framåt och
jämförelse mot pri[n_obs:] -- görs i tools/forecast_eval.py och behöver inte torch.

Bara varianter med deterministisk fortsättning: stagger och ds_det_det. För slumpad
ordning eller slumpad dwell är prediktionen en fördelning, inte en följd.
"""
import argparse
import json
from pathlib import Path

import numpy as np

VARIANTER = ("stagger", "ds_det_det")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=150, help="emittrar per variant")
    ap.add_argument("--n-obs", type=int, default=512, help="pulser modellen ser (träningsfönstret)")
    ap.add_argument("--n-pred", type=int, default=128, help="pulser att predicera")
    ap.add_argument("--drop", type=float, default=0.0, help="bortfall i observationen (0 = rent)")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    katalog = Path(a.run)
    run_cfg = json.load(open(katalog / "config.json"))
    from transformer_post_generator.config import cfg
    from tools.reeval import _lagg_pa_cfg
    _lagg_pa_cfg(cfg, run_cfg)

    import torch
    from transformer_post_generator import all_emitters as ae
    from transformer_post_generator.data import collate, label_to_tokens, make_channels
    from transformer_post_generator.model import build_model
    from transformer_post_generator.vocab import ids_to_tokens

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    model.load_state_dict(torch.load(katalog / "model.pt", map_location=device))
    model.eval()

    # längre signal än fönstret: generatorn får göra n_obs + n_pred pulser, modellen
    # ser bara de första n_obs. Facitet (ankrat vid fönstrets start) påverkas inte.
    ae.GEN.n_pulses = a.n_obs + a.n_pred

    poster = []
    for v in VARIANTER:
        rng = np.random.default_rng([a.seed, ae.VARIANTS.index(v)])
        data = ae.create_emitter_data(a.n, 1, a.drop, None, rng, only=[v])
        rows = []
        for seqs, lab in data:
            pri = np.asarray(seqs[0], dtype=float)
            tok_true = label_to_tokens(lab, 0)
            # observationen: de första n_obs INTERVALLEN. Med bortfall är antalet
            # intervall färre än pulser; vi tar det som finns upp till n_obs.
            rows.append((pri, pri[:a.n_obs], tok_true, lab["variant"]))
        for i0 in range(0, len(rows), a.batch):
            chunk = rows[i0:i0 + a.batch]
            src, _, _ = collate([(make_channels(obs), tt) for _, obs, tt, _ in chunk])
            src = {k: t.to(device) for k, t in src.items()}
            with torch.no_grad():
                out = model.greedy(src, max_new=cfg.eval_max_new).cpu()
            for j, (pri, obs, tt, var) in enumerate(chunk):
                poster.append(dict(variant=var, n_obs=int(len(obs)), pri=pri.tolist(),
                                   tokens_pred=ids_to_tokens(out[j]), tokens_true=tt))
        print(f"{v}: {len(rows)} signaler")

    tag = a.tag or (f"drop{a.drop:.2f}" if a.drop > 0 else "ren")
    ut = katalog / f"forecast_{tag}.json"
    with open(ut, "w") as f:
        json.dump(dict(n_obs=a.n_obs, n_pred=a.n_pred, drop=a.drop, seed=a.seed,
                       poster=poster), f)
    print(f"skrev {ut} ({len(poster)} poster)")


if __name__ == "__main__":
    main()
