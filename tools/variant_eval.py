"""
Evalmåtten per VARIANT och bortfallsnivå, på en sparad checkpoint.

    python -m tools.variant_eval runs/<mapp> --n 200 --drops 0.0 0.05 0.1 0.2
    python -m tools.variant_eval runs/<mapp> --matt exact n_levels_ok n_states_ok

Samma emittrar i alla bortfallsnivåer (make_variant_eval_sets), samma mått som
train/reeval (runlog.compare). Skriver en tabell per mått och variant_eval.json
i körningsmappen.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=200, help="emittrar per variant")
    ap.add_argument("--drops", type=float, nargs="+", default=(0.0, 0.05, 0.1, 0.2))
    ap.add_argument("--matt", nargs="+", default=("exact", "n_levels_ok", "n_states_ok", "states_ok"),
                    help="mått att tabellera (alla sparas i json)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    katalog = Path(a.run)
    run_cfg = json.load(open(katalog / "config.json"))
    from transformer_post_generator.config import cfg
    from tools.reeval import _lagg_pa_cfg
    _lagg_pa_cfg(cfg, run_cfg)

    import torch
    from transformer_post_generator.all_emitters import make_variant_eval_sets
    from transformer_post_generator.data import collate
    from transformer_post_generator.model import build_model
    from transformer_post_generator.runlog import evaluate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    model.load_state_dict(torch.load(katalog / "model.pt", map_location=device))
    model.eval()

    sets = make_variant_eval_sets(a.n, 1, p_drops=a.drops, seed=a.seed)
    res = evaluate(model, sets, device, collate, batch=a.batch)

    varianter = sorted({k.split("/")[0] for k in res})
    for m in a.matt:
        if not any(m in v for v in res.values()):
            continue
        print(f"\n{m}")
        print(f"{'variant':<14}" + "".join(f"{p:>8.2f}" for p in a.drops))
        for v in varianter:
            print(f"{v:<14}" + "".join(f"{res[f'{v}/drop_{p:.2f}'].get(m, float('nan')):>8.3f}" for p in a.drops))
        print(f"{'medel':<14}" + "".join(
            f"{sum(res[f'{v}/drop_{p:.2f}'].get(m, 0) for v in varianter) / len(varianter):>8.3f}" for p in a.drops))

    with open(katalog / "variant_eval.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nskrev {katalog / 'variant_eval.json'}")


if __name__ == "__main__":
    main()
