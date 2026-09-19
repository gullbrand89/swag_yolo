"""
Diagnos av cykelhuvudena och ORDER-blocket, per variant, på en sparad checkpoint.

    python -m tools.cyc_diag runs/<mapp>
    python -m tools.cyc_diag runs/<mapp> --n 200 --drops 0.0 0.2

Frågan
------
cyc planade ut kring 0.5-0.65 i träningsloggen. Två helt olika orsaker ger den
siffran, och de kräver olika åtgärder:

  A) encodern har FASEN men inte ANKARET: den vet var i cykeln pulsen är, relativt,
     men inte var den kanoniska rotationen (lexikografiskt minsta) börjar. Då är
     predikterna rätt upp till en KONSTANT förskjutning per sekvens.
     -> cyc_shift (rätt upp till per-sekvens-rotation) hög, cyc låg.
     Åtgärd: byt ankare i facit till något lokalt avläsbart, t.ex. fönstrets första puls.

  B) encodern har inte periodiciteten alls.
     -> både cyc och cyc_shift låga.
     Åtgärd: kanaler/kapacitet, inte facit.

Samma uppdelning görs för det avkodade ORDER-blocket: rätt teckenidentiskt (order),
och rätt upp till rotation (order_rot). Skiljer sig de två har avkodaren cykeln men
inte startpunkten.

Kolumner per variant och bortfall:
  n         antal sekvenser med fast ordning (bara de har cyc-facit)
  cyc       per-puls-träffsäkerhet, kanonisk position
  cyc_shift per-puls-träffsäkerhet efter bästa konstanta förskjutning per sekvens (mod P)
  seq_cyc   andel sekvenser där ALLA pulser är rätt
  seq_shift andel sekvenser där alla pulser är rätt upp till förskjutning
  P_acc     periodhuvudets träffsäkerhet;  P_mae medelfel i antal
  order     ORDER-blocket teckenidentiskt (greedy avkodning med gällande villkor)
  order_rot ORDER-blocket rätt upp till rotation
  order|cyc ORDER rätt givet att seq_cyc var rätt -- läser avkodaren av huvudet?
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _order_block(tokens):
    if "ORDER" not in tokens:
        return None
    i = tokens.index("ORDER") + 1
    if i >= len(tokens) or tokens[i] != "FIXED":
        return None
    j = i + 1
    while j < len(tokens) and tokens[j] != "DWELL":
        j += 1
    return tuple(tokens[i + 1:j])


def _is_rotation(a, b):
    if a is None or b is None or len(a) != len(b) or not a:
        return False
    return any(a[k:] + a[:k] == b for k in range(len(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=200, help="emittrar per variant")
    ap.add_argument("--drops", type=float, nargs="+", default=(0.0, 0.2))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    katalog = Path(a.run)
    run_cfg = json.load(open(katalog / "config.json"))
    from transformer_post_generator.config import cfg
    from tools.reeval import _lagg_pa_cfg
    _lagg_pa_cfg(cfg, run_cfg)

    import torch
    from transformer_post_generator import all_emitters as ae
    from transformer_post_generator.data import (collate, label_to_tokens, as_signals,
                                                 make_channels, n_levels_of, n_order_of,
                                                 AUX_IGNORE)
    from transformer_post_generator.model import build_model
    from transformer_post_generator.vocab import ids_to_tokens

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    model.load_state_dict(torch.load(katalog / "model.pt", map_location=device))
    model.eval()
    har_cyc = model.aux is not None and getattr(model.aux, "cyc", None) is not None
    print(f"{katalog}: aux_cyc={cfg.aux_cyc} (huvud finns: {har_cyc}), "
          f"constrain_period={cfg.constrain_period}, {device}")
    seed = cfg.eval_seed if a.seed is None else a.seed

    rader = []
    for v in ae.VARIANTS:
        if ae.GEN.weights[v] <= 0:
            continue
        for p in a.drops:
            rng = np.random.default_rng([seed, ae.VARIANTS.index(v)])
            data = ae.create_emitter_data(a.n, 1, p, None, rng, only=[v], with_aux=True)
            pairs = []
            for seqs, lab in data:
                for i, (s, ax) in enumerate(zip(as_signals(seqs), lab["aux"])):
                    tok = label_to_tokens(lab, i)
                    pairs.append((make_channels(s, ax, n_levels_of(tok), n_order_of(tok)), tok))

            st = defaultdict(list)
            for i0 in range(0, len(pairs), a.batch):
                src, _, tgt_out = collate(pairs[i0:i0 + a.batch])
                src = {k_: t.to(device) for k_, t in src.items()}
                with torch.no_grad():
                    mem = model.encode(src)
                    aux = model.aux(mem, src["mask"]) if model.aux is not None else {}
                    out = model.greedy(src, max_new=cfg.eval_max_new).cpu()
                tgt_cyc = src["aux_cyc"].cpu(); tgt_P = src["aux_period"].cpu()
                pred_cyc = aux["cyc"].argmax(-1).cpu() if "cyc" in aux else None
                pred_P = aux["period"].argmax(-1).cpu() if "period" in aux else None
                for i in range(out.size(0)):
                    tt = ids_to_tokens(tgt_out[i]); pt = ids_to_tokens(out[i])
                    P_true = int(tgt_P[i])
                    if P_true == AUX_IGNORE:
                        continue                      # slumpad ordning: inget cyc-facit
                    valid = tgt_cyc[i] != AUX_IGNORE
                    ob_t, ob_p = _order_block(tt), _order_block(pt)
                    order_ok = ob_t is not None and ob_p == ob_t
                    st["order"].append(order_ok)
                    st["order_rot"].append(_is_rotation(ob_p, ob_t))
                    if pred_P is not None:
                        st["P_acc"].append(int(pred_P[i]) == P_true)
                        st["P_mae"].append(abs(int(pred_P[i]) - P_true))
                    if pred_cyc is None or not bool(valid.any()):
                        continue
                    t = tgt_cyc[i][valid].numpy(); q = pred_cyc[i][valid].numpy()
                    acc = float((t == q).mean())
                    # bästa konstanta förskjutning mod P: har den fasen men inte ankaret?
                    shift = max(float((((t + s) % P_true) == q).mean()) for s in range(P_true))
                    st["cyc"].append(acc); st["cyc_shift"].append(shift)
                    st["seq_cyc"].append(acc == 1.0); st["seq_shift"].append(shift == 1.0)
                    if acc == 1.0:
                        st["order|cyc"].append(order_ok)
            if st["order"]:
                rader.append((v, p, {k_: (float(np.mean(x)) if x else float("nan"))
                                     for k_, x in st.items()}, len(st["order"])))

    kol = ("cyc", "cyc_shift", "seq_cyc", "seq_shift", "P_acc", "P_mae", "order", "order_rot", "order|cyc")
    print(f"\n{'variant':<14}{'drop':>5}{'n':>5}" + "".join(f"{k:>10}" for k in kol))
    for v, p, m, n in rader:
        print(f"{v:<14}{p:>5.2f}{n:>5}" + "".join(f"{m.get(k, float('nan')):>10.3f}" for k in kol))
    print("\nLäsning: cyc_shift >> cyc  => fasen finns, ankaret saknas (byt ankare i facit)."
          "\n         båda låga         => periodiciteten saknas i encodern."
          "\n         order_rot >> order => avkodaren har cykeln men fel startpunkt.")


if __name__ == "__main__":
    main()
