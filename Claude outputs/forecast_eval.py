"""
Prediktion ur emittermodellen: hur väl förklarar den signalens FORTSÄTTNING?

    python -m tools.forecast_eval runs/20260920_092634/forecast_ren.json

Läser posterna från tools/forecast_run.py. För varje signal:
  1. post + observation (pri[:n_obs]) -> emittermodell (tools.biblioteksformat)
  2. lambda rullas framåt från fönstrets sista besök: återstående dwell i det
     pågående tillståndet, sedan cykelns positioner i tur och ordning med sin
     dwell (vanligaste observerade), PRI = mu för positionens nivå
  3. jämförs mot pri[n_obs:], puls för puls

Samma sak görs med FACITPOSTEN i stället för modellens (oraklet). Skillnaden mellan
modell och orakel är transformerns bidrag till felet; oraklets eget fel är
konverterarens + utrullningens.

Mått per horisont h (pulser framåt): andel signaler där puls h ligger inom tol_us av
den verkliga. Per signal: "hel horisont rätt" = alla n_pred pulser inom tol, och
TOA-driften efter n_pred pulser (summan av felen, i µs).

Behöver inte torch. Skriver forecast_<tag>.png (exempel + sammanfattning) och
forecast_<tag>_summary.json bredvid indata.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _vanligast(stod, w):
    if not stod:
        return None
    return int(stod[int(np.argmax(w))])


def rulla_ut(r, n_pred):
    """
    -> (pred_pri: array(n_pred) eller None, metod: str)
    Utrullning av lambda framåt ur konverterarens resultat r (till_bibliotek).
    """
    th, mu = r["theta"], r["Phi"]["mu"]
    lam, c = th["lambda"], th["c"]
    if not lam or lam[-1] < 0 or any(m is None for m in mu):
        return None, "otilldelat slut"

    P = th["period"]
    if P is not None and th["cykel"] is not None:
        cykel = th["cykel"]                                  # [(nivå, stöd, w)] kanonisk
        pos = th["lambda_tillstand"][-1]
        niva_av = lambda p: cykel[p][0]
        dwell_av = lambda p: _vanligast(cykel[p][1], cykel[p][2])
        metod = "period ur lambda"
    elif th["cykel_post"] is not None and th["c_tillstand"] is not None:
        # ingen period avläst (t.ex. bortfall): postens ORDER-block som hypotes,
        # med fasen från _dwell_per_tillstand
        cykel_post, cs, ws = th["cykel_post"], th["c_tillstand"], th["w_tillstand"]
        P = len(cykel_post)
        # fasen: den förskjutning som får flest besök att stämma
        obs = lam
        tratt = [sum(1 for i, v in enumerate(obs) if cykel_post[(i + o) % P] == v) for o in range(P)]
        o = int(np.argmax(tratt))
        pos = (len(lam) - 1 + o) % P
        niva_av = lambda p: cykel_post[p]
        dwell_av = lambda p: _vanligast(cs[p], ws[p])
        metod = "postens cykel"
    else:
        return None, "ingen cykel"

    # reservdwell per nivå när positionen saknar observation
    c_stod, w_stod = th["c_stod"], th["w"]
    def dwell(p):
        d = dwell_av(p)
        if d is None:
            k = niva_av(p)
            d = _vanligast(c_stod[k], w_stod[k]) if c_stod and c_stod[k] else 1
        return max(1, d)

    ut = []
    # pågående tillstånd: återstående dwell
    rest = dwell(pos) - c[-1]
    ut += [mu[niva_av(pos)]] * max(0, rest)
    while len(ut) < n_pred:
        pos = (pos + 1) % P
        ut += [mu[niva_av(pos)]] * dwell(pos)
    return np.asarray(ut[:n_pred], dtype=float), metod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--tol-us", type=float, default=1.0)
    ap.add_argument("--n-exempel", type=int, default=4)
    a = ap.parse_args()

    from tools.biblioteksformat import till_bibliotek

    d = json.load(open(a.json))
    n_pred = d["n_pred"]
    katalog, tag = Path(a.json).parent, Path(a.json).stem.replace("forecast_", "")

    rader = []
    for post in d["poster"]:
        pri = np.asarray(post["pri"], dtype=float)
        n_obs = post["n_obs"]
        obs, sant = pri[:n_obs], pri[n_obs:n_obs + n_pred]
        rad = dict(variant=post["variant"], n_levels=(post["tokens_true"].index("ORDER") - 1) // 2)
        for namn, tok in (("modell", post["tokens_pred"]), ("orakel", post["tokens_true"])):
            try:
                r = till_bibliotek(obs, tok)
                pred, metod = rulla_ut(r, n_pred)
            except Exception as e:                       # oparsbar post etc.
                pred, metod = None, f"fel: {type(e).__name__}"
            if pred is None:
                fel = np.full(n_pred, np.inf)
            else:
                fel = np.abs(pred - sant[:len(pred)])
                if len(fel) < n_pred:
                    fel = np.concatenate([fel, np.full(n_pred - len(fel), np.inf)])
            rad[namn] = dict(pred=None if pred is None else pred.tolist(), metod=metod,
                             inom=(fel <= a.tol_us).tolist(),
                             helt=bool(np.all(fel <= a.tol_us)),
                             drift=float(np.sum(pred - sant)) if pred is not None and len(pred) == n_pred else None)
        rad["exact_post"] = post["tokens_pred"] == post["tokens_true"]
        rad["obs_svans"] = obs[-64:].tolist(); rad["sant"] = sant.tolist()
        rader.append(rad)

    # ---- sammanfattning
    summ = {}
    for v in sorted({r_["variant"] for r_ in rader}):
        rv = [r_ for r_ in rader if r_["variant"] == v]
        s = dict(n=len(rv), exact_post=float(np.mean([r_["exact_post"] for r_ in rv])))
        for namn in ("modell", "orakel"):
            inom = np.array([r_[namn]["inom"] for r_ in rv], dtype=float)   # (n, n_pred)
            s[namn] = dict(
                hel_horisont=float(np.mean([r_[namn]["helt"] for r_ in rv])),
                inom_per_h=inom.mean(0).tolist(),
                inom_h1=float(inom[:, 0].mean()), inom_h16=float(inom[:, min(15, n_pred-1)].mean()),
                inom_h128=float(inom[:, -1].mean()),
                metoder=dict(zip(*np.unique([r_[namn]["metod"] for r_ in rv], return_counts=True))),
                drift_median_abs=float(np.median([abs(r_[namn]["drift"]) for r_ in rv if r_[namn]["drift"] is not None]))
                if any(r_[namn]["drift"] is not None for r_ in rv) else None)
            s[namn]["metoder"] = {k: int(x) for k, x in s[namn]["metoder"].items()}
        summ[v] = s
        print(f"\n{v}  (n={s['n']}, post exact {s['exact_post']:.3f})")
        for namn in ("modell", "orakel"):
            m = s[namn]
            print(f"  {namn:<7} hel horisont {m['hel_horisont']:.3f}   inom {a.tol_us} µs vid h=1/16/{n_pred}: "
                  f"{m['inom_h1']:.3f}/{m['inom_h16']:.3f}/{m['inom_h128']:.3f}   "
                  f"|drift| median {m['drift_median_abs']}   metoder {m['metoder']}")
    with open(katalog / f"forecast_{tag}_summary.json", "w") as f:
        json.dump(summ, f, indent=1)

    # ---- plottar
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    varianter = sorted(summ)
    fig, axes = plt.subplots(1, len(varianter), figsize=(5.2 * len(varianter), 3.6), squeeze=False)
    h = np.arange(1, n_pred + 1)
    for ax, v in zip(axes[0], varianter):
        ax.plot(h, summ[v]["modell"]["inom_per_h"], color="#2A6FDB", lw=2, label="modellens post")
        ax.plot(h, summ[v]["orakel"]["inom_per_h"], color="#999999", lw=1.5, ls="--", label="facitpost (orakel)")
        ax.set_ylim(0, 1.02); ax.set_xlabel("pulser framåt"); ax.set_title(v)
        ax.set_ylabel(f"andel inom {a.tol_us:g} µs"); ax.grid(alpha=0.3); ax.legend(loc="lower left")
    fig.suptitle("Prediktion ur emittermodellen: andel signaler med rätt PRI per horisont", y=1.02)
    fig.tight_layout(); fig.savefig(katalog / f"forecast_{tag}_horisont.png", dpi=150, bbox_inches="tight")

    # exempel: per variant två lyckade och ett misslyckat
    ex = []
    for v in varianter:
        rv = [r_ for r_ in rader if r_["variant"] == v]
        bra = [r_ for r_ in rv if r_["modell"]["helt"]]
        dal = [r_ for r_ in rv if not r_["modell"]["helt"] and r_["modell"]["pred"] is not None]
        # välj rika exempel: flest nivåer
        bra.sort(key=lambda r_: -r_["n_levels"]); dal.sort(key=lambda r_: -r_["n_levels"])
        ex += bra[:2] + dal[:1]
    if ex:
        fig, axes = plt.subplots(len(ex), 1, figsize=(11, 2.3 * len(ex)), squeeze=False)
        for ax, r_ in zip(axes[:, 0], ex):
            svans = np.asarray(r_["obs_svans"]); sant = np.asarray(r_["sant"]); pred = np.asarray(r_["modell"]["pred"])
            n0 = len(svans); x_obs = np.arange(-n0, 0); x_p = np.arange(0, len(sant))
            ax.plot(x_obs, svans, ".-", color="#444444", ms=3, lw=0.8, label="observerat")
            ax.plot(x_p, sant, ".-", color="#999999", ms=3, lw=0.8, label="verklig fortsättning")
            ax.plot(x_p[:len(pred)], pred, "x", color="#2A6FDB", ms=4, label="predikterat")
            fel = np.abs(pred - sant[:len(pred)]) > a.tol_us
            if fel.any():
                ax.plot(x_p[:len(pred)][fel], pred[fel], "x", color="#D6453D", ms=6)
            ax.axvline(0, color="k", lw=0.8, ls=":")
            ok = "rätt hela horisonten" if r_["modell"]["helt"] else f"första fel vid h={int(np.argmax(fel))+1}"
            ax.set_title(f"{r_['variant']}, {r_['n_levels']} nivåer — {ok}", fontsize=10, loc="left")
            ax.set_ylabel("PRI (µs)"); ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("puls relativt fönstrets slut"); axes[0, 0].legend(loc="upper right", fontsize=8, ncol=3)
        fig.tight_layout(); fig.savefig(katalog / f"forecast_{tag}_exempel.png", dpi=150, bbox_inches="tight")
    print(f"\nskrev forecast_{tag}_summary.json, forecast_{tag}_horisont.png, forecast_{tag}_exempel.png i {katalog}")


if __name__ == "__main__":
    main()
