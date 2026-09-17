"""
Visuell inspektion av facit mot verklig signal.

Användning i notebook:

    from inspect_label import plot_label, simulate_from_label
    import numpy as np
    from transformer_post_generator.data import create_emitter_data
    from transformer_post_generator.labels import to_tokens

    data = create_emitter_data(4, 1, 0.0, None, np.random.default_rng(0))
    seqs, lab = data[0]
    tokens = to_tokens(lab["levels"], lab["lengths"], lab["order_fixed"], lab["length_fixed"])
    plot_label(seqs[0], tokens).show()

Två paneler:
  övre  – den verkliga pulsföljden, med facitets nivåer som horisontella linjer
  undre – en sekvens simulerad från det PARSADE facitet
Ser panelerna likadana ut beskriver facitet emittern korrekt.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from transformer_post_generator.labels import parse
from transformer_post_generator.vocab import bin_of, pri_of_bin


# ------------------------------------------------------------------ simulering
def simulate_from_label(d, n_pulses, rng=None, start_phase=0):
    """
    Rullar ut en pulsföljd från en parsad facit-dict.
    d : dict från parse(tokens)
    Returnerar (pri, switch_idx) där switch_idx är pulsindex för varje nivåbyte.
    """
    rng = rng or np.random.default_rng(0)
    levels = d["order"] if d["order_fixed"] else sorted(d["levels"])
    lengths = d["lengths"]

    pri, switches = [], []
    i = j = start_phase                      # position i nivå- resp längdcykeln
    prev = None
    while len(pri) < n_pulses:
        lvl = levels[i % len(levels)] if d["order_fixed"] else rng.choice(levels)
        # undvik samma nivå två gånger i rad vid slumpad ordning
        if not d["order_fixed"] and prev is not None and len(levels) > 1:
            while lvl == prev:
                lvl = rng.choice(levels)
        n = lengths[j % len(lengths)] if d["length_fixed"] else rng.choice(
            [x for x in lengths if x is not None] or [1])
        if n is None:                        # INF: static
            n = n_pulses - len(pri)
        if pri:
            switches.append(len(pri))
        pri.extend([lvl] * int(n))
        prev = lvl
        i += 1; j += 1
    return np.asarray(pri[:n_pulses], dtype=float), [s for s in switches if s < n_pulses]


# ------------------------------------------------------------------ plott
def plot_label(pri_real, tokens, n_sim=None, rng=None, title=None):
    """
    pri_real : array med observerade (eller rena) PRI-värden i µs
    tokens   : facit som token-lista
    """
    pri_real = np.asarray(pri_real, dtype=float)
    n = len(pri_real)
    n_sim = n_sim or n

    try:
        d = parse(tokens)
        parse_err = None
    except Exception as e:
        d, parse_err = None, str(e)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=("verklig signal", "simulerad från facit"
                        if parse_err is None else f"PARSE-FEL: {parse_err}"))

    # ---- övre: verklig signal
    fig.add_trace(go.Scatter(
        x=np.arange(n), y=pri_real, mode="lines+markers", name="verklig PRI",
        marker=dict(size=4), line=dict(width=1)), row=1, col=1)

    if d is not None:
        # facitets nivåer: bin-mitt som linje + band på en halv binbredd
        half_bin = abs(pri_of_bin(1) - pri_of_bin(0)) / 2
        for k, lv in enumerate(sorted(d["levels"])):
            for r in (1, 2):
                fig.add_hrect(y0=lv - half_bin, y1=lv + half_bin,
                              fillcolor="rgba(200,0,0,0.10)", line_width=0, row=r, col=1)
                fig.add_hline(y=lv, line=dict(color="rgba(200,0,0,0.45)", width=1, dash="dot"),
                              row=r, col=1)
                fig.add_annotation(x=0, y=lv, text=f"L{k} (bin {bin_of(lv)})", showarrow=False,
                                   xanchor="right", font=dict(size=9, color="rgb(160,0,0)"),
                                   row=r, col=1)

        # ---- undre: simulerad signal
        pri_sim, switches = simulate_from_label(d, n_sim, rng)
        fig.add_trace(go.Scatter(
            x=np.arange(len(pri_sim)), y=pri_sim, mode="lines+markers", name="simulerad PRI",
            marker=dict(size=4), line=dict(width=1, color="rgb(0,120,90)")), row=2, col=1)
        for s in switches:
            fig.add_vline(x=s, line=dict(color="rgba(0,0,0,0.18)", width=1), row=2, col=1)

    head = title or " ".join(tokens)
    if len(head) > 140:
        head = head[:140] + " …"
    fig.update_layout(
        height=620, showlegend=False,
        title=dict(text=head, font=dict(size=12)),
        margin=dict(l=60, r=20, t=90, b=45))
    fig.update_yaxes(title_text="PRI (µs)", row=1, col=1)
    fig.update_yaxes(title_text="PRI (µs)", row=2, col=1)
    fig.update_xaxes(title_text="pulsindex", row=2, col=1)
    return fig


def plot_order(pri_real, tokens, tol_bins=1, n_show=None, title=None):
    """
    Ordningen som en trappa: y-axeln är nivåINDEX (L0, L1, L2 ...) i stället för µs.
    Varje puls mappas till närmaste nivå i facitet; pulser som inte matchar någon
    nivå ritas som röda kryss längst upp (spuriösa värden, t.ex. 2x PRI vid bortfall).

    Under trappan ritas facitets förväntade nivåföljd (ORDER FIXED) upprepad, med
    den startfas som passar den verkliga signalen bäst. Ligger de i takt stämmer
    ordningen; glider de isär ser du exakt var takten tappas.
    """
    pri_real = np.asarray(pri_real, dtype=float)
    if n_show:
        pri_real = pri_real[:n_show]
    n = len(pri_real)

    try:
        d = parse(tokens); parse_err = None
    except Exception as e:
        d, parse_err = None, str(e)
    if d is None:
        fig = go.Figure()
        fig.update_layout(title=f"PARSE-FEL: {parse_err}", height=380)
        return fig

    lab = sorted(d["levels"])
    lab_bins = [bin_of(v) for v in lab]

    # varje puls -> nivåindex, eller None om ingen nivå är nära nog
    obs = []
    for p in pri_real:
        b = bin_of(p)
        j = int(np.argmin([abs(b - lb) for lb in lab_bins]))
        obs.append(j if abs(b - lab_bins[j]) <= tol_bins else None)

    x_ok = [i for i, v in enumerate(obs) if v is not None]
    y_ok = [obs[i] for i in x_ok]
    x_bad = [i for i, v in enumerate(obs) if v is None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_ok, y=y_ok, mode="lines+markers", name="verklig ordning",
        line=dict(width=1, shape="hv"), marker=dict(size=5)))
    if x_bad:
        fig.add_trace(go.Scatter(
            x=x_bad, y=[len(lab) - 0.5] * len(x_bad), mode="markers",
            name="matchar ingen nivå",
            marker=dict(symbol="x", size=7, color="rgb(200,60,0)")))

    # facitets ordning, bästa startfas
    if d["order_fixed"]:
        idx = {v: i for i, v in enumerate(lab)}
        cycle = [idx[v] for v in d["order"]]
        best, best_hits = 0, -1
        for r in range(len(cycle)):
            rot = cycle[r:] + cycle[:r]
            hits = sum(1 for i, v in enumerate(obs)
                       if v is not None and v == rot[i % len(rot)])
            if hits > best_hits:
                best, best_hits = r, hits
        rot = cycle[best:] + cycle[:best]
        exp = [rot[i % len(rot)] for i in range(n)]
        fig.add_trace(go.Scatter(
            x=list(range(n)), y=exp, mode="lines", name="facitets ordning",
            line=dict(width=1, dash="dot", shape="hv", color="rgba(200,0,0,0.6)")))
        frac = best_hits / max(1, len(x_ok))
        note = f"facitcykel {'→'.join('L'+str(c) for c in rot)} · i takt {frac:.0%}"
    else:
        note = "ORDER RANDOM – ingen förväntad följd att jämföra mot"

    fig.update_layout(
        height=420, title=dict(text=title or note, font=dict(size=12)),
        margin=dict(l=60, r=20, t=60, b=45))
    fig.update_yaxes(title_text="nivå", tickmode="array",
                     tickvals=list(range(len(lab))),
                     ticktext=[f"L{i} ({v:.0f} µs)" for i, v in enumerate(lab)],
                     range=[-0.6, len(lab) - 0.2])
    fig.update_xaxes(title_text="pulsindex")
    return fig


def _collapse(obs):
    """[0,0,0,1,1,2,...] -> ([0,1,2,...], [3,2,1,...]) : en post per dwell."""
    seq, runs = [], []
    for v in obs:
        if seq and v == seq[-1]:
            runs[-1] += 1
        else:
            seq.append(v); runs.append(1)
    return seq, runs


def plot_visits(pri_real, tokens, tol_bins=1, n_visits=40, title=None):
    """
    En prick per DWELL, inte per puls. x-axeln är besöksnummer, y-axeln nivåindex.
    Långa och korta dwells tar lika mycket plats, så ordningen går att läsa över
    många varv. Prickens storlek visar hur många pulser besöket varade.

    Övre spåret  = den verkliga signalen, kollapsad.
    Undre spåret = facitets ordning, också en prick per besök.
    Ligger de i takt stämmer ordningen.
    """
    pri_real = np.asarray(pri_real, dtype=float)
    try:
        d = parse(tokens); parse_err = None
    except Exception as e:
        d, parse_err = None, str(e)
    if d is None:
        fig = go.Figure(); fig.update_layout(title=f"PARSE-FEL: {parse_err}", height=360)
        return fig

    lab = sorted(d["levels"]); lab_bins = [bin_of(v) for v in lab]

    obs = []
    for p in pri_real:
        b = bin_of(p)
        j = int(np.argmin([abs(b - lb) for lb in lab_bins]))
        obs.append(j if abs(b - lab_bins[j]) <= tol_bins else -1)   # -1 = ingen match

    seq, runs = _collapse(obs)
    seq, runs = seq[:n_visits], runs[:n_visits]
    x = list(range(len(seq)))
    size = [8 + 20 * min(r, 40) / 40 for r in runs]

    fig = go.Figure()
    ok = [i for i, v in enumerate(seq) if v >= 0]
    fig.add_trace(go.Scatter(
        x=[x[i] for i in ok], y=[seq[i] for i in ok], mode="lines+markers",
        name="verklig ordning", line=dict(width=1, color="rgba(0,90,160,0.5)"),
        marker=dict(size=[size[i] for i in ok], color="rgb(0,90,160)", opacity=0.75),
        text=[f"{runs[i]} pulser" for i in ok],
        hovertemplate="besök %{x}: L%{y}<br>%{text}<extra></extra>"))
    bad = [i for i, v in enumerate(seq) if v < 0]
    if bad:
        fig.add_trace(go.Scatter(
            x=[x[i] for i in bad], y=[len(lab) - 0.5] * len(bad), mode="markers",
            name="matchar ingen nivå",
            marker=dict(symbol="x", size=8, color="rgb(200,60,0)")))

    if d["order_fixed"]:
        idx = {v: i for i, v in enumerate(lab)}
        cycle = [idx[v] for v in d["order"]]
        best, best_hits = 0, -1
        for r in range(len(cycle)):
            rot = cycle[r:] + cycle[:r]
            hits = sum(1 for i, v in enumerate(seq)
                       if v >= 0 and v == rot[i % len(rot)])
            if hits > best_hits:
                best, best_hits = r, hits
        rot = cycle[best:] + cycle[:best]
        fig.add_trace(go.Scatter(
            x=x, y=[rot[i % len(rot)] for i in x], mode="lines+markers",
            name="facitets ordning",
            line=dict(width=1, dash="dot", color="rgba(200,0,0,0.6)"),
            marker=dict(size=6, symbol="circle-open", color="rgb(200,0,0)")))
        note = ("facitcykel " + "→".join("L" + str(c) for c in rot) +
                f" · i takt {best_hits / max(1, len(ok)):.0%} · {len(seq)} besök")
    else:
        note = f"ORDER RANDOM · {len(seq)} besök"

    fig.update_layout(height=430, title=dict(text=title or note, font=dict(size=12)),
                      margin=dict(l=90, r=20, t=60, b=45))
    fig.update_yaxes(title_text="nivå", tickmode="array", tickvals=list(range(len(lab))),
                     ticktext=[f"L{i} ({v:.0f} µs)" for i, v in enumerate(lab)],
                     range=[-0.6, len(lab) - 0.2])
    fig.update_xaxes(title_text="besöksnummer (en prick per dwell)")
    return fig


def plot_levels(pri_real, tokens, tol_bins=1, title=None):
    """
    Bara nivåerna, ingen tidsaxel: varje distinkt observerad PRI blir en prick.
    Prickens storlek = hur många pulser som hamnade där, så spuriösa värden
    (t.ex. 2x PRI efter bortfall) syns som små prickar.

    Gör det lätt att avgöra om facitets nivåer stämmer, utan att längder och
    ordning skymmer bilden.
    """
    pri_real = np.asarray(pri_real, dtype=float)
    obs_bins = np.array([bin_of(p) for p in pri_real])
    uniq, counts = np.unique(obs_bins, return_counts=True)
    y = np.array([pri_of_bin(b) for b in uniq])

    try:
        d = parse(tokens); parse_err = None
    except Exception as e:
        d, parse_err = None, str(e)
    lab = sorted(d["levels"]) if d is not None else []

    # matcha observerade toppar mot facitets nivåer
    matched = np.zeros(len(uniq), dtype=bool)
    if lab:
        for i, b in enumerate(uniq):
            matched[i] = any(abs(b - bin_of(lv)) <= tol_bins for lv in lab)

    half_bin = abs(pri_of_bin(1) - pri_of_bin(0)) / 2
    size = 8 + 26 * (counts / counts.max())

    fig = go.Figure()
    for m, name, color in ((matched, "matchar facit", "rgb(0,120,90)"),
                           (~matched, "saknas i facit", "rgb(200,60,0)")):
        if not m.any():
            continue
        fig.add_trace(go.Scatter(
            x=np.zeros(m.sum()), y=y[m], mode="markers", name=name,
            marker=dict(size=size[m], color=color, opacity=0.65),
            text=[f"bin {b}, {c} pulser" for b, c in zip(uniq[m], counts[m])],
            hovertemplate="%{y:.1f} µs<br>%{text}<extra></extra>"))

    for k, lv in enumerate(lab):
        fig.add_hrect(y0=lv - half_bin, y1=lv + half_bin,
                      fillcolor="rgba(200,0,0,0.10)", line_width=0)
        fig.add_hline(y=lv, line=dict(color="rgba(200,0,0,0.45)", width=1, dash="dot"))
        fig.add_annotation(x=0.35, y=lv, text=f"L{k} (bin {bin_of(lv)})", showarrow=False,
                           xanchor="left", font=dict(size=9, color="rgb(160,0,0)"))

    n_lab, n_obs = len(lab), len(uniq)
    head = title or (f"{n_obs} distinkta bins i signalen, {n_lab} nivåer i facit"
                     + (f"   PARSE-FEL: {parse_err}" if parse_err else ""))
    fig.update_layout(height=520, title=dict(text=head, font=dict(size=12)),
                      showlegend=True, margin=dict(l=60, r=20, t=60, b=40))
    fig.update_xaxes(range=[-1, 1.6], showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="PRI (µs)")
    return fig


def describe(tokens):
    """Facitet i klartext, för utskrift bredvid plotten."""
    try:
        d = parse(tokens)
    except Exception as e:
        return f"PARSE-FEL: {e}\n{' '.join(tokens)}"
    lines = [" ".join(tokens), ""]
    lines.append(f"nivåer ({len(d['levels'])}): " +
                 ", ".join(f"L{i}={v:.1f} µs (bin {bin_of(v)})"
                           for i, v in enumerate(sorted(d["levels"]))))
    if d["order_fixed"]:
        names = {v: f"L{i}" for i, v in enumerate(sorted(d["levels"]))}
        lines.append("ordning: FAST  " + " → ".join(names[v] for v in d["order"]) +
                     f"   (period {len(d['order'])})")
    else:
        lines.append("ordning: SLUMPAD")
    kind = "FAST cykel" if d["length_fixed"] else "SLUMPAD ur mängd"
    vals = ", ".join("INF" if x is None else str(x) for x in d["lengths"])
    lines.append(f"längder: {kind}  [{vals}]   (period {len(d['lengths'])})")
    return "\n".join(lines)


# ------------------------------------------------------------------ batch
def plot_many(data, to_tokens_fn, k=6, rng=None):
    """
    data : lista av (pri_sequences, label_dict) från create_emitter_data
    to_tokens_fn : funktion label_dict -> tokens (t.ex. create_emitter_label)
    Returnerar en lista av figurer; anropa .show() på var och en i notebooken.
    """
    figs = []
    for seqs, lab in data[:k]:
        tokens = to_tokens_fn(lab)
        print(describe(tokens)); print("-" * 70)
        figs.append(plot_label(seqs[0], tokens, rng=rng))
    return figs
