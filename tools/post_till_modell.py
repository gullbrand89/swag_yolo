"""
Post -> emittermodell, UTAN signalen.

    from tools.post_till_modell import post_till_modell, rulla_ut_post
    m = post_till_modell(tokens)          # samma form som biblioteksformat.emittermodell
    pred = rulla_ut_post(m, n_pred)       # PRI-följd framåt från fönstrets slut, eller None

Formatet per tillstånd (labels_tillstand) bär hela modellen, så det här är en ren
omskrivning: inget mäts, inget antas ur signalen. Posten är skriven bakåt i tiden
(tillstånd j = besöket j steg före fönstrets slut); här vänds den till en följd
framåt, se lambda och rulla_ut_post. Lika (komponent, dwell) på flera
cykelpositioner är ETT tillstånd; lambda säger att det besöks flera gånger.
Det som INTE finns i posten och
därför sätts konventionellt:
    mu      binmitten (kvantiserat till bin-bredden, ~0,25 µs)
    sigma2  None (ingen brusmodell i posten ännu)
    phi     1/K vid *, annars 1
    RANGE   likformig över a..b
Jämför mot biblioteksformat.till_bibliotek(pri, tokens)["modell"] för att se vad de
konventionerna kostar när posten är rätt.
"""
from transformer_post_generator.labels_tillstand import parse


def post_till_modell(tokens):
    d = parse(tokens)
    K = len(d["levels"])

    def komp(k, phi, dw):
        c = dict(niva=k, mu=float(d["levels"][k]), sigma2=None, phi=phi)
        if dw[0] == "D":
            c["c"], c["w"] = [dw[1]], [1.0]
        elif dw[0] == "R":
            n = dw[2] - dw[1] + 1
            c["c"], c["w"] = list(range(dw[1], dw[2] + 1)), [1.0 / n] * n
        else:
            c["c"], c["w"] = [], []
            c["dwell_min"] = d["seen"]
        return c

    # Ett tillstånd är en (komponent, dwell). Två cykelpositioner med samma
    # komponent och samma dwell är SAMMA tillstånd -- att det besöks två gånger per
    # varv är lambdas sak, inte tillståndslistans. Därför slås lika ihop: N_s är
    # antalet unika, och lambda är följden av tillstånds-id över cykeln.
    P = len(d["states"])
    ids, tillstand = {}, []
    pos_id = []
    for comp, dw in d["states"]:
        nyckel = (comp, dw)
        if nyckel not in ids:
            ids[nyckel] = len(tillstand)
            if comp is None:
                tillstand.append(dict(komponenter=[komp(k, 1.0 / K, dw) for k in range(K)]))
            else:
                tillstand.append(dict(komponenter=[komp(comp, 1.0, dw)]))
        pos_id.append(ids[nyckel])
    # Posten är skriven bakåt i tiden (position j = besöket j steg före slutet).
    # lambda framåt från nuläget: position 0, sedan P-1, P-2, ..., 1 -- ett varv.
    framat = [0] + list(range(P - 1, 0, -1))
    lam = [pos_id[j] for j in framat]
    return {"N_s": len(tillstand), "tillstand": tillstand, "lambda": lam, "period": P,
            "grund": "post", "seen": d["seen"]}


def rulla_ut_post(m, n_pred):
    """
    PRI-följd n_pred pulser framåt ur posten ensam, eller None om följden inte är
    deterministisk (* eller RANGE i något tillstånd). INF: konstant.
    Följer lambda: först resten av lambda[0], sedan lambda[1], lambda[2], ... cykliskt.
    """
    ts, lam = m["tillstand"], m["lambda"]
    if any(len(t["komponenter"]) != 1 for t in ts):
        return None
    k0 = ts[lam[0]]["komponenter"][0]
    if not k0["c"]:                                   # INF
        return [k0["mu"]] * n_pred
    if any(len(t["komponenter"][0]["c"]) != 1 for t in ts):
        return None
    ut = [k0["mu"]] * max(0, k0["c"][0] - (m["seen"] or 0))
    i = 0
    while len(ut) < n_pred:
        i = (i + 1) % len(lam)
        k = ts[lam[i]]["komponenter"][0]
        ut += [k["mu"]] * k["c"][0]
    return ut[:n_pred]
