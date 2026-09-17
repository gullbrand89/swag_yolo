"""
Fortsätt en pulsföljd utifrån ett facit.

    python forecast.py

Ett facit är inte en beskrivning av en observation -- det är en SPECIFIKATION av en
emitter. Nivåerna är tillstånd, dwelltiderna är varaktigheter, ordningsblocket är
övergångsstrukturen. Det är en dold semi-Markovmodell, och en sådan går att köra
framåt.

Det ger en utvärdering som inte går via token alls:

    ta modellens facit  ->  ställ in fasen mot en observerad snutt
                        ->  generera fortsättningen
                        ->  jämför mot vad som faktiskt hände

Två facit kan skilja sig på flera token och ändå beskriva samma emitter, eller vara
nästan identiska och beskriva helt olika beteenden. Prognosen bryr sig bara om det
senare.

PROGNOSHORISONTEN
-----------------
Det mest användbara måttet här är hur LÅNGT prognosen håller, inte hur bra den är i
snitt. Ett korrekt facit för en deterministisk emitter förutsäger hur länge som
helst. Är dwelltiden fel med en enda puls glider prognosen ur fas, och horisonten
kollapsar till ungefär en cykel. Felets storlek läses alltså direkt av horisonten,
mycket känsligare än en tokenjämförelse.

VAD SOM INTE GÅR ATT FÖRUTSÄGA
------------------------------
`ORDER RANDOM` och `DWELL RANGE` är stokastiska. Där finns ingen enskild sann
fortsättning, och en prognos kan bara jämföras FÖRDELNINGSMÄSSIGT -- vilka nivåer
som förekommer och hur långa besöken är. Skriptet skiljer på de två fallen och
vägrar rapportera en horisont för en emitter där horisonten inte betyder något.
"""
import numpy as np

from transformer_post_generator.config import cfg


# =====================================================================
# Facit -> körbar specifikation
# =====================================================================
def fran_facit(facit):
    """
    Tar en tokenlista (modellens utdata) eller en redan tolkad dict.
    -> spec med nivåer i µs, ordningscykel, längdcykel eller intervall.
    """
    if isinstance(facit, (list, tuple)):
        from transformer_post_generator.labels import parse
        d = parse(list(facit))
    else:
        d = dict(facit)

    nivaer = list(d["order"]) if d["order_fixed"] and d["order"] else list(d["levels"])
    langder = [x for x in d["lengths"] if x is not None]

    return dict(
        nivaer=nivaer,                       # ordningscykeln, eller bara mängden
        alla_nivaer=sorted(set(d["levels"])),
        order_fixed=bool(d["order_fixed"]),
        length_fixed=bool(d["length_fixed"]),
        langder=langder,
        statisk=len([x for x in d["lengths"] if x is None]) > 0 and not langder,
        deterministisk=bool(d["order_fixed"]) and bool(d["length_fixed"]),
    )


# =====================================================================
# Fasinställning mot en observerad snutt
# =====================================================================
def _bin(pri):
    b = int((pri - cfg.pri_min) / (cfg.pri_max - cfg.pri_min) * (cfg.n_bins - 1))
    return max(0, min(cfg.n_bins - 1, b))


def besok(pri, tol=1):
    """-> [(bin, antal pulser)] för varje obrutet besök på samma nivå."""
    b = [_bin(p) for p in pri]
    ut, start = [], 0
    for i in range(1, len(b)):
        if abs(b[i] - b[i - 1]) > tol:
            ut.append((b[start], i - start))
            start = i
    ut.append((b[start], len(b) - start))
    return ut


def _basta_rotation(observerad, cykel):
    """
    Vilken position i cykeln motsvarar det SENAST observerade elementet?

    Matchar bakåt från slutet så långt det stämmer, för varje möjlig rotation, och
    väljer den som stämmer längst. Med en entydig cykel räcker ett element; med
    upprepningar i cykeln behövs flera, och då gör bakåtmatchningen jobbet.

    -> (index i cykeln, hur många element som stämde)
    """
    if not cykel:
        return 0, 0
    bast, bast_poang = 0, -1
    for r in range(len(cykel)):
        poang = 0
        for k in range(min(len(observerad), 3 * len(cykel))):
            if observerad[-1 - k] == cykel[(r - k) % len(cykel)]:
                poang += 1
            else:
                break
        if poang > bast_poang:
            bast, bast_poang = r, poang
    return bast, bast_poang


def stall_in(spec, prefix, tol_bins=2):
    """
    Var i sin cykel befinner sig emittern när snutten tar slut?

    -> dict med nivåindex, hur många pulser som redan spenderats på nuvarande nivå,
       och positionen i längdcykeln.

    Det SISTA besöket är avskuret av snittet, så dess längd är en undre gräns för
    dwelltiden -- vi vet att minst så många pulser har gått, inte att besöket är slut.
    """
    rutor = besok(prefix)
    if not rutor:
        raise ValueError("tom snutt")

    niva_bins = [_bin(v) for v in spec["nivaer"]]

    def till_index(b):
        bast, avst = 0, None
        for i, nb in enumerate(niva_bins):
            d = abs(b - nb)
            if avst is None or d < avst:
                bast, avst = i, d
        return bast if avst is not None and avst <= tol_bins else None

    observerad = [till_index(b) for b, _ in rutor]
    kanda = [x for x in observerad if x is not None]
    if not kanda:
        raise ValueError("ingen av snuttens nivåer finns i facitet")

    if spec["order_fixed"] and len(spec["nivaer"]) > 0:
        cykel = list(range(len(spec["nivaer"])))
        pos, traff = _basta_rotation([x for x in observerad if x is not None], cykel)
    else:
        pos, traff = kanda[-1], 0

    # positionen i längdcykeln: lika många fullbordade besök som vi sett, minus det
    # avskurna sista
    fullbordade = max(0, len(rutor) - 1)
    langd_pos = fullbordade % max(1, len(spec["langder"]))

    return dict(niva_index=pos, spenderat=rutor[-1][1],
                langd_pos=langd_pos, fasmatchning=traff, besok=len(rutor))


# =====================================================================
# Kör emittern framåt
# =====================================================================
def fortsatt(spec, prefix, n_pulser, rng=None, tol_bins=2):
    """
    -> (pri, nivaindex) för de n_pulser som kommer EFTER snutten.

    Stokastiska val (ORDER RANDOM, DWELL RANGE) dras ur rng. Är emittern
    deterministisk används rng aldrig och resultatet är reproducerbart.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    fas = stall_in(spec, prefix, tol_bins)

    nivaer = spec["nivaer"] if spec["nivaer"] else spec["alla_nivaer"]
    k = len(nivaer)
    i = fas["niva_index"]
    langd_pos = fas["langd_pos"]

    def nasta_dwell(pos):
        if spec["statisk"]:
            return 10 ** 9                      # lämnar aldrig nivån
        if spec["length_fixed"] and spec["langder"]:
            return int(spec["langder"][pos % len(spec["langder"])])
        if spec["langder"]:
            lo, hi = min(spec["langder"]), max(spec["langder"])
            return int(rng.integers(lo, hi + 1))
        return 1

    kvar = nasta_dwell(langd_pos) - fas["spenderat"]
    if kvar < 0:
        kvar = 0                                # snutten visade en längre dwell än facit

    pri, idx = [], []
    while len(pri) < n_pulser:
        if kvar <= 0:
            if spec["order_fixed"]:
                i = (i + 1) % k
            elif k > 1:
                val = [j for j in range(k) if j != i]   # ingen omedelbar upprepning
                i = int(rng.choice(val))
            langd_pos += 1
            kvar = nasta_dwell(langd_pos)
        pri.append(nivaer[i])
        idx.append(i)
        kvar -= 1

    return np.array(pri[:n_pulser], dtype=float), np.array(idx[:n_pulser])


# =====================================================================
# Utvärdering
# =====================================================================
def horisont(sant, prognos, tol_bins=2):
    """
    Antal pulser innan prognosen först går fel.

    Ett korrekt facit för en deterministisk emitter ger horisont = hela längden.
    En dwelltid som är en puls fel ger en horisont på ungefär en cykel, oavsett hur
    rätt nivåerna är -- fasglidningen dominerar.
    """
    n = min(len(sant), len(prognos))
    for i in range(n):
        if abs(_bin(sant[i]) - _bin(prognos[i])) > tol_bins:
            return i
    return n


def utvardera(sant, prognos, spec, tol_bins=2):
    """-> dict. Rapporterar horisont bara när den betyder något."""
    n = min(len(sant), len(prognos))
    s, p = np.asarray(sant[:n], float), np.asarray(prognos[:n], float)
    traff = np.array([abs(_bin(a) - _bin(b)) <= tol_bins for a, b in zip(s, p)])

    rapport = dict(
        pulser=n,
        andel_ratt=float(traff.mean()),
        medelfel_us=float(np.abs(s - p).mean()),
        deterministisk=spec["deterministisk"],
    )

    if spec["deterministisk"]:
        rapport["horisont"] = horisont(s, p, tol_bins)
    else:
        # stokastisk emitter: ingen enskild sann fortsättning finns, så bara
        # fördelningsmått är meningsfulla
        rapport["horisont"] = None
        rapport["nivamangd_stammer"] = (
            sorted({_bin(x) for x in s}) == sorted({_bin(x) for x in p}))
        rapport["mediandwell_sant"] = float(np.median([L for _, L in besok(s)]))
        rapport["mediandwell_prognos"] = float(np.median([L for _, L in besok(p)]))
    return rapport


# =====================================================================
# Demo
# =====================================================================
def _bygg_signal(nivaer, langder, n):
    pri, i, j = [], 0, 0
    while len(pri) < n:
        for _ in range(langder[j % len(langder)]):
            pri.append(nivaer[i % len(nivaer)])
            if len(pri) >= n:
                break
        i += 1; j += 1
    return np.array(pri[:n], dtype=float)


if __name__ == "__main__":
    def niva(andel):
        b = int((cfg.n_bins - 1) * andel)
        return cfg.pri_min + (b + 0.5) / (cfg.n_bins - 1) * (cfg.pri_max - cfg.pri_min)

    nivaer = [niva(0.20), niva(0.45), niva(0.70)]
    langder = [4, 7, 5]
    hel = _bygg_signal(nivaer, langder, 160)
    snutt, facit_fortsattning = hel[:40], hel[40:]

    print(f"emitter: 3 nivåer {[round(v, 1) for v in nivaer]} µs, dwell {langder}")
    print(f"snutt {len(snutt)} pulser, verklig fortsättning {len(facit_fortsattning)}\n")

    ratt = dict(levels=nivaer, order=nivaer, lengths=langder,
                order_fixed=True, length_fixed=True)
    spec = fran_facit(ratt)
    fas = stall_in(spec, snutt)
    print(f"fasinställning: nivåindex {fas['niva_index']}, "
          f"{fas['spenderat']} pulser spenderade, {fas['besok']} besök i snutten, "
          f"bakåtmatchning {fas['fasmatchning']} element")

    prognos, _ = fortsatt(spec, snutt, len(facit_fortsattning))
    r = utvardera(facit_fortsattning, prognos, spec)
    print(f"\nRÄTT FACIT")
    print(f"  andel rätt {r['andel_ratt']:.3f}   medelfel {r['medelfel_us']:.2f} µs"
          f"   horisont {r['horisont']}/{r['pulser']} pulser")

    fel = dict(ratt, lengths=[4, 8, 5])          # en dwell en puls för lång
    spec_fel = fran_facit(fel)
    prognos_fel, _ = fortsatt(spec_fel, snutt, len(facit_fortsattning))
    rf = utvardera(facit_fortsattning, prognos_fel, spec_fel)
    print(f"\nEN DWELL EN PULS FEL  (7 -> 8)")
    print(f"  andel rätt {rf['andel_ratt']:.3f}   medelfel {rf['medelfel_us']:.2f} µs"
          f"   horisont {rf['horisont']}/{rf['pulser']} pulser")
    print("\n  Ett enda felaktigt tal, och prognosen glider ur fas. Horisonten fångar"
          "\n  det som en tokenjämförelse knappt skulle notera.")

    slump = dict(ratt, order_fixed=False, length_fixed=False, lengths=[4, 7])
    spec_s = fran_facit(slump)
    prognos_s, _ = fortsatt(spec_s, snutt, len(facit_fortsattning),
                            rng=np.random.default_rng(1))
    rs = utvardera(facit_fortsattning, prognos_s, spec_s)
    print(f"\nSTOKASTISK EMITTER (ORDER RANDOM, DWELL RANGE)")
    print(f"  horisont: {rs['horisont']}  -- rapporteras inte, den betyder ingenting")
    print(f"  nivåmängden stämmer: {rs['nivamangd_stammer']}")
    print(f"  mediandwell sant {rs['mediandwell_sant']:.0f} mot "
          f"prognos {rs['mediandwell_prognos']:.0f} pulser")
