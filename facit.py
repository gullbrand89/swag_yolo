"""
Hur facitet byggs och valideras, steg för steg.

    python facit.py

Skriven för att LÄSAS. Explicita loopar, ett steg per funktion, även där det går
att göra kortare. Den riktiga koden i labels.py gör samma sak kompaktare.

Sist i filen jämförs den här versionen mot labels.to_tokens. Går de isär är
beskrivningen inte längre sann om koden, och då säger den till.

Vad ett facit är
----------------
En bibliotekspost som beskriver emittern, inte observationen:

    LEVELS  L0 B102  L1 B255  L2 B408         vilka PRI-nivåer den använder
    ORDER FIXED L0 L1 L2                      i vilken ordning den besöker dem
    DWELL FIXED D10 D5                        hur många pulser den stannar
    END

B-token är nivåer (bins i emitterrymden), D-token är dwelltider (antal pulser).
Skilda rymder: en bin är en mätning med ändlig noggrannhet, en dwelltid ett exakt
antal, och de tål inte samma utjämning. Nivåblocket har inget antalstoken --
antalet går att läsa av genom att räkna par tills ORDER dyker upp.

Ordnings- och längdcyklerna är separata block med oberoende perioder: en emitter
kan ha tre nivåer men bara två skilda dwelltider.

Den svåra delen är inte formatet utan KANONISERINGEN. En emitter måste ha exakt
ETT giltigt facit. Finns det två ser modellen identisk indata med olika mål, och
skillnaden blir ett golv i förlustfunktionen som ingen träning kommer under.
"""
import numpy as np

from config import cfg

MAX_LEVELS = cfg.max_levels


# =====================================================================
# Steg 0 -- från µs till bin
# =====================================================================
def bin_av(pri):
    """
    En PRI i µs -> ett heltal i [0, n_bins - 1].

    OBS: detta är FACITETS binning över emitterrymden [pri_min, pri_max], inte
    inputens binning över observationsrymden. De är två olika skalor och det är
    lätt att blanda ihop dem.
    """
    andel = (pri - cfg.pri_min) / (cfg.pri_max - cfg.pri_min)
    b = int(andel * (cfg.n_bins - 1))
    if b < 0:
        b = 0
    if b > cfg.n_bins - 1:
        b = cfg.n_bins - 1
    return b


def pri_av_bin(b):
    """Tillbaka till µs -- mitten av binen."""
    return cfg.pri_min + (b + 0.5) / (cfg.n_bins - 1) * (cfg.pri_max - cfg.pri_min)


# =====================================================================
# Steg 1 -- kanonisering A: nollbrett intervall är en fast dwell
# =====================================================================
def steg1_nollbrett_intervall(lengths, length_fixed):
    """
    'Dwelltiden lottas ur intervallet [2, 2]' och 'dwelltiden är alltid 2'
    beskriver samma emitter. Utan den här regeln finns två giltiga facit.

    Villkoret att inget värde får vara None är inte kosmetiskt: None betyder INF,
    alltså en nivå som aldrig lämnas, och den informationen får inte tyst
    försvinna i en kollaps.
    """
    if length_fixed:
        return lengths, True
    if len(lengths) == 0:
        return lengths, False
    for x in lengths:
        if x is None:
            return lengths, False

    minsta = min(lengths)
    storsta = max(lengths)
    if minsta == storsta:
        return [lengths[0]], True      # blev en fast dwell
    return lengths, False


# =====================================================================
# Steg 2 -- namnge nivåerna
# =====================================================================
def steg2_namnge(levels):
    """
    Nivåerna får namn L0, L1, ... efter STIGANDE bin-värde.

    Varför inte i den ordning de dök upp: då skulle namnen bero på var i signalen
    observationen började, alltså på en egenskap hos observationen och inte hos
    emittern. Två inspelningar av samma emitter skulle få olika facit.

    -> (namn_per_bin, sekvens_av_namn, unika_bins)
    """
    bins = []
    for v in levels:
        bins.append(bin_av(v))

    unika = sorted(set(bins))

    if len(unika) > MAX_LEVELS:
        raise AssertionError(f"{len(unika)} nivåer, max {MAX_LEVELS}")

    # Två SKILDA nivåvärden får inte hamna i samma bin -- då blir facitet
    # oobserverbart. Upprepningar av samma värde i cykeln är däremot tillåtna.
    distinkta_varden = set()
    for v in levels:
        distinkta_varden.add(round(float(v), 9))
    if len(unika) != len(distinkta_varden):
        raise AssertionError(f"bin-kollision: {sorted(distinkta_varden)} -> bins {unika}")

    namn = {}
    for i in range(len(unika)):
        namn[unika[i]] = f"L{i}"

    sekvens = []
    for b in bins:
        sekvens.append(namn[b])

    return namn, sekvens, unika


# =====================================================================
# Steg 3 -- kanonisering B: ordningen vid en eller två nivåer
# =====================================================================
def steg3_pataglig_ordning(order_fixed, unika, namn):
    """
    Med EN nivå sker inga byten alls. Med TVÅ nivåer alternerar den observerade
    följden alltid -- en omedelbar upprepning smälter ihop med föregående besök
    och syns inte i pulsföljden. En slumpad ordning ger då exakt samma signal som
    cykeln [L0, L1].

    ORDER RANDOM och ORDER FIXED beskriver alltså samma emitter, och utan den här
    regeln tvingas modellen gissa mellan två lika riktiga svar.

    -> (order_fixed, sekvens_eller_None, paatvingad)
    """
    if order_fixed or len(unika) > 2:
        return order_fixed, None, False

    sekvens = []
    for b in unika:
        sekvens.append(namn[b])
    return True, sekvens, True


# =====================================================================
# Steg 4 -- kanonisering C: minsta period
# =====================================================================
def steg4_minsta_period(cykel):
    """
    [10, 10, 10] -> [10]   och   [5, 8, 5, 8] -> [5, 8]

    En cykel och samma cykel upprepad två gånger beskriver samma beteende.
    """
    n = len(cykel)
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        stammer = True
        for i in range(n):
            if cykel[i] != cykel[i % p]:
                stammer = False
                break
        if stammer:
            return cykel[:p]
    return cykel


# =====================================================================
# Steg 5 -- kanonisering D: minsta rotation
# =====================================================================
def steg5_minsta_rotation(cykel):
    """
    [L1, L2, L0] -> [L0, L1, L2]

    Var i cykeln observationen råkade börja är en egenskap hos inspelningen, inte
    hos emittern. Vi väljer alltid den rotation som kommer först alfabetiskt, så
    att alla startfaser ger samma facit.
    """
    if len(cykel) == 0:
        return cykel

    basta = None
    basta_index = 0
    for start in range(len(cykel)):
        roterad = cykel[start:] + cykel[:start]
        if basta is None or roterad < basta:
            basta = roterad
            basta_index = start
    return cykel[basta_index:] + cykel[:basta_index]


# =====================================================================
# Steg 6 -- sätt ihop tokensekvensen
# =====================================================================
def till_tokens(levels, lengths, order_fixed, length_fixed):
    """Facit-dict -> tokenlista. Alla kanoniseringar tillämpas här."""
    levels = list(levels)
    lengths_rena = []
    for x in lengths:
        lengths_rena.append(None if x is None else int(x))
    lengths = lengths_rena

    lengths, length_fixed = steg1_nollbrett_intervall(lengths, length_fixed)
    namn, sekvens, unika = steg2_namnge(levels)
    order_fixed, tvingad_sekvens, paatvingad = steg3_pataglig_ordning(
        order_fixed, unika, namn)
    if tvingad_sekvens is not None:
        sekvens = tvingad_sekvens

    # ---- nivådefinitionerna
    #
    # Inget antalstoken: nivånamn och ORDER är disjunkta tokenklasser, så parsern
    # läser par tills ORDER dyker upp. Ett explicit antal var dessutom det fält
    # modellen var sämst på, och ett fel i det gjorde hela posten otolkbar vid
    # avkodning även när varenda nivå var rätt.
    tokens = ["LEVELS"]
    for b in unika:
        tokens.append(namn[b])
        tokens.append(f"B{b}")

    sekvens_min = steg4_minsta_period(sekvens)
    if cfg.compress_lengths:
        langder_min = steg4_minsta_period(lengths)
    else:
        langder_min = list(lengths)

    # ---- ordningsblocket
    langder_ut = None
    if order_fixed:
        samma_period = len(langder_min) == len(sekvens_min)
        # En PÅTVINGAD ordning vet ingenting om vilken längd som hör till vilken
        # nivå. Roterar man ihop dem hittar man på en koppling generatorn aldrig
        # uppgav, så den grenen får bara tas när ordningen var uppgiven.
        if length_fixed and not paatvingad and samma_period:
            par = []
            for i in range(len(sekvens_min)):
                par.append((sekvens_min[i], langder_min[i]))
            par = steg5_minsta_rotation(par)
            sekvens_ut = [p[0] for p in par]
            langder_ut = [p[1] for p in par]
        else:
            sekvens_ut = steg5_minsta_rotation(sekvens_min)
        tokens.append("ORDER")
        tokens.append("FIXED")
        tokens.extend(sekvens_ut)
    else:
        tokens.append("ORDER")
        tokens.append("RANDOM")

    # ---- dwellblocket
    if length_fixed:
        if langder_ut is None:
            langder_ut = steg5_minsta_rotation(langder_min)
        tokens.append("DWELL")
        tokens.append("FIXED")
        for n in langder_ut:
            tokens.append("INF" if n is None else f"D{int(n)}")
    else:
        if len(lengths) == 0:
            raise AssertionError("DWELL RANGE utan längder")
        for x in lengths:
            if x is None:
                raise AssertionError(f"DWELL RANGE kan inte innehålla INF: {lengths}")
        tokens.append("DWELL")
        tokens.append("RANGE")
        tokens.append(f"D{min(lengths)}")
        tokens.append(f"D{max(lengths)}")

    tokens.append("END")
    return tokens


# =====================================================================
# Steg 7 -- tillbaka från tokens
# =====================================================================
def fran_tokens(tokens):
    """Tokenlista -> dict. Speglar till_tokens."""
    i = 0

    def ar_niva(t):
        return t.startswith("L") and t[1:].isdigit()

    def bin_tal(t):
        if not (t.startswith("B") and t[1:].isdigit()):
            raise ValueError(f"väntade nivåbin, fick {t}")
        return int(t[1:])

    def dur_tal(t):
        if not (t.startswith("D") and t[1:].isdigit()):
            raise ValueError(f"väntade dwelltid, fick {t}")
        return int(t[1:])

    if tokens[i] != "LEVELS":
        raise ValueError("saknar LEVELS")
    i += 1

    # Inget antal att läsa: par tills ORDER. Avgränsningen är entydig eftersom
    # nivånamn (L...) och ORDER inte kan förväxlas.
    namn_till_pri = {}
    while ar_niva(tokens[i]):
        namn = tokens[i]; i += 1
        namn_till_pri[namn] = pri_av_bin(bin_tal(tokens[i])); i += 1
    if not namn_till_pri:
        raise ValueError("tom nivådefinition")

    if tokens[i] != "ORDER":
        raise ValueError(f"väntade ORDER, fick {tokens[i]}")
    i += 1

    ordning = []
    order_fixed = tokens[i] == "FIXED"
    i += 1
    if order_fixed:
        while ar_niva(tokens[i]):
            ordning.append(namn_till_pri[tokens[i]]); i += 1

    if tokens[i] != "DWELL":
        raise ValueError(f"väntade DWELL, fick {tokens[i]}")
    i += 1
    length_fixed = tokens[i] == "FIXED"
    i += 1

    langder = []
    while tokens[i] != "END":
        langder.append(None if tokens[i] == "INF" else dur_tal(tokens[i]))
        i += 1

    return dict(levels=sorted(namn_till_pri.values()), order=ordning,
                lengths=langder, order_fixed=order_fixed, length_fixed=length_fixed)


# =====================================================================
# VALIDERING 1 -- rundturen
# =====================================================================
def validera_rundtur(levels, lengths, order_fixed, length_fixed):
    """
    Facit -> tokens -> tolkning -> tokens ska ge samma sträng.

    NÖDVÄNDIG MEN LÅNGT IFRÅN TILLRÄCKLIG. Rundturen bevarar de flaggor den får
    och ifrågasätter dem aldrig. Skickar du in ett felaktigt length_fixed går
    rundturen igenom med glans och facitet är ändå fel.
    """
    t1 = till_tokens(levels, lengths, order_fixed, length_fixed)
    d = fran_tokens(t1)
    kalla = d["order"] if d["order_fixed"] else d["levels"]
    t2 = till_tokens(kalla, d["lengths"], d["order_fixed"], d["length_fixed"])
    return t1 == t2


# =====================================================================
# VALIDERING 2 -- entydighet
# =====================================================================
def validera_entydighet():
    """
    Testet rundturen INTE gör: två likvärdiga beskrivningar av samma emitter ska
    ge identiska tokensekvenser.

    Det är den sortens fel som sätter ett golv i lossen utan att synas någonstans
    -- modellen får två olika mål för identisk indata och kan bara gissa.
    """
    def niva(andel):
        return pri_av_bin(int((cfg.n_bins - 1) * andel))

    fall = []

    a = till_tokens([niva(0.2), niva(0.5)], [4, 4], True, False)   # RANGE [4,4]
    b = till_tokens([niva(0.2), niva(0.5)], [4], True, True)       # FIXED 4
    fall.append(("nollbrett intervall == fast dwell", a == b, a, b))

    a = till_tokens([niva(0.2), niva(0.5)], [7, 7, 7], True, True)
    b = till_tokens([niva(0.2), niva(0.5)], [7], True, True)
    fall.append(("upprepad cykel == minsta period", a == b, a, b))

    a = till_tokens([niva(0.2), niva(0.5), niva(0.8)], [3, 9, 4], True, True)
    b = till_tokens([niva(0.5), niva(0.8), niva(0.2)], [9, 4, 3], True, True)
    fall.append(("roterad cykel == originalet", a == b, a, b))

    a = till_tokens([niva(0.3), niva(0.7)], [6], False, True)      # RANDOM
    b = till_tokens([niva(0.3), niva(0.7)], [6], True, True)       # FIXED
    fall.append(("2 nivåer: RANDOM == FIXED", a == b, a, b))

    a = till_tokens([niva(0.2), niva(0.5), niva(0.8)], [4], False, True)
    b = till_tokens([niva(0.2), niva(0.5), niva(0.8)], [4], True, True)
    fall.append(("3 nivåer: RANDOM != FIXED", a != b, a, b))

    return fall


# =====================================================================
# VALIDERING 3 -- mot signalen
# =====================================================================
def besok(pri, tolerans=1):
    """
    Dela signalen i besök: obrutna följder av pulser på samma nivå.
    -> lista av (bin, antal_pulser)

    Det här är den viktigaste funktionen i hela filen, för det är BESÖK och inte
    pulser som avgör om facitet går att härleda. En nivå som förekommer i
    tvåtusen pulser men bara besöks en gång säger ingenting om ordningen.
    """
    b = [bin_av(p) for p in pri]
    resultat = []
    start = 0
    for i in range(1, len(b)):
        if abs(b[i] - b[i - 1]) > tolerans:
            resultat.append((b[start], i - start))
            start = i
    resultat.append((b[start], len(b) - start))
    return resultat


def validera_mot_signal(pri, levels, min_varv=3):
    """
    Beskriver facitet faktiskt den signal det påstår sig beskriva?

    Tre frågor:
      1. Syns varje facitnivå i signalen?
      2. Finns det värden i signalen som inte finns i facitet?
      3. Hinner cykeln upprepas tillräckligt för att gå att fastställa?

    Fråga 3 är den som brukar svika. Svaret nej betyder inte att modellen är
    dålig -- det betyder att den delen av facitet är obestämbar och hör hemma i
    en redovisad taklista, inte bland felen.
    """
    facit_bins = sorted({bin_av(v) for v in levels})
    obs_bins = sorted({bin_av(p) for p in pri})
    rutor = besok(pri)

    saknade = []
    for fb in facit_bins:
        hittad = False
        for ob in obs_bins:
            if abs(ob - fb) <= 4:
                hittad = True
                break
        if not hittad:
            saknade.append(fb)

    extra = []
    for ob in obs_bins:
        nara = False
        for fb in facit_bins:
            if abs(ob - fb) <= 4:
                nara = True
                break
        if not nara:
            extra.append(ob)

    varv = len(rutor) / max(1, len(facit_bins))
    return dict(
        nivaer=len(facit_bins),
        besok=len(rutor),
        varv=varv,
        saknade_nivaer=saknade,
        oforklarade_bins=extra,
        cykeln_bestambar=varv >= min_varv,
    )


# =====================================================================
# Kontroll mot den riktiga koden
# =====================================================================
def jamfor_med_riktiga(n=200, seed=1):
    try:
        from labels import to_tokens
    except ImportError as e:
        print(f"\nkan inte jämföra mot labels.py ({e}) -- hoppar över")
        return True

    rng = np.random.default_rng(seed)
    lika = True
    for _ in range(n):
        k = int(rng.integers(1, 6))
        bins = sorted(rng.choice(np.arange(1, cfg.n_bins - 1), size=k, replace=False))
        lv = [pri_av_bin(int(b)) for b in bins]
        dw = [int(x) for x in rng.integers(1, 30, size=k)]
        of = bool(rng.integers(0, 2))
        lf = bool(rng.integers(0, 2))
        if not lf and k == 1:
            dw = [dw[0], dw[0] + 5]          # RANGE behöver bredd för att vara RANGE
        try:
            a = till_tokens(lv, dw, of, lf)
            b = to_tokens(lv, dw, of, lf)
        except AssertionError:
            continue
        if a != b:
            print(f"  SKILJER\n    enkel:  {' '.join(a)}\n    riktig: {' '.join(b)}")
            lika = False
            break
    print(f"\n{'ok  ' if lika else 'FEL '} enkel version == labels.to_tokens "
          f"({n} slumpade facit)")
    return lika


if __name__ == "__main__":
    def niva(andel):
        return pri_av_bin(int((cfg.n_bins - 1) * andel))

    levels = [niva(0.20), niva(0.45), niva(0.70)]
    lengths = [4, 4, 4]

    print(f"emitterrymd [{cfg.pri_min}, {cfg.pri_max}] µs, {cfg.n_bins} bins "
          f"({(cfg.pri_max - cfg.pri_min) / (cfg.n_bins - 1):.4f} µs/bin)\n")
    print(f"in: nivåer {[round(v, 1) for v in levels]} µs, längder {lengths}, "
          f"ORDER fast, DWELL fast")

    _, sekvens, unika = steg2_namnge(levels)
    print(f"\n  steg 2  bins {unika}, namn i stigande ordning -> {sekvens}")
    print(f"  steg 4  minsta period av längderna {lengths} -> "
          f"{steg4_minsta_period(lengths)}")
    print(f"  steg 5  minsta rotation av {sekvens} -> "
          f"{steg5_minsta_rotation(sekvens)}")

    tokens = till_tokens(levels, lengths, True, True)
    print(f"\nfacit:  {' '.join(tokens)}")
    print(f"tolkat: {fran_tokens(tokens)['lengths']} längder, "
          f"{len(fran_tokens(tokens)['levels'])} nivåer")

    print("\nVALIDERING 1 -- rundturen")
    print(f"  {'ok' if validera_rundtur(levels, lengths, True, True) else 'FEL'}")

    print("\nVALIDERING 2 -- entydighet (det rundturen inte ser)")
    for namn, ok, a, b in validera_entydighet():
        print(f"  {'ok  ' if ok else 'FEL '} {namn}")
        if not ok:
            print(f"        {' '.join(a)}\n        {' '.join(b)}")

    print("\nVALIDERING 3 -- mot signalen")
    pri = []
    for _varv in range(4):
        for v in levels:
            pri.extend([v] * 4)
    rapport = validera_mot_signal(np.array(pri), levels)
    for k, v in rapport.items():
        print(f"  {k:<20} {v}")

    jamfor_med_riktiga()
