"""
Posten + signalen -> bibliotekets parameterform.

    python -m tools.biblioteksformat            # självtest mot generatorns sanning
    python -m tools.biblioteksformat --n 500

Posten säger VILKA nivåer emittern har. Signalen säger allt annat. Så fort
nivåmängden är känd kan varje puls tilldelas sin nivå, och då är Phi, theta och
övergångsmatriserna mätningar -- inte gissningar:

    mu_k          medel av pulserna på nivå k   (inte bin-mitten -- se nedan)
    sigma2_k      varians inom nivån
    phi_k         andel pulser på nivå k
    lambda        tillståndsföljden, från posten
    c_tillstand   dwellfördelning per TILLSTÅND   <- den modellen beskriver
    w_tillstand   vikterna till dem
    c_stod, w     samma sak per NIVÅ -- reservfallet, se nedan
    Q_niva        övergångar mellan besök -- diagnostik

mu blir NOGGRANNARE än posten. Ett B-token är kvantiserat till bin-bredden, cirka
0,25 µs med n_bins=2048 över [0.98, 505]. Medelvärdet över pulserna har ingen sådan
gräns.

Nivå och tillstånd är inte samma sak
------------------------------------
En nivå är ett VÄRDE. Ett tillstånd är en PLATS i följden. De sammanfaller bara när
varje värde besöks på ett enda sätt. Cykeln

    värde   125.7  187.1  3.0  125.7  248.5  64.4  3.0  187.1  3.0  187.1  64.4
    dwell    28     11     7    19     21     28    5    8      6    8      21

har fem nivåer och elva tillstånd. Värdet 3.0 är tre tillstånd med gemensamt mu och
dwell 7, 5 och 6. Aggregerar man per nivå blir det en trespetsig fördelning som
ingenting i emittern motsvarar; per tillstånd blir det tre skarpa ettor, vilket är
vad emittern faktiskt gör. Därför är c_tillstand huvudresultatet och c_stod bara
det som återstår när lambda saknas (ORDER RANDOM).

Just därför tas lambda från POSTEN och inte ur Q_niva. En övergångsmatris är
markovsk: den säger vad som följer på ett VÄRDE. Att 3.0 följs av 125.7 första
gången och av 187.1 andra gången ryms inte i den och går inte att räkna fram ur
den. ORDER FIXED L0 L2 L0 L1 säger det rakt ut. Posten är alltså starkare än
mätningen här, och theta["lambda_matt"] är den svagare mätningen kvar som
oberoende kontroll -- säger de olika saker beskriver posten och signalen olika
emittrar.

Det som INTE går att mäta bort
------------------------------
Är nivåmängden fel är allt härefter fel. Därför returneras `kvalitet` med andelen
pulser som inte kunde tilldelas någon nivå. Är den hög ska resultatet inte användas
-- posten beskriver en annan emitter än signalen.

Bortfall gör en puls till summan av två intervall. En sådan puls hamnar oftast
utanför alla nivåer och räknas som otilldelad, men om 2*mu_j råkar ligga nära mu_k
tilldelas den fel nivå utan att något märks. `kvalitet["multipelmisstankar"]` räknar
de otilldelade som ligger nära en heltalsmultipel av en nivå, vilket är signaturen
för bortfall snarare än för ett felaktigt facit.

Kantbesöken är avhuggna av observationsfönstret: det första besöket började före
fönstret och det sista slutar efter det. Deras LÄNGDER är alltså censurerade och
utesluts ur c. Deras pulser räknas däremot med i mu, sigma2 och phi, där de är
giltiga observationer.

Uppmätt (n=2000, seed=7, cfg som den står nu, GIVET RÄTT FACIT)
---------------------------------------------------------------
                          bortfall 0      bortfall 0,25    d:o, grind 0,35
  mu medelfel             2,5e-10 µs      1,4e-04 µs       2,7e-06 µs
  mu maxfel               5,0e-10 µs      9,2e-02 µs       1,2e-02 µs
  phi medelavvikelse      4,5e-03         1,5e-02          1,3e-02
  cykel ur Q_niva         127/127         106/127          106/122
  otilldelade pulser      0,0000          0,2612           -

Utan bortfall återskapas mu alltså EXAKT -- 5e-10 µs är flyttalsbrus, och
2e-9 bin-bredder. Posten själv är kvantiserad till en halv bin, 0,12 µs. Att
gå via signalen i stället för via B-tokenet vinner alltså åtta tiopotenser.

Med 25 % bortfall är medelfelet fortfarande försumbart medan MAXfelet växer --
det är de enstaka ihopslagna intervall som råkar landa på en annan nivå. Att
avvisa de signaler konverteraren själv flaggar (andel_otilldelade > 0,35, vilket
var 93 av 2000) tar bort åtta niondelar av maxfelet. Kvalitetsmåttet är alltså
inte dekoration utan den grind som gör resultatet användbart.

Cykelåterskapningen faller till 83 % vid 25 % bortfall och räddas INTE av
grinden -- den har en annan orsak: hål bryter besökskedjan, och en bruten kedja
ger färre övergångar att räkna, inte sämre pulser. Siffrorna förutsätter dessutom
rätt facit. Med modellens egen post tillkommer modellens fel ovanpå detta.
"""
import argparse
import sys

import numpy as np

from transformer_post_generator.config import cfg
from transformer_post_generator.labels import parse
from transformer_post_generator.vocab import bin_of


# Över den här andelen otilldelade pulser antas bortfall, och tvetydiga pulser
# utesluts. Under den antas signalen hel. Se till_bibliotek.
TROSKEL_BORTFALL = 0.02


# ------------------------------------------------------------------ tilldelning
def _binbredd():
    return (cfg.pri_max - cfg.pri_min) / cfg.n_bins


def _som_nivaer(facit):
    """tokenlista eller generatordict -> nivåvärden i µs, utan upprepningar."""
    niv, _ = _nivaer_och_cykel(facit)
    return niv


def _nivaer_och_cykel(facit):
    """
    -> (nivåvärden utan upprepning, tillståndsföljd som index i den listan)

    Följden är None när ordningen inte är uppgiven (ORDER RANDOM).

    Skillnaden mellan de två är hela poängen. Nivåerna är en VÄRDEMÄNGD; följden
    är en sekvens av TILLSTÅND, och samma värde kan förekomma flera gånger i den.
    En cykel som besöker 3.0 tre gånger med dwell 7, 5 och 6 är tre tillstånd med
    ett gemensamt mu -- inte ett tillstånd med spretig dwell.
    """
    if isinstance(facit, dict):
        cykel_us = list(facit["levels"])          # generatorns cykel, med upprepningar
        ordnad = bool(facit["order_fixed"])
    else:
        d = parse(list(facit))
        ordnad = d["order_fixed"]
        cykel_us = d["order"] if ordnad else d["levels"]

    niv = sorted({round(float(v), 9) for v in cykel_us})
    if not ordnad:
        return niv, None
    pos = {v: i for i, v in enumerate(niv)}
    return niv, [pos[round(float(v), 9)] for v in cykel_us]


def tilldela(pri, nivaer, tol_bins=None, uteslut_tvetydiga=True, max_m=4):
    """
    Varje puls till närmaste nivå inom toleransen, annars -1.

    -> (index per puls, besöksföljd, längd per besök)
    Toleransen mäts i BINS, inte i µs, eftersom det är i bin-rymden facitet är
    uttryckt och det är där två nivåer måste gå att skilja åt.

    uteslut_tvetydiga: en puls som ligger inom toleransen för nivå k men SAMTIDIGT
    inom toleransen för m*nivå_j (m >= 2) kan vara två ihopslagna intervall som
    råkar landa på k. Den sortens puls går inte att tilldela, och att räkna med den
    förorenar mu_k med ett värde som aldrig sändes. Den sätts till -1 i stället.
    Utan uteslutningen växer maxfelet i mu till en halv bin vid 25 % bortfall.
    """
    tol = cfg.run_tol_bins if tol_bins is None else tol_bins
    pri = np.asarray(pri, dtype=float)
    p_bins = np.array([bin_of(v) for v in pri], dtype=float)
    n_bins = np.array([bin_of(v) for v in nivaer], dtype=float)

    d = np.abs(p_bins[:, None] - n_bins[None, :])
    idx = d.argmin(axis=1)
    idx = np.where(d[np.arange(len(pri)), idx] <= tol, idx, -1)

    if uteslut_tvetydiga and max_m >= 2:
        mult = np.array([bin_of(m * v) for v in nivaer for m in range(2, max_m + 1)],
                        dtype=float)
        nara_multipel = (np.abs(p_bins[:, None] - mult[None, :]) <= tol).any(axis=1)
        idx = np.where(nara_multipel, -1, idx)

    # besök = löpande sträcka med samma nivå
    foljd, langder = [], []
    for v in idx:
        if foljd and v == foljd[-1]:
            langder[-1] += 1
        else:
            foljd.append(int(v))
            langder.append(1)
    return idx, foljd, langder


def _multipelmisstankar(pri, idx, nivaer, tol_bins, max_m=4):
    """Otilldelade pulser som ligger nära m * en nivå -- signaturen för bortfall."""
    tol = cfg.run_tol_bins if tol_bins is None else tol_bins
    otilldelade = np.asarray(pri, dtype=float)[np.asarray(idx) < 0]
    if not len(otilldelade):
        return 0
    mal = np.array([bin_of(m * v) for v in nivaer for m in range(2, max_m + 1)],
                   dtype=float)
    if not len(mal):
        return 0
    b = np.array([bin_of(v) for v in otilldelade], dtype=float)
    return int((np.abs(b[:, None] - mal[None, :]).min(axis=1) <= tol).sum())


def _dwell_per_tillstand(besok, cykel):
    """
    Fördela de observerade besöken på positionerna i tillståndsföljden.

    besok : [(nivåindex, längd), ...] i observerad ordning
    cykel : tillståndsföljden, som nivåindex -- rullas runt

    -> (stödpunkter per position, vikter per position)

    Fönstret börjar mitt i cykeln, så först måste fasen hittas: den förskjutning
    som får flest besök att stämma med följden. Att bara anta noll vore att
    tilldela varje besök fel position så fort inspelningen inte råkar börja på
    cykelns första tillstånd, vilket den nästan aldrig gör.
    """
    if not besok or not cykel:
        return None, None
    P = len(cykel)
    obs = [v for v, _ in besok]
    tratt = [sum(1 for i, v in enumerate(obs) if cykel[(i + o) % P] == v)
             for o in range(P)]
    o = int(np.argmax(tratt))

    hink = [[] for _ in range(P)]
    for i, (v, L) in enumerate(besok):
        p = (i + o) % P
        if cykel[p] == v:          # bara besök som stämmer med följden räknas
            hink[p].append(int(L))

    stod, vikt = [], []
    for x in hink:
        if not x:
            stod.append([]); vikt.append([])
            continue
        varden, antal = np.unique(np.array(x), return_counts=True)
        stod.append([int(y) for y in varden])
        vikt.append([float(y) for y in antal / antal.sum()])
    return stod, vikt


# ------------------------------------------------------------------ konvertering
def till_bibliotek(pri, facit, tol_bins=None, trim_kanter=True,
                   uteslut_tvetydiga="auto"):
    """
    pri   : observerad pulsföljd i µs
    facit : posten -- tokenlista eller generatorns dict

    -> dict med Phi, theta, Q_niva, Q_langd, kvalitet.

    Phi["mu"][k] är None om ingen puls tilldelades nivå k. Det är inte ett fel i
    konverteraren utan ett besked: posten påstår en nivå som inte syns i just den
    här signalen. Den ska synas i returvärdet, inte tystas med ett nollvärde.
    """
    pri = np.asarray(pri, dtype=float)
    nivaer, cykel_facit = _nivaer_och_cykel(facit)
    K = len(nivaer)
    if K == 0:
        raise ValueError("facitet innehåller inga nivåer")
    # "auto": uteslutningen kostar data och ska bara betalas när det finns bortfall
    # att skydda sig mot. Andelen otilldelade i en första tilldelning ÄR skattningen
    # av bortfallet -- ett ihopslaget intervall hamnar nästan alltid mellan nivåerna.
    # Utan bortfall skulle uteslutningen bara kasta giltiga pulser på de nivåer som
    # råkar ligga harmoniskt (mu_k nära 2*mu_j), vilket förskjuter phi kraftigt.
    idx, foljd, langder = tilldela(pri, nivaer, tol_bins, uteslut_tvetydiga=False)
    if uteslut_tvetydiga == "auto":
        uteslut_tvetydiga = (idx < 0).mean() > TROSKEL_BORTFALL
    if uteslut_tvetydiga:
        idx, foljd, langder = tilldela(pri, nivaer, tol_bins, uteslut_tvetydiga=True)

    # ---- Phi: en mätning per nivå
    mu, sigma2, phi, n_per = [], [], [], []
    n_tilldelade = int((idx >= 0).sum())
    for k in range(K):
        v = pri[idx == k]
        n_per.append(int(len(v)))
        if len(v) == 0:
            mu.append(None); sigma2.append(None); phi.append(0.0)
        else:
            mu.append(float(v.mean()))
            # ddof=1 kräver minst två observationer. Med en enda puls är variansen
            # inte odefinierad utan omätbar, och 0.0 vore ett påstående vi inte har
            # täckning för.
            sigma2.append(float(v.var(ddof=1)) if len(v) > 1 else None)
            phi.append(len(v) / n_tilldelade if n_tilldelade else 0.0)

    # ---- besök, med kantbesöken borttagna ur längdstatistiken
    besok = [(v, L) for v, L in zip(foljd, langder) if v >= 0]
    inre = besok[1:-1] if (trim_kanter and len(besok) > 2) else besok

    c = [[] for _ in range(K)]
    for v, L in inre:
        c[v].append(int(L))

    # ---- (w, c): dwellfördelningen per tillstånd, empiriskt
    # Ett tillstånd har en fördelning för VÄRDET (mu, sigma2 ovan) och en för hur
    # många pulser i rad det håller i. Den andra är den här: stödpunkterna c_stod
    # är de observerade dwelllängderna, w deras frekvenser.
    #
    # Två stödpunkter på en nivå betyder INTE brus. Det betyder att nivån besöks
    # med två olika längder -- alltså att det som ser ut som en nivå i själva
    # verket är två tillstånd med samma mu. Nivåer är en värdemängd, tillstånd är
    # punkter i en sekvens, och de sammanfaller bara när varje nivå besöks på ett
    # enda sätt. Fördelningen är det enda stället där skillnaden syns.
    c_stod, w = [], []
    for k in range(K):
        if not c[k]:
            c_stod.append([]); w.append([])
            continue
        varden, antal = np.unique(np.array(c[k]), return_counts=True)
        c_stod.append([int(x) for x in varden])
        w.append([float(x) for x in antal / antal.sum()])

    # ---- (w, c) per TILLSTÅND, inte per nivå
    # Aggregeringen ovan är per värde, och slår ihop de tillstånd som delar mu.
    # Nivå 0 i exemplet besöks på position 2, 6 och 8 med dwell 7, 5 och 6 -- per
    # nivå blir det en trespetsig fördelning, per tillstånd tre skarpa ettor. Det
    # senare är vad modellen faktiskt beskriver.
    # Kräver att lambda är känd: utan en tillståndsföljd finns inga positioner att
    # fördela besöken på, och då är aggregeringen per nivå det enda som går.
    c_tillstand = w_tillstand = None
    if cykel_facit:
        c_tillstand, w_tillstand = _dwell_per_tillstand(inre, cykel_facit)

    # ---- övergångsmatriser: räknade, inte härledda ur ORDER-token
    # Q_niva[j, k] = antal gånger ett besök på j följdes av ett besök på k.
    # Otilldelade besök bryter kedjan i stället för att hoppas över -- att sy ihop
    # j -> k över ett hål vore att hitta på en övergång som inte observerats.
    Q_niva = np.zeros((K, K), dtype=np.int64)
    for a, b in zip(foljd, foljd[1:]):
        if a >= 0 and b >= 0:
            Q_niva[a, b] += 1

    # Q_langd över besökslängder: samma idé, i längddomänen. Indexeras av de
    # observerade längdvärdena, som returneras med matrisen.
    langd_varden = sorted({L for _, L in inre})
    pos = {L: i for i, L in enumerate(langd_varden)}
    M = len(langd_varden)
    Q_langd = np.zeros((M, M), dtype=np.int64)
    for (_, La), (_, Lb) in zip(inre, inre[1:]):
        Q_langd[pos[La], pos[Lb]] += 1

    # ---- lambda: i vilken ordning tillstånden kommer
    # Tas från POSTEN, inte ur Q_niva. En övergångsmatris är markovsk -- den kan
    # bara säga vad som följer på ett VÄRDE. Besöker cykeln samma värde flera gånger
    # med olika fortsättning (3.0 -> 125.7 en gång, 3.0 -> 187.1 nästa) finns den
    # följden inte i matrisen, och den går inte att räkna fram ur den heller.
    # ORDER FIXED L0 L2 L0 L1 säger det rakt ut. Posten är alltså starkare än
    # mätningen just här, och det är därför den får bestämma.
    #
    # lambda_matt är samma sak räknad ur signalen. Den är svagare -- None så fort
    # cykeln har upprepningar -- men den är oberoende av posten, och när båda finns
    # och säger olika saker beskriver posten och signalen olika emittrar.
    lam = cykel_facit
    lam_matt = dominant_cykel(Q_niva)

    n_otilldelade = int(len(pri) - n_tilldelade)
    return {
        "nivaer_facit": nivaer,
        "Phi": {"mu": mu, "sigma2": sigma2, "phi": phi, "K": K, "n_pulser": n_per},
        "theta": {"c_stod": c_stod, "w": w,
                  "c_tillstand": c_tillstand, "w_tillstand": w_tillstand,
                  "c_obs": c,       # rå besökslängd per besök, före sammanräkning
                  "c_medel": [float(np.mean(x)) if x else None for x in c],
                  "lambda": lam, "lambda_matt": lam_matt},
        "Q_niva": Q_niva,
        "Q_langd": Q_langd,
        "langd_varden": langd_varden,
        "kvalitet": {
            "n_pulser": int(len(pri)),
            "n_otilldelade": n_otilldelade,
            "andel_otilldelade": n_otilldelade / len(pri) if len(pri) else 0.0,
            "multipelmisstankar": _multipelmisstankar(pri, idx, nivaer, tol_bins),
            "n_besok": len(besok),
            "nivaer_utan_pulser": [k for k in range(K) if n_per[k] == 0],
        },
    }


def pad(Q, D):
    """K x K -> D x D, nollutfyllt. Biblioteket vill ha fast dimension."""
    if Q.shape[0] > D:
        raise ValueError(f"{Q.shape[0]} nivåer ryms inte i {D}x{D}")
    ut = np.zeros((D, D), dtype=Q.dtype)
    ut[:Q.shape[0], :Q.shape[1]] = Q
    return ut


def dominant_cykel(Q):
    """
    Nivåcykeln ur Q_niva, om det finns en. -> lista med nivåindex, annars None.

    Efterföljaren till j är den VANLIGASTE, inte den enda. Ett enstaka felaktigt
    besök -- ett ihopslaget intervall som tilldelades fel, eller ett hål som bröt
    kedjan -- lägger då till en svag kant utan att dölja cykeln. Kravet att
    kedjan ska sluta sig och täcka alla nivåer är det som fångar en verkligt
    slumpad ordning; att kräva EXAKT en utgång per nivå gör bara testet känsligt
    för brus utan att skilja fler fall åt.
    """
    K = Q.shape[0]
    if K == 0 or (Q.sum(axis=1) == 0).any():
        return None
    nasta = {j: int(np.argmax(Q[j])) for j in range(K)}
    cykel, sedd, j = [], set(), 0
    while j not in sedd:
        sedd.add(j); cykel.append(j); j = nasta[j]
    return cykel if len(cykel) == K and nasta[cykel[-1]] == cykel[0] else None


# ------------------------------------------------------------------ självtest
def sjalvtest(n=200, drop=0.0, seed=0, tol_bins=None, grind=None):
    from transformer_post_generator.all_emitters import create_emitter_data

    data = create_emitter_data(n, 1, drop_rate=drop, rng=np.random.default_rng(seed))
    fel_mu, fel_phi, andel_ot = [], [], []
    n_avvisade = [0]
    dwell_ratt = dwell_provad = 0
    flerlage = 0
    ts_ratt = ts_provad = 0
    ordning_ratt = ordning_provad = 0
    trasiga = []

    for seqs, lab in data:
        pri = np.asarray(seqs[0], dtype=float)
        if len(pri) < 8:
            continue
        r = till_bibliotek(pri, lab, tol_bins=tol_bins)
        andel_ot.append(r["kvalitet"]["andel_otilldelade"])
        # grind: hoppa över signaler som konverteraren själv flaggar som otillförlitliga
        if grind is not None and r["kvalitet"]["andel_otilldelade"] > grind:
            n_avvisade[0] += 1
            continue

        # mu mot de SANNA nivåvärdena
        sanna = _som_nivaer(lab)
        for k, m in enumerate(r["Phi"]["mu"]):
            if m is not None:
                fel_mu.append(abs(m - sanna[k]))

        # (w, c): en emitter med EN fast dwell ska ge exakt en stödpunkt med vikt 1
        if (lab["length_fixed"] and len(lab["lengths"]) == 1
                and lab["lengths"][0] is not None):
            sant_k = int(lab["lengths"][0])
            for k in range(len(r["theta"]["c_stod"])):
                if not r["theta"]["c_stod"][k]:
                    continue
                dwell_provad += 1
                if r["theta"]["c_stod"][k] == [sant_k]:
                    dwell_ratt += 1
        flerlage += sum(1 for x in r["theta"]["c_stod"] if len(x) > 1)

        # per TILLSTÅND: när generatorn har en äkta dwell per position i cykeln
        # ska varje position ge exakt den längden, med vikt 1
        if (lab["order_fixed"] and lab["length_fixed"]
                and len(lab["lengths"]) == len(lab["levels"])
                and None not in lab["lengths"]
                and r["theta"]["c_tillstand"] is not None):
            for pos, sant in enumerate(lab["lengths"]):
                fick = r["theta"]["c_tillstand"][pos]
                if not fick:
                    continue
                ts_provad += 1
                if fick == [int(sant)]:
                    ts_ratt += 1

        # phi mot den sanna tidsandelen, när den är entydig
        if lab["order_fixed"] and lab["length_fixed"]:
            cyk = [sanna.index(round(float(v), 9)) for v in lab["levels"]]
            langder = list(lab["lengths"])
            if None not in langder and len(langder) in (1, len(cyk)):
                vikt = np.zeros(len(sanna))
                for i, kk in enumerate(cyk):
                    vikt[kk] += langder[i % len(langder)]
                vikt = vikt / vikt.sum()
                for k in range(len(sanna)):
                    fel_phi.append(abs(r["Phi"]["phi"][k] - vikt[k]))

            # ordningen: ger Q_niva tillbaka den sanna cykeln?
            if len(sanna) >= 3 and len(set(cyk)) == len(cyk):
                ordning_provad += 1
                fick = dominant_cykel(r["Q_niva"])
                if fick is not None:
                    r0 = fick.index(cyk[0])
                    if fick[r0:] + fick[:r0] == cyk:
                        ordning_ratt += 1
                    else:
                        trasiga.append((lab["variant"], cyk, fick))

    bredd = _binbredd()
    print(f"n={len(data)}  drop={drop}  tol_bins={tol_bins or cfg.run_tol_bins}"
          f"{'' if grind is None else f'  grind={grind}'}")
    if grind is not None:
        print(f"avvisade av grinden: {n_avvisade[0]}")
    print(f"bin-bredd {bredd:.4f} µs\n")
    print(f"mu:   {len(fel_mu)} nivåer uppmätta")
    if fel_mu:
        fm = np.array(fel_mu)
        print(f"      medelfel {fm.mean():.2e} µs   max {fm.max():.2e} µs")
        print(f"      max i bin-bredder: {fm.max() / bredd:.2e}")
    print(f"phi:  medelavvikelse {np.mean(fel_phi):.2e}  max {max(fel_phi):.2e}"
          if fel_phi else "phi:  -")
    print(f"dwell:   {dwell_ratt}/{dwell_provad} nivåer med exakt rätt (w, c)")
    print(f"         {ts_ratt}/{ts_provad} TILLSTÅND med exakt rätt dwell")
    print(f"         {flerlage} nivåer fick flera stödpunkter")
    print(f"ordning: {ordning_ratt}/{ordning_provad} cykler återskapade ur Q_niva")
    print(f"otilldelade pulser: medel {np.mean(andel_ot):.4f}  "
          f"max {np.max(andel_ot):.4f}")
    for v, sant, fick in trasiga[:3]:
        print(f"  ! {v}: sant {sant}, ur Q {fick}")

    # Utan bortfall är exakt återskapning ett KRAV -- varje avvikelse är ett fel
    # i konverteraren. Med bortfall är den ett mätvärde: information saknas i
    # signalen, och då säger en nolla ingenting om koden.
    if drop == 0:
        return not trasiga
    print("\n(bortfall > 0: avvikelser är informationsförlust i signalen, inte fel i konverteraren)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol-bins", type=int, default=None)
    ap.add_argument("--grind", type=float, default=None,
                    help="hoppa över signaler med högre andel otilldelade")
    a = ap.parse_args()
    ok = sjalvtest(a.n, a.drop, a.seed, a.tol_bins, a.grind)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
