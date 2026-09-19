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
    lambda        SIGNALENS tillståndsföljd: nivåindex per besök, observerat
    c, kant       dwell per besök, och om den är censurerad av fönsterkanten
    period, cykel härlett ur lambda: minsta period och den kanoniska cykeln
                  med dwellfördelning per position. None om ingen period.
    cykel_post    postens ORDER-block som hypotes; cykel_stammer jämför
    c_stod, w     dwellfördelning per NIVÅ (bakåtkompatibelt)
    Q_niva        övergångar mellan besök -- diagnostik
    modell        slidens form: N_s tillstånd x K_i komponenter, lambda över tillstånd
                  (emittermodell); bred_approximation för en klocka över ett tillstånd

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

Vad lambda ÄR
-------------
lambda beskriver signalen, inte emittern: nivåindex per besök, i den ordning de
observerades. Sliden skriver p(x | {Phi, theta}_x) -- parametrar för DEN HÄR
signalen -- och det är den läsningen. Två fönster av samma emitter ger två olika
lambda (fas, längd); det är priset för att vara förlustfri.

Att följden har en period är då ett faktum om lambda, inte en flagga bredvid den.
period_av läser av den, och cykel är den emitter-invarianta formen: minsta
periodiska enhet, kanoniskt roterad, med dwellfördelning per position. Den blir
densamma oavsett fönster. ORDER FIXED/RANDOM i posten är alltså härledbart, och
postens cykel ligger kvar som cykel_post -- en hypotes att jämföra mot. Under
bortfall kan den vara bättre än observationen: modellen ser genom hål som
period_av inte gör (ett enda otilldelat besök ger period None).

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

Variant-signaturer (600 emittrar, ren data, givet rätt facit)
--------------------------------------------------------------
                 period    cykel==post   skarp dwell    N_s
  ds_det_det      97 %       79/83          98 %       = cykellängd
  ds_det_rand    100 %       82/85           6 %       = cykellängd, bred dwell
  ds_rand_det      7 %         -           100 %       1, K komponenter, skarp dwell var
  ds_rand_rand     9 %         -             0 %       1, K komponenter, bred dwell
  jitter           4 %         -           100 %       1, K täta komponenter, dwell 1
  stagger        100 %       81/83         100 %       = cykellängd
  static         100 %       89/89         100 %       1, censurerad dwell

Perioden söks i paret (nivå, dwell): två alternerande nivåer med fjorton olika
dwelltider är period 14 med fjorton skarpa tillstånd, inte period 2 med en
fjortonspetsig dwell. De få ds_det_det-missarna är fönster med färre än två
hela varv av parföljden.

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

c_tillstand tål INTE bortfall
-----------------------------
    dwell per tillstånd, exakt      554/554 vid bortfall 0    17/451 vid 0,25
    dwell per nivå, exakt         4473/4473 vid bortfall 0  3477/4299 vid 0,25

Per tillstånd är det alltså exakt på hel signal och obrukbart vid 25 % bortfall.
Orsaken är fasen. Positionen härleds ur BESÖKSNUMRET -- besök i hör till
cykelposition (i + o) mod P -- och det förutsätter att varje besök svarar mot
exakt en position. Ett hål som delar ett besök i två, eller smälter ihop två,
förskjuter numreringen, och därefter sitter allt fel. Felen ligger också bara i
de LÅNGA signalerna, vilket är samma sak sett från andra hållet: ju fler besök,
desto större chans att fasen glidit någonstans på vägen.

Aggregeringen per nivå har inte det problemet -- den bryr sig inte om ordningen
-- och faller bara till 81 %. Så länge bortfallet är verkligt är c_stod alltså
det robusta valet och c_tillstand det exakta. Att göra c_tillstand robust kräver
att positionen hittas per besök i stället för att räknas fram, alltså en
inpassning mot cykeln som tillåter hopp. Det är inte gjort.
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
def period_av(folj, min_varv=2):
    """
    Minsta period P i en OBSERVERAD följd, eller None.

    Skiljer sig från labels._min_period på två sätt som båda kommer av att det
    här är ett fönster och inte en cykel: längden behöver inte vara en multipel
    av P (sista varvet får vara påbörjat), och det krävs minst min_varv hela varv
    innan något kallas periodiskt -- annars är varje följd trivialt periodisk
    med P = längden. Ett hål (-1) bryter allt: ett fönster med bortfall får
    ingen period, för vi vet inte vad som stod i hålet.
    """
    n = len(folj)
    def hal(v):                            # -1, eller ett par vars nivå är -1
        return (v[0] if isinstance(v, tuple) else v) < 0
    if n == 0 or any(hal(v) for v in folj):
        return None
    for P in range(1, n // min_varv + 1):
        if all(folj[i] == folj[i + P] for i in range(n - P)):
            return P
    return None


def _rot_min(x):
    """Minsta rotation, samma kanonisering som labels._rot."""
    x = list(x)
    r = min(range(len(x)), key=lambda k: x[k:] + x[:k])
    return x[r:] + x[:r]


def emittermodell(r):
    """
    Konverterarens resultat i slidens form.

        N_s tillstånd, tillstånd i har K_i komponenter (mu, sigma2, phi) och en
        dwellfördelning (w, c) PER KOMPONENT. lambda är en följd över tillstånd.

    Indexen på sliden -- {mu_{1:K_i}}_i och {(w, c)_{1:K_i}}_i -- säger just det:
    ett tillstånd får vara flertoppigt, och dwellen hör till komponenten. Det ger
    en enda regel för alla varianter:

      period finns   -> ett tillstånd per cykelposition, K_i = 1, lambda = cykeln
      ingen period   -> ETT tillstånd med alla nivåer som komponenter, lambda = [0]

    Andra grenen är kollapsen: slumpad ordning betyder att ordningen inte finns,
    inte att nivåerna inte finns. Komponenterna ligger kvar, skarpa, med var sin
    dwell -- så ds_rand_det behåller att nivå 3 alltid har dwell 7. Jitter blir
    samma sak med täta komponenter; en bred klocka över dem är en approximation
    (bred_approximation) och inte en del av mätningen.

    Statiska emittrar: period 1, ett tillstånd, en komponent, och dwellen är
    censurerad av fönstret -- c är tom och dwell_min säger hur långt vi såg.
    """
    th, Phi = r["theta"], r["Phi"]
    kant_max = th.get("dwell_kant_max")

    def komp(niv, phi, c, w):
        d = dict(niva=niv, mu=Phi["mu"][niv], sigma2=Phi["sigma2"][niv], phi=phi,
                 c=list(c), w=list(w))
        if not c and kant_max is not None:
            d["dwell_min"] = kant_max            # bara sett censurerat: minst så lång
        return d

    if th["period"] is not None:
        tillstand = [dict(komponenter=[komp(niv, 1.0, c, w)]) for niv, c, w in th["cykel"]]
    else:
        komps = [komp(k, Phi["phi"][k], th["c_stod"][k], th["w"][k])
                 for k in range(Phi["K"]) if Phi["n_pulser"][k] > 0]
        tillstand = [dict(komponenter=komps)]

    return dict(N_s=len(tillstand), tillstand=tillstand,
                **{"lambda": th["lambda_tillstand"]}, period=th["period"])


def bred_approximation(r, tillstand_i):
    """
    (mu, sigma2) för ett tillstånds ALLA komponenter ihopslagna -- en klocka över
    spannet. Förlustgivande: lägger massa mellan lägena. Till för jitter-liknande
    tillstånd med många täta komponenter, och bara när den som läser biblioteket
    vill ha det så. För ett likformigt spann [a, b] blir sigma (b-a)/sqrt(12).
    """
    m = r["modell"]["tillstand"][tillstand_i]
    nivaer = {k["niva"] for k in m["komponenter"]}
    idx = r["_idx"]; pri = r["_pri"]
    v = pri[np.isin(idx, list(nivaer))]
    if len(v) < 2:
        return None
    return float(v.mean()), float(v.var(ddof=1))


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

    # ---- lambda: signalens tillståndsföljd, som den observerades
    # lambda beskriver SIGNALEN, inte emittern: nivåindex per besök, i ordning,
    # -1 där besöket inte gick att tilldela. Det är "det här hände". Att följden
    # har en period är då ett FAKTUM om lambda som läses av ur den, inte en
    # flagga bredvid den -- ORDER FIXED/RANDOM i posten är alltså härledbart.
    #
    # Två fönster av samma emitter ger två olika lambda (olika fas, olika längd).
    # Det är priset för att vara förlustfri. Den emitter-invarianta formen är
    # cykel: minsta periodiska enhet, roterad till kanonisk form, med dwell per
    # position -- den blir densamma oavsett fönster, när en period finns.
    lam = [int(v) for v in foljd]
    c_per_besok = [int(L) for L in langder]
    kant = [False] * len(foljd)
    if trim_kanter and len(foljd) > 2:
        kant[0] = kant[-1] = True          # censurerade dwell, se kommentaren ovan
    elif trim_kanter and len(foljd) <= 2:
        kant = [True] * len(foljd)         # ett-två besök: båda kanterna avhuggna

    # Perioden söks i PARET (nivå, dwell), inte i nivåföljden ensam. Två nivåer
    # som alternerar med fjorton olika dwelltider är en emitter med period 14 och
    # fjorton skarpa tillstånd -- inte period 2 med en fjortonspetsig dwell. Det
    # är README:ns "olika perioder" löst: parföljden bär båda cyklerna på en gång.
    # Kantbesöken har censurerad dwell och ingår inte i parsökningen; de får bara
    # vara med i nivåföljden. Är paren inte periodiska (slumpad dwell) faller vi
    # tillbaka på nivåperioden, och dwellen blir en fördelning per position.
    P_niv = period_av(lam)
    if P_niv is None and len(lam) == 1 and lam[0] >= 0:
        P_niv = 1                          # static: ett besök hela fönstret, aldrig lämnat
    P = None
    cykel = None
    if P_niv is not None:
        inre_i = [i for i, k in enumerate(kant) if not k]
        par = [(lam[i], c_per_besok[i]) for i in inre_i]
        P_par = period_av(par)
        # parperioden måste vara en multipel av nivåperioden; annars är den brus
        P = P_par if (P_par is not None and P_par % P_niv == 0) else P_niv
        skarp = P_par is not None and P == P_par
        # kanonisk rotation av nivåföljden med period P
        start = inre_i[0] if skarp and inre_i else 0
        bas = lam[start:start + P]
        rot = min(range(P), key=lambda k: bas[k:] + bas[:k])
        niv_cykel = bas[rot:] + bas[:rot]
        # positionen i cykeln för besök i är (i - start - rot) mod P
        hink = [[] for _ in range(P)]
        for i, (v, L, k) in enumerate(zip(lam, c_per_besok, kant)):
            if not k:
                hink[(i - start - rot) % P].append(L)
        cykel = []
        for pos_i, dl in enumerate(hink):
            if dl:
                varden, antal = np.unique(np.array(dl), return_counts=True)
                cykel.append((niv_cykel[pos_i], [int(x) for x in varden],
                              [float(x) for x in antal / antal.sum()]))
            else:
                cykel.append((niv_cykel[pos_i], [], []))

    # tillståndsföljden över TILLSTÅND (inte nivåer), för emittermodell():
    # med period är tillstånd = cykelposition; utan period finns bara tillstånd 0
    if P is not None:
        lam_tillstand = [((i - start - rot) % P) if v >= 0 else -1
                         for i, v in enumerate(lam)]
    else:
        lam_tillstand = [0 if v >= 0 else -1 for v in lam]
    kant_langder = [L for L, k, v in zip(c_per_besok, kant, lam) if k and v >= 0]
    dwell_kant_max = max(kant_langder) if kant_langder else None

    # postens ORDER-block, som hypotes att jämföra mot -- under bortfall kan den
    # vara bättre än observationen, för modellen ser genom hål som period_av inte gör
    cykel_post = cykel_facit
    if cykel is not None and cykel_post is not None:
        cykel_stammer = _rot_min([c[0] for c in cykel]) == _rot_min(cykel_post)
    else:
        cykel_stammer = None
    lam_matt = dominant_cykel(Q_niva)

    n_otilldelade = int(len(pri) - n_tilldelade)
    r = {
        "nivaer_facit": nivaer,
        "Phi": {"mu": mu, "sigma2": sigma2, "phi": phi, "K": K, "n_pulser": n_per},
        "theta": {
            # --- signalens tillståndsföljd (primärt)
            "lambda": lam,               # nivåindex per besök, -1 = otilldelat
            "c": c_per_besok,            # dwell per besök, i samma ordning
            "kant": kant,                # True = censurerad dwell (fönsterkant)
            # --- härlett: emitterns cykel, om följden är periodisk
            "period": P,
            "cykel": cykel,              # [(nivåindex, dwell_stod, dwell_w), ...] kanonisk
            "cykel_post": cykel_post,    # vad posten påstod (ORDER FIXED), som hypotes
            "cykel_stammer": cykel_stammer,
            "lambda_tillstand": lam_tillstand,
            "dwell_kant_max": dwell_kant_max,
            # --- bakåtkompatibelt: per nivå och per postens tillståndsposition
            "c_stod": c_stod, "w": w,
            "c_tillstand": c_tillstand, "w_tillstand": w_tillstand,
            "c_obs": c,
            "c_medel": [float(np.mean(x)) if x else None for x in c],
            "lambda_post": cykel_facit, "lambda_matt": lam_matt},
        "_pri": pri, "_idx": idx,        # för bred_approximation
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
    r["modell"] = emittermodell(r)
    return r


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
