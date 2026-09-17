"""
Allt som händer med datan innan den når transformern, steg för steg.

    python forbehandling.py

Den här filen är skriven för att LÄSAS, inte för att vara snabb. Varje steg är en
egen funktion med en explicit loop, även där numpy kunde göra samma sak på en rad.
Den riktiga koden i data.py gör exakt samma sak men vektoriserat.

Sist i filen finns en kontroll som jämför den här långsamma versionen mot den
riktiga. Går de isär är det ett fel i en av dem, och då säger den till. Det är
poängen med att ha båda: en läsbar beskrivning som inte kan glida ifrån koden.

Resan en puls gör
-----------------
    PRI i µs
      -> klipps till observationsrymden [in_min, in_max]
      -> normaliseras till [0, 1]
      -> blir fyra parallella kanaler:  bins, cont, toa, rl
      -> paddas ihop med andra signaler till en batch, med en mask
      -> transformern ser (bins, cont, toa, rl, mask)

Modellen får alltså aldrig se PRI-värdet direkt. Den ser en diskret bin, ett
kontinuerligt värde, en tidpunkt och en räknare.
"""
import numpy as np

from config import cfg


# =====================================================================
# Steg 0 -- en exempelsignal att följa genom kedjan
# =====================================================================
def exempelsignal():
    """
    Tre nivåer, fyra pulser på varje, två varv. Nivåerna hämtas ur cfg så att
    exemplet alltid ligger inom den konfigurerade rymden.

    -> (pri, nivåer)   pri är 24 PRI-värden i µs
    """
    lo, hi = cfg.pri_min, cfg.pri_max
    nivaer = [lo + (hi - lo) * andel for andel in (0.20, 0.45, 0.70)]

    pri = []
    for _varv in range(2):
        for niva in nivaer:
            for _puls in range(4):
                pri.append(niva)
    return np.array(pri, dtype=float), nivaer


# =====================================================================
# Steg 1 -- klipp till observationsrymden
# =====================================================================
def steg1_klipp(pri):
    """
    Allt utanför [in_min, in_max] dras in till kanten.

    Varför: bin-indexet räknas som en andel av det intervallet, och ett värde
    utanför skulle ge ett index utanför tabellen.

    VARNING: klippning är tyst. Ligger en stor del av datan utanför rymden blir
    den till en enda bin utan att något klagar, och modellen kan då omöjligt
    skilja de värdena åt. Det är därför input_fidelity.py mäter andelen klippta
    pulser separat -- ett upplösningstest ser det inte.
    """
    ut = np.zeros(len(pri))
    for i in range(len(pri)):
        if pri[i] < cfg.in_min:
            ut[i] = cfg.in_min
        elif pri[i] > cfg.in_max:
            ut[i] = cfg.in_max
        else:
            ut[i] = pri[i]
    return ut


# =====================================================================
# Steg 2 -- normalisera till [0, 1]
# =====================================================================
def steg2_normalisera(pri_klippt):
    """
    in_min blir 0.0, in_max blir 1.0, linjärt däremellan.

    Linjärt och inte logaritmiskt: mätning på korpusen visade att avstånden
    mellan nivåer är ABSOLUTA (ungefär lika många µs oavsett nivåns storlek),
    inte relativa. Hade de varit relativa vore logaritmisk skala rätt.
    """
    spann = cfg.in_max - cfg.in_min
    x = np.zeros(len(pri_klippt))
    for i in range(len(pri_klippt)):
        x[i] = (pri_klippt[i] - cfg.in_min) / spann
    return x


# =====================================================================
# Steg 3 -- kanal "bins": diskret index
# =====================================================================
def steg3_bins(pri, x):
    """
    x i [0, 1] blir ett heltal i [0, in_bins - 2].

    Den SISTA binen, in_bins - 1, är reserverad som overflow: dit hamnar allt
    som låg på eller över in_max. Därför används in_bins - 2 som skala -- annars
    hade det största giltiga värdet krockat med overflow-binen.

    Varför en diskret kanal alls, när cont redan bär värdet? För att modellen
    ska kunna slå upp en inbäddning per nivå. Ett kontinuerligt tal måste gå
    genom ett linjärt lager och kan inte ge varje nivå en egen representation.
    """
    bins = np.zeros(len(x), dtype=np.int64)
    for i in range(len(x)):
        b = int(x[i] * (cfg.in_bins - 2))

        if b < 0:
            b = 0
        if b > cfg.in_bins - 2:
            b = cfg.in_bins - 2

        if pri[i] >= cfg.in_max:        # overflow går före allt annat
            b = cfg.in_bins - 1

        bins[i] = b
    return bins


# =====================================================================
# Steg 4 -- kanal "cont": kontinuerligt värde
# =====================================================================
def steg4_cont(x):
    """
    Samma värde som bins bygger på, men obeskuret, skalat till [-1, 1].

    Varför båda: binningen kastar bort allt inom en bin. Den kontinuerliga
    kanalen bevarar det, så att modellen kan skilja två pulser som råkat hamna
    i samma bin. Bin-kanalen ger identitet, cont-kanalen ger precision.
    """
    cont = np.zeros(len(x), dtype=np.float32)
    for i in range(len(x)):
        cont[i] = x[i] * 2.0 - 1.0
    return cont


# =====================================================================
# Steg 5 -- kanal "toa": när pulsen kom
# =====================================================================
def steg5_toa(pri):
    """
    Ankomsttid = summan av alla PRI hittills, delat med toa_scale.

    PRI är AVSTÅND mellan pulser. Summerar man dem får man TIDPUNKTER. Modellen
    behöver båda: nivåerna är avstånd, men dwelltider och ordning handlar om var
    i tiden något händer.

    toa_scale ska ligga nära korpusens medel-PRI. Då avancerar toa ungefär ett
    steg per puls, alltså samma storleksordning som pulsindex -- vilket är den
    arbetspunkt RoPE:s positionskodning är byggd för. Ligger skalan fel blir
    tidsaxeln antingen ihoptryckt eller utsmetad.
    """
    toa = np.zeros(len(pri), dtype=np.float32)
    summa = 0.0
    for i in range(len(pri)):
        summa = summa + pri[i]
        toa[i] = summa / cfg.toa_scale
    return toa


# =====================================================================
# Steg 6 -- kanal "rl": hur länge vi legat still
# =====================================================================
def steg6_rl(bins, flaggor=None):
    """
    För varje puls: hur många pulser i rad har legat på samma bin?

    Nollställs så fort binen byter med mer än run_tol_bins. Första pulsen är
    alltid 0, eftersom det inte finns någon föregående att jämföra med.

    Varför kanalen finns: dwelltider mäts i ANTAL pulser. Utan räknaren måste
    modellen härleda dem genom att attention räknar positioner, vilket är en
    beräkning transformers är dåliga på. Med räknaren står svaret redan i
    indatan och behöver bara läsas av.

    En flaggad puls (från en detektor för saknade pulser) räknas som två, för
    att en saknad puls betyder att intervallet egentligen var två intervall.
    """
    rl = np.zeros(len(bins), dtype=np.int64)
    for i in range(1, len(bins)):
        if flaggor is not None and flaggor[i]:
            rl[i] = rl[i - 1] + 2
        else:
            hopp = abs(int(bins[i]) - int(bins[i - 1]))
            if hopp <= cfg.run_tol_bins:
                rl[i] = rl[i - 1] + 1
            else:
                rl[i] = 0
    return rl


# =====================================================================
# Hela kedjan för EN signal
# =====================================================================
def forbehandla(pri):
    """En PRI-sekvens -> dict med de fyra kanalerna."""
    pri = np.asarray(pri, dtype=float)
    if pri.ndim != 1:
        raise ValueError(f"väntade EN pulsföljd (1-D), fick shape {pri.shape}")

    pri_klippt = steg1_klipp(pri)
    x = steg2_normalisera(pri_klippt)

    return dict(
        bins=steg3_bins(pri, x),
        cont=steg4_cont(x),
        toa=steg5_toa(pri),
        rl=steg6_rl(steg3_bins(pri, x)),
        flag=np.zeros(len(pri), dtype=np.int64),
        pri=pri.astype(np.float32),
    )


# =====================================================================
# Steg 7 -- slå ihop flera signaler till en batch
# =====================================================================
def bygg_batch(signaler):
    """
    Signaler är olika långa, men en tensor är rektangulär. Kortare signaler
    fylls ut med nollor, och en mask talar om vilka positioner som är påhittade.

    mask[i, t] == True  betyder padding, alltså "titta inte här".
    mask[i, t] == False betyder en riktig puls.

    Polariteten är lätt att få omvänd, och gör man det attenderar modellen på
    padding och ignorerar datan -- utan att krascha.
    """
    kanaler = forbehandla(signaler[0]).keys()
    B = len(signaler)
    T = max(len(s) for s in signaler)

    batch = {k: np.zeros((B, T), dtype=np.float32) for k in kanaler}
    batch["bins"] = np.zeros((B, T), dtype=np.int64)
    batch["rl"] = np.zeros((B, T), dtype=np.int64)
    batch["flag"] = np.zeros((B, T), dtype=np.int64)
    mask = np.ones((B, T), dtype=bool)

    for i in range(B):
        kanal = forbehandla(signaler[i])
        n = len(signaler[i])
        for namn in kanaler:
            for t in range(n):
                batch[namn][i, t] = kanal[namn][t]
        for t in range(n):
            mask[i, t] = False

    batch["mask"] = mask
    return batch


# =====================================================================
# Utskrift
# =====================================================================
def visa(pri, n=12):
    pri_klippt = steg1_klipp(pri)
    x = steg2_normalisera(pri_klippt)
    bins = steg3_bins(pri, x)
    cont = steg4_cont(x)
    toa = steg5_toa(pri)
    rl = steg6_rl(bins)

    print(f"observationsrymd [{cfg.in_min}, {cfg.in_max}] µs, {cfg.in_bins} bins "
          f"({(cfg.in_max - cfg.in_min) / (cfg.in_bins - 2):.4f} µs/bin), "
          f"toa_scale {cfg.toa_scale}")
    print(f"\nde {min(n, len(pri))} första pulserna av {len(pri)}:\n")
    print(f"{'#':>3}{'PRI µs':>11}{'x':>9}{'bins':>8}{'cont':>9}{'toa':>9}{'rl':>5}")
    print("-" * 54)
    for i in range(min(n, len(pri))):
        print(f"{i:>3}{pri[i]:>11.2f}{x[i]:>9.4f}{bins[i]:>8}"
              f"{cont[i]:>9.4f}{toa[i]:>9.2f}{rl[i]:>5}")
    print("\nrl nollställs vid varje nivåbyte -- det är dwelltiden, avläsbar direkt.")


# =====================================================================
# Kontroll mot den riktiga koden
# =====================================================================
def jamfor_med_riktiga(n_signaler=20, seed=0):
    """
    Den här filen och data.make_channels ska ge exakt samma siffror. Går de isär
    har en av dem ändrats utan den andra, och då är den här beskrivningen inte
    längre sann om koden.
    """
    try:
        from data import make_channels
    except ImportError as e:
        print(f"\nkan inte jämföra mot data.py ({e}) -- hoppar över")
        return True

    rng = np.random.default_rng(seed)
    allt_lika = True
    for _ in range(n_signaler):
        n = int(rng.integers(20, 200))
        pri = rng.uniform(cfg.in_min, cfg.in_max * 1.2, n)   # även värden över in_max
        min_ = forbehandla(pri)
        riktig = make_channels(pri)
        for namn in ("bins", "cont", "toa", "rl"):
            if not np.allclose(min_[namn], riktig[namn], atol=1e-5):
                print(f"  SKILJER i kanal {namn}")
                allt_lika = False
    print(f"\n{'ok  ' if allt_lika else 'FEL '} enkel version == data.make_channels "
          f"({n_signaler} slumpade signaler)")
    return allt_lika


if __name__ == "__main__":
    pri, nivaer = exempelsignal()
    print(f"exempel: 3 nivåer {[round(v, 1) for v in nivaer]} µs, "
          f"4 pulser på varje, 2 varv\n")
    visa(pri)

    batch = bygg_batch([pri, pri[:10]])
    print(f"\nbatch: {batch['bins'].shape[0]} signaler paddade till "
          f"{batch['bins'].shape[1]} positioner")
    print(f"  mask rad 1: {batch['mask'][1][:14].astype(int)} ... "
          f"(1 = padding, hit ska modellen inte titta)")

    jamfor_med_riktiga()