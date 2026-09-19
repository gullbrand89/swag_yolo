"""
dict <-> tokens.

Format (nivå- och längdcykel som separata block, oberoende perioder):
  LEVELS  L0 B<bin>  L1 B<bin> ...
  ORDER   FIXED L0 L2 L1   |  ORDER RANDOM
  DWELL   FIXED D10 D5     |  DWELL RANGE D5 D12   (min, max)
  END
Längd: D<k> pulser eller INF (static).

Nivåer och dwelltider har SKILDA tokenrymder, B respektive D. En bin är en mätning
med ändlig noggrannhet, en dwelltid ett exakt antal -- de tål inte samma utjämning.

Nivåblocket har INGET antalstoken. Antalet är redundant -- nivånamn och ORDER är
disjunkta tokenklasser, så parsern läser par tills den ser ORDER. Ett explicit antal
var dessutom det fält modellen var sämst på, och ett fel i det gjorde hela posten
otolkbar vid avkodning även när alla nivåer var rätt.

RANGE används bara när intervallet har bredd. Ett nollbrett intervall beskriver samma
emitter som en fast dwell och normaliseras till DWELL FIXED, så att en emitter aldrig
kan ha två giltiga facit.
"""
from .config import cfg
from .vocab import LEVEL_NAMES, TOK2ID, bin_of, pri_of_bin


def _min_period(x):
    """[10,10,10] -> [10]  och  [5,8,5,8] -> [5,8]. Oförändrad om ingen upprepning."""
    x = list(x)
    n = len(x)
    for p in range(1, n + 1):
        if n % p == 0 and all(x[i] == x[i % p] for i in range(n)):
            return x[:p]
    return x


def _rot_k(x):
    """Förskjutningen k som ger den minsta rotationen: x[k:] + x[:k]."""
    return min(range(len(x)), key=lambda k: x[k:] + x[:k])


def _rot(x):
    r = _rot_k(x)
    return x[r:] + x[:r]

def _L(n):
    return "INF" if n is None else f"D{int(n)}"


def to_tokens(levels, lengths, order_fixed, length_fixed, compress_lengths=None, start=None):
    """
    start : generatorns cykelindex för fönstrets FÖRSTA besök (all_emitters ger det
            per signal i label["start"]), eller None. Med start och fast ordning med
            minst tre positioner ANKRAS ORDER- och DWELL FIXED-blocken vid fönstrets
            början i stället för vid den lexikografiskt minsta rotationen. Se _canon.
    """
    return _canon(levels, lengths, order_fixed, length_fixed, compress_lengths, start)[0]


def canon_info(levels, lengths, order_fixed, length_fixed, compress_lengths=None, start=None):
    """
    -> dict(P, k, forced_order, order_fixed) för hjälp-målet aux_cyc.

    P : ORDER-cykelns längd i posten (minsta period av nivåföljden)
    k : rotationen to_tokens valde: seq_out[j] = seq_min[(j + k) % P]. En puls vars
        besök ligger på råposition r i generatorns cykel har alltså kanonisk position
        ((r mod P) - k) mod P -- det är vad encodern ska lära sig per puls.
    forced_order : ordningen tvingades till FIXED av kanoniseringen (<= 2 nivåer);
        då finns ingen råposition, och positionen ges av nivåns index i stället.
    """
    return _canon(levels, lengths, order_fixed, length_fixed, compress_lengths, start)[1]


def _canon(levels, lengths, order_fixed, length_fixed, compress_lengths=None, start=None):
    """
    levels : nivåcykeln i µs (kan innehålla upprepningar)
    lengths: längdcykeln i pulser (None = INF). Behöver INTE ha samma period som levels.
    compress_lengths : reducera längdcykeln till sin minsta period, så att
                       [10,10,10] blir [10] och [5,8,5,8] blir [5,8].
                       None -> använd cfg.compress_lengths (default True).

    Separata block:
      ORDER FIXED L0 L1 L2   DWELL FIXED N10 N5      (två cykler, ev. olika period)
      ORDER FIXED L0 L1 L2   DWELL RANGE N5 N12      (slumpad längd, min och max)
      ORDER RANDOM           DWELL FIXED N10 N5 N8
      ORDER RANDOM           DWELL RANGE N5 N12

    Kanonisering:
      * nivåer namnges L0, L1, ... efter bin-värde i stigande ordning
      * varje cykel reduceras till sin minsta period
      * varje cykel roteras till sin minsta rotation -- UTOM med start given och
        fast ordning med P >= 3: då roteras den så att position 0 är fönstrets
        första besök (ankring vid fönstret, se nedan)
      * har cyklerna samma period roteras de TILLSAMMANS, så att kopplingen
        nivå <-> längd bevaras

    Ankring vid fönstret (start):
      Den lexikografiskt minsta rotationen är en GLOBAL beräkning: för att veta
      att en puls står på position 0 måste man hitta cykelns lägsta nivå och, vid
      lika, jämföra framåt. tools.cyc_diag på körning 20260919_171551 visade att
      encodern inte lär sig det (cyc 0.27 på stagger, och lika lågt upp till
      rotation -- den har inte ens fasen). Med ankring vid fönstrets första besök
      är position j helt enkelt "nivån på besök j från start", som avkodaren kan
      slå upp via besökskanalen (data.make_channels, cfg.use_visit). Posten
      beskriver då DEN HÄR signalens observerade följd, i linje med att lambda i
      emittermodellen är signalens tillståndsföljd; konverteraren kanoniserar
      perioden själv och påverkas inte.
      P <= 2 ankras inte: med två nivåer är ordningen inte observerbar, och den
      tvingade ordningen (forced_order) och en äkta tvåcykel måste ge samma post.
      * ett nollbrett intervall skrivs om till en fast dwell: length_fixed=False
        med min == max blir DWELL FIXED
      * med en eller två nivåer är ordningen inte observerbar och skrivs alltid som
        ORDER FIXED -- se kommentaren vid forced_order
    """
    if compress_lengths is None:
        compress_lengths = getattr(cfg, "compress_lengths", True)

    levels = list(levels)
    lengths = [None if x is None else int(x) for x in lengths]

    # ---- kanonisering: ett nollbrett intervall ÄR en fast dwell
    # Utan den här regeln har samma emitter två giltiga facit -- DWELL RANGE N2 N2
    # och DWELL FIXED N2 -- och då ser modellen identiska signaler med olika mål.
    # Det blir ett golv i lossen som ingen träning tar bort.
    # Villkoret None not in lengths är inte kosmetiskt: None betyder INF, alltså en
    # nivå som aldrig lämnas, och den informationen får inte tyst filtreras bort.
    if not length_fixed and lengths and None not in lengths:
        if min(lengths) == max(lengths):
            length_fixed = True
            if compress_lengths:
                lengths = [lengths[0]]

    # ---- nivådefinitioner
    bins = [bin_of(v) for v in levels]
    uniq = sorted(set(bins))
    assert len(uniq) <= cfg.max_levels, f"{len(uniq)} nivåer, max {cfg.max_levels}"
    # Två skilda nivåvärden får inte hamna i samma bin -- då blir facitet oobserverbart.
    # (Upprepningar i cykeln är däremot tillåtna, jfr ds_revisit.)
    assert len(uniq) == len({round(float(v), 9) for v in levels}), \
        f"bin-kollision mellan nivåer: {sorted(set(levels))} -> bins {uniq}"
    name = {b: LEVEL_NAMES[i] for i, b in enumerate(uniq)}
    seq = [name[b] for b in bins]

    # ---- kanonisering: ordningen är inte observerbar med färre än tre nivåer
    # Med EN nivå sker inga byten alls. Med TVÅ nivåer alternerar den observerade
    # följden alltid -- en omedelbar upprepning smälter ihop med föregående besök och
    # syns inte i signalen, så en slumpad ordning ger exakt samma pulsföljd som
    # cykeln [L0, L1]. ORDER RANDOM och ORDER FIXED beskriver då samma emitter, och
    # utan den här regeln får identiska signaler två olika facit.
    forced_order = not order_fixed and len(uniq) <= 2
    if forced_order:
        order_fixed = True
        seq = [name[b] for b in uniq]

    t = ["LEVELS"]
    for b in uniq:
        t += [name[b], f"B{b}"]

    # ---- reducera cyklerna till minsta period
    seq_min = _min_period(seq)
    len_min = _min_period(lengths) if compress_lengths else list(lengths)

    # ---- ordning + längder
    k_rot = 0
    # ankring vid fönstret: besök j från start ligger på råposition (start + j) i
    # generatorns cykel, alltså på seq_min[(start + j) % P] -- rotationen är start % P
    anchored = start is not None and order_fixed and not forced_order and len(seq_min) >= 3
    if order_fixed:
        # forced_order: ordningen är påtvingad av kanoniseringen ovan, inte uppgiven av
        # generatorn. Då finns ingen känd koppling nivå <-> längd att bevara, och att
        # rotera ihop dem skulle hitta på en.
        if length_fixed and not forced_order and len(len_min) == len(seq_min):
            # samma period: rotera ihop så att kopplingen bevaras
            par = list(zip(seq_min, len_min))
            k_rot = (int(start) % len(par)) if anchored else _rot_k(par)
            pairs = par[k_rot:] + par[:k_rot]
            seq_out = [p[0] for p in pairs]
            len_out = [p[1] for p in pairs]
        else:
            k_rot = (int(start) % len(seq_min)) if anchored else _rot_k(seq_min)
            seq_out, len_out = seq_min[k_rot:] + seq_min[:k_rot], None
        t += ["ORDER", "FIXED"] + seq_out
    else:
        seq_out, len_out = None, None
        t += ["ORDER", "RANDOM"]

    if length_fixed:
        if len_out is None:
            if anchored:
                # olika period: dwellcykeln ankras på samma sätt. Generatorn rullar
                # längderna med samma löpande index som nivåerna (lphase = phase), så
                # besök j har dwell len_min[(start + j) % len(len_min)].
                kl = int(start) % len(len_min)
                len_out = len_min[kl:] + len_min[:kl]
            else:
                len_out = _rot(len_min)
        t += ["DWELL", "FIXED"] + [_L(n) for n in len_out]
    else:
        # Den gamla koden filtrerade bort None här och gav "N0 N0" om allt var None.
        # Båda är tyst informationsförlust: INF betyder "lämnar aldrig nivån" och går
        # inte att uttrycka som ett intervall, så ett RANGE med INF är ett
        # självmotsägande facit. Bättre att smälla än att skicka nonsens till modellen.
        assert lengths, "DWELL RANGE utan längder"
        assert None not in lengths, f"DWELL RANGE kan inte innehålla INF: {lengths}"
        t += ["DWELL", "RANGE", f"D{min(lengths)}", f"D{max(lengths)}"]

    t += ["END"]

    bad = [x for x in t if x not in TOK2ID]
    assert not bad, f"tokens saknas i vokabulär: {bad}"
    assert len(t) + 2 <= cfg.max_tgt, f"facit {len(t)+2} tokens > max_tgt {cfg.max_tgt}"
    info = dict(P=len(seq_min) if order_fixed else None, k=k_rot,
                forced_order=forced_order, order_fixed=order_fixed, anchored=anchored,
                bins=list(uniq))       # sorterade nivåbins; index = nivånamnets nummer
    return t, info


def parse(tokens):
    it = iter(tokens)
    def nxt():
        try: return next(it)
        except StopIteration: raise ValueError("oväntat slut")
    def bin_num(tok):
        if not (tok.startswith("B") and tok[1:].isdigit()):
            raise ValueError(f"väntade nivåbin, fick {tok}")
        return int(tok[1:])
    def dur_num(tok):
        if not (tok.startswith("D") and tok[1:].isdigit()):
            raise ValueError(f"väntade dwelltid, fick {tok}")
        return int(tok[1:])
    def length(tok): return None if tok == "INF" else dur_num(tok)
    def is_level(tok): return tok.startswith("L") and tok[1:].isdigit()

    if nxt() != "LEVELS": raise ValueError("saknar LEVELS")
    # Inget antalstoken: läs (nivånamn, bin)-par tills ORDER dyker upp. Nivånamn och
    # ORDER är disjunkta tokenklasser, så avgränsningen är entydig.
    name2pri = {}
    tok = nxt()
    while is_level(tok):
        name2pri[tok] = pri_of_bin(bin_num(nxt()))
        tok = nxt()
    if not name2pri: raise ValueError("tom nivådefinition")

    if tok != "ORDER": raise ValueError(f"väntade ORDER, fick {tok}")
    ot = nxt(); order, lengths = [], []
    order_fixed = length_fixed = None
    tok = nxt()
    if ot == "FIXED":
        order_fixed = True
        while is_level(tok):
            if tok not in name2pri: raise ValueError(f"odefinierad nivå {tok}")
            order.append(name2pri[tok]); tok = nxt()
        if not order: raise ValueError("tom ordning")
    elif ot == "RANDOM":
        order_fixed = False
    else:
        raise ValueError(f"okänd ORDER-typ {ot}")

    if tok != "DWELL": raise ValueError(f"väntade DWELL, fick {tok}")
    dt = nxt()
    if dt not in ("FIXED", "RANGE"): raise ValueError(f"okänd DWELL-typ {dt}")
    length_fixed = dt == "FIXED"; tok = nxt()
    while tok != "END":
        lengths.append(length(tok)); tok = nxt()
    if not lengths: raise ValueError("tomt DWELL-block")
    if not length_fixed and len(lengths) != 2:
        raise ValueError(f"DWELL RANGE kräver min och max, fick {len(lengths)}")

    if tok != "END": raise ValueError(f"väntade END, fick {tok}")
    if length_fixed is None: raise ValueError("saknar längdinformation")
    return dict(levels=sorted(name2pri.values()), order=order, lengths=lengths,
                order_fixed=order_fixed, length_fixed=length_fixed)


def roundtrip_ok(levels, lengths, order_fixed, length_fixed):
    t = to_tokens(levels, lengths, order_fixed, length_fixed)
    d = parse(t)
    src = d["order"] if d["order_fixed"] else d["levels"]
    return to_tokens(src, d["lengths"], d["order_fixed"], d["length_fixed"]) == t
