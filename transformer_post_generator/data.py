"""
Datapipeline runt generatorn.

Vilken generator som används styrs av `cfg.emitter` (modulnamn, default "all_emitters").
Modulen måste exportera:

    create_emitter_data(n_emitters, n_signals, drop_rate, noise_level, rng)
        -> lista av (pri_sequences, label)
    pri_sequences : lista med n_signals observerade PRI-arrayer (µs), samma emitter, olika startfas,
                    bortfall/brus redan applicerat
    label         : dict med levels, lengths, order_fixed, length_fixed

Samma rng-seed ska ge samma emittrar OCH samma signaler oavsett drop_rate/noise_level;
all_emitters gör det med tre separata strömmar (emitter / signal / bortfall).

Valfritt:  detect_missing(pri_obs) -> bool-array, från din missing-pulse-detektor.

Valfritt:  create_emitter_data(..., with_aux=True) lägger facit per puls i
           label["aux"] (en dict per signal med new, pos, merge). Används av
           hjälp-lossen när cfg.aux_weight > 0. En generator som saknar parametern
           fungerar ändå: målen fylls då med AUX_IGNORE och hjälp-lossen blir noll.
"""
import importlib
import inspect

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .vocab import safe_ids, PAD, BOS, EOS
from .labels import to_tokens

from .config import cfg
_emitter = importlib.import_module(cfg.emitter,package=__package__)
create_emitter_data = _emitter.create_emitter_data
detect_missing = getattr(_emitter, "detect_missing", None)

AUX_IGNORE = -100          # ignore_index i loss.aux_loss; även utfyllnad i _pack
_AUX = ("aux_new", "aux_pos", "aux_merge", "aux_cyc")
_AUX_OK = "with_aux" in inspect.signature(create_emitter_data).parameters


def label_to_tokens(label, i=0):
    """
    Posten för signal i under etiketten. Facitet är per SIGNAL sedan ORDER-blocket
    ankras vid fönstret (labels.to_tokens, start): label["start"][i] är fönstrets
    första cykelposition. Saknas nyckeln (äldre generator) blir posten den
    kanoniska, som förut.
    """
    start = label.get("start")
    st = None if start is None else start[i]
    return to_tokens(label["levels"], label["lengths"], label["order_fixed"],
                     label["length_fixed"], start=st)


# ---------------- input-kanaler
def run_length(bins, flags=None):
    """Hur många pulser i rad som legat på samma bin. Sekventiell till sin natur."""
    rl = np.zeros(len(bins), dtype=np.int64)
    for i in range(1, len(bins)):
        if flags is not None and flags[i]:
            rl[i] = rl[i - 1] + 2                      # flaggat intervall räknas som två pulser
        elif abs(int(bins[i]) - int(bins[i - 1])) <= cfg.run_tol_bins:
            rl[i] = rl[i - 1] + 1
        else:
            rl[i] = 0
    return rl


def recur_lag(bins, rl):
    """
    Per puls: hur många BESÖK sedan den här nivån senast besöktes. 0 = aldrig.

    Ett besök börjar där rl == 0. Nivån identifieras med besökets första bin, inom
    cfg.run_tol_bins -- samma tolerans som räknaren, så de två kanalerna är
    konsekventa om vad som är "samma nivå".

    Varför den finns
    ----------------
    ORDER FIXED betyder att besöksföljden är periodisk. Med P unika nivåer i cykeln
    får varje puls lag = P, konstant. Med slumpad ordning varierar lagen. Det är
    alltså FIXED/RANDOM-beslutet utskrivet som en kanal -- en enda skalär på den
    (andelen pulser på den vanligaste lagen) skiljer klasserna med 0.80 på ren data.

    Utan kanalen måste encodern räkna besök själv och jämföra position i med den
    puls som ligger p BESÖK tillbaka, vilket är ett varierande PULSavstånd. RoPE
    ger fasta pulsavstånd. Modellen stod på klassprioren (order_type_ok 0.67) i
    20 000 steg, med alla lossvikter, och med bortfallet avstängt. Den kunde inte.

    Bortfall
    --------
    Ett ihopslaget intervall hamnar på en bin som inte är en nivå, blir ett falskt
    besök, och rubbar lagen. Kanalen är därför brusig vid bortfall, och det är
    modellens sak att lära sig när den går att lita på -- den har aux_merge, som
    hittar två av tre sammanslagningar. Kanalen levererar råvaran, inte svaret.

    Sekventiell, som run_length. Ordboken senast[bin] är liten (antal nivåer).
    """
    n = len(bins)
    lag = np.zeros(n, dtype=np.int64)
    if n == 0:
        return lag
    tol = cfg.run_tol_bins
    senast = {}                 # besökets bin -> index på det senaste besöket dit
    besok = 0
    cur_bin = None
    cur_lag = 0
    for i in range(n):
        if i == 0 or rl[i] == 0:
            if cur_bin is not None:
                senast[cur_bin] = besok
            besok += 1
            cur_bin = int(bins[i])
            traffar = [v for b, v in senast.items() if abs(b - cur_bin) <= tol]
            cur_lag = besok - max(traffar) if traffar else 0
        lag[i] = cur_lag
    return lag


def n_levels_of(tokens):
    """Antal nivåer i en post = antal nivånamn i LEVELS-blocket. Det är exakt det
    antal avkodaren ska skriva, så det är målet för räknehuvudet -- inte
    len(set(levels)), som skulle kunna skilja sig om två nivåer delade bin."""
    n = 0
    for t in tokens[1:]:
        if t == "ORDER":
            break
        if t[0] == "L" and t[1:].isdigit():
            n += 1
    return n


def n_order_of(tokens):
    """Cykelns längd P i posten = antal nivånamn i ORDER FIXED-blocket. None vid
    RANDOM. Målet för periodhuvudet, av samma skäl som n_levels_of är räknemålet."""
    i = tokens.index("ORDER") + 1
    if tokens[i] != "FIXED":
        return None
    n = 0
    for t in tokens[i + 1:]:
        if t == "DWELL":
            break
        n += 1
    return n


def make_channels(pri_obs, aux=None, n_levels=None, period=None):
    """
    aux      : dict med new/pos/merge från generatorn, eller None (-> ignoreras i lossen).
    n_levels : antal nivåer i facitet, mål för räknehuvudet. None -> ignoreras.
    period   : ORDER-cykelns längd, mål för periodhuvudet. None -> ignoreras.
    """
    p = np.asarray(pri_obs, dtype=float)
    if p.ndim != 1:
        raise ValueError(
            f"make_channels väntade EN pulsföljd (1-D), fick shape {p.shape}. "
            f"Skickas hela listan med signaler in i stället för en enskild?")

    x = (np.clip(p, cfg.in_min, cfg.in_max) - cfg.in_min) / (cfg.in_max - cfg.in_min)

    bins = np.clip((x * (cfg.in_bins - 2)).astype(np.int64), 0, cfg.in_bins - 2)
    bins[p >= cfg.in_max] = cfg.in_bins - 1            # overflow-bin
    cont = (x * 2 - 1).astype(np.float32)
    toa = (np.cumsum(p) / cfg.toa_scale).astype(np.float32)

    flags = None
    if cfg.use_drop_flag and detect_missing is not None:
        flags = np.asarray(detect_missing(p), dtype=bool)
    rl = run_length(bins, flags) if cfg.use_counter else np.zeros(len(bins), dtype=np.int64)
    flag = flags.astype(np.int64) if flags is not None else np.zeros(len(bins), dtype=np.int64)
    # recur bygger på rl:s besöksgränser. Är räknaren av finns inga gränser att
    # räkna på, så kanalen är meningslös utan den -- använd rl räknad ändå.
    if cfg.use_recur:
        rl_for_recur = rl if cfg.use_counter else run_length(bins, flags)
        recur = recur_lag(bins, rl_for_recur)
    else:
        recur = np.zeros(len(bins), dtype=np.int64)
    # Besökskanalen: löpnummer på besöket räknat från fönstrets början, 0, 1, 2, ...
    # (tak cfg.max_visit). Posten ankras vid fönstret, så ORDER FIXED plats j är
    # nivån på besök j -- med den här kanalen är det en uppslagning för avkodarens
    # cross-attention i stället för en räkning. Byggd på rl:s besöksgränser som recur.
    if cfg.use_visit:
        rl_v = rl if cfg.use_counter else run_length(bins, flags)
        visit = np.minimum(np.cumsum(rl_v == 0) - 1, cfg.max_visit).astype(np.int64)
    else:
        visit = np.zeros(len(bins), dtype=np.int64)

    # Facit till hjälp-lossen. Det är MÅL, inte input: modellen läser aldrig de här
    # nycklarna (InputEmbedding tar bara bins, cont, rl, flag), de följer bara med i
    # src-dicten för att hamna på samma device som resten.
    ch_aux = {}
    for k in _AUX:
        a = None if aux is None else aux.get(k[4:])
        if a is None:
            a = np.full(len(p), AUX_IGNORE, dtype=np.int64)
        a = np.asarray(a, dtype=np.int64)
        assert len(a) == len(p), f"{k}: {len(a)} mål för {len(p)} pulser"
        ch_aux[k] = a

    # Räknehuvudets mål: ett tal per SEKVENS, inte per puls. Samma AUX_IGNORE.
    ch_aux["aux_count"] = np.int64(AUX_IGNORE if n_levels is None else n_levels)
    ch_aux["aux_period"] = np.int64(AUX_IGNORE if period is None else period)

    # pri följer med rå: RL-belöningen mäter mot signalen, inte mot facitet
    return dict(bins=bins, cont=cont, toa=toa, rl=rl, flag=flag, recur=recur, visit=visit,
                pri=p.astype(np.float32), **ch_aux)


_CHANNELS = ("bins", "cont", "toa", "rl", "flag", "recur", "visit")


def as_signals(seqs):
    """
    Normalisera till en lista med 1-D-pulsföljder. Generatorer skiljer sig här:
    lista med arrayer, 2-D-array (n_signals, n_pulser), eller en naken 1-D-array
    när n_signals == 1.
    """
    if isinstance(seqs, (list, tuple)):
        return [np.asarray(s, dtype=float) for s in seqs]
    a = np.asarray(seqs, dtype=float)
    return [a] if a.ndim == 1 else [row for row in a]


def make_pairs(rng, n_emitters=1, p_drop=None, samples=None, **kw):
    """-> lista av (channels, tokens), en post per signal."""
    if cfg.aux_weight > 0 and _AUX_OK:
        kw = dict(kw, with_aux=True)
    data = create_emitter_data(n_emitters, samples or cfg.samples_per_emitter, p_drop,
                               cfg.noise_level, rng, **kw)
    out = []
    for seqs, label in data:
        sigs = as_signals(seqs)
        auxs = label.get("aux") or [None] * len(sigs)
        for i, (s, a) in enumerate(zip(sigs, auxs)):   # en signal i taget, aldrig hela arrayen
            tokens = label_to_tokens(label, i)          # per signal: ankrad vid fönstret
            k = n_levels_of(tokens) if cfg.aux_count else None
            P = n_order_of(tokens) if cfg.aux_cyc else None
            out.append((make_channels(s, a, k, P), tokens))
    return out

class StreamDataset(Dataset):
    """
    Obegränsad data; varje __getitem__ ger en ny emitter med alla dess startfaser,
    alltså en LISTA med cfg.samples_per_emitter par. collate plattar ut den, så
    DataLoader ska ha batch_size = cfg.batch // cfg.samples_per_emitter.

    Indexet ignoreras (strömmen är oändlig), så shuffle=True gör ingen nytta.
    """
    def __init__(self, n, seed):
        self.n, self.seed, self.rng = n, seed, None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.rng is None:
            w = get_worker_info()
            self.rng = np.random.default_rng([self.seed, w.id if w else 0])
        p = float(self.rng.uniform(0, cfg.p_drop)) if cfg.randomize_p_drop else cfg.p_drop
        return make_pairs(self.rng, p_drop=p)


def make_eval_sets():
    """Samma emittrar OCH samma signaler i varje bortfallsvariant."""
    return {f"drop_{p:.2f}": make_pairs(np.random.default_rng(cfg.eval_seed),
                                       cfg.eval_n_emitters, p,
                                       samples=cfg.eval_samples_per_emitter)
            for p in cfg.eval_p_drops}


# ---------------- collate
def _flatten(batch):
    flat = []
    for b in batch:
        flat.extend(b if isinstance(b, list) else [b])
    return flat


def _pack(flat):
    B = len(flat)
    T = max(len(ch["bins"]) for ch, _ in flat)
    tg = [[BOS] + safe_ids(tok) + [EOS] for _, tok in flat]
    L = max(map(len, tg))

    src = dict(
        bins=torch.zeros(B, T, dtype=torch.long),
        cont=torch.zeros(B, T),
        toa=torch.zeros(B, T),
        rl=torch.zeros(B, T, dtype=torch.long),
        flag=torch.zeros(B, T, dtype=torch.long),
        recur=torch.zeros(B, T, dtype=torch.long),
        visit=torch.zeros(B, T, dtype=torch.long),
        mask=torch.ones(B, T, dtype=torch.bool),
    )
    for k in _AUX:                                      # utfyllnad = ignoreras i lossen
        src[k] = torch.full((B, T), AUX_IGNORE, dtype=torch.long)
    src["aux_count"] = torch.tensor([int(ch["aux_count"]) for ch, _ in flat],
                                    dtype=torch.long)
    src["aux_period"] = torch.tensor([int(ch["aux_period"]) for ch, _ in flat],
                                     dtype=torch.long)
    tgt = torch.full((B, L), PAD, dtype=torch.long)
    for i, (ch, _) in enumerate(flat):
        n = len(ch["bins"])
        for k in _CHANNELS + _AUX:
            src[k][i, :n] = torch.as_tensor(ch[k])
        src["mask"][i, :n] = False
        tgt[i, :len(tg[i])] = torch.tensor(tg[i])
    return src, tgt


def collate(batch):
    src, tgt = _pack(_flatten(batch))
    return src, tgt[:, :-1], tgt[:, 1:]


def collate_rl(batch):
    """
    Som collate, men lämnar även de råa PRI-sekvenserna. GRPO behöver dem: belöningen
    mäter det genererade facitet mot SIGNALEN, inte mot tgt_out.
    -> (src, tgt_in, tgt_out, [pri_0, pri_1, ...])
    """
    flat = _flatten(batch)
    src, tgt = _pack(flat)
    pris = [np.asarray(ch["pri"], dtype=float) for ch, _ in flat]
    return src, tgt[:, :-1], tgt[:, 1:], pris
