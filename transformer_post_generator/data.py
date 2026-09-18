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
_AUX = ("aux_new", "aux_pos", "aux_merge")
_AUX_OK = "with_aux" in inspect.signature(create_emitter_data).parameters


def label_to_tokens(label):
    return to_tokens(label["levels"], label["lengths"], label["order_fixed"], label["length_fixed"])


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


def make_channels(pri_obs, aux=None):
    """aux : dict med new/pos/merge från generatorn, eller None (-> ignoreras i lossen)."""
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

    # pri följer med rå: RL-belöningen mäter mot signalen, inte mot facitet
    return dict(bins=bins, cont=cont, toa=toa, rl=rl, flag=flag,
                pri=p.astype(np.float32), **ch_aux)


_CHANNELS = ("bins", "cont", "toa", "rl", "flag")


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
        tokens = label_to_tokens(label)
        sigs = as_signals(seqs)
        auxs = label.get("aux") or [None] * len(sigs)
        for s, a in zip(sigs, auxs):        # en signal i taget, aldrig hela arrayen
            out.append((make_channels(s, a), tokens))
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
        mask=torch.ones(B, T, dtype=torch.bool),
    )
    for k in _AUX:                                      # utfyllnad = ignoreras i lossen
        src[k] = torch.full((B, T), AUX_IGNORE, dtype=torch.long)
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
