"""
Två varianter med samma gränssnitt:
    model(src, tgt_in) -> logits
    model.greedy(src)  -> token-ids
src är dicten från collate (bins, cont, toa, rl, flag, mask).

Ändringar mot föregående version, båda i encoderns attention:

  * QK-normalisering. Ingenting begränsade tidigare q·k, så attention-logitarna
    kunde växa med qkv-vikternas norm. Softmaxen blir då successivt spetsigare tills
    varje position i praktiken attenderar på en enda nyckel, gradienten genom
    attention kollapsar och modellen tappar uttrycksförmåga -- för ALLA uppgifter
    samtidigt, även de redan lösta. Encodern kör över 1024 positioner, vilket är den
    regim där det inträffar. Styrs av cfg.qk_norm (default True) så att den går att
    ablera.

  * RoPE i fp32. apply_rope multiplicerade cos/sin med q och k, som under autocast är
    bfloat16 -- åtta mantissabitar, alltså ungefär tre decimalers precision på
    rotationen. Rotationen görs nu i fp32 och castas tillbaka.

Hjälp-huvuden (cfg.aux_weight > 0): tre linjära lager på encoderns utdata som per puls
svarar på "nytt besök?", "position i uppehållet" och "antal ihopslagna intervall".
De används bara i träningen -- model(src, tgt_in, return_aux=True) -- och påverkar
varken generate() eller greedy(). Se loss.aux_loss.

Decodern använder nn.TransformerDecoderLayer och har ingen QK-normalisering. Den kör
över ~40 token i stället för 1024 och är därför betydligt mindre utsatt; vill man ha
det även där måste lagret skrivas för hand. Detsamma gäller IndexModel.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import cfg
from .vocab import VOCAB, PAD, BOS, EOS, TOK2ID, LEVEL_NAMES, IS_BIN, BIN_START

_ORDER_ID = TOK2ID["ORDER"]
_LEVELS_ID = TOK2ID["LEVELS"]
_LVL_LO, _LVL_HI = TOK2ID[LEVEL_NAMES[0]], TOK2ID[LEVEL_NAMES[-1]]


# ---------------- gemensamt
class SinusoidalIndex(nn.Module):
    def __init__(self, d, maxlen):
        super().__init__()
        pe = torch.zeros(maxlen, d); pos = torch.arange(maxlen)[:, None]
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, n):
        assert n <= self.pe.size(0), f"sekvens {n} > maxlen {self.pe.size(0)}"
        return self.pe[:n]

class SinusoidalTOA(nn.Module):
    def __init__(self, d, max_period=10000.0):
        super().__init__()
        self.register_buffer("div", torch.exp(torch.arange(0, d, 2) * (-math.log(max_period) / d)))
    def forward(self, t):
        ang = t[..., None] * self.div
        return torch.cat([ang.sin(), ang.cos()], -1)

class InputEmbedding(nn.Module):
    """bins + cont + (räknare) + (flagga) + (återbesökslag)"""
    def __init__(self, d):
        super().__init__()
        self.bin = nn.Embedding(cfg.in_bins, d)
        self.cont = nn.Linear(1, d)
        self.rl = nn.Embedding(cfg.max_run + 1, d) if cfg.use_counter else None
        self.flag = nn.Embedding(2, d) if cfg.use_drop_flag else None
        # återbesökslag, se data.recur_lag och cfg.use_recur
        self.recur = nn.Embedding(cfg.max_recur + 1, d) if cfg.use_recur else None
    def forward(self, src):
        x = self.bin(src["bins"]) + self.cont(src["cont"][..., None])
        if self.rl is not None:
            x = x + self.rl(src["rl"].clamp(max=cfg.max_run))
        if self.flag is not None:
            x = x + self.flag(src["flag"])
        if self.recur is not None:
            x = x + self.recur(src["recur"].clamp(max=cfg.max_recur))
        return x

class Decoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.emb = nn.Embedding(len(VOCAB), d, padding_idx=PAD)
        self.pos = SinusoidalIndex(d, cfg.max_tgt)
        layer = nn.TransformerDecoderLayer(d, cfg.nhead, cfg.ff, cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, cfg.dec_layers, norm=nn.LayerNorm(d))
        self.out = nn.Linear(d, len(VOCAB))
    def forward(self, mem, mask, tgt_in):
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), 1)
        y = self.emb(tgt_in) + self.pos(T)
        h = self.dec(y, mem, tgt_mask=causal, tgt_key_padding_mask=(tgt_in == PAD),
                     memory_key_padding_mask=mask)
        return self.out(h)


class AuxHeads(nn.Module):
    """
    Hjälp-huvuden på encoderns utdata. -> dict med logits.
      new, pos, merge : per puls, (B, T, klasser)
      count           : per SEKVENS, (B, max_levels + 1) -- antal nivåer, klass K = K

    count är räknehuvudet. Bakgrund: n_levels_ok stod på 0.68 genom tre
    avkodningsvillkor och hoppade till 0.99 när antalet gavs utifrån (tools.reeval
    --count). Modellen kan välja rätt bins men inte avgöra hur många. Avkodaren
    kan inte räkna autoregressivt; encodern får göra det i ett svep, med recur-
    kanalen tillgänglig (max lag ~ antal nivåer vid fast ordning) och hela
    signalen i blickfånget. Poolningen är ett maskat medel över pulserna.
    """
    def __init__(self, d):
        super().__init__()
        self.new = nn.Linear(d, 2)                     # nytt besök: nej / ja
        self.pos = nn.Linear(d, cfg.max_dur + 1)       # position i uppehållet, 0..max_dur
        self.merge = nn.Linear(d, 4)                   # 0, 1, 2, 3+ tappade pulser
        self.count = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                   nn.Linear(d, cfg.max_levels + 1)) if cfg.aux_count else None
    def forward(self, mem, mask=None):
        out = dict(new=self.new(mem), pos=self.pos(mem), merge=self.merge(mem))
        if self.count is not None:
            if mask is None:
                pooled = mem.mean(1)
            else:
                keep = (~mask).float()[..., None]                    # mask=True är utfyllnad
                pooled = (mem * keep).sum(1) / keep.sum(1).clamp(min=1.0)
            out["count"] = self.count(pooled)
        return out


def _aux_heads(d):
    return AuxHeads(d) if cfg.aux_weight > 0 else None


class Base(nn.Module):
    def forward(self, src, tgt_in, return_aux=False):
        """return_aux=True -> (logits, aux), där aux är None om huvudena är avstängda."""
        mem = self.encode(src)
        logits = self.decoder(mem, src["mask"], tgt_in)
        if not return_aux:
            return logits
        return logits, (self.aux(mem, src["mask"]) if self.aux is not None else None)

    @torch.no_grad()
    def generate(self, src, n=1, greedy=True, temperature=1.0, top_k=0,
                 max_new=None, generator=None):
        """
        Avkoda n sekvenser per exempel.

        -> (B*n, L) token-ids som börjar med BOS. Rad i*n + j är sampel j av
           exempel i, alltså ligger varje grupp sammanhängande -- det är den
           ordningen GRPO förutsätter när den räknar gruppens baslinje.

        Encodern körs en gång och minnet upprepas, så n sampel kostar n gånger
        avkodaren men bara en gång encodern.
        """
        mem, mask = self.encode(src), src["mask"]
        if n > 1:
            mem = mem.repeat_interleave(n, 0)
            mask = mask.repeat_interleave(n, 0)

        B, dev = mem.size(0), mem.device
        steps = min(max_new or cfg.max_tgt - 1, cfg.max_tgt - 1)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)

        # Villkorad avkodning i nivåblocket, se _level_constraint. Tillståndet per rad:
        # är vi fortfarande före ORDER, och vilken bin skrevs senast.
        in_levels = torch.ones(B, dtype=torch.bool, device=dev)
        last_bin = torch.full((B,), BIN_START - 1, dtype=torch.long, device=dev)
        is_bin = IS_BIN.to(dev)
        present = None
        k_hat = None
        kluster = None
        if cfg.constrain_levels and cfg.constrain_present:
            present = self._present_bins(src)
            if cfg.constrain_count:
                har_huvud = self.aux is not None and getattr(self.aux, "count", None) is not None
                if cfg.count_source == "model" and har_huvud:
                    # räknehuvudet: K ur encodern, i ett svep. Tål bortfall.
                    # mem/mask är redan upprepade n gånger, så k_hat blir (B*n,) direkt.
                    k_hat = self.aux(mem, mask)["count"].argmax(-1).clamp(min=1)
                    # ingen kluster-per-val-koppling: med bortfall finns fler kluster än
                    # nivåer, och vilka K av dem som är riktiga är modellens sak att välja
                    kluster = None
                else:
                    # klusterräkning: exakt på ren data, överskattar under bortfall
                    k_hat = self._count_present(present)
                    kluster = self._cluster_index(present)
                    if n > 1:
                        k_hat = k_hat.repeat_interleave(n, 0)
                        kluster = kluster.repeat_interleave(n, 0)
            if n > 1:
                present = present.repeat_interleave(n, 0)
        n_names = torch.zeros(B, dtype=torch.long, device=dev)

        for _ in range(steps):
            logits = self.decoder(mem, mask, ys)[:, -1].float()
            logits[:, PAD] = -1e30                      # PAD och BOS är aldrig giltiga
            logits[:, BOS] = -1e30                      # utdata mitt i en sekvens
            if cfg.constrain_levels:
                logits = self._level_constraint(logits, ys[:, -1], in_levels, last_bin,
                                                is_bin, present, k_hat, n_names, kluster)
            if greedy:
                nxt = logits.argmax(-1)
            else:
                logits = logits / max(float(temperature), 1e-6)
                if top_k > 0:
                    kth = logits.topk(min(top_k, logits.size(-1)), -1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, -1e30)
                nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)[:, 0]
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            # uppdatera avkodningstillståndet
            wrote_bin = is_bin[nxt]
            last_bin = torch.where(wrote_bin, nxt, last_bin)
            wrote_name = (nxt >= _LVL_LO) & (nxt <= _LVL_HI) & in_levels
            n_names = n_names + wrote_name.long()
            in_levels &= nxt != _ORDER_ID
            if done.all():
                break
        return ys

    @staticmethod
    def _present_bins(src):
        """
        (B, V) bool: vilka B-token som motsvarar en bin som FÖREKOMMER i insignalen,
        utvidgat med +-cfg.run_tol_bins.

        En nivå är per definition en PRI som sänds, alltså en bin som finns bland
        pulserna. Input- och facitbins delar index (config.in_max är vald så, och
        tests.py vaktar det), så mängden är direkt avläsbar ur src["bins"]. Bins
        >= n_bins ligger över facitrymden -- det är sammanslagna intervall -- och
        tas inte med.

        På ren data ÄR den här mängden nivåmängden. Under bortfall är den en
        övermängd, och modellens jobb blir att välja bland det som finns i stället
        för att räkna upp ur 2048 möjliga. Det senare gick sönder över åtta nivåer:
        mängden rätt i 12 % vid 9-16 nivåer, 0 % vid 17-32, med sex FELPLACERADE
        nivåer per post -- bins som ingen puls låg på.
        """
        bins, pad = src["bins"], src["mask"]
        B, dev = bins.size(0), bins.device
        V = len(VOCAB)
        present = torch.zeros(B, V, dtype=torch.bool, device=dev)
        valid = (~pad) & (bins < cfg.n_bins)
        rows = torch.arange(B, device=dev)[:, None].expand_as(bins)
        tol = cfg.run_tol_bins
        for off in range(-tol, tol + 1):
            tok = (bins + off).clamp(0, cfg.n_bins - 1) + BIN_START
            present[rows[valid], tok[valid]] = True
        return present

    @staticmethod
    def _count_present(present):
        """
        (B,) antal bin-KLUSTER i present-mängden: en löpande sträcka av tillåtna
        B-token räknas som ett kluster. Med +-run_tol_bins utvidgning blir varje
        nivå en sträcka på 2*tol+1 bins, och två nivåer längre isär än så blir två.

        På ren data är det här antalet nivåer, exakt. Under bortfall är det en
        ÖVERSKATTNING -- sammanslagna intervall under pri_max ger egna kluster.
        Värdet är alltså ett tak att mäta mot, inte ett facit att lita på; en
        modell som får sätta antalet själv gör det bättre vid bortfall. Se
        cfg.constrain_count.
        """
        p = present[:, BIN_START:BIN_START + cfg.n_bins]
        start = p & ~torch.cat([torch.zeros_like(p[:, :1]), p[:, :-1]], 1)
        return start.sum(1)

    @staticmethod
    def _cluster_index(present):
        """
        (B, V) long: för varje B-token, vilket kluster (0, 1, 2, ...) i present det
        tillhör. -1 för token utanför present. Klustren räknas i stigande bin-ordning,
        samma räkning som _count_present.
        """
        B, V = present.shape
        p = present[:, BIN_START:BIN_START + cfg.n_bins]
        start = p & ~torch.cat([torch.zeros_like(p[:, :1]), p[:, :-1]], 1)
        idx = torch.cumsum(start.long(), 1) - 1
        idx = torch.where(p, idx, torch.full_like(idx, -1))
        ut = torch.full((B, V), -1, dtype=torch.long, device=present.device)
        ut[:, BIN_START:BIN_START + cfg.n_bins] = idx
        return ut

    @staticmethod
    def _level_constraint(logits, prev, in_levels, last_bin, is_bin, present=None,
                          k_hat=None, n_names=None, kluster=None):
        """
        Nivåblocket är en MÄNGD, skriven strikt stigande: LEVELS L0 B10 L1 B25 ... ORDER.
        Efter ett nivånamn får därför bara en bin STÖRRE än den senaste följa. Utan
        det här villkoret stammar modellen -- L5 B482 L6 B482 L7 B482 -- när den inte
        kan bestämma sig för att sluta: en upprepning av senaste bin är billigast under
        ordinalutjämningen, och ingenting säger avkodaren att den är ogiltig.

        I run C (recur, steg 4000) stammade 43 % av posterna och nivåmängden var rätt i
        51 %, trots att 99,3 % av nivåerna syns i fönstret. Villkoret tar bort
        upprepningen och låter den bin modellen hade som tvåa -- ofta nästa riktiga
        nivå -- komma fram.

        "Strikt över" betyder över last_bin + 2*run_tol_bins. present utvidgar varje
        nivå till ett kluster på 2*tol+1 bins; tog modellen klustrets nederkant
        (B481 för en nivå på B482) sträcker sig klustret till B483, så gapet måste
        vara hela klusterbredden. Med gränsen vid last_bin ensam kunde B481 följas
        av B482 -- två "nivåer" ur samma kluster. Med räkningen
        påslagen trängde den dubbletten ut en riktig nivå i slutet: 330 av 1000
        poster, 5 felplacerade bins per post, i den första --count-körningen.

        Med present (cfg.constrain_present) begränsas valet dessutom till bins som
        förekommer i insignalen, se _present_bins.

        Med k_hat (cfg.constrain_count) styrs även STOPPET: efter en bin tvingas ett
        nytt nivånamn så länge färre än k_hat namn skrivits och det finns tillåtna
        bins kvar, och ORDER tvingas när k_hat är nått. Bakgrund: med de två
        villkoren ovan steg level_precision 0.889 -> 0.938 medan n_levels_ok stod
        kvar på 0.68 -- modellen slutar för tidigt oavsett kandidater.

        Rent avkodningsvillkor: kräver ingen omträning och ändrar ingenting i loss.
        Slås av med cfg.constrain_levels = False, då är avkodningen som förut.
        """
        V = logits.size(-1)
        ids = torch.arange(V, device=logits.device)

        # ---- efter en bin: nästa är ett nivånamn eller ORDER. Med k_hat styrs valet.
        if k_hat is not None and n_names is not None:
            # ...eller direkt efter LEVELS: samma val, noll namn skrivna
            prev_is_bin = (is_bin[prev] | (prev == _LEVELS_ID)) & in_levels
            if bool(prev_is_bin.any()):
                is_name = (ids >= _LVL_LO) & (ids <= _LVL_HI)
                # finns det alls en tillåten bin kvar över den senaste?
                rest = is_bin[None, :] & (ids[None, :] > last_bin[:, None] + 2 * cfg.run_tol_bins)
                if present is not None:
                    rest = rest & present
                room = rest.any(-1)
                # fortsätt: färre namn än k_hat OCH något att fortsätta med -> ORDER bort
                forts = prev_is_bin & (n_names < k_hat) & room
                # stanna: k_hat nått -> namnen bort, ORDER blir kvar
                stanna = prev_is_bin & (n_names >= k_hat)
                logits = logits.masked_fill(forts[:, None] & (ids[None, :] == _ORDER_ID), -1e30)
                logits = logits.masked_fill(stanna[:, None] & is_name[None, :], -1e30)

        # ---- efter ett nivånamn: nästa är en bin, strikt över den senaste
        prev_is_name = (prev >= _LVL_LO) & (prev <= _LVL_HI) & in_levels
        if not bool(prev_is_name.any()):
            return logits
        # (B, V): tillåtet = är en bin OCH större än radens senaste bin
        allowed = is_bin[None, :] & (ids[None, :] > last_bin[:, None] + 2 * cfg.run_tol_bins)
        # ...OCH förekommer i insignalen, om det villkoret är på
        if present is not None:
            allowed = allowed & present
        # ...OCH, med räkningen på, ur rätt kluster: val nummer j kommer ur kluster j.
        # Med k_hat kluster och exakt k_hat val i stigande ordning finns ingen annan
        # lösning, och det stänger den sista läckan: att hoppa FÖRBI en nivå. Utan
        # det gick modellen 314 -> 1210 förbi 613, och 613 var sedan oåtkomlig.
        # Namnet före binnen är redan skrivet och räknat: när bin nummer j (från 0)
        # väljs är n_names = j + 1. Därav -1. Med == n_names hoppades kluster 0 över
        # och allt sköts ett steg -- precision föll från 0.969 till 0.866.
        if kluster is not None and n_names is not None:
            allowed = allowed & (kluster == (n_names - 1)[:, None])
        # rader vars senaste bin redan är den högsta kan inte fortsätta blocket; då
        # lämnas logits orörda så att modellen kan avsluta med ORDER som vanligt
        has_room = allowed.any(-1)
        apply = prev_is_name & has_room
        logits = logits.masked_fill(apply[:, None] & ~allowed, -1e30)
        return logits

    @torch.no_grad()
    def greedy(self, src, max_new=None):
        return self.generate(src, n=1, greedy=True, max_new=max_new)


# ---------------- index-modell
class IndexModel(Base):
    def __init__(self):
        super().__init__()
        d = cfg.d_model
        self.inp = InputEmbedding(d)
        self.pos_idx = SinusoidalIndex(d, cfg.max_src)
        self.pos_toa = SinusoidalTOA(d)
        layer = nn.TransformerEncoderLayer(d, cfg.nhead, cfg.ff, cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, cfg.enc_layers, norm=nn.LayerNorm(d))
        self.decoder = Decoder(d)
        self.aux = _aux_heads(d)

    def encode(self, src):
        x = self.inp(src) + self.pos_idx(src["bins"].size(1)) + self.pos_toa(src["toa"])
        return self.enc(x, src_key_padding_mask=src["mask"])


# ---------------- rope-modell
class HybridRoPE(nn.Module):
    def __init__(self, head_dim, time_frac, base=10000.0):
        super().__init__()
        n_pairs = head_dim // 2
        n_time = int(round(n_pairs * time_frac)); n_idx = n_pairs - n_time
        self.register_buffer("ft", base ** (-torch.arange(n_time) / max(n_time, 1)))
        self.register_buffer("fi", base ** (-torch.arange(n_idx) / max(n_idx, 1)))
    def forward(self, toa, idx):
        # fp32 hela vägen: vinklarna når flera tusen radianer för långa sekvenser och
        # tål inte bfloat16:s åtta mantissabitar.
        ang = torch.cat([toa.float()[..., None] * self.ft.float(),
                         idx.float()[..., None] * self.fi.float()], -1)
        return ang.cos()[:, None], ang.sin()[:, None]

def apply_rope(x, cos, sin):
    """Rotationen görs i fp32 och castas tillbaka till x:s dtype.

    Under autocast är x bfloat16, och utan den här konverteringen castas cos/sin ner
    dit -- ungefär tre decimalers precision på en rotation som ska vara exakt."""
    d = x.size(-1) // 2
    x1, x2 = x[..., :d].float(), x[..., d:].float()
    cos, sin = cos.float(), sin.float()
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return out.to(x.dtype)

class RoPEAttention(nn.Module):
    def __init__(self, d, nhead, dropout):
        super().__init__()
        self.h, self.dh = nhead, d // nhead
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d); self.drop = dropout
        # QK-norm håller attention-logitarna i schack. Utan den växer q·k med
        # qkv-vikternas norm, softmaxen kollapsar mot one-hot och gradienten genom
        # attention dör -- långsamt, och utan att bry sig om learning rate.
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.qn = nn.LayerNorm(self.dh)
            self.kn = nn.LayerNorm(self.dh)
    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            # före rotationen: RoPE är ortogonal och bevarar normen, så ordningen
            # spelar roll bara för att LayerNorm annars skulle blanda ihop de
            # roterade komponenterna
            q, k = self.qn(q), self.kn(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        am = (~mask)[:, None, None, :] if mask is not None else None
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am,
                                           dropout_p=self.drop if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, T, -1))

class RoPELayer(nn.Module):
    def __init__(self, d, nhead, ff, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = RoPEAttention(d, nhead, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
    def forward(self, x, cos, sin, mask):
        x = x + self.drop(self.attn(self.n1(x), cos, sin, mask))
        return x + self.drop(self.ff(self.n2(x)))

class RoPEModel(Base):
    def __init__(self):
        super().__init__()
        d = cfg.d_model
        self.inp = InputEmbedding(d)
        self.rope = HybridRoPE(d // cfg.nhead, cfg.time_frac)
        self.layers = nn.ModuleList([RoPELayer(d, cfg.nhead, cfg.ff, cfg.dropout)
                                     for _ in range(cfg.enc_layers)])
        self.norm = nn.LayerNorm(d)
        self.decoder = Decoder(d)
        self.aux = _aux_heads(d)

    def encode(self, src):
        B, T = src["bins"].shape
        x = self.inp(src)
        idx = torch.arange(T, device=x.device, dtype=torch.float32)[None].expand(B, T)
        cos, sin = self.rope(src["toa"], idx)
        for layer in self.layers:
            x = layer(x, cos, sin, src["mask"])
        return self.norm(x)


def build_model():
    return {"rope": RoPEModel, "index": IndexModel}[cfg.model]()
