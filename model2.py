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

Decodern använder nn.TransformerDecoderLayer och har ingen QK-normalisering. Den kör
över ~40 token i stället för 1024 och är därför betydligt mindre utsatt; vill man ha
det även där måste lagret skrivas för hand. Detsamma gäller IndexModel.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from vocab import VOCAB, PAD, BOS, EOS


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
    """bins + cont + (räknare) + (flagga)"""
    def __init__(self, d):
        super().__init__()
        self.bin = nn.Embedding(cfg.in_bins, d)
        self.cont = nn.Linear(1, d)
        self.rl = nn.Embedding(cfg.max_run + 1, d) if cfg.use_counter else None
        self.flag = nn.Embedding(2, d) if cfg.use_drop_flag else None
    def forward(self, src):
        x = self.bin(src["bins"]) + self.cont(src["cont"][..., None])
        if self.rl is not None:
            x = x + self.rl(src["rl"].clamp(max=cfg.max_run))
        if self.flag is not None:
            x = x + self.flag(src["flag"])
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


class Base(nn.Module):
    def forward(self, src, tgt_in):
        return self.decoder(self.encode(src), src["mask"], tgt_in)

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

        for _ in range(steps):
            logits = self.decoder(mem, mask, ys)[:, -1].float()
            logits[:, PAD] = -1e30                      # PAD och BOS är aldrig giltiga
            logits[:, BOS] = -1e30                      # utdata mitt i en sekvens
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
            if done.all():
                break
        return ys

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
        self.qk_norm = getattr(cfg, "qk_norm", True)
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
