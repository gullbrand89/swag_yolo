"""
All konfiguration på ett ställe. Ändra här, importera `cfg` överallt.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    # ---------------- datarymd (µs)
    # Värdena nedan är mätta med tools/input_fidelity mot all_emitters så som den
    # ser ut nu. Ändrar du GEN-parametrarna i all_emitters.py måste de mätas om --
    # särskilt toa_scale, som ska ligga på korpusens medel-PRI.
    pri_min: float = 0.98           # emitterns rymd (facit)
    pri_max: float = 505.0
    n_bins: int = 2048              # facitets binning, linjär (0.2462 µs/bin)
    in_min: float = 0.98            # observationsrymd (input)
    in_max: float = 1010.0          # ~2 x pri_max: marginal för ihopslagna pulser
    in_bins: int = 4096             # inputens binning, LINJÄR (inte log)
    toa_scale: float = 143.43        # µs per positionsenhet ~ korpusens medel-PRI

    # ---------------- vokabulär
    max_levels: int = 32
    max_dur: int = 32             # största dwelltid i pulser. 1023 täcker allt när
                                    # fönstret är 1024 pulser -- höj om fönstret växer
    # Bredden på den ordinala utjämningen för nivåbins. Vid 1.0 lägger målet bara
    # 0.399 på rätt bin och 0.242 på vardera grannen -- modellen belönas alltså för
    # att tveka ±1 bin, och det syns i predikterna: nivåblocket är rätt inom ±1 i
    # 53% av posterna men teckenidentiskt i 18%. Facitets bin är exakt, så
    # utjämningen är ett inlärningsstöd och inte en brusmodell. Vid 0.5 blir det
    # 0.787 på rätt bin.
    # OBS: golvet för loss_num faller från 1.19 till 0.67 nat enbart av det här.
    # loss_num går alltså ner ~0.52 gratis och är INTE jämförbar med förra körningen.
    sigma_bin: float = 0.5
    sigma_dur: float = 0.5          # utjämning på dwelltider, i pulser
    max_run: int = 64               # räknarens tak
    compress_lengths: bool = True   # [10,10,10] -> [10] i facit

    # ---------------- generator
    emitter: str = ".all_emitters"   # modulnamn; måste ha create_emitter_data
                                    # "all_emitters" | "dwell_emitter" | "stagger_emitter"

    # ---------------- störningar
    p_drop: float = 0.25             # bortfall vid träning (per exempel: uniform(0, p_drop) om randomize)
    randomize_p_drop: bool = True
    noise_level: Optional[float] = None   # skickas vidare till create_emitter_data
    run_tol_bins: int = 1           # räknarens tolerans

    # ---------------- input-kanaler
    use_counter: bool = True
    use_drop_flag: bool = False     # från din missing-pulse-detektor

    # ---------------- data
    samples_per_emitter: int = 1    # startfaser per emitter från create_emitter_data
    # Fyra fönster av SAMMA emitter är inte fyra oberoende mätningar -- variationen
    # mellan emittrar dominerar, så de kostar fyra gånger avkodningen utan att minska
    # standardfelet. Augmentering hör hemma i träningen, inte i mätningen.
    eval_samples_per_emitter: int = 1
    eval_n_emitters: int = 200
    # Greedy kör annars max_tgt-1 = 255 steg. Längsta facit på 30000 emittrar är 89
    # token, och en enda rad som aldrig når EOS drar hela batchen till taket.
    eval_max_new: int = 112
    eval_p_drops: tuple = (0.0, 0.05, 0.1, 0.2)
    eval_seed: int = 123

    # ---------------- modell
    model: str = "rope"             # "rope" | "index"
    d_model: int = 256
    nhead: int = 8
    enc_layers: int = 4
    dec_layers: int = 4
    ff: int = 1024
    dropout: float = 0.1
    max_tgt: int = 256
    max_src: int = 4096             # bara index-modellen
    time_frac: float = 0.5          # bara rope-modellen

    # ---------------- träning (SFT, train.py)
    steps: int = 20000
    batch: int = 64                # sekvenser per steg (delas på samples_per_emitter)
    lr: float = 3e-4
    warmup: int = 500
    weight_decay: float = 0.01
    # Säkerhetsventil, inte en del av optimeraren. Vid 4.0 löste den ut på 0.95% av
    # stegen och p99 låg på 3.96 -- alltså vid kanten. Ett skarpare mål ger större
    # gradienter, så 4.0 skulle riskera att bli aktivt och återinföra just den
    # stegrande lossen vi felsökte bort. Mät om med grad_norm i loggen.
    clip: float = 8.0
                                    # den typiska gradientnormen gör klippningen aktiv
                                    # varje steg och frikopplar stegstorleken från
                                    # lr -- det var orsaken till den stigande lossen.
                                    # Logga grad_norm och sätt ~5 x medianen.
    struct_weight: float = 0.5      # lossvikt på strukturtokens
    smooth_width: int = 2           # fönstret för utjämningen; >= 3 x sigma
    seed: int = 0
    num_workers: int = 16
    eval_every: int = 2000
    log_every: int = 10
    run_root: str = "runs"

    # ---------------- RL (GRPO, grpo.py)
    # Verklig batch = rl_prompts * rl_group sekvenser genom avkodaren per steg.
    rl_init: str = ""               # sökväg till SFT-checkpoint; tomt = träna från slumpvikter
    rl_steps: int = 3000
    rl_prompts: int = 16            # signaler per steg
    rl_group: int = 8               # G, sampel per signal (gruppen som ger baslinjen)
    rl_lr: float = 1e-5
    rl_warmup: int = 50
    rl_temp: float = 1.0            # samplingstemperatur för rollouts
    rl_top_k: int = 0               # 0 = av
    rl_max_new: int = 64            # längsta facit att generera (verkliga är ~20-45 tokens)
    rl_clip: float = 0.2            # PPO-klippning; verkningslös när rl_inner_epochs == 1
    rl_beta: float = 0.02           # KL-vikt mot referensmodellen
    rl_inner_epochs: int = 1        # µ i GRPO; 1 = on-policy, ratio == 1
    rl_norm_adv: bool = True        # True: dela med gruppens std (DeepSeek)
                                    # False: bara centrera (Dr. GRPO, tar bort svårighetsbias)
    rl_seq_mean: bool = False       # True: medel per sekvens sen över gruppen (DeepSeek)
                                    # False: medel över alla tokens (ingen längdbias)
    rl_eval_every: int = 250
    rl_log_every: int = 10
    rl_run_root: str = "runs_rl"

    def dict(self): return asdict(self)


cfg = Config()

# Ett fält UTAN typannotering blir ingen dataclass-fält, bara ett klassattribut. Det
# går att läsa som vanligt, så inget smäller -- men det hamnar aldrig i asdict() och
# därmed aldrig i körningens config.json. Då loggar du en konfiguration du inte körde.
_oannoterade = [k for k, v in vars(Config).items()
                if not k.startswith("_") and not callable(v)
                and k not in Config.__dataclass_fields__]
assert not _oannoterade, (
    f"saknar typannotering och hamnar därför inte i config.json: {_oannoterade}. "
    f"Skriv t.ex. '{_oannoterade[0]}: int = ...' i stället för '{_oannoterade[0]} = ...'."
    if _oannoterade else "")
