"""
All konfiguration på ett ställe. Ändra här, importera `cfg` överallt.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    # ---------------- datarymd (µs)
    pri_min: float = 100.0          # emitterns rymd (facit)
    pri_max: float = 1000.0
    n_bins: int = 512               # facitets binning, linjär
    in_min: float = 100.0           # observationsrymd (input)
    in_max: float = 4000.0
    in_bins: int = 512              # inputens binning, log
    toa_scale: float = 550.0        # µs per positionsenhet

    # ---------------- vokabulär
    max_levels: int = 32
    max_int: int = 511              # >= n_bins-1 och >= största dwell-längd
    max_run: int = 64               # räknarens tak
    compress_lengths: bool = True   # [10,10,10] -> [10] i facit

    # ---------------- generator
    emitter: str = "all_emitters"   # modulnamn; måste ha create_emitter_data
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
    samples_per_emitter: int = 4    # startfaser per emitter från create_emitter_data
    eval_n_emitters: int = 500
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
    batch: int = 256                # sekvenser per steg (delas på samples_per_emitter)
    lr: float = 3e-4
    warmup: int = 200
    weight_decay: float = 0.01
    struct_weight: float = 0.5      # lossvikt på strukturtokens
    smooth_width: int = 2           # ordinal smoothing ±bins
    seed: int = 0
    num_workers: int = 4
    eval_every: int = 1000
    log_every: int = 50
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
