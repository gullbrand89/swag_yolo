# PRI → bibliotekspost

En encoder-decoder-transformer som läser en observerad pulsföljd (PRI i µs) och
skriver en bibliotekspost som beskriver emittern:

```
LEVELS N3 L0 N113 L1 N198 L2 N283 ORDER FIXED L0 L2 L1 DWELL FIXED N10 N5 N8 END
```

Två träningslägen: cross-entropy mot facit (`train.py`), och RL mot en verifierare
som mäter posten mot signalen (`grpo.py`).

```bash
pip install -r requirements.txt
python tests.py                                # innan varje längre körning
python train.py                                # SFT
python grpo.py --init runs/<tidsstämpel>/model.pt   # RL
```

---

## Filerna

| | |
|---|---|
| `config.py` | all konfiguration. Ändra här, importera `cfg` överallt. |
| `vocab.py` | vokabulär och binning |
| `labels.py` | dict ↔ tokens, med kanonisering |
| `loss.py` | cross-entropy med ordinal smoothing, plus fältvis uppdelning |
| `model.py` | RoPE- och indexmodellen bakom ett gemensamt gränssnitt |
| `data.py` | pipeline: generator → kanaler → batch |
| `train.py` | SFT-loopen |
| `reward.py` | **verifierbar belöning** — mäter en post mot signalen, utan facit |
| `grpo.py` | **RL-loopen** (GRPO, DeepSeek-stil) |
| `runlog.py` | körningsmapp, loggar, fältvisa mått |
| `verify.py` | mäter facit mot signal; `verify_stats` för en hel batch |
| `tests.py` | snabb testsvit |

Generatorer, valbara med `cfg.emitter`:

| | |
|---|---|
| `all_emitters.py` | **standard.** Nio mönstervarianter, viktade i `GEN.weights` |
| `dwell_emitter.py` | avgränsade experiment på LÄNGDER, en ratt i taget |
| `stagger_emitter.py` | avgränsade experiment på ORDNING |

Verktyg: `check_metrics.py` (är måtten konsekventa?), `show_failures.py` (visa facit
som inte stämmer med sin signal), `diag_rotation.py` (var brister den parvisa
matchningen?), `inspect_label.py` (plotly-paneler för notebook).

`LOSS.md` går igenom lossfunktionen rad för rad, med siffror.

---

## Byta generator

```python
# config.py
emitter: str = "all_emitters"    # "dwell_emitter" | "stagger_emitter" | din egen modul
```

Modulen behöver bara exportera

```python
create_emitter_data(n_emitters, n_signals, drop_rate, noise_level, rng)
    -> [(pri_sequences, label_dict), ...]
```

där `label_dict` har `levels`, `lengths`, `order_fixed`, `length_fixed`. Valfritt
`detect_missing(pri) -> bool-array` plockas upp automatiskt om `cfg.use_drop_flag`.

Två krav på generatorn, båda kontrollerade av `tests.py`:

* **Nivå- och längdcykeln måste rullas med samma löpande index vid fast ordning.**
  Med oberoende startfas beskriver facitet en annan parning än signalen, och då
  tränar du på brus i just det facitet ska lära ut.
* **Bortfallet ska dras ur en egen slumpström.** Annars förskjuts alla efterföljande
  signaler så fort `drop_rate > 0`, och bortfallskurvan mäter byte av signal.

---

## RL: vad och varför

`train.py` optimerar sannolikheten för nästa token givet att allt före var rätt.
`LOSS.md` §9 beskriver vad det missar: ett fel i antalet nivåer kostar ett token under
teacher forcing, men förstör hela posten vid avkodning.

`grpo.py` optimerar posten i stället. För varje signal samplas `rl_group` svar, varje
svar får en belöning, och gruppens medel är baslinjen:

```
A_i = (r_i − medel(r)) / std(r)
```

Ingen värdemodell — det är hela skillnaden mot PPO, och anledningen att det går att
köra på en modell i den här storleken.

Belöningen kommer från `reward.py`, som mäter posten mot **signalen**, inte mot
facitet. Det ger tre saker:

1. Fel som förstör posten straffas som det de är, inte en gång per token.
2. Modellen tränas på sin egen avkodning — exposure bias försvinner.
3. Belöningen behöver inget facit, så loopen går att köra på inspelad data.

Delbelöningarna och vikterna står i `reward.RW`. Kör `python reward.py` för en
egentest som visar att ett sant facit ligger på 1,0 och att degenererade svar
(en enda nivå, `ORDER RANDOM`, maximalt `DWELL RANGE`) ligger lågt.

### Läsa loggen

`runs_rl/<tidsstämpel>/train.csv`:

| kolumn | vad du tittar efter |
|---|---|
| `reward` | ska stiga. Gör den inte det: höj `rl_temp` eller sänk `rl_beta` |
| `parsed` | ska nå ~1,0 inom några hundra steg |
| `reward_std` | spridningen **inom** grupperna. Går den mot 0 finns ingen gradient kvar |
| `uniq` | antal olika svar i batchen. Faller den mot 1 har policyn kollapsat |
| `kl` | driver iväg = modellen glider från SFT-lösningen. Höj `rl_beta` |

Och i `eval.json`, som är den enda platsen facit används:

> **Stiger `reward` men inte `exact` eller `level_recall` har modellen hittat ett
> kryphål i verifieraren.** Det är den enda kontroll som räknas. Två kryphål är redan
> stängda och beskrivna överst i `reward.py`; hittar du ett tredje hör det hemma där.

### Rattarna

| | |
|---|---|
| `rl_group` | G. Fler sampel = mindre brus i baslinjen, linjärt dyrare |
| `rl_beta` | KL mot SFT-modellen. 0 = fri, 0,02 = normalt, högre = konservativt |
| `rl_temp` | rollout-temperatur. För låg → ingen spridning → ingen gradient |
| `rl_norm_adv` | `False` ger Dr. GRPO-varianten utan std-normalisering, som annars ger lätta exempel för stort inflytande |
| `rl_seq_mean` | `False` (default) viktar per token i stället för per sekvens, vilket tar bort längdbiasen |
| `rl_inner_epochs` | µ. Med 1 är kvoten exakt 1 och klippningen verkningslös |

Hela RL-körningen går i `eval()`-läge. Det stänger av dropout med flit: annars skiljer
sig log-sannolikheterna mellan rollout och uppdatering, och kvoten mäter brus i stället
för policyändring.

---

## Kända begränsningar

* När nivå- och längdcykeln har **olika period** kan formatet inte uttrycka den
  relativa fasen, trots att den är observerbar i signalen. Två emittrar som beter sig
  olika får då samma facit. Åtgärd vore ett offset-token.
* `cfg.smooth_width = 2` ger `num`-lossen ett golv på **1,372** — se `LOSS.md` §6.
  Ett `num` på 2,05 är alltså ett överskott på 0,68, inte ett dåligt värde i sig.
* Samma smoothing-bredd används för nivåbins och dwell-längder trots att skalorna är
  helt olika. ±2 bins av 512 är försumbart; ±2 pulser av en dwell på 4 är 50 % fel.
* `reward.py` rekonstruerar ihopslagna intervall upp till `RW.max_merge = 3`. Vid
  högre bortfall än ~25 % börjar ordnings- och längdbelöningen tappa upplösning.
