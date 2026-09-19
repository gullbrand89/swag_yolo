# PRI → bibliotekspost

En encoder-decoder-transformer som läser en observerad pulsföljd (PRI i µs) och
skriver en bibliotekspost som beskriver emittern:

```
LEVELS L0 B113 L1 B198 L2 B283 ORDER FIXED L0 L2 L1 DWELL FIXED D10 D5 D8 END
```

Nivåer och dwelltider har **skilda tokenrymder**, `B` respektive `D`. En bin är en
mätning med ändlig noggrannhet, en dwelltid ett exakt antal pulser — de tål inte
samma ordinala utjämning.

```bash
pip install -r requirements.txt
python tests.py                        # innan varje längre körning
python -m transformer_post_generator.train
```

---

## Struktur

```
transformer_post_generator/   biblioteket — allt som träningen behöver
tools/                        diagnostik och analys, körs som moduler
docs/LOSS.md                  lossfunktionen rad för rad, med siffror
tests.py                      testsviten, körs från roten
train_fixed.py                bisektion: samma modell, FAST data
_arkiv/                       parkerat, se längst ned
```

Allt i `tools/` importerar paketet absolut (`from transformer_post_generator.config
import cfg`), så de körs som moduler från projektroten:

```bash
python -m tools.run_report
python -m tools.baseline --n 200
```

### transformer_post_generator/

| | |
|---|---|
| `config.py` | all konfiguration. Ändra här, importera `cfg` överallt |
| `vocab.py` | vokabulär och binning. `B`-rymd för nivåer, `D`-rymd för dwelltider |
| `labels.py` | dict ↔ tokens, med kanonisering |
| `loss.py` | cross-entropy med ordinal smoothing, fältvis uppdelning, `aux_loss` |
| `model.py` | RoPE- och indexmodellen bakom ett gemensamt gränssnitt, plus hjälp-huvudena |
| `data.py` | pipeline: generator → kanaler → batch |
| `all_emitters.py` | **generatorn.** Nio mönstervarianter, viktade i `GEN.weights` |
| `train.py` | SFT-loopen |
| `runlog.py` | körningsmapp, loggar, fältvisa mått |
| `verify.py` | mäter facit mot signal; `verify_stats` för en hel batch |

`__init__.py` löser `cfg.emitter` som ett relativt modulnamn (`".all_emitters"`) mot
paketet, så en egen generator byts in utan att importvägarna ändras.

### Hjälp-lossen

`model.forward(src, tgt_in, return_aux=True)` ger, utöver logits, tre linjära huvuden
på encoderns utdata som per puls svarar på *nytt besök?*, *position i uppehållet* och
*antal ihopslagna intervall*. De vägs in med `cfg.aux_weight` och används **bara i
träningen** — `generate()` och `greedy()` rör dem inte. Se `loss.aux_loss`.

### tools/

Verifiering av att uppgiften är lösbar, innan modellen får skulden:

| | |
|---|---|
| `label_coverage.py` | hur mycket av facitet syns faktiskt i signalen? |
| `input_fidelity.py` | överlever nivåerna `make_channels`, eller klipps de bort? |
| `check_dwell.py` | stämmer DWELL-blocket med vad signalen visar? |
| `check_order.py` | går FIXED att skilja från RANDOM överhuvudtaget? |
| `check_visits.py` | räcker observationsfönstret för att se nivåcykeln? |
| `check_drift.py` | driver dataströmmen över tid? (härmar DataLoaderns processer) |
| `measure_levels.py` | mät binningsparametrarna ur datan i stället för att härleda dem |

Utvärdering av en körning:

| | |
|---|---|
| `run_report.py` | träningskurvan och eval-måtten bredvid varandra |
| `classify_errors.py` | delar upp felen i preds-filerna efter vad som skiljer |
| `baseline.py` | samma facit, utan modell — golvet att slå |
| `num_breakdown.py` | delar upp num-lossen på nivåbins och dwelltider |
| `sharpness.py` | är modellen för skarp eller för platt mot sitt mål? |
| `show_failures.py` | facit som inte stämmer med sin signal, i detalj |

Läsbara genomgångar och visualisering:

| | |
|---|---|
| `forbehandling.py` | allt som händer med datan innan transformern, steg för steg |
| `facit.py` | hur facitet byggs och valideras, steg för steg |
| `forecast.py` | fortsätt en pulsföljd utifrån ett facit (dold semi-Markov) |
| `inspect_label.py` | plotly-paneler för notebook |

`forbehandling.py` och `facit.py` är skrivna för att **läsas** — explicita loopar, ett
steg per funktion, även där det går kortare. Den riktiga koden gör samma sak kompaktare.

---

## Byta generator

```python
# config.py
emitter: str = ".all_emitters"    # relativt modulnamn, löses mot paketet
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

## Läsa en körning

`runs/<tidsstämpel>/` innehåller `config.json` (hela cfg som körningen såg den),
`train.csv`, `eval.json` och `preds_drop_*.txt`. Enklast via `python -m tools.run_report`,
som lägger träningskurvan bredvid eval-måtten och svarar på frågan: när `loss_num`
steg — blev modellen sämre, eller bara mindre säker?

`config.json` gör körningarna jämförbara i efterhand. `runs/` ligger i `.gitignore`.

---

## Kända begränsningar

* När nivå- och längdcykeln har **olika period** kan formatet inte uttrycka den
  relativa fasen, trots att den är observerbar i signalen. Två emittrar som beter sig
  olika får då samma facit. Åtgärd vore ett offset-token.
* `cfg.smooth_width` sätter ett golv på `num`-lossen — se `docs/LOSS.md` §6. Ett
  `num` på 2,05 kan alltså vara ett litet överskott, inte ett dåligt värde i sig.
* Samma smoothing-bredd används för nivåbins och dwelltider trots att skalorna är
  olika. `vocab.py` ger dem skild `sigma` (`cfg.sigma_bin`, `cfg.sigma_dur`), men
  bredden är gemensam.
* `tests.py` säger att steg 6 hoppas över utan torch. Det stämmer inte längre —
  `data.py` importerar torch på modulnivå, så hela sviten kräver torch.

---

## _arkiv/

Parkerat, inte borttaget. Ingenting i det aktiva kodträdet importerar härifrån.

| | |
|---|---|
| `grpo.py`, `reward.py` | RL mot en verifierare — **future work**, se nedan |
| `dwell_emitter.py`, `stagger_emitter.py` | avgränsade generatorer, en ratt i taget |
| `diag_rotation.py`, `check_metrics.py` | diagnostik, ersatt av `tools/` |
| `swag.py`, `xx.py`, `verify_range.py` | engångsskript |

Idén i `grpo.py`/`reward.py`: `train.py` optimerar sannolikheten för nästa token givet
att allt före var rätt. `docs/LOSS.md` §9 beskriver vad det missar — ett fel i antalet
nivåer kostar ett token under teacher forcing men förstör hela posten vid avkodning.
RL-loopen optimerar posten i stället, med en belöning som mäter den mot **signalen**
och alltså inte behöver något facit. Filerna är inte uppdaterade för den delade
B/D-tokenrymden och kör inte som de ligger.
