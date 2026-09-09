# Hur lossfunktionen fungerar

Dokumentet går igenom `loss.py` rad för rad, med konkreta siffror. Målet är att du ska
kunna svara på "varför blev lossen 2,05?" utan att gissa.

---

## 1. Vad lossen egentligen mäter

Modellen är en språkmodell över biblioteksposter. Facitet är en sekvens av tokens:

```
LEVELS N3 L0 N113 L1 N198 L2 N283 ORDER FIXED L0 L1 L2 DWELL FIXED N10 N5 N8 END
```

Decodern genererar den ett token i taget. Vid varje position ger den en
sannolikhetsfördelning över hela vokabulären (~1300 tokens), och lossen mäter hur
mycket sannolikhet den lade på **rätt** token.

Lossen är alltså inte "hur bra är posten" utan "hur säker var modellen på nästa token,
i genomsnitt". Det är en viktig skillnad som vi återkommer till i avsnitt 9.

---

## 2. Cross-entropy, med siffror

Säg att vokabulären bara har fem tokens och facit vid en position är token nr 2.
Modellen ger fördelningen:

| token | 0 | 1 | **2** | 3 | 4 |
|---|---|---|---|---|---|
| sannolikhet | 0,05 | 0,10 | **0,60** | 0,20 | 0,05 |

Lossen för den positionen är `-log(0,60) = 0,51`.

Några referenspunkter som är bra att ha i huvudet:

| modellen lade på rätt token | loss |
|---|---|
| 0,99 | 0,01 |
| 0,90 | 0,11 |
| 0,60 | 0,51 |
| 0,37 | 1,00 |
| 0,14 | 2,00 |
| 0,05 | 3,00 |

Och slumpnivån: om modellen gissar helt jämnt över `V` tokens blir lossen `ln(V)`.
Med 512 talalternativ är det `ln(512) ≈ 6,2`.

Tabellen ovan gäller dock bara **one-hot**-mål. Taltokens har ordinal smoothing, och
då är golvet inte noll utan målfördelningens entropi — med `w = 2` är det `1,372`
(avsnitt 6). Ett `num` på 2,05 är alltså ett överskott på `0,68` över golvet, inte
"13 % sannolikhet på rätt tal". Räkna aldrig `-log p` baklänges på `num`.

---

## 3. Teacher forcing: vad modellen får se

Under träning matas decodern med **facitets** tokens, inte sina egna gissningar.

```
position:   0        1     2    3     4     ...
tgt_in:   <bos>   LEVELS   N3   L0   N113   ...
tgt_out:  LEVELS    N3     L0  N113   L1    ...
```

`tgt_in` är facitet förskjutet ett steg (det modellen läser), `tgt_out` är facitet
(det den ska förutsäga). Den förskjutningen görs i `collate`:

```python
return src, tgt[:, :-1], tgt[:, 1:]
```

Konsekvensen: **varje position bedöms som om allt före den var rätt.** Om modellen
gissar fel antal nivåer vid position 1 straffas det en gång, och sedan får den ändå
det korrekta `N3` att fortsätta från. Det är därför lossen kan se hyfsad ut medan
greedy-avkodning ger katastrof — se avsnitt 9.

---

## 4. Formerna på tensorerna

```python
logits    # (B, T, V)  – råa poäng, en per token i vokabulären
tgt_out   # (B, T)     – rätt token-id per position
```

`B` = antal exempel i batchen, `T` = längsta facitet i batchen, `V` = `len(VOCAB)`.
Kortare facit fylls ut med `PAD`.

---

## 5. Steg ett: log_softmax

```python
logp = F.log_softmax(logits, -1)      # (B, T, V)
```

Gör om råa poäng till log-sannolikheter längs vokabulär-axeln. Att räkna i log-rummet
direkt är numeriskt stabilare än `log(softmax(x))`.

---

## 6. Steg två: målfördelningen (ordinal smoothing)

Här ligger den enda icke-standardiserade delen. Vanlig cross-entropy jämför mot en
**one-hot**-vektor: 1,0 på rätt token, 0 på alla andra. Problemet är att våra
taltokens har en *ordning*. Om facit är `N12` är `N11` nästan rätt, medan `N400` är
helt fel — men one-hot behandlar dem lika.

`ordinal_targets` bygger därför en mjukare målfördelning för taltokens:

```python
soft = torch.zeros(B, T, V)
soft.scatter_(2, tgt[..., None], 1.0)          # börja med one-hot överallt
isnum = IS_NUM[tgt] & (tgt != PAD)             # vilka positioner är N-tokens?
```

För de positionerna ersätts one-hot av en gaussisk klocka över grannarna:

```python
for off in range(-w, w + 1):                   # w = cfg.smooth_width, nu 2
    j = idx + off
    ok = (j >= NUM_START) & (j < NUM_START + len(NUM))   # håll dig inom N-blocket
    rows[ar[ok], j[ok]] += math.exp(-0.5 * (off / sigma) ** 2)
rows = rows / rows.sum(-1, keepdim=True)       # normalisera till summa 1
```

Att `idx + off` fungerar beror på att `N0 … N511` ligger **sammanhängande** i
vokabulären. Att lägga till 1 på token-id:t flyttar dig till nästa tal. Villkoret `ok`
ser till att smetningen inte läcker ut i strukturtokens vid kanterna (`N0`, `N511`).

Med `w = 2` och `sigma = 1` blir målet för facit `N12`:

| token | N10 | N11 | **N12** | N13 | N14 |
|---|---|---|---|---|---|
| vikt före normalisering | 0,135 | 0,607 | **1,000** | 0,607 | 0,135 |
| efter normalisering (summa 2,4836) | 0,054 | 0,244 | **0,403** | 0,244 | 0,054 |

Två konsekvenser att förstå:

**Lossen kan aldrig bli noll för taltokens.** Lossen är `-Σ p_mål · log p_modell`.
Lägger modellen all sannolikhet på `N12` blir termerna för grannarna
`-0,244·log(0) = ∞`. Den optimala strategin är i stället att **matcha målfördelningen
exakt**, vilket ger loss lika med målfördelningens entropi:

| `w` | golv för `num` |
|---|---|
| 0 | 0,000 |
| 1 | 1,068 |
| **2** | **1,372** |
| 3 | 1,417 |

> Det här är viktigt när du tolkar `num`. Ett `num` på 1,4 kan alltså vara nära perfekt,
> inte dåligt. Vill du att `num` ska gå mot noll, sätt `smooth_width = 0`. Ett alternativ är
> att logga `num − golv` direkt, så att noll faktiskt betyder noll.

**Bredden måste matcha den precision du bryr dig om.** `w = 2` betyder ±2 bins för
nivåer (≈3,5 µs) och ±2 pulser för dwell-längder. Det senare är i grövsta laget när
dwells är 4–32 pulser långa. Se avsnitt 10.

Strukturtokens (`LEVELS`, `ORDER`, `L0`, `END` …) behåller one-hot. De har ingen
ordning — `L0` är inte "nästan" `L1`.

---

## 7. Steg tre: fältvikter och maskering

```python
per_tok = -(ordinal_targets(tgt_out) * logp).sum(-1)     # (B, T)
mask = tgt_out != PAD
w = torch.where(IS_NUM[tgt_out], 1.0, cfg.struct_weight) # 1,0 för tal, 0,5 för struktur
return (per_tok * w * mask).sum() / (w * mask).sum()
```

Rad för rad:

1. **`per_tok`** — multiplicera målfördelningen med modellens log-sannolikheter och
   summera över vokabulären. Ger ett tal per position: korsentropin där.
2. **`mask`** — padding ska inte räknas. `PAD`-positioner nollställs.
3. **`w`** — strukturtokens väger 0,5, taltokens 1,0. Skälet: strukturtokens är många
   och lätta. Utan viktning kan lossen sjunka snyggt medan nivåer och längder förblir
   dåliga, bara för att modellen blivit bra på att skriva `ORDER` på rätt ställe.
4. **Normaliseringen** — dividera med summan av vikterna, inte med antalet positioner.
   Det gör lossen jämförbar mellan batchar med olika facitlängder.

---

## 8. `loss_by_field`: tre siffror i stället för en

Totalen döljer var problemet sitter, så `field_masks` delar upp positionerna i tre:

| fält | vilka tokens | vad det mäter |
|---|---|---|
| **num** | alla `N…` | nivåernas bins, dwell-längder, antal nivåer |
| **order** | `L…` inne i `ORDER`-blocket, samt `FIXED`/`RANDOM` | i vilken ordning nivåerna spelas, och fast vs slumpad |
| **grammar** | allt annat: `LEVELS`, `ORDER`, `DWELL`, `END`, `INF`, `L…` i definitionerna | ren form |

Uppdelningen görs så här:

```python
after_order = torch.cumsum((tgt == _ORDER).long(), 1) > 0     # har vi passerat ORDER?
before_end  = torch.cumsum(((tgt == _DWELL) | (tgt == _END)).long(), 1) == 0
in_order    = after_order & before_end                         # inne i ORDER-blocket
order   = valid & in_order & (is_lvl | is_type)
grammar = valid & ~isnum & ~order
```

Kumulativ summa används för att avgöra var i posten varje position ligger, utan loopar.

Notera villkoret `in_order` på `is_type`. Utan det matchar `is_type` även `FIXED`/`RANGE`
efter `DWELL`, och `order` blandar då ihop nivåernas ordning med om dwellen är fast
eller slumpad — två egenskaper som lärs vid olika tidpunkter.

**Så tolkar du dem:**

- `grammar` ska falla mot nära noll inom några hundra till tusen steg. Gör den inte
  det är något fel i pipelinen, inte i uppgiften.
- `order` sjunker när modellen börjar läsa mönstret ur pulserna. Den kan ligga still
  länge om modellen skriver för få nivåer — då finns knappt några ordningstokens att
  ha fel på.
- `num` är den långsamma. Nivåbins lärs relativt tidigt, dwell-längder sist.

Notera att `loss_by_field` **inte** använder fältvikterna — den rapporterar oviktad
medel-loss per fält, så talen är jämförbara med varandra.

---

## 9. Vad lossen inte mäter

Det här avsnittet är det viktigaste i dokumentet.

**Fel som får konsekvenser straffas bara en gång.** Om modellen skriver `N4` när
facit är `N3` (antal nivåer) kostar det ett token under träning. Under greedy-avkodning
matas modellens eget `N4` vidare och hela resten av posten byggs kring fel antal.
Det heter *exposure bias*, och det är varför `n_levels_ok = 0,198` kan samexistera med
en loss som ser hyfsad ut.

**Lossen är positionsbunden, inte mängdbaserad.** Om modellen lägger in en extra nivå
i mitten förskjuts allt efter den, och även korrekta värden hamnar på fel position.
Utvärderingen i `runlog.compare` gör i stället matchning på nivåmängden, vilket är
varför `level_precision` och `level_recall` säger något annat än lossen.

**Semantiskt likvärdiga poster kan få olika loss.** Kanoniseringen i `to_tokens`
finns just för att undvika det, men om två olika beskrivningar av samma emitter kan
uppstå tränar du modellen på brus och lossen fastnar på ett golv.

**Sjunkande loss garanterar inte att modellen använder inputen.** En decoder kan sänka
lossen betydligt genom att bara lära sig facitens statistik. Testet för det är inte
lossen utan antalet unika prediktioner i `preds_*.txt`:

```bash
grep '^P ' runs/<stamp>/preds_drop_0.00.txt | sort -u | wc -l
```

Är svaret 1 ignorerar modellen encodern helt, oavsett vad lossen visar.

---

## 10. Rattarna

| parameter | vad den gör | när du rör den |
|---|---|---|
| `cfg.smooth_width` | bredden på ordinal smoothing (bins/pulser) | 0 om du vill att `num` ska gå mot noll; 1 om ±2 pulser är för slappt för dwell-längder |
| `cfg.struct_weight` | vikt på strukturtokens | höj mot 1,0 om grammatiken inte sätter sig; sänk om struktur dominerar lossen |
| `sigma` i `ordinal_targets` | klockans skärpa inom bredden | sällan; bredden är den effektiva ratten |

**En känd svaghet:** samma bredd används för nivåbins och dwell-längder, trots att
skalorna är helt olika. ±2 bins av 512 är en försumbar avvikelse för en nivå; ±2 pulser
av en dwell på 4 är 50 % fel. Vill du åtgärda det får `ordinal_targets` skilja på
positionstyp på samma sätt som `field_masks` gör, och använda olika bredd.

**En andra svaghet:** antalet nivåer i `LEVELS N<n>` är kategoriskt, inte ordinalt —
`N5` är inte "nästan" `N6` på ett meningsfullt sätt — men det smetas ändå. Antingen
undanta den positionen, eller ta bort `LEVELS N<n>` ur formatet och låt
nivådefinitionerna löpa till `ORDER`.

---

## 11. Sammanfattning i en mening

Lossen är viktad cross-entropy per token med teacher forcing, där taltokens får en
uppmjukad målfördelning över grannbins och strukturtokens väger hälften — den mäter
hur säker modellen är på nästa token givet att allt före var rätt, vilket är bra för
gradienten men **inte** ett mått på hur bra biblioteksposten blir, och därför måste
läsas tillsammans med `level_recall`, `n_lengths_ok` och antalet unika prediktioner.
