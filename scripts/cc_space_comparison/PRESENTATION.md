# Presentation plan: D6.002 vs D6.007

A 9-slide core deck with a backup section. Figures are in
`output/D6.002_vs_D6.007/`. Numbers and full argument in `RESULTS.md`.

**The arc**: ask the question → reframe what the change actually *was* →
show four independent lines of evidence → give the counter-case an honest
hearing → recommend. The reframe has to come second. Every result afterwards
is read differently once the audience knows D6.007 lost 69% of its proteins,
and if you show AUROC deltas first they will spend the rest of the talk
assuming this is a story about 84 extra compounds.

---

## Core deck

Each slide below gives the figure, the bullets to put on it (kept under 100
characters so they stay one line at projector size), and what to say.

### 1 — The question
**No figure.**

**Slide bullets**
- D6.002: proteomics space from the DeepCoverMoA screen (Mitchell et al.)
- D6.007: the same space extended with more proteomics datasets
- Did the extension make it a better bioactivity space?

**Speaker notes.** Both spaces were fitted sign0→sign3 in the same local CC
instance, so nothing that follows is a pipeline difference — it is the data.
Validation follows the CC Protocols paper throughout, so every test here has a
published precedent rather than being invented for the occasion.

### 2 — What actually changed
**No figure yet** — a two-number slide (see *Gap* below).

**Slide bullets**
- Compounds: 872 → 956 (+9.6%), and D6.002’s set is a strict subset
- Proteins measured: 9,802 → 3,083 (−68.5%)
- “All Shared Proteins” intersects protein sets: a trade, not more data

**Speaker notes.** This is the slide the rest of the talk depends on. The
compound gain is clean and purely additive — every D6.002 compound is still
there. The feature loss is the part nobody costed: keeping only proteins
measured across all integrated datasets threw away more than two-thirds of
them. Ask the room to hold that number; it explains almost every result
coming.

### 3 — A measurement caution
**Figure:** `moa_atc_roc_overlay.png`

**Slide bullets**
- Two different MoA/ATC AUROCs exist per run, and they disagree on the winner
- Canvas metric (`moa_roc.pkl`): recapitulates B1/E1 nearest neighbours
- `validation_stats.json`: separates a fixed external annotation pair set

**Speaker notes.** The figure makes it self-evident — blue over orange on the
top row, orange over blue on the bottom. Keep this short: it is a
don’t-trust-the-obvious-number caveat, not a result. Say which one you are
using (the canvas metric, because it is the paper’s) and move on. If you are
tight on time this can drop to backup, unless someone in the room has already
seen the `validation_stats.json` numbers — then it pre-empts the objection.

### 4 — Does each space preserve its own raw data?
**Figure:** `signature_recovery_lollipop.png` (Fig. 3b)

**Slide bullets**
- Fig. 3b: do the higher signatures still recover sign0’s nearest neighbours?
- D6.007 wins every step ending at sign2 (0.931 vs 0.883 from sign0)
- But sign0→sign3, the one that matters, reverses: 0.682 vs 0.698

**Speaker notes.** The strongest slide in the deck, because it is the one that
could have gone the other way. Walk left to right. The intermediate wins are
exactly what slide 2 predicts — 3,083 features compress into a 128-dimensional
sign2 more faithfully than 9,802 do. Then stop on sign0→sign3 and make the
point explicitly: sign3 is what the CC actually serves, and better
intermediate compression of a thinner input did not produce a better final
signature. Likely reason: sign3 is inferred by a network trained across all CC
spaces, so D6.007 enters that step with a weaker proteomics signal.

### 5 — How do they compare against the rest of the CC?
**Figure:** `across_roc_scatter.png`

**Slide bullets**
- D6.007 has the lower AUROC in 25 of 25 CC spaces (mean −0.011)
- Largest losses are D-level: D1.001 −0.019, D5.001 −0.015, D2.001 −0.014
- A uniform loss everywhere is lost resolution, not gained orthogonality

**Speaker notes.** Left panel for the shape — everything sits under the
diagonal. Right panel for magnitude and identity. Worth naming the D-level
losses: Table S2 says a D-level space should recapitulate its own level well,
so that is the least favourable place to lose ground. This slide also kills
the orthogonality objection, so have the sentence ready: a space that had
gained orthogonal signal would win somewhere, and this one wins nowhere.

### 6 — Do the two spaces still agree with each other?
**Figure:** `recap_roc_DeepCoverMoA_NN.png`

**Slide bullets**
- Cross-space recapitulation is 0.61 at p=0.01 — barely above chance
- The paper gets 0.72–0.74 comparing transcriptomics with proteomics
- At p=0.001 it rises to 0.71: the closest neighbours do survive

**Speaker notes.** Lead with the comparison that stings: two genuinely
different data types agree with each other more than these two versions of the
same space do. Then point at the second curve and soften it honestly — the
tightest neighbourhoods are preserved, it is the mid-range structure that was
reorganised. Show one direction only; the reverse is 0.605 against 0.606, so
just say it is symmetric and keep that figure in backup.

### 7 — Did the new compounds reach new chemistry?
**Figure:** `input_compound_tsne_overlay.png`

**Slide bullets**
- The 84 new compounds scatter through the existing 872
- They add density, not coverage — no empty region gets filled
- Membership is from sign0: real measured data, not inference

**Speaker notes.** This is the Extended Data Fig. 2/5 test and it reads
instantly, so let the picture do the work. The one technical point worth
making: this uses sign0 membership because sign3 is inferred for all ~1.2M CC
compounds — a sign3 version of this plot would mark everything as shared and
tell you nothing.

### 8 — The case for D6.007
**Figure:** `confidence_overlay.png` (+ optionally the DBSCAN panel cropped
from `structure_metrics_bar.png`)

**Slide bullets**
- Mean sign3 confidence improves: 0.164 → 0.223
- DBSCAN clusters rise 113 → 147, which Table S2 reads as finer structure
- But unclustered compounds rise too (14.5% → 16.5%): partly fragmentation

**Speaker notes.** Give the counter-evidence a real slide rather than a
footnote — presenting it yourself is what makes the conclusion credible, and
someone will find these numbers anyway. Then close it honestly: the extra
clusters come with more unclustered compounds and a larger top cluster, and
higher confidence is expected when more compounds have real data. Confidence
measures inference support, not whether the space resolves biology.

### 9 — Conclusion
**No figure.**

**Slide bullets**
- Keep D6.002 as the primary proteomics space
- D6.007 is a higher-confidence, lower-resolution companion, not a replacement
- For more compounds, try a protein union with missing-value handling

**Speaker notes.** Tie it back to slide 2: the 84 extra compounds do not reach
new bioactivity space and do not offset losing 69% of the proteins. End
forward-looking rather than negative — the current construction is
compound-limited by design and feature-limited as a side effect, so a union
with missing-value handling is the more promising next build. Flag the one
caveat before anyone else does: the whole-universe AUROCs are computed over
each space’s entire inferred population, so the targeted tests on slides 4, 6
and 7 are what the conclusion actually rests on.

---

## Backup slides

Keep these out of the main flow; pull them up if asked.

| Figure | Pull it up when someone asks... |
|---|---|
| `headline_metrics_bar.png` | "what about the other metrics?" — all 10 at once |
| `recap_roc_All_Shared_Proteins_NN.png` | "is the comparison symmetric?" |
| `structure_metrics_bar.png` | "what about redundancy / orthogonality?" |
| `key_coverage_comparison.png` | "did the new compounds connect to the CC?" (they did — distributions are superimposed) |
| *Non-discriminative metrics* | "the outlier rate / coverage looked different to me" |

That last one has no figure and does not need one — one slide of text:
`pct_outliers` is pinned at exactly 10% by a hardcoded
`IsolationForest(contamination=0.1)`, and `moa_cov`/`atc_cov` measure coverage
of a fixed external validation file, so they are bit-identical for any two
whole-universe sign3 spaces. Neither can discriminate; both have been cited as
if they could.

---

## Practical notes

**Gap — slide 2 has no figure.** The single most important fact in the
analysis is a pair of numbers with no graphic. A simple two-bar panel
(compounds and features, side by side, with the −68.5% called out) would carry
that slide far better than a table. Nothing in the pipeline produces it yet —
worth adding if you want it.

**Dense figures need cropping.** `headline_metrics_bar.png` (10 panels) and
`structure_metrics_bar.png` (6 panels) are built for reading at desk size, not
from the back of a room. For slide 8, crop the single DBSCAN panel rather than
shrinking all six. **Note the side effect:** both grids now carry one shared
legend across the top instead of repeating the space names under every panel,
so a cropped single panel arrives with no colour key — add one in the slide,
or say “blue is the original” out loud when it appears.

**Consistency already holds.** Blue is always D6.002 and orange always D6.007,
in every figure, assigned by position rather than by label. All bars and
lollipops carry their exact value, so you never have to ask the audience to
compare heights on differences of 0.01–0.02.

**Two numbers to have memorised** for questions: **9,802 → 3,083** proteins,
and **25/25** CC spaces worse. Nearly every challenge routes back to one of
those.

**One caveat to state rather than be caught by.** The whole-universe AUROCs
(MoA, ATC, across-CC) are computed over each space's entire ~1.23M-compound
*inferred* population, so the two are not compared at equal difficulty. The
targeted tests — slides 4, 6 and 7 — carry more weight, and the conclusion
rests on those.
