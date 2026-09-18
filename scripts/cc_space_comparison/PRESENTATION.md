# Presentation plan: D6.002 vs D6.007

A 9-slide core deck with a backup section, plus a figure guide (§*How to read
every figure*) holding the background you need to explain each plot. Figures
are in `output/D6.002_vs_D6.007/`. Numbers and full argument in `RESULTS.md`.

**The arc**: ask the question → reframe what the change actually *was* →
show four independent lines of evidence → give the counter-case an honest
hearing → recommend. The reframe has to come second. Every result afterwards
is read differently once the audience knows D6.007 lost 69% of its proteins,
and if you show AUROC deltas first they will spend the rest of the talk
assuming this is a story about 84 extra compounds.

**Bullet style.** Facts and numbers, never questions, and no artifact names —
file names, pickle keys and function names belong in the speaker notes and in
`RESULTS.md`, where someone can act on them; on a slide they cost a line of
width and buy nothing an audience can use.

Use `->` only where there is a real term→definition or cause→consequence
relation to show. Where the arrow would just stand in for "is", "says" or a
colon, use a colon — a slide full of arrows that all mean different things is
harder to read than plain statements, not easier. And never put a `->` on a
bullet that already uses `→` for a numeric change; write "rises 113 to 147"
instead. Slide 3 is the one slide where nearly every bullet earns an arrow,
because it really is a set of definitions plus their consequence.

---

## Core deck

Each slide below gives the figure, the bullets to put on it (kept under 100
characters so they stay one line at projector size), and a speaker script —
sentences you can say more or less as they stand, four or five beats a slide.
Anything that is background rather than delivery lives in *How to read every
figure* below; anything that is Q&A material sits in a short *If asked* block.

### 1 — Scope
**No figure.**

**Slide bullets**
- D6.002 -> proteomics space from the DeepCoverMoA screen alone (Mitchell et al.)
- D6.007 -> same space extended with additional proteomics datasets
- Both fitted sign0→sign3 in one CC instance -> differences are data, not pipeline
- Validation follows CC Protocols: Fig. 3, Ext. Data Fig. 2/5/8, Table S2

**Speaker script**
* Today I'm comparing two proteomics bioactivity spaces we built in the
  Chemical Checker, D6.002 and D6.007.
* D6.002 comes from the DeepCoverMoA screen alone; D6.007 is the same space
  extended with additional proteomics datasets and more compounds.
* The question is whether that extension made it a better bioactivity space.
* Both were fitted sign0 to sign3 in the same CC instance, so nothing you'll
  see is a pipeline difference — it's the data. And every test follows the CC
  Protocols paper's own validation.

### 2 — Datasets summary
**Figure:** the signature-matrix table below (this is the deck's existing
"Datasets Summary" slide, with the Sign3 column filled in).

| | DeepCoverMOA (D6.002) | All Shared Proteins (D6.007) |
|---|---|---|
| Input FeatMat | (872, 9.802) | (956, 3.083) |
| Intersection with CC | 829 | 906 |
| Sign0 FeatMat | (872, 9.802) | (956, 3.083) |
| Sign1 FeatMat | (872, 543) | (956, 410) |
| Sign2 FeatMat | (872, 128) | (956, 128) |
| **Sign3 FeatMat** | **(1.230.708, 128)** | **(1.230.715, 128)** |

**Slide bullets**
- Compounds: 872 → 956 (+84, +9.6%), D6.002 a strict subset of D6.007
- Proteins: 9,802 → 3,083 (−6,719, −68.5%), panels intersected not merged
- Sign3 matrices: (1.230.708, 128) and (1.230.715, 128)
- New to the CC universe: 43 compounds from D6.002, 50 from D6.007

**Speaker script**
* Before any results, what actually changed — because it reframes everything
  that follows.
* Compounds went from 872 to 956. That part is clean: D6.002's set is a strict
  subset, so nothing was lost.
* Proteins went from 9,802 down to 3,083. "All Shared Proteins" keeps only the
  proteins measured across all the integrated datasets, so the panels were
  intersected, not merged — to gain 84 compounds we gave up 69% of the
  features. Please hold that number; it explains almost every result coming.
* The sign3 row is new. A sign3 key set is the CC universe plus whatever
  compounds that dataset brings that the CC didn't already have — 43 for
  D6.002, 50 for D6.007, on top of a shared universe of 1,230,665. That's why
  the two matrices differ by 7 rows.

**If asked.** Both 43 and 50 are measured against the same fixed reference —
the original CC, without any proteomics — so they are directly comparable and
that is the number to quote. The 43 are a *subset* of the 50 (D6.007 inherits
all of D6.002's compounds), so do not add them or read 50 as "50 more than
D6.002". And avoid framing the difference as "the extension adds 7 molecules":
that quietly makes CC + D6.002 the baseline, when the whole point is that the
two spaces are alternative additions to the same CC. The full split:

| | in core CC (A1–E5) | new molecule to CC | total |
|---|---|---|---|
| also in D6.002 | 829 | 43 | 872 |
| new in D6.007 | 77 | 7 | 84 |
| total | 906 | 50 | 956 |

"In core CC" means the 25 exemplary spaces, **excluding both of our proteomics
builds** — the universe is the union of the other spaces' sign2 keys,
1,230,665 molecules. Keep the two senses of "new" apart: **all 84 are new
proteomics measurements**; only 7 are molecules the CC had never seen. The
other 77 were known to the CC through other data types and now gain real
proteomics. Table S2 gives 128
features and exactly this key set as the pass condition for a type III
signature.

### 3 — Neighborhood and annotation ROC curves
**Figure:** `moa_atc_roc_overlay.png`

**Slide bullets**
- Two MoA/ATC AUROCs per diagnosis run -> opposite winners
- Neighbourhood AUROC -> B1/E1 nearest-neighbour recapitulation (the paper's)
- Annotation AUROC -> separation of a fixed curated annotated-pair set
- Same metric name, different test -> conclusion flips silently

**Speaker script**
* A quick caution about measurement, because this one caught me out.
* Every CC diagnosis produces two different MoA and ATC AUROCs, and they
  disagree about which space wins.
* The top row is the neighbourhood metric — how well each space reproduces the
  nearest-neighbour structure of B1, mechanism of action, and E1, therapeutic
  areas. The bottom row separates a fixed curated set of annotated compound
  pairs.
* You can see the reversal: blue above orange on top, orange above blue below.
  Both gaps are about 0.01. I'll use the neighbourhood metric from here,
  because it's the paper's.

**If asked.** The neighbourhood AUROC comes from `Diagnosis.cross_roc()`,
stored in `moa_roc.pkl`/`atc_roc.pkl`; the annotation AUROC comes from
`signature_base.validate()`, stored in `validation_stats.json`. They share the
field names `moa_auc` and `atc_auc` and measure different things, which is
exactly why this slide exists. If you are tight on time it can drop to backup,
unless someone has already seen the `validation_stats.json` numbers — then it
pre-empts the objection.

### 4 — Signature recovery
**Figure:** `signature_recovery_lollipop.png`

**Slide bullets**
- Fig. 3b test -> higher signature scored on recovering sign0's neighbours
- Thinner input (3,083 features) -> more faithful 128-dim compression
- Steps ending at sign2: D6.007 ahead, 0.931 vs 0.883 from sign0
- But sign0→sign3 reverses: 0.682 vs 0.698

**Speaker script**
* This is the paper's Figure 3b test, run inside each space: take the nearest
  neighbours at a lower signature type, ask whether the higher one still
  recovers them. One is perfect, 0.5 is chance.
* D6.007 wins the steps ending at sign2 — 0.931 against 0.883 from the raw
  data.
* That's what the previous slide predicts: 3,083 features lose less than 9,802
  when you squeeze them into 128 dimensions.
* But sign0 to sign3 reverses, 0.682 against 0.698 — and that's the step that
  matters, because sign3 is what the Chemical Checker actually serves.
* Better compression of a thinner input did not give a better final signature.
  sign3 is inferred by a network trained across all 25 spaces, and D6.007
  enters that step with a weaker proteomics signal.

### 5 — AUROC against CC spaces
**Figure:** `across_roc_scatter.png`

**Slide bullets**
- D6.007 lower in 25 of 25 CC spaces, mean −0.011, no single win
- Largest losses are D-level: D1.001 −0.019, D5.001 −0.015, D2.001 −0.014
- Direction is robust across 25 tests; individual magnitudes carry no error bar
- Table S2: own-level recapitulation is the expected good outcome
- Uniform loss everywhere -> lost resolution, not gained orthogonality

**Speaker script**
* Here I ask how well each space recapitulates the neighbourhood structure of
  all 25 other CC spaces.
* On the left, each dot is one space — D6.002 on x, D6.007 on y, dashed line is
  equality. Every dot sits below it: 25 out of 25, mean minus 0.011.
* The largest losses are all at the D level, transcriptomics worst at minus
  0.019. That's the least favourable place to lose, because Table S2 says a
  space should recapitulate its own level well — though I'd hold that one
  loosely, because these values carry no error bars and when I re-measured
  D1.001 with them, the gap disappeared. It's the 25-out-of-25 direction I'd
  stand behind, not any single number.
* This also closes off the orthogonality objection: a space that had gained
  orthogonal signal would win somewhere. This one wins nowhere.

### 6 — Recapitulating transcriptomics neighbourhoods
**Figure:** both spaces side by side — `recap_roc_D1.001_by_DeepCoverMoA.png`
and `recap_roc_D1.001_by_All_Shared_Proteins.png`. The two spaces compared
against *each other* move to backup, along with the paper's own panels.

**Slide bullets**
- D1.001 (transcriptomics) defines the neighbours; each panel is one of our spaces
- 0.605 vs 0.607 at p=0.01, and 0.729 vs 0.727 at p=0.001
- Differences ≤0.002 against a ±0.007–0.018 spread -> indistinguishable
- The paper's own D6.001 scores 0.60 and 0.72: our build lands on it

**Speaker script**
* Here I'm asking how well each of our proteomics spaces recovers the
  neighbourhood structure of an *external* space — D1.001, transcriptomics.
* Transcriptomics defines which molecules are neighbours; each panel asks how
  well one of our spaces can tell those pairs apart. Green is the closest 1%
  of pairs, blue the closest 0.1%.
* The two panels are effectively identical: 0.605 and 0.607 at the 1% cutoff,
  0.729 and 0.727 at 0.1%. Those gaps are two thousandths, against a subsample
  spread of seven to eighteen thousandths. Neither space recovers
  transcriptomics better than the other.
* The more valuable result is the comparison with the paper. It runs this
  exact test with its own proteomics space and reports 0.60 and 0.72 — we land
  on both. Our build behaves the way the protocol says it should.
* So this slide does two jobs: it's the external validation that the pipeline
  is sound, and it's a clean null result for the comparison — this axis does
  not separate our two spaces.

**The rounding trap on this slide.** The printed legends read 0.60 and 0.61,
which looks like a win for D6.007. It is not — the underlying values are
0.6050 and 0.6067, a gap of 0.002 against a standard deviation of 0.007. If
anyone points at the legend, say the two are the same number to the precision
this test supports.

**Watch the colours here.** Green and blue are *not* the two spaces — within
each panel they are the two p-value cutoffs, in the paper's own colours. The
two spaces are the two *panels*.

**If asked** how the two spaces compare against *each other* rather than
against transcriptomics: 0.61 at p=0.01 and 0.71 at p=0.001, with the reverse
direction giving 0.61 and 0.73 — symmetric, and again matching the paper's own
equivalent values (0.60/0.61 and 0.72/0.74 in Ext. Data Fig. 8g,h). Figures in
backup. That comparison was on this slide in an earlier draft and came off
because it is self-referential: both spaces are proteomics built from
overlapping data, so agreement between them is close to guaranteed and tells
you less than an external reference does.

### 7 — New compounds representation
**Figure:** `input_compound_tsne_overlay.png`

**Slide bullets**
- The 84 new compounds scatter through the existing 872 -> added density, not coverage
- Membership from real measured data (sign0), not inferred signatures

Two bullets on purpose. The picture carries this slide, and the CC-membership
split (77 already in the original CC, 7 not) is a *different* question —
universe growth, not new chemistry — which slide 2 already answers with 43 vs
50. It is kept in the notes below for questions rather than put on the slide.

**Speaker script**
* This asks whether the 84 new compounds reached new chemistry. It's the test
  the paper runs in Extended Data Figures 2 and 5.
* Grey is the 872 compounds present in both, orange the 84 only D6.007 has, all
  in one shared embedding.
* If the extension had opened new territory, the orange points would cluster
  where the grey ones don't reach. They're scattered right through instead —
  density, not coverage.
* One technical point worth making: membership here comes from each dataset's
  real measured data — sign0 — not from sign3. sign3 is inferred for all 1.2
  million CC compounds, so a sign3 version of this plot would mark everything
  as shared and tell you nothing.

**If the CC-membership split comes up.** Of the 84, 77 were already in the
original CC and 7 were not. Two rules for quoting it:

1. **"Already in the CC" means the original CC** — the 25 exemplary A1–E5
   spaces, *not* D6.002 and not anything we built. The reference is the union
   of those spaces' sign2 keys, 1,230,665 molecules, and it contains no
   proteomics at all (verified: no D6 space is in it).
2. **Never say "the extension adds 7 molecules to the CC."** That treats
   CC + D6.002 as the baseline, which is wrong for this comparison: D6.002 and
   D6.007 are two candidate additions to the *same* proteomics-free CC, and
   you would deploy one or the other. Measured against that fixed reference,
   D6.002 brings 43 new molecules and D6.007 brings 50 — that is the
   comparison, and it is on slide 2.

**If asked** "surely all 84 are new to the CC as well?" — no, and this is the
most likely confusion in the room. The CC universe is ~1.23 M molecules drawn
from ChEMBL, DrugBank and similar, i.e. anything with *any* recorded
bioactivity. Proteomics screens test approved drugs and tool compounds, which
is exactly that population, so ~95% of any such dataset is already in the CC —
your own sign0 slides say 829/872 and 906/956, both 95%. The 84 are compounds
the second study screened and the first didn't; they are mostly well-known
drugs, not novel chemical matter. Checked directly: the 77 appear in 4 to 24
of the 25 original CC spaces (most in 15–22), while the 7 appear in none.
"New to the proteomics dataset" and "new molecule to the CC" are independent
properties, and for well-studied drugs the first is common and the second is
rare.



### 8 — Confidence and cluster structure
**Figure:** `confidence_overlay.png` (+ optionally the DBSCAN panel cropped
from `structure_metrics_bar.png`)

**Slide bullets**
- Mean sign3 confidence rises 0.164 to 0.223 -> better-supported inference
- DBSCAN clusters rise 113 to 147: Table S2 reads this as finer structure
- But unclustered rises 14.5% to 16.5%, and the top cluster 8.3% to 11.6%
- Confidence -> ranking within a space, no published pass threshold

**Speaker script**
* I want to give the case *for* D6.007 a proper slide, because there is one.
* Confidence is the strongest point: mean sign3 applicability goes from 0.164
  to 0.223 and the whole distribution shifts right. More compounds with real
  data means the inference is better supported.
* Clustering is the second: 147 DBSCAN clusters against 113, which Table S2
  reads as finer structure.
* But unclustered compounds rise from 14.5% to 16.5% and the largest cluster
  grows from 8.3% to 11.6%, so some of that is fragmentation. And confidence
  ranks compounds *within* a space — there's no published threshold that makes
  a space reliable.

### 9 — Conclusion
**No figure.**

**Slide bullets**
- Keep D6.002 as the primary proteomics space
- D6.007 -> higher-confidence, lower-resolution companion, not a replacement
- More compounds -> protein union with missing-value handling, not intersection
- The conclusion rests on signature recovery, the 25 CC spaces and the overlay

**Speaker script**
* My recommendation is to keep D6.002 as our primary proteomics space.
* The 84 extra compounds don't reach new bioactivity space, and they don't
  offset losing 69% of the measured proteins — which shows up as uniform
  degradation across all 25 CC spaces.
* I wouldn't call D6.007 a failure, though. It's a higher-confidence,
  lower-resolution companion, best evaluated alongside D6.002 rather than
  instead of it.
* If we want more compounds, the better route is a *union* of proteins with
  missing-value handling rather than an intersection.
* One caveat I'd rather state than be caught by: the whole-universe AUROCs are
  computed over each space's entire inferred population, so the two aren't
  compared at equal difficulty. Signature recovery and the compound overlay are
  the targeted tests, and they're what this rests on.

---

## Backup slides

Keep these out of the main flow; pull them up if asked.

| Figure | Pull it up when someone asks... |
|---|---|
| `headline_metrics_bar.png` | "what about the other metrics?" — all 10 at once |
| `recap_roc_*_NN.png` (2 figs) | "how do the two spaces compare against each other?" — 0.61 / 0.71 and 0.61 / 0.73, symmetric |
| `docs/ext_data_fig8_gh.png` | "what exactly does the paper report?" — its own panels g,h |
| `structure_metrics_bar.png` | "what about redundancy / orthogonality?" |
| `key_coverage_comparison.png` | "did the new compounds connect to the CC?" (they did — distributions are superimposed) |
| *Non-discriminative metrics* | "the outlier rate / coverage looked different to me" |

That last one has no figure and does not need one — one slide of text:
`pct_outliers` is pinned at exactly 10% by a hardcoded
`IsolationForest(contamination=0.1)` (exactly 1,000 of 10,000 keys flagged in
both datasets, verified), and `moa_cov`/`atc_cov` measure coverage of a fixed
external validation file, so they are bit-identical for any two whole-universe
sign3 spaces. Neither can discriminate; both have been cited as if they could.

---

## How to read every figure

Background for each plot: what is drawn, how it was computed, what the paper
does with it, and what it shows here. This section is for you, not the slides.

### Common machinery: the NN-recapitulation AUROC

Four of the figures are the same test with different inputs, so it is worth
understanding once. It comes from the paper's Fig. 3 and is used throughout
its Extended Data figures.

1. Pick a **reference** representation and a **query** representation of the
   same set of molecules.
2. Sample a set of molecules (here 2,500 per subsample, 5 subsamples, matching
   the paper) and compute all pairwise **cosine distances** in the reference.
3. Call a pair **nearest neighbours** if its reference distance falls below
   the *P*-th percentile of that subsample's own distance distribution.
   P = 0.01 means "the closest 1% of pairs". This threshold is recomputed per
   subsample from the reference's own distances, not fixed globally.
4. Score every pair by its cosine distance in the **query** representation,
   and ask how well that score separates the NN pairs from the rest. That
   separation is the AUROC.

Read it as: **how much of the reference's neighbourhood structure survives in
the query.** 1.0 = perfectly preserved, 0.5 = the query knows nothing about
the reference's neighbourhoods. A tighter *P* asks only about the very closest
pairs, which is why AUROC always rises as *P* falls.

Two framings matter here, and they are not interchangeable:

- **Within a space** (slide 4): reference = a lower signature type, query = a
  higher one, both for the same dataset. Asks *did the pipeline preserve this
  dataset's own raw structure?*
- **Between two spaces** (slide 6): reference = one dataset's sign3, query =
  the other's. Asks *are these two spaces describing the same biology?*

### `moa_atc_roc_overlay.png` — slide 3

**What is drawn.** Four ROC panels in a 2×2 grid. Columns are MoA (against
B1.001, mechanism of action) and ATC (against E1.001, therapeutic area). Rows
are the two different metrics. Each panel overlays both spaces; the legend
carries the AUROC.

**How computed.** *Top row (neighbourhood):* the CC diagnosis's own
`cross_roc()` — take 10,000 molecules shared with B1/E1, call each molecule's
k=5 nearest neighbours *in B1.001/E1.001 sign3* a positive pair, score pairs
by cosine distance in the space being diagnosed. This is the number the CC
diagnosis canvas prints in its panel title. *Bottom row (annotation):*
`signature_base.validate()` — read a fixed external validation set shipped
with CC, and separate pairs that share a MoA/ATC annotation from pairs that do
not.

**Paper backing.** Table S2's "MoA" and "ATC" rows describe the top row only:
"Recapitulation of the nearest neighbors in the B1 space (MoA) using the
current signature… quantified by AUROC… indicated in the figure title." The
bottom row is a separate CC validation routine with no Table S2 entry.

**How to read it.** Higher is better, but Table S2 explicitly declines to set
a pass mark: "a low AUROC is not necessarily a negative outcome, as it may
instead reflect orthogonality." That escape hatch is available here in
principle and is closed by slide 5, not by this figure.

**What it shows here.** Top row: D6.002 ahead (0.761 vs 0.747 MoA; 0.736 vs
0.726 ATC). Bottom row: D6.007 ahead (0.729 vs 0.741; 0.699 vs 0.706). Both
gaps are ~0.01, so the honest statement is that the two spaces are close on
annotation recovery and the *direction* depends on which test you run.

### `signature_recovery_lollipop.png` — slide 4

**What is drawn.** Six groups of two lollipops, one group per signature-type
pair. Height is the NN-recapitulation AUROC of the higher signature against
the lower, with an error bar over 5 subsamples and the value printed above.
The dashed line at 0.5 is chance. Blue = D6.002, orange = D6.007.

**How computed.** The common machinery above, run separately inside each
dataset over that dataset's own real input compounds (872 and 956). Pools
smaller than n_random=2,500 are subsampled at 80% per repetition so the error
bar is meaningful.

**Paper backing.** This is Fig. 3b, the paper's own signature-cascade
validation, and it is the reason the six pairs are exactly these six.

**How to read it.** Each bar answers *does signature type Y still know what
type X considered similar?* A healthy cascade decays gently: near-1.0 for
sign0→sign1 (a linear projection retaining 90% of variance should lose almost
nothing), lower for sign2 (a fixed 128-dim harmonisation), lower still for
sign3 (inferred from the whole CC, not from this space alone).

**What it shows here.** Both cascades are healthy in shape. D6.007 wins the
three steps involving sign2 (0.931 vs 0.883, 0.932 vs 0.917, 0.781 vs 0.748)
and loses the three that end at sign3 from sign0 or sign1 (0.682 vs 0.698,
0.669 vs 0.696). The single most load-bearing number on this slide is
sign0→sign3, because that is the end-to-end question: is what the CC serves
still a description of what was measured?

### `across_roc_scatter.png` — slide 5

**What is drawn.** Two panels. Left: a scatter with D6.002's AUROC on x and
D6.007's on y, one dot per CC space, plus the y=x diagonal; dots below the
line are spaces where D6.007 is worse. Right: the same 25 numbers as a sorted
horizontal bar chart of the difference (D6.007 − D6.002), each labelled.

**How computed.** Read from the diagnosis artifact `across_roc.pkl`, which
stores a full ROC/PR curve per CC space. Each space's AUROC is that space's
sign3 nearest neighbours recapitulated by the space being diagnosed.

**Paper backing.** Table S2's "ROC across CC" row: "Recapitulation of the
nearest neighbors of all CC spaces in terms of the AUROC, sorted in descending
order." Its good outcome: "If the created space belongs to a predefined CC
type (e.g. B, targets), we typically expect to observe decent recapitulation
of the same level spaces."

**How to read it.** The scatter is for the *shape* of the difference: a space
that gained orthogonal information would be scattered on both sides of the
diagonal — worse against the spaces it now diverges from, better against
whatever it moved towards. A space that simply lost resolution sits uniformly
below. The bar chart is for magnitude and identity.

**What it shows here.** 25 of 25 below the diagonal, mean −0.011, range
−0.0043 (E5.001) to −0.0193 (D1.001). No space is a win. D-level spaces —
D6's own level, and the ones Table S2 says should be recapitulated well —
supply three of the four largest losses. This is the figure that closes off
the orthogonality reading of slide 3.

### `recap_roc_*_NN.png` — backup (two figures)

**What is drawn.** A ROC curve with a ±1 std shaded band across 5 subsamples,
plotted twice on the same axes at two NN cutoffs (P = 1e-2 and P = 1e-3), with
`pval:X.Xe-0Y - AUROC±std` in a boxed lower-right legend. Green = the looser
cutoff, blue = the stricter one. The dashed diagonal is chance. The two files
are the two directions of the comparison.

**Why these are backup rather than slide 6.** Both spaces are proteomics built
from overlapping data, so a high level of agreement between them is close to
guaranteed and carries little information either way. The
`recap_roc_D1.001_by_*.png` pair below asks the same question against an
*external* reference, which is both a harder test and the one with a published
value to check against, so that pair took the slide.

The colours, axis labels (FPR/TPR), quarter-step ticks, dashed grid, square
aspect and legend format are all chosen to match Ext. Data Fig. 8g,h, so the
two can be read against each other directly if the question comes up. They are
deliberately *not* shown side by side on the slide: the values agree, so
putting both up invites a methodology discussion that changes no conclusion.

**How computed.** The common machinery above, with reference = one dataset's
sign3 and query = the other's, over the 1,230,708-compound key intersection
(a 50,000-compound random pool, then 2,500 molecules × 5 subsamples).

**Paper backing.** Ext. Data Fig. 8g,h runs this same test between two sign3
spaces, at these two p-values, with this legend format. Fig. 3g,h is the same
plot applied within one dataset across signature types.

**Reading the caption's grammar** — "Recapitulation of kNN **at X** type III
signature level **using Y** type III signatures":

- *at X* → X defines the neighbours. X is the **reference**, the answer key.
- *using Y* → Y is scored on recovering them. Y is the **query**.

So the paper's panels are `g: reference D1.001 ← query D6.001` and
`h: reference D1.002 ← query D6.001`. **The query is D6.001 in both**; the
paper varies the reference, asking how well proteomics recovers the
neighbourhoods of each of two transcriptomics versions. It is not a
bidirectional test.

**Ours is bidirectional**, which is a deliberate difference: `reference D6.002
← query D6.007` in one file and the roles swapped in the other. The per-panel
quantity is identical, so the magnitudes are comparable as a benchmark of what
this test returns — but do not describe the paper's g and h as "two
directions", because they are not.

**Why sign3 and not sign0 for this one.** sign3 is the only representation
both spaces share over a large common key set; sign0 overlaps on just 872
compounds. The whole-universe framing is deliberate and matches the paper —
but it does mean this test is dominated by *inferred* signatures, which is why
slide 7 exists as the complementary real-data test.

**How to read it — and the cutoff trap.** 0.5 is "these two spaces know
nothing about each other's neighbourhoods"; 1.0 is "identical structure". The
absolute value is only interpretable against a like-for-like cutoff, and this
is exactly where an earlier draft of this project went wrong: it set our
P=1e-2 number against the paper's P=1e-3 number and concluded our spaces
agreed unusually badly. **Always compare P to P.**

The paper's actual values, read off Ext. Data Fig. 8g,h (saved as
`docs/ext_data_fig8_gh.png`, extracted from the article PDF):

| reference ← query | P = 1e-2 | P = 1e-3 |
|---|---|---|
| paper g: D1.001 ← D6.001 | 0.60 ± 0.01 | 0.72 ± 0.03 |
| paper h: D1.002 ← D6.001 | 0.61 ± 0.01 | 0.74 ± 0.04 |
| ours: D6.002 ← D6.007 | 0.61 ± 0.00 | 0.71 ± 0.01 |
| ours: D6.007 ← D6.002 | 0.61 ± 0.01 | 0.73 ± 0.02 |

**What it shows here.** 0.61 ± 0.00 at P = 1e-2 and 0.71 ± 0.01 at P = 1e-3
(D6.002 ← D6.007); 0.61 ± 0.01 and 0.73 ± 0.02 in reverse. The near-symmetry
says neither space's neighbour structure is the more reproducible one. Our
values land on the paper's at both cutoffs, inside its own subsample spread.
One observation this permits, offered weakly:
you might expect two versions of the *same* proteomics space to agree with
each other more than proteomics agrees with transcriptomics, and they do not
measurably — but the paper's own spread is ±0.03–0.04, so this is a hint at
most and should not be reported as a finding. So ~0.61 at the 1% cutoff is
the *normal* result for two related sign3 spaces, not a red flag — the paper
reports it without comment. This figure is therefore **neutral** between
D6.002 and D6.007: it confirms the comparison behaves as published, and it
should not be cited as evidence either for or against the extension. The
argument against D6.007 rests on slides 4 and 5.

### `input_compound_tsne_overlay.png` — slide 7

**What is drawn.** A 2D t-SNE of both datasets' real input compounds, embedded
from their shared sign3 vectors. Grey = present in both (n=872). Orange and
larger = D6.007 only (n=84).

**How computed.** Membership comes from each dataset's **sign0** keys, i.e.
compounds with real, non-inferred measurements. The coordinates come from the
sign3 vectors of those compounds in D6.007's space, so both groups live in one
embedding and are directly comparable.

**Paper backing.** Extended Data Figs. 2a, 5a and 8a do this when comparing
two versions of a related space (B1.001 vs B1.002; D1.001 vs D1.002 vs
D6.001): project the newer space and highlight the points that also have a
sign0 entry in the older one.

**Why membership must come from sign0.** sign3 exists for every one of the
~1.2M CC compounds regardless of whether a dataset measured them, so a sign3
membership test would mark ~1.23M of 1.23M compounds as "shared" and show
nothing. This distinction is the single most important technical point in the
whole comparison.

**How to read it.** Two outcomes are possible and they mean opposite things.
If the orange points cluster in regions the grey ones do not reach, the new
data opened new bioactivity territory — the extension added coverage. If they
interleave with the grey, the new compounds resemble compounds already there —
the extension added density.

**What it shows here.** Interleaved throughout, with no orange-only region.
Density, not coverage. Slide 2's arithmetic says the same thing from a
completely different direction: 77 of the 84 were already CC compounds.

### `confidence_overlay.png` — slide 8

**What is drawn.** Overlaid density histograms of the sign3 confidence
(applicability) score for the two spaces, with a dashed line at each mean and
the means in the legend.

**How computed.** Read from the `confidences.pkl` diagnosis artifact, over the
10,000-key diagnosis subsample.

**Paper backing.** Table S2's "Confidence (I)" row, which gives *no*
good-or-bad outcome and instead notes that "Confidence values are
space-dependent and enable compound ranking — from the least confident to the
most confident one." Fig. 3c reports this distribution for a new space.

**How to read it — and the trap.** There is **no published 0.6 threshold**.
That number has been attributed to Table S2 in earlier drafts of this project
and it is not there; the row is explicit that the scale is space-dependent and
the values are for ranking *within* a space. So this plot supports "D6.007's
signatures are better supported by the rest of the CC" and does **not** support
"D6.007's signatures are reliable and D6.002's are not". Do not let a
questioner push you into the second claim. (Both means, 0.16 and 0.22, are far
below 0.6 anyway — and the distributions extend slightly below zero, so treat
this as a relative score, not a probability.)

**What it shows here.** D6.007's distribution is shifted right: mean 0.164 →
0.223, median 0.160 → 0.223. This is the clearest single point in D6.007's
favour, and it has a mundane explanation — more compounds with real data makes
the Siamese inference better supported for the surrounding universe.

### `structure_metrics_bar.png` — slide 8 crop / backup

**What is drawn.** Six labelled bar pairs: redundant signatures %, DBSCAN
cluster count, unclustered %, CC ranks agreement (RBO), mean CC spaces per
compound, and % of compounds in fewer than 5 CC spaces. One shared legend
across the top; every bar carries its value.

**Paper backing, per panel** (Table S2 wording, and note the reading direction
is *not* uniformly "higher is better"):

| Panel | Table S2 row | Direction |
|---|---|---|
| Redundant signatures % | "Redund" | "A low percentage of redundancy is considered favorable" |
| DBSCAN clusters | "Cluster sizes" | "A large number of clusters with low proportions of compounds each may indicate a continuous representation of the chemical space, capturing fine details" |
| Unclustered % | "Cluster sizes" | "Those compounds not assigned to any cluster correspond to outliers" — so lower is better |
| CC ranks agreement (RBO) | "CC ranks agreement (I)/(II)" | **Lower is better**: "Low RBO values typically indicate that the descriptors are correctly encoding the specificity of the current space, showing a certain degree of orthogonality" |
| Mean CC spaces / compound | "Sign wrt CC" | "A relatively high representation of the compounds in other bioactivity spaces… will lead to higher confidence type III signatures" |
| Compounds in <5 CC spaces | "Key Coverage (I)" | Bad outcome is "a high proportion of molecules present in less than 5 CC spaces" |

**What it shows here.** Redundancy (2.494 vs 2.527%), RBO (0.0857 vs 0.0878),
mean CC spaces (7.728 vs 7.760) and <5-space fraction (0.51 vs 0.50%) are all
effectively ties. The two panels that move are clusters (113 → 147) and
unclustered (14.48 → 16.49%) — which is why slide 8 presents them together:
the extra granularity comes with extra fragmentation, so the Table S2 "fine
details" reading is only half available.

**One caveat when quoting small deltas from this figure.** These all come from
each run's own 10,000-key diagnosis subsample, and the two runs' subsamples
overlap on only **2,157 of 10,000** keys. The comparison is unpaired: fine for
distributions, not for reading a 0.03-point difference as real.

### `recap_roc_D1.001_by_*.png` — slide 6 (two figures)

**What is drawn.** The same ROC-with-band panel as slide 6, but with **D1.001
(transcriptomics) as the reference** and each of our two spaces as the query
in turn. One figure per space, identical styling.

**Why this is on slide 6.** It asks the recapitulation question against an
*external* reference instead of between two spaces built from overlapping
proteomics data, so it is both the harder test and the only one in the deck
with a published value to check against. The two panels are the two spaces on
one axis, which also makes it read as a comparison slide rather than as two
directions of one comparison.

**Why this is not slide 5 either.** Slide 5's scatter also carries
a D1.001 number for each space (0.7361 and 0.7168), but it comes from a
*different test*. `across_roc` loops `Diagnosis.cross_roc()`, which defines
positives as each molecule's **k=5 nearest neighbours** in the reference and
draws a **balanced** negative set. This test defines positives by a
**distance percentile** (1% of pairs at P=0.01) with every other pair
negative. Different positive sets, very different class balance — the two
AUROCs are on different scales (0.74 versus 0.60 for the same pair of spaces)
and **must never be quoted against each other**.

**Paper backing.** This is Ext. Data Fig. 8g exactly: reference D1.001, query
a proteomics sign3 space. The paper's query is its D6.001; ours is D6.002 and
D6.007. So these two figures are directly comparable to the paper's published
panel, which the slide-6 figures are not.

| reference ← query | P = 1e-2 | P = 1e-3 |
|---|---|---|
| paper g: D1.001 ← D6.001 | 0.60 ± 0.01 | 0.72 ± 0.03 |
| ours: D1.001 ← D6.002 | 0.605 ± 0.007 | 0.729 ± 0.014 |
| ours: D1.001 ← D6.007 | 0.607 ± 0.007 | 0.727 ± 0.018 |

**What it shows here — two things, and the first is the more valuable.**

1. **The pipeline validates.** Both our spaces land within ~0.01 of the
   paper's own D6.001 at both cutoffs. If anyone asks whether the build is
   sound, this is the figure to reach for — it is the only external,
   published benchmark available.
2. **It does not separate the two spaces.** 0.605 vs 0.607 and 0.729 vs
   0.727, against standard deviations of 0.007–0.018. No effect.

**The rounding trap.** The legends print 0.60 and 0.61, which reads as a win
for D6.007. The underlying values are 0.6050 and 0.6067 — a gap of 0.002
against a standard deviation of 0.007. Quote three decimals, or say "the same
number", but do not let the printed legend carry a comparison.

**The honest caveat it raises for slide 5.** Slide 5 names D1.001 as D6.007's
single largest loss (−0.0193). Re-measured here *with error bars*, that gap
does not reproduce. The 25/25 sign pattern on slide 5 still stands — twenty-five
tests all landing negative is strong however small each one is — but the
individual magnitudes have no error bars behind them, and the one we have now
re-measured vanished. If you are asked about a specific CC space on slide 5,
say the direction is robust and the size is not.

### `key_coverage_comparison.png` — backup

**What is drawn.** Overlaid histograms of how many *other* CC spaces each
compound also appears in, with a dashed line at 5.

**Paper backing.** Table S2's "Key Coverage (I)" row. Good outcome: "A high
proportion of molecules is present in 5-6 (minimum) datasets, corresponding to
the CC chemical spaces (A1-A5) and the target binding space (B4)." Bad:
"A high proportion of molecules is present in less than 5 CC spaces, typically
indicating little overlap with the CC universe (leading to unconfident type
III signatures)." The row also notes "Values around 10 datasets are also
common, due to the high overlap between B4 and C3-C5."

**What it shows here.** Both distributions are essentially superimposed, both
peak at 9 with a second mode at 6, and 0.5% of compounds fall below 5 in each.
This is a **pass, not a difference** — the new compounds connected to the CC
universe normally. Use it to answer "did the new data land as isolated null
signatures?" (no), not as evidence either way in the comparison.

### `headline_metrics_bar.png` — backup

**What is drawn.** Ten small-multiple panels, each its own y-axis, each bar
labelled: MoA/ATC AUROC under both metrics, mean and median confidence, MoA/ATC
KS D statistic, mean anomaly score, and outliers flagged %.

**Why small multiples and not one chart.** An earlier version put a ~1.2e6
molecule count on the same axis as 0–1 AUROCs, which flattened every
meaningful bar to invisibility. Count-like columns are now excluded and each
metric gets its own scale. The differences on this figure are mostly 0.01–0.02,
which is exactly why every bar carries its printed value.

**Two panels to disregard.** "Outliers flagged (%)" reads 10.0 for both, by
construction — the CC diagnosis fits `IsolationForest(contamination=0.1)`, so
exactly 1,000 of 10,000 keys are flagged in every dataset ever run. Only the
neighbouring "Mean anomaly score" panel carries signal (−0.0262 vs −0.0201).
`moa_cov`/`atc_cov` were dropped from this figure for the same reason: they
measure coverage of a fixed external file and are bit-identical.

---

## Practical notes

**Reconciling this plan with the deck as it stands.** The current PDF
(`docs/1_CC_GlobalPerturbProtDatsets.pdf`) already has the sign0/sign1/sign2
walkthrough for all six datasets, the two-dataset summary, the two sign3
diagnosis canvases, and then slides titled "Neighborhood and Annotation ROC
Curves", "Signature Recovery", "AUROC against CC spaces" (×2), "New Compounds
Representation", "Confidence Scores" and "Further work ideas" — i.e. slides
3, 4, 5, 7 and 8 of the plan above, in that order. Two gaps:

- **Slide 6 (cross-space recapitulation) has no slide in the deck yet.** It is
  the test with the single most important number in the comparison and it has
  no substitute among the others — worth adding.
- **There is no conclusion slide** (slide 9); the deck ends on "Further work
  ideas".

Also: the page-number footers in the deck are placeholders from the sign3
section onwards (23, 23, 46, 46, 46, 46, 47), so they will need a pass.

**Dense figures need cropping.** `headline_metrics_bar.png` (10 panels) and
`structure_metrics_bar.png` (6 panels) are built for reading at desk size, not
from the back of a room. For slide 8, crop the single DBSCAN panel rather than
shrinking all six. **Note the side effect:** both grids carry one shared
legend across the top instead of repeating the space names under every panel,
so a cropped single panel arrives with no colour key — add one in the slide,
or say "blue is the original" out loud when it appears.

**Colour consistency, precisely.** Blue is D6.002 and orange is D6.007 in
every figure where both spaces appear as series: the ROC overlay, the recovery
lollipop, both metric grids, the confidence overlay and the key-coverage
histogram. Two figures do **not** follow that rule and should not be described
as if they did:

- `across_roc_scatter.png` — single-colour; the marks are per-CC-space
  *differences*, not one space or the other.
- `recap_roc_*_NN.png` — two shades of blue for two p-value cutoffs of one
  directional comparison (see slide 6).

All bars and lollipops carry their exact value, so you never have to ask the
audience to compare heights on differences of 0.01–0.02.

**Three numbers to have memorised** for questions: **9,802 → 3,083** proteins,
**25/25** CC spaces worse, and **0.682 vs 0.698** for sign0→sign3. Nearly every
challenge routes back to one of those. Note the deliberate change here: the
cross-space recapitulation AUROC used to be on this list, and it came off when
the p-value mismatch was found — it no longer discriminates.

**One caveat to state rather than be caught by.** The whole-universe AUROCs
(MoA, ATC, across-CC) are computed over each space's entire ~1.23M-compound
*inferred* population, so the two are not compared at equal difficulty. The
targeted tests — slides 4 and 7 — carry more weight, and the conclusion rests
on those together with the 25/25 result on slide 5. Slide 6 is a
*reproducibility* check rather than a discriminating test: it shows our
comparison behaves exactly as the published one does, and it favours neither
space.
