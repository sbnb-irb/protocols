# D6.002 ("DeepCoverMoA") vs D6.007 ("All Shared Proteins")

Comparison of two CC proteomics bioactivity spaces built in the same local
instance (`local_CC_D6`), following the validation methodology of
Comajuncosa-Creus et al., *Nat. Protoc.* 2025.

Reproduce with:

```bash
cd scripts/cc_space_comparison/
./run_compare_claudecode.sh --local-cc-dir /scratch/sbnb/sayala/protocols/local_CC_D6 \
    --dataset-a D6.002 --label-a "DeepCoverMoA" \
    --dataset-b D6.007 --label-b "All Shared Proteins" \
    --output-dir output/D6.002_vs_D6.007 --random-state 42
```

Runtime ~175 s (last run: 174.9 s wall). All outputs referenced below are in
`output/D6.002_vs_D6.007/`.

The original `run_compare.sh` / `compare_cc_spaces.py` / `cc_compare.py` are
kept unchanged alongside, for reference. The `*_claudecode` files are the
reworked versions this write-up is based on; the original ones still report
the MoA/ATC winner from `validation_stats.json` (see §2) and will not
reproduce these numbers.

---

## Bottom line

**D6.007 is not a better proteomics space than D6.002, and it is not simply
"D6.002 plus more data". It is a different, more weakly-resolved space bought
at a steep cost in measured features.**

The evidence points one way on the tests that measure how well the space
resolves bioactivity structure end-to-end -- the end-to-end signature recovery
(§3) and the uniform degradation across all 25 CC spaces (§4) -- and the
metrics that favour D6.007, better intermediate compression and higher
confidence, are explainable by the same underlying change and do not survive
to the signature the CC actually serves.

**The case rests on §3 and §4, not on §5.** The cross-space recapitulation was
previously the headline argument here; corrected for a p-value mismatch
against the paper (see §5) it is neutral between the two spaces and no longer
supports any conclusion. There is a real argument that D6.007 is
*complementary* rather than strictly worse, but it is weaker than the evidence
against it, and it is not the argument the headline numbers alone would
make.

---

## 1. The fact that reframes everything: D6.007 trades features for compounds

From `sign0_membership.json` — real, non-inferred input data:

| | D6.002 | D6.007 | change |
|---|---|---|---|
| Input compounds (`sign0` keys) | 872 | 956 | **+84 (+9.6%)** |
| Measured features (proteins) | **9,802** | **3,083** | **−6,719 (−68.5%)** |

D6.002's compounds are a **strict subset** of D6.007's (`a_is_subset_of_b:
true`, `n_only_a: 0`), so the extension is purely additive in compounds. But
"All Shared Proteins" means the protein set was **intersected** across the
integrated datasets: to gain 84 compounds, the space gave up more than
two-thirds of its proteins.

The full signature-matrix cascade, read directly from the CC instance
(`cc.get_signature(<type>, "full", <code>).shape`):

| | D6.002 (DeepCoverMoA) | D6.007 (All Shared Proteins) |
|---|---|---|
| sign0 | (872, 9 802) | (956, 3 083) |
| sign1 | (872, 543) | (956, 410) |
| sign2 | (872, 128) | (956, 128) |
| **sign3** | **(1 230 708, 128)** | **(1 230 715, 128)** |

Both sign3 matrices are 128-dimensional, which is Table S2's stated pass
condition for a type III signature ("The number of features should be 128").

**The 7-key difference between the two sign3 matrices is informative, and it
is not noise.** Per Table S2, a sign3 key set is the CC universe *joined with*
the new dataset's own compounds. The deck's own intersection counts are
829/872 for D6.002 and 906/956 for D6.007, so:

```
|sign3_A| = |CC universe| + (872 - 829) = |U| + 43 = 1,230,708  ->  |U| = 1,230,665
|sign3_B| = |CC universe| + (956 - 906) = |U| + 50 = 1,230,715   (consistent)
```

Verified directly against the key sets rather than inferred from the counts,
with the universe *U* pinned independently as
``sign3(D1.001) & sign3(D6.002) & sign3(D6.007)`` = 1,230,665 rather than
taken from the deck's intersection counts. The full cross-tabulation, with
every margin machine-checked:

| | in core CC (A1–E5) | new molecule to CC | total |
|---|---|---|---|
| **also in D6.002** | 829 | 43 | **872** |
| **new in D6.007** | 77 | 7 | **84** |
| **total** | 906 | 50 | **956** |

Identities confirmed: ``sign3(D6.002) == U | A``, ``sign3(D6.007) == U | B``,
``A ⊂ B``, and ``sign3(D6.007) - sign3(D6.002)`` is *exactly* the set of 7.

**The trap in this table, and it is an easy one.** The 43 and the 50 are not
two separate contributions to be compared: because D6.002's compounds are a
strict subset of D6.007's, ``(A - U) ⊂ (B - U)`` -- **the 43 are inside the
50**. Every molecule D6.002 brings that the CC lacked, D6.007 brings as well.
The extension's own contribution to the universe is therefore the difference,
50 - 43 = 7, not 50.

**The clearest way to see this**: run the ordinary intersection-with-CC
analysis on the *new compounds alone* (``B - A``) rather than on all 956, and
it returns 7 directly -- no subtraction needed
(``scripts/cc_space_comparison/verify_compound_split.py``):

```
D6.002  (all 872)          in core CC = 829    NOT in core CC = 43
D6.007  (all 956)          in core CC = 906    NOT in core CC = 50
the 84 NEW compounds only  in core CC =  77    NOT in core CC =  7

D6.007's 50 not-in-CC  =  43 inherited from D6.002  +  7 from the 84
```

The 956-compound run reports 50 because it compares D6.007 against the CC
*only*; it has no knowledge of D6.002, so the 43 that D6.002 already
contributed are counted again.

**The 43-vs-50 framing is the correct one, and the 7 should not be used as a
headline.** Both 43 and 50 are measured against the same fixed,
proteomics-free reference, so they are directly comparable: D6.002 brings 43
molecules the CC lacked, D6.007 brings 50. Describing the difference as "the
extension adds 7 molecules to the CC" quietly promotes CC + D6.002 to the
baseline, which is wrong for this comparison -- D6.002 and D6.007 are
*alternative* additions to the same original CC, not a base and an increment.
The 7 remains useful for one narrow purpose only: it explains why the two
sign3 matrices differ by exactly 7 rows.

**What this does and does not mean.** Two senses of "new" are in play and
they must be kept apart:

- **New measurements: all 84.** D6.007 measures 84 compounds D6.002 never did.
  That is real new proteomics data, without qualification.
- **New molecules to the CC: 7.** Of those 84, 77 were already in the **core
  CC** -- the 25 exemplary A1–E5 spaces -- through chemistry, target, cell or
  clinical data, and so already carried an inferred sign3 signature there. For
  those 77 the extension replaces inference with real measured proteomics,
  which is a genuine point in D6.007's favour. Only 7 were unknown to the CC
  entirely.

  **Phrasing to avoid:** "77 already had CC signatures" and "77 are already
  present in D6.002's sign3" both invite the reading that those signatures
  came from *our* DeepCoverMoA build. They did not. The universe used
  throughout this section excludes both D6.002 and D6.007 by construction
  (union of the other exemplary spaces' sign2 keys); the two are set-equal
  here only because the 84 are disjoint from D6.002's compounds.

Against the fixed proteomics-free reference the two builds contribute 43 and
50 new molecules respectively -- that is the comparable pair, and neither is
evidence about whether the new data is informative. The no-new-territory
conclusion in §6 rests on the t-SNE overlay alone.

This single fact predicts nearly every result below, and it should be stated
before any AUROC delta. A comparison that reported only the +9.6% compound
gain would describe half the change — and the less consequential half.

## 2. The headline MoA/ATC numbers were being read from the wrong artifact

There are **two different MoA/ATC AUROCs** in a CC diagnosis run. They measure
different things and they disagree about which space wins:

| Metric | Source | D6.002 | D6.007 | Winner |
|---|---|---|---|---|
| MoA AUROC (neighborhood) | `moa_roc.pkl` | **0.7605** | 0.7471 | D6.002 |
| ATC AUROC (neighborhood) | `atc_roc.pkl` | **0.7356** | 0.7261 | D6.002 |
| MoA AUROC (annotation) | `validation_stats.json` | 0.7292 | **0.7411** | D6.007 |
| ATC AUROC (annotation) | `validation_stats.json` | 0.6994 | **0.7062** | D6.007 |

Confirmed from the CC source, not inferred:

- **Neighborhood metric** (`Diagnosis.cross_roc()`): takes 10,000 shared
  molecules, calls each molecule's k=5 nearest neighbours *in B1.001/E1.001
  sign3* a positive pair, and scores pairs by cosine distance in the space
  being diagnosed. This is the number printed in the diagnosis canvas title
  and the Fig. 3 / Ext. Data Fig. 8 NN-recapitulation family. **This is the
  paper-faithful headline.**
- **Annotation metric** (`signature_base.validate()` →
  `Plot.vector_validation()`): reads a **fixed external validation set** from
  CC's `tests/validation_sets/`, and separates pairs that share a MoA/ATC
  annotation from pairs that do not.

Verification: the trapezoid AUC of `moa_sign3_auc_validation.tsv` reproduces
`validation_stats.json`'s `moa_auc` to six decimals (0.729215), while
`moa_roc.pkl['auc']` = 0.760537 matches the canvas title. Different curves
(43,307 vs 34,103 points), different tests, same names.

Both are now reported side by side in `summary_table.csv` and in all four
panels of `moa_atc_roc_overlay.png`, where the reversal is visible directly:
blue above orange on the top row, orange above blue on the bottom.

**How to read the disagreement.** D6.007 is *better* at separating curated
same-annotation compound pairs but *worse* at reproducing the global
neighbourhood structure of B1/E1. Both are consistent with the feature
intersection: restricting to the ~3,000 proteins measured everywhere plausibly
keeps the strongest, most annotation-aligned signal while discarding the
finer-grained protein detail that carries the rest of the neighbourhood
structure. One caveat in D6.007's favour: the annotation metric is
deterministic on a fixed external pair set, whereas the neighborhood metric
uses an unseeded random subsample, so the annotation metric is the better
*controlled* of the two even though it is not the paper's headline.

## 3. Signature recovery within each space: better intermediates, no better end-to-end

This is the Fig. 3b test, run separately for each space over its own real
input compounds (872 and 956). Nearest neighbours are defined at the lower
signature type and the higher one is scored on recovering them
(`signature_recovery.csv`, `signature_recovery_lollipop.png`):

| Pair | D6.002 | D6.007 | Better |
|---|---|---|---|
| sign0 → sign1 | **0.986** ± 0.001 | 0.968 ± 0.003 | D6.002 |
| sign0 → sign2 | 0.883 ± 0.004 | **0.931** ± 0.003 | D6.007 |
| **sign0 → sign3** | **0.698** ± 0.012 | 0.682 ± 0.007 | D6.002 |
| sign1 → sign2 | 0.917 ± 0.006 | **0.932** ± 0.002 | D6.007 |
| sign1 → sign3 | **0.696** ± 0.006 | 0.669 ± 0.004 | D6.002 |
| sign2 → sign3 | 0.748 ± 0.006 | **0.781** ± 0.005 | D6.007 |

**This result is genuinely mixed, and it is the most interesting thing in the
comparison after §1.**

D6.007 compresses *better at every step that ends in sign2*: 0.931 vs 0.883
from raw data, 0.932 vs 0.917 from sign1, and it carries more of that
structure into sign3 (0.781 vs 0.748). That is the expected consequence of
§1 — with 3,083 features instead of 9,802 there is simply less information to
lose when squeezing into a 128-dimensional sign2, so the compression is more
faithful.

But **the end-to-end path does not inherit the gain**. sign0 → sign3, the
question that actually matters for a CC space (does the inferred signature
the resource serves still reflect what was measured?), is *worse* for D6.007:
0.682 vs 0.698. Same for sign1 → sign3 (0.669 vs 0.696). Better intermediate
compression of a thinner input does not produce a better final signature.

The most likely reading: sign3 is inferred by a network trained across all
CC spaces, so its output is substantially shaped by the rest of the universe
rather than by this space alone. D6.007 enters that step with a weaker, less
distinctive proteomics signal, and ends up pulled further from its own raw
data despite the cleaner intermediates. This is consistent with §4, the losses
across all 25 CC spaces. (It is *not* supported by §5, which at matched
cutoffs turns out to be neutral.)

Note also that **both spaces lose a great deal between sign0 and sign3**
(0.70 and 0.68 from near-1.0 at sign0 → sign1). That is a property of the
protocol rather than a defect of either space, but it is worth stating
plainly: most of what distinguishes these two datasets at the raw level does
not survive to the signature the CC actually serves.

## 4. D6.002 wins on all 25 CC spaces

`across_roc_comparison.csv` / `across_roc_scatter.png`: **D6.007 has the lower
NN-recapitulation AUROC in 25 of 25 CC spaces**, mean delta −0.0110, range
−0.0043 (E5.001) to −0.0193 (D1.001).

This is the single most robust result here. It is 25 near-independent tests
all pointing the same way, it corroborates the neighborhood metric's direction
in §2, and a uniform shift of this shape is what a loss of feature resolution
looks like — not what a gain of 84 compounds looks like.

Worth noting for a D-level (cells) space: the largest single loss is
**D1.001 (transcriptomics, −0.0193)**, and D5.001 (−0.0146) and D2.001
(−0.0143) are also among the worst. Per Table S2's "ROC across CC" row, decent
recapitulation of same-level spaces is the expected good outcome, so D6.007
degrading most against its own level would be the least favourable place to
lose. **But see §5b**: re-measured with error bars under a
distance-percentile positive definition, the D1.001 gap does not reproduce
(0.605 vs 0.607). These `across_roc` values carry no error bar, so the sign
pattern across all 25 is the robust part of this section and the individual
magnitudes are not.

A methodological note that matters for §5b: `across_roc` loops
`Diagnosis.cross_roc()`, which defines positive pairs as each molecule's
**k=5 nearest neighbours** in the reference space and samples a *balanced*
negative set. That is a different test from the paper's Fig. 3 / Ext. Data
Fig. 8 percentile-cutoff design, and the two AUROCs are on different scales --
0.74 here versus 0.60 there, for the same pair of spaces.

## 5. Cross-space recapitulation: in line with the paper's own benchmark

`shared_key_recapitulation.json`, over the 1,230,708-compound sign3
intersection (50,000-compound random pool, 2,500 molecules × 5 subsamples):

| Direction | AUROC | NN cutoff (cosine) |
|---|---|---|
| D6.002 neighbours recovered by D6.007 | 0.606 ± 0.004 | 0.4113 |
| D6.007 neighbours recovered by D6.002 | 0.605 ± 0.007 | 0.3959 |

**This section previously reported the opposite conclusion, from a
cutoff-mismatched comparison.** An earlier draft set our P=0.01 AUROC (~0.61)
against "0.72–0.74 from Ext. Data Fig. 8g,h" and concluded that our two spaces
agree with each other *less* than transcriptomics agrees with proteomics.
Those 0.72–0.74 values are the paper's **P=0.001** numbers. Read at matched
cutoffs, the conclusion reverses.

Extended Data Fig. 8g,h, read directly off the figure extracted from the
article PDF (`docs/ext_data_fig8_gh.png`):

The caption reads "Recapitulation of kNN **at X** type III signature level
**using Y** type III signatures": *at X* names the **reference** that defines
the neighbours, *using Y* names the **query** scored on recovering them.

| reference ← query | P = 1e-2 | P = 1e-3 |
|---|---|---|
| paper g: D1.001 ← D6.001 | 0.60 ± 0.01 | 0.72 ± 0.03 |
| paper h: D1.002 ← D6.001 | 0.61 ± 0.01 | 0.74 ± 0.04 |
| ours: D6.002 ← D6.007 | 0.61 ± 0.00 | **0.71 ± 0.01** |
| ours: D6.007 ← D6.002 | 0.61 ± 0.01 | **0.73 ± 0.02** |

Note the two designs differ, even though the per-panel quantity is the same.
The paper holds the **query** fixed at D6.001 and varies the reference across
two transcriptomics versions; its g and h are *not* two directions of one
comparison. Ours swaps reference and query on a single pair, so it *is*
bidirectional. The magnitudes remain comparable as a benchmark of what this
test returns.

**Our numbers are statistically indistinguishable from the paper's at both
cutoffs.** So ~0.61 at P=0.01 is not a warning sign; it is
simply what this test returns for two related-but-distinct sign3 spaces, and
the paper reports the same value without remark. The AUROC rising to ~0.72 at
the stricter cutoff is likewise the expected shape, not a rescue.

What this does *not* support is any claim that D6.007's similarity structure
has shifted anomalously. The two spaces agree with each other about as well as
the paper's own pair of related spaces do. This test, on its own, is neutral
between the two datasets — it neither favours D6.007 nor counts against it.

One genuine observation survives: the near-symmetry of the two directions
(0.606 vs 0.605 at P=0.01) says neither space's neighbour structure is the
more reproducible one, with a small asymmetry at P=0.001 (0.71 vs 0.73) in
D6.002's favour as the *recapitulating* space — well inside the paper's own
±0.03–0.04 subsample spread, so not worth reading into.

The cutoffs themselves differ slightly: D6.007's 1st-percentile cosine
distance is tighter (0.396 vs 0.411), i.e. its signatures are somewhat more
compressed in cosine space.

## 5b. Against the transcriptomics space: the pipeline validates, the spaces don't separate

`external_reference_recapitulation.csv`,
`recap_roc_D1.001_by_{DeepCoverMoA,All_Shared_Proteins}.png`. This puts our
two spaces on **exactly** the paper's Ext. Data Fig. 8g axis: reference
D1.001, query = each of our spaces in turn, same two cutoffs, same ROC-band
plot.

| reference ← query | P = 1e-2 | P = 1e-3 |
|---|---|---|
| paper g: D1.001 ← D6.001 | 0.60 ± 0.01 | 0.72 ± 0.03 |
| ours: D1.001 ← D6.002 | 0.605 ± 0.007 | 0.729 ± 0.014 |
| ours: D1.001 ← D6.007 | 0.607 ± 0.007 | 0.727 ± 0.018 |

**Two things follow, and the first is the more valuable.**

1. **The pipeline validates against a published benchmark.** Both of our
   proteomics spaces recover D1.001's neighbourhood structure to within
   ~0.01 of the value the paper reports for its own D6.001, at both cutoffs.
   Nothing about our sign0→sign3 build is anomalous; it lands where the
   protocol says it should. This is the strongest external check in the whole
   comparison, and it is worth more than any of the internal deltas.
2. **The test does not discriminate the two spaces.** 0.605 vs 0.607 at
   P=1e-2 and 0.729 vs 0.727 at P=1e-3 -- differences of 0.002, against
   subsample standard deviations of 0.007-0.018. There is no effect here.

**This sits in mild tension with §4, and the tension is worth stating.**
`across_roc` reports D1.001 as D6.007's *largest* single loss (0.7361 →
0.7168, −0.0193); the banded test above finds nothing. The two are not the
same measurement (§4's note): `Diagnosis.cross_roc()` defines positives as
each molecule's **k=5 nearest neighbours** with a *balanced* negative set,
whereas this test defines them by a **distance percentile** with every other
pair negative. Under k-NN every molecule contributes exactly 5 positives,
including isolated ones; under a percentile cutoff dense-region molecules
contribute many and isolated ones none. That is a real difference in what is
being asked.

Two consequences for how §4 should be read:

- The **25/25 sign pattern stands.** Twenty-five near-independent tests all
  landing negative is strong on a sign test whatever the individual
  magnitudes, and that is what §4's argument rests on.
- **Individual per-space deltas should not be over-read.** `across_roc` is a
  single run with no error bar, and the one delta we have now re-measured
  with error bars -- D1.001, the largest of the 25 -- does not reproduce.
  The claim "D6.007 degrades most against its own level" is therefore weaker
  than it first appeared and should be dropped or heavily hedged.

## 6. The new compounds do not open new territory

`input_compound_tsne_overlay.png` projects both datasets' **sign0** compounds
through a shared sign3 space (sign0, not sign3 — sign3 is inferred for the
whole ~1.2M universe and would mark everything as shared).

The 84 D6.007-only compounds are **scattered throughout the existing
distribution**, interleaved with the 872 shared ones. They add density, not
coverage. There is no previously-empty region of bioactivity space that the
extension fills.

This is the Ext. Data Fig. 2/5/8a test, and it is the most direct check on
"did integrating more data reach new chemistry". The answer is no.

## 7. Structure metrics: mixed, and the one point in D6.007's favour

From `diagnosis_extras.csv` / `structure_metrics_bar.png`, with Table S2's
reading direction (which is *not* uniformly "higher is better"):

| Metric | D6.002 | D6.007 | Better | Table S2 direction |
|---|---|---|---|---|
| Redundant signatures | 2.494% | 2.527% | D6.002 (tie) | lower better |
| DBSCAN clusters | 113 | **147** | **D6.007** | more = finer structure |
| Unclustered / outlying | 14.48% | 16.49% | D6.002 | lower better |
| CC ranks agreement (RBO) | 0.0857 | 0.0878 | D6.002 (tie) | lower = more orthogonal |
| Mean CC spaces per compound | 7.728 | 7.760 | tie | higher better |
| Mean confidence | 0.1645 | **0.2228** | **D6.007** | ranking only, no threshold |

**The genuine case for D6.007** rests on two numbers: it produces 147 DBSCAN
clusters against D6.002's 113 (Table S2: "a large number of clusters with low
proportions of compounds each may indicate a continuous representation of the
chemical space, capturing fine details"), and its mean sign3 confidence is 35%
higher. Higher confidence is expected — more compounds with real data makes
inference better supported — and is a real benefit.

That case is weakened by the fact that D6.007's extra clusters come with more
*unclustered* compounds (16.49% vs 14.48%), which Table S2 identifies as
outliers, and its largest cluster is also bigger (11.62% vs 8.30%). So the
extra granularity is partly fragmentation, not purely finer resolution.

Table S2 is also explicit that a lower MoA/ATC AUROC "is not necessarily a
negative outcome, as it may instead reflect orthogonality". That is the
strongest available argument that D6.007 is complementary rather than worse.
It does not survive contact with §4: a space that had gained orthogonal signal
would lose against *some* CC spaces and gain against others. Losing against
all 25, uniformly, is loss of resolution, not gained orthogonality.

## 8. Three metrics that cannot discriminate these spaces — and why

These were previously cited as comparative evidence. They are not capable of
being evidence, confirmed from the CC source:

- **`pct_outliers` (was reported as 90.0% for both).** `Diagnosis.outliers()`
  fits `IsolationForest(contamination=0.1)` — a hardcoded rate. Exactly
  1,000 of 10,000 keys are flagged in *every* dataset. The old 90.0% figure
  was also **the inlier percentage**: the stored `scores` use scikit-learn's
  convention (negative = outlier) while CC's plotter negates them
  (`scs = -results["scores"]`), so Table S2's "scores above 0 are outliers"
  refers to the *plotted* sign. Only the score distribution carries signal
  (D6.002 −0.0262 ± 0.0195, D6.007 −0.0201 ± 0.0153).
- **`moa_cov` / `atc_cov` (98.15% / 95.98%, bit-identical).** These are the
  `frac` return of `vector_validation` — coverage of the fixed external
  validation file, not of each dataset's own compounds. Both sign3 spaces
  span the whole universe, so they are structurally identical for this
  purpose. (The two key sets are not *literally* identical -- they differ by
  the 7 compounds derived in §1 -- but 7 keys in 1.23 M cannot move a coverage
  figure at 2 d.p.) Now excluded from the summary table.
- **`across_coverage`.** `vs_overlap == 1.0` for all 25 CC spaces in both
  datasets. Per Table S2's "CC wrt Sign" row this is the *expected pass* for
  a type III signature, not a difference.

`key_coverage_comparison.png` is a fourth, softer case: the two distributions
are essentially superimposed (mean 7.73 vs 7.76, 0.5% in <5 spaces for both).
The new compounds connected to the CC universe normally — which is reassuring,
but not discriminating.

---

## Caveats

1. **The whole-universe AUROCs are not a like-for-like difficulty
   comparison.** `moa_auroc_*`, `atc_auroc_*` and `across_roc` are all
   computed over each space's entire ~1.23M-compound *inferred* population.
   They should carry less weight than the sign0 overlay (§6) and the
   within-space signature recovery (§3), which are targeted tests. The
   cross-space recapitulation (§5) is targeted too, but having been corrected
   it now discriminates nothing and should not be cited either way.
2. **Comparisons over the 10,000-key diagnosis artifacts are unpaired.**
   Within a dataset all artifacts describe one identical subsample; *between*
   datasets the two runs drew different subsamples overlapping by only 2,157
   of 10,000. Fine for comparing distributions, but small deltas
   (redundancy, ranks agreement, key coverage) should not be over-read.
3. **No absolute confidence threshold is applied.** A "mean confidence > 0.6
   indicates reliable signatures" bar has been attributed to Table S2, but it
   is **not in Table S2** — row 9 gives no good/bad outcome and states that
   confidence values are "space-dependent" and used to rank compounds *within*
   a space. Both spaces are far below 0.6 (0.16 and 0.22; 0.04% and 0.08% of
   compounds above it), which is worth knowing, but this write-up does not
   claim the paper sets 0.6 as a pass mark. **If that threshold exists
   elsewhere in the paper, it would be worth pinning down** — it would make
   both spaces formally unreliable and change how §7's confidence gain reads.
4. **D6.007's `global_ranks_agreement` contains one NaN**, for the single
   compound present in no other CC space. All reductions are now `nan`-safe;
   a plain `np.mean` returns NaN for the whole space.
5. **The paper PDF's body text could not be machine-read** (CID-encoded
   fonts), so methodology claims here rest on `CLAUDE.md`'s specification,
   Table S2 (read in full from the supplementary workbook), and the CC source.
   Every Table S2 quotation and reading-direction used in this document has
   been re-read verbatim from the workbook and is reproduced in
   `PRESENTATION.md`'s figure guide.

## Recommendation

Keep **D6.002** as the primary proteomics space. D6.007's 84 extra compounds
do not reach new bioactivity space (§6) and do not offset the loss of 69% of
measured proteins, which shows up as a uniform degradation across all 25 CC
spaces (§4).

If the goal of the integration was more compounds, the more promising route is
a **union** of proteins with missing-value handling rather than an
intersection — the current construction is compound-limited by design and
feature-limited as a side effect. If D6.007 is kept, it is best framed as a
higher-confidence, lower-resolution companion to D6.002, not a replacement,
and the two should be evaluated together rather than one chosen.
