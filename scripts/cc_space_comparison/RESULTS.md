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

Runtime ~70 s. All outputs referenced below are in `output/D6.002_vs_D6.007/`.

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

The evidence points one way on every test that measures how well the space
resolves bioactivity structure end-to-end, and the metrics that favour D6.007
-- better intermediate compression, higher confidence -- are explainable by
the same underlying change and do not survive to the signature the CC
actually serves. There is a real argument
that D6.007 is *complementary* rather than strictly worse, but it is weaker
than the evidence against it, and it is not the argument the headline numbers
alone would make.

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
data despite the cleaner intermediates. This is consistent with §4 (losses
across all 25 CC spaces) and §5 (a shifted similarity structure).

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
degrading most against its own level is the least favourable place to lose.

## 5. Cross-space recapitulation: the neighbourhoods really have moved

`shared_key_recapitulation.json`, over the 1,230,708-compound sign3
intersection (50,000-compound random pool, 2,500 molecules × 5 subsamples):

| Direction | AUROC | NN cutoff (cosine) |
|---|---|---|
| D6.002 neighbours recovered by D6.007 | 0.606 ± 0.004 | 0.4113 |
| D6.007 neighbours recovered by D6.002 | 0.605 ± 0.007 | 0.3959 |

At P=0.01 this is **barely above the 0.5 random baseline**, and well below the
0.72–0.74 that Ext. Data Fig. 8g,h reports for D1.001 vs D6.001 — two
genuinely *different* data types. Two versions of the conceptually same
proteomics space agree with each other less than transcriptomics agrees with
proteomics. The similarity structure has substantially shifted.

**But the ROC-with-band plots add an important nuance.** In
`recap_roc_DeepCoverMoA_NN.png`, the stricter cutoff does much better:

- P = 1e-2 → AUROC 0.61 ± 0.00
- P = 1e-3 → AUROC **0.71 ± 0.01**

So the **tightest** neighbourhoods are substantially preserved; it is the
looser, 1%-level neighbourhood structure that has been reorganised. That is
the signature of a space that kept its strongest relationships and lost
resolution in the mid-range — again exactly what dropping 69% of features
would do. It is a meaningfully softer conclusion than the P=0.01 number alone
supports, and it should be reported alongside it.

The near-symmetry of the two directions (0.606 vs 0.605) says neither space's
neighbour structure is the more reproducible one. The cutoffs differ, though:
D6.007's 1st-percentile distance is tighter (0.396 vs 0.411), i.e. its
signatures are more compressed in cosine space.

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
  span the whole universe, so they are structurally identical. Now excluded
  from the summary table.
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
   cross-space recapitulation (§5), which are targeted tests.
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
