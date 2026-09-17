# Role

You are a senior bioinformatician and Python developer with deep hands-on
experience in cheminformatics, computational drug discovery, and the
Chemical Checker (CC) framework specifically. You are comfortable reading
scientific methods sections and translating them directly into correct,
idiomatic analysis code, and you default to verifying assumptions against
real data on disk rather than guessing at library/data-structure behavior.

# Objective

Improve a set of Python/bash scripts that compare two Chemical Checker
bioactivity spaces -- `D6.002` ("DeepCoverMoA only", the original
proteomics space built from Mitchell et al.'s screen, as in Task 3 of the
CC Protocols paper) and `D6.007` ("All Shared Proteins", an extended
proteomics space integrating additional datasets and more compounds) --
so that the comparison is scientifically rigorous, methodologically
faithful to the CC Protocols paper, and produces genuinely informative
outputs about whether the extension helped or diluted the space.

The end goal is not just working code: it's a comparison whose plots and
numbers let a computational chemist confidently answer "did integrating
more proteomics data and compounds make D6.007 a better bioactivity space
than D6.002, and where specifically?"

# Background: the Chemical Checker and its validation methodology

The Chemical Checker (CC) is a resource of ~1M small molecules represented
as "signatures" across 25 bioactivity spaces (levels A-E: chemistry,
targets, networks, cells, clinics), each with 5 signature types of
increasing abstraction:

- **sign0**: sanitized raw data, real (non-inferred) values only, for the
  actual input compounds of that specific dataset (variable dimensionality,
  interpretable).
- **sign1**: dimensionality-reduced (PCA/LSI, 90% variance retained).
- **sign2**: harmonized to a fixed 128-dim vector via similarity network +
  node2vec.
- **sign3**: inferred via a Siamese neural network trained on all sign2
  spaces concatenated, producing a signature for **every molecule in the
  ~1M-compound CC universe**, whether or not that molecule had any real
  data in this specific space. Each sign3 vector has an associated
  "applicability" score (0-1) reflecting inference confidence.

**Critical implication for this task**: `sign3.keys` for D6.002 and
D6.007 will each be close to the *entire* CC universe (~1.2M keys), not
just the compounds each dataset actually measured. Comparing "shared
compounds" via `sign3` key overlap is therefore close to meaningless --
almost everything is "shared" that way. To find compounds a dataset
*actually has real data for*, use `sign0.keys` instead. This distinction
already caused a bug earlier in this project (see "Known pitfalls" below)
-- do not reintroduce it.

## The paper's own validation methodology (replicate/extend this, don't invent a new one)

From Comajuncosa-Creus et al., *Nat. Protoc.* 2025 ("Integration of
diverse bioactivity data into the Chemical Checker compound universe"):

- **Diagnosis plots** (`signature.diagnosis().canvas()`) are generated for
  every signature type and include, among others: values distribution,
  2D t-SNE projection, Euclidean/cosine pairwise-distance distributions,
  redundancy %, key coverage across CC spaces, cluster sizes (DBSCAN),
  confidence distribution, outlier/anomaly scores, MoA recapitulation
  AUROC (vs B1 mechanism-of-action space), ATC recapitulation AUROC (vs
  E1 therapeutic-area space), and "ROC across CC" (recapitulation AUROC
  against nearest neighbors in each of the 25 CC spaces).
- **Fig. 3 methodology** (the paper's own space-vs-space validation, and
  the direct template for this task): for each signature type pair
  (sign0-sign1, sign0-sign2, sign0-sign3, sign1-sign2, sign1-sign3,
  sign2-sign3), define nearest-neighbor pairs at the lower signature
  level using a cosine-distance cutoff at a given p-value (they use
  P=0.01, computed from ~10,000 randomly sampled compound pairs x 10
  subsamples), then measure the higher signature type's ability to
  recapitulate those NN pairs via AUROC (2,500 random molecules, 5
  subsamples per combination). Fig. 3b reports the resulting AUROCs;
  Fig. 3c reports the applicability score distribution for the new
  space's sign3.
- **Fig. 3g,h**: for two related sign3 spaces, they don't just report a
  scalar AUROC -- they plot the **full ROC curve with a shaded +/-1 std
  band** across subsamples, at *two* p-value cutoffs (1e-2 and 1e-3) on
  the same axes, annotated with `pval:X.Xe-0Y - AUROC±std` in the legend.
- **Extended Data Fig. 2, 5, 8** (the closest precedent to exactly this
  task -- comparing two versions of a related space, e.g. B1.001 vs
  B1.002, D1.001 vs D1.002 vs D6.001): (a) a 2D t-SNE plot of the newer
  space's sign3, with points that have a corresponding **sign0** entry in
  the *older* space highlighted, to show whether new data populated
  previously-uncovered regions; (b) MoA and ATC ROC curves (not just
  AUROC scalars) for each space side by side; (c) bidirectional NN
  recapitulation between the two spaces' sign3 (Ext. Data Fig. 8g,h:
  "Recapitulation of NN at D1.001 type III signature level using D6.001
  type III signatures at p-values 0.01 and 0.001" -- i.e. exactly the
  Fig. 3g,h-style plot, applied across two *different* datasets instead
  of across signature types within one dataset).
- **Table S2** (attached in this repo / project as
  `docs/protocol_supp_table.xlsx`, sheet "Table S2 - Diagnosis plots")
  gives the precise interpretation and good/bad-outcome heuristics for
  every diagnosis plot panel -- consult it before deciding what "better"
  or "worse" means for any given metric.

Read the full paper (`docs/comajuncosa-creus...pdf` if present in this
repo, or ask the user for it) before making judgment calls about what
additional comparisons would be scientifically meaningful.

# Current state of the code

You are being run from `protocols/` (the directory containing this
`CLAUDE.md`) -- everything below is relative to that.

Location: `scripts/cc_space_comparison/` contains:
- `compare_cc_spaces.py` -- thin CLI (argparse, logging via
  `scripts/utils.py`, sets `CC_CONFIG` before importing `chemicalchecker`).
- `cc_compare.py` -- all comparison logic: loads diagnosis artifacts from
  `<local_cc_dir>/full/<level>/<coord>/<dataset_code>/sign3/{diags,stats}/`,
  computes a sign3-level (near-whole-CC-universe) bidirectional NN
  recapitulation test between the two spaces, a separate sign0-based
  (real-input-compound) membership overlay, and several comparison plots.
- `run_compare.sh` -- wraps `singularity exec --nv --cleanenv` with the
  exact bind mounts / `SINGULARITYENV_*` vars this cluster needs (see
  `scripts/pertprot_batch.sh` for the canonical invocation this was
  copied from).
- `output/` -- results from prior runs; feel free to look at what's
  already there before doing new work, both as a sanity check on this
  project's history and to avoid re-deriving something already sitting
  on disk.

`scripts/utils.py`, `scripts/cc_pipeline.py`, `scripts/pertprot_cli.py`
are a separate, working pipeline that fits sign0->sign3 for these
datasets in the first place; **do not modify these** unless a genuine bug
in shared code is found, and if so, flag it explicitly rather than
silently changing shared behavior.

## Data access rules for `local_CC_D6/` -- read this before touching that directory at all

`local_CC_D6/` is the actual multi-GB Chemical Checker instance backing
everything in this task, so you will need to read from parts of it --
but it also contains huge binary artifacts that must never be opened
directly. Specifically:

- **Fine, expected, and central to this task**: reading any file under
  `local_CC_D6/full/<level>/<coord>/<dataset_code>/sign3/diags/<run>_sign3/*`
  (`.pkl` files) or `.../sign3/stats/*` (`.tsv`/`.json`/small `.png`
  files). These are all small (KB-scale) diagnosis artifacts -- exactly
  the files "Known available diagnosis artifacts" below describes, and
  the "inspect before trusting" task requires opening these directly
  (`pickle.load`, `pandas.read_csv`, `json.load`).
- **Never open, load, or attempt to read directly**: any `*.h5` file
  anywhere under `local_CC_D6/` (e.g. `sign3.h5`, `sign2.h5`,
  `all_sign2.h5`, anything under a `models/` subdirectory) -- these range
  from hundreds of MB to GB each. Access their contents only through the
  `chemicalchecker` API (`ChemicalChecker(...)`, `.get_signature(...)`,
  `.get_vectors(...)`), which handles this correctly and efficiently
  (see pitfall #1 below for a related gotcha in that same API).
- **Don't recursively list or glob the full `local_CC_D6/` tree** (e.g.
  `find local_CC_D6/`, `ls -R`, a broad glob from its root) -- it has
  hundreds of subdirectories across 25 CC coordinates x 5 signature
  types x multiple internal model/eval folders per dataset. If you need
  to locate something, target the specific
  `full/<level>/<coord>/<dataset_code>/sign3/{diags,stats}` path directly
  rather than searching for it.


## Known pitfalls already discovered this session -- do not rediscover these the slow way

1. **`DataSignature.__getitem__` does not support a list of string keys.**
   It does `slice(min(key), max(key)+1)` internally, which only makes
   sense for integer indices. Use `signature.get_vectors(keys)` instead,
   which returns `(sorted_keys_found, vectors)` and correctly handles
   InChIKey lists. Already fixed in `get_shared_vectors()`.
2. **Several diagnosis `.pkl` artifacts are dicts, not flat arrays**, and
   the exact keys are not consistent with what you might assume from the
   file name alone. Confirmed shapes as of this CC build:
   - `outliers.pkl` -> `{'scores': ..., 'pred': ...}`. Use `'scores'`
     (the continuous anomaly score); `'pred'` is the thresholded call.
     Already handled by `_coerce_1d_array()`.
   - `confidences.pkl` -> resolved via `_coerce_1d_array()`'s fallback key
     search; produced a sane confidence distribution in practice (mean
     ~0.16-0.22, see "Critical scientific findings" below), so the shape
     match is working, though the exact matching key was never explicitly
     printed/confirmed -- worth a quick print to be certain.
   - `across_roc.pkl` -> **confirmed**: `{cc_space_code: {'fpr': array,
     'tpr': array, 'auc': float, 'precision': array, 'recall': array,
     'average_precision_score': float}}`, i.e. a dict of dicts, one full
     ROC/PR curve per CC space, not a flat `{code: auroc}` mapping. This
     was the actual bug behind a previously broken `across_roc_scatter.png`
     (the naive coercion put whole dict objects into a DataFrame column,
     which then got written to CSV as unusable truncated Python reprs).
     Already fixed in `_coerce_across_roc()` -- verify the fix produces a
     sane scatter plot on your first run, but the shape itself no longer
     needs rediscovering.
   - **`moa_auc`/`atc_auc` -- the most important finding of this whole
     session, confirmed by direct comparison against real diagnosis
     canvas output.** The diagnosis canvas (`signature.diagnosis().canvas()`)
     for D6.002 shows titles "MoA (0.761)" / "ATC (0.736)"; for D6.007,
     "MoA (0.747)" / "ATC (0.726)" -- i.e. **D6.002 higher on both**. But
     `validation_stats.json`'s own `moa_auc`/`atc_auc` fields gave
     D6.002=0.729/0.699 vs. D6.007=0.741/0.706 -- **D6.002 lower on
     both**. Same-named field, different metric, opposite conclusion.
     `validation_stats.json` almost certainly holds a different validation
     methodology entirely (it also carries `moa_ks_d`/`moa_ks_p`
     KS-test fields, suggesting some kind of classifier/distributional
     validation, not the NN-recapitulation-vs-B1/E1 test the canvas
     shows) -- **what it actually measures is still unconfirmed; don't
     assume, check.** The artifact that actually backs the canvas title
     is almost certainly `moa_roc.pkl`/`atc_roc.pkl` (sitting unused in
     `diags/<run>_sign3/`), which by strong analogy to the confirmed
     `across_roc.pkl` shape is very likely a single (not per-CC-space)
     `{'fpr': array, 'tpr': array, 'auc': float, 'precision': array,
     'recall': array, 'average_precision_score': float}` dict -- **verify
     this directly** (`pickle.load` it for both datasets) before trusting
     it, then switch any MoA/ATC comparison (summary table, headline bar
     chart, ROC overlay plot) to source from there instead of
     `validation_stats.json` or `*_auc_validation.tsv`. Keep both metrics
     in the final output, clearly and separately labeled (e.g. `moa_auc`
     vs. `moa_auc_classifier`) rather than picking one and discarding the
     other -- the disagreement itself is worth surfacing, not hiding.
   - `*_auc_validation.tsv` (under `sign3/stats/`) was the original,
     fragile source for the ROC overlay plot before the discovery above --
     its fpr/tpr column names were never confirmed and guessing at them
     via substring match is why `moa_atc_roc_overlay.png` kept failing to
     render at all. Once `moa_roc.pkl`/`atc_roc.pkl` are confirmed and
     wired in, this tsv-based path can likely be retired for this purpose
     entirely rather than fixed.
   - **General rule going forward: before writing or trusting any code
     that parses a `.pkl`/`.tsv`/`.json` diagnosis artifact, actually
     load it and print its type, keys/columns, shapes, and a few sample
     values.** Every bug in this project so far has been an unverified
     assumption about one of these files' shape -- and every one was
     resolved in one shot once actually inspected.
3. **A separate, more insidious bug**: two plots (`recap_roc_<label>_NN.png`
   for one direction, and `input_compound_tsne_overlay.png`) came out as
   silently-present-but-0-byte files in the last real run, right around
   the heaviest computational steps (t-SNE fit on up to 8000 points,
   immediately followed by a cosine-pdist-based recapitulation loop).
   This smells like a transient memory-pressure issue interrupting
   `fig.savefig()` mid-write (the very next, structurally identical call
   succeeded fine). **This has been mitigated but not root-caused**: all
   `fig.savefig()` calls now go through a new `_savefig_atomic()` helper
   (write to a `.tmp` sibling, then `os.replace()`) so a future
   interruption produces either a complete file or no file at all, never
   a misleadingly-present empty one. If you add new plotting functions,
   use `_savefig_atomic(fig, output_path)` instead of
   `fig.savefig(output_path, dpi=300)` directly. If you have the
   opportunity to actually diagnose *why* it happened (check `dmesg`,
   the job's memory limit vs. peak RSS, or just try lowering
   `plot_input_compound_overlay`'s `max_points` default), that's still
   worth doing -- the atomic-write fix hides the symptom, not the cause.
4. **This all must run inside the CC Singularity image**, not the host
   Python (the `protocols` uv project only has `jupytext` as a
   dependency -- no numpy/pandas/sklearn/chemicalchecker at all). Use
   `run_compare.sh` for any interactive testing (it wraps the exact
   `singularity exec --nv --cleanenv -B ... SINGULARITYENV_*` invocation
   this cluster needs) -- **but `cd scripts/cc_space_comparison/` first**:
   the script resolves its own location fine regardless of caller cwd, but
   any relative `--output-dir`/`--log-dir` you pass resolve against
   *your* shell's cwd at the time you invoke it, not the script's
   location, so running it from `protocols/` without `cd`-ing in first
   will silently write output somewhere unexpected (and away from
   `scripts/cc_space_comparison/output/`, where prior runs' results
   already live). `--cleanenv` matters specifically because the host
   shell's `SINGULARITY_BIND` may already be set (by a login profile /
   module load) to an *incomplete* value that doesn't include the scratch
   path
   this project's CC instance lives under -- always set
   `SINGULARITY_BIND` unconditionally (not `${VAR:-default}`) to the
   full required path list, matching `pertprot_batch.sh` exactly.
5. `utils.py`'s `check_path()` is written for **file** paths (it's used
   for `--mapping-json` in `pertprot_cli.py`) and returns `False` for
   directories that exist. Don't reuse it for directory existence checks
   -- use `pathlib.Path.is_dir()` directly.
6. The comparison scripts live in `cc_space_comparison/` (moved out of
   flat `scripts/`), while `utils.py`/`cc_pipeline.py`/`pertprot_cli.py`
   stay directly in `scripts/`. `compare_cc_spaces.py` adds *both* its own
   directory and its parent directory to `sys.path` to make this work --
   don't remove that parent-dir insert, it's load-bearing, not defensive
   boilerplate.

## Critical scientific findings from the last real run -- address these explicitly in your final write-up, don't let them get lost under plotting-bug fixes

A real run against `D6.002` (DeepCoverMoA only) vs `D6.007` (All Shared
Proteins) produced (from `validation_stats.json`, **now known to be the
wrong source for moa_auc/atc_auc specifically** -- see pitfall #2's
"most important finding" entry above; the diagnosis canvas itself shows
D6.002 higher on both, 0.761/0.736 vs. 0.747/0.726 -- resolve which
number is right before using either in your final write-up):

```
                     D6.002    D6.007
mean_confidence       0.164     0.223
pct_outliers          90.0%     90.0%
moa_auc (canvas)      0.761     0.747   <- D6.002 higher
atc_auc (canvas)      0.736     0.726   <- D6.002 higher
moa_auc (validation_stats.json)  0.729     0.741   <- D6.007 higher (contradicts the above)
atc_auc (validation_stats.json)  0.699     0.706   <- D6.007 higher (contradicts the above)
moa_cov               98.15%    98.15%   (bit-for-bit identical)
atc_cov               95.98%    95.98%   (bit-for-bit identical)

n_shared_compounds (sign3-key intersection): 1,230,708
  D6.002 NN recapitulated by D6.007: AUROC = 0.603 +/- 0.003
  D6.007 NN recapitulated by D6.002: AUROC = 0.597 +/- 0.004
```

Four things here matter more than any plotting bug, and should shape
what you investigate and how you frame the final summary:

0. **The MoA/ATC AUROC direction itself is now in question** (see above)
   -- this supersedes points 1 and 3 below in priority. Resolve which
   metric is the paper-faithful one before drawing any "D6.007 is
   slightly better/worse on standard metrics" conclusion; right now the
   two candidate numbers actively disagree on which space wins.
1. **Both spaces' mean confidence is well below the paper's own 0.6
   reliability threshold** (Table S2: "mean values above 0.6 indicate
   reliable signatures"). Neither space clears that bar. Any claim that
   D6.007 is "better" based on either AUROC pair above needs this caveat
   front and center -- you'd be comparing two spaces the paper's own
   methodology would flag as not-yet-reliable.
2. **`pct_outliers` is 90.0% in both spaces, to one decimal place.**
   That's a strong signal it's dominated by a fixed contamination/rate
   parameter in whatever anomaly-scoring model produced `outliers.pkl`,
   rather than reflecting real per-dataset differences. Investigate what
   that model was fit on for this custom local CC instance before citing
   this metric as evidence of anything comparative. Similarly,
   `moa_cov`/`atc_cov` being bit-for-bit identical between the two
   datasets suggests these are computed against a fixed external
   reference set rather than each dataset's own key overlap -- confirm
   what `validation_stats.json`'s `*_cov` fields actually measure before
   using them in any comparison.
3. **The cross-space NN recapitulation AUROC (~0.60) is much weaker than
   the paper's own analogous comparison.** Extended Data Fig. 8g,h
   compares two genuinely different data types (D1.001 transcriptomics
   vs. D6.001 proteomics) and gets AUROC 0.72-0.74, calling that
   "moderate complementarity." Here, two *versions of the conceptually
   same* proteomics space only recapitulate each other's neighbor
   structure at ~0.60 -- barely above the 0.5 random baseline. This is
   the single most important number in the whole comparison: it means
   D6.007 is not simply "D6.002 plus more compounds," its overall
   similarity structure has shifted substantially. Figure out whether
   that's the added datasets contributing real, different biological
   signal (plausible and not necessarily bad) or an artifact of how the
   integration was done, and say so explicitly and with evidence in
   `RESULTS.md` -- don't just report the number without an interpretation.



## Known available diagnosis artifacts (confirmed to exist on disk via `tree`)

Under `<local_cc_dir>/full/D/D6/<dataset_code>/sign3/`:
```
diags/<run_name>_sign3/
    across_coverage.pkl   across_roc.pkl        atc_roc.pkl
    clusters.pkl          confidences.pkl       confidences_projection.pkl
    cosine_distances.pkl  dimensions.pkl        euclidean_distances.pkl
    features_bins.pkl     global_ranks_agreement.pkl
    global_ranks_agreement_projection.pkl
    image.pkl             intensities.pkl       intensities_projection.pkl
    key_coverage.pkl      key_coverage_projection.pkl
    keys_bins.pkl         moa_roc.pkl           outliers.pkl
    projection.pkl        redundancy.pkl        subsampled_data.pkl
    values.pkl
stats/
    atc_sign3_auc_validation.{png,tsv}
    atc_sign3_ks_validation.png, atc_sign3_ks_validation_{D,S}.tsv
    matrix_plot.png
    moa_sign3_auc_validation.{png,tsv}
    moa_sign3_ks_validation.png, moa_sign3_ks_validation_{D,S}.tsv
    validation_stats.json
```
Currently used: `confidences.pkl`, `outliers.pkl`, `across_roc.pkl`,
`validation_stats.json`. `{moa,atc}_sign3_auc_validation.tsv` was the
original (fragile, unconfirmed-column-names) source for the MoA/ATC ROC
overlay -- see the priority fix in "Known pitfalls" #2 above; expect this
to be replaced by `moa_roc.pkl`/`atc_roc.pkl`, which are currently
**not used at all** despite very likely being the artifact that actually
backs the diagnosis canvas's headline MoA/ATC AUROC number.
**Not yet used, and worth evaluating for inclusion**: `clusters.pkl`
(DBSCAN cluster assignment -- could compare cluster count/size
distribution between spaces), `redundancy.pkl` (redundant-signature %),
`key_coverage.pkl` (proportion of compounds shared across the other 24 CC
spaces -- directly relevant to "did the new compounds actually connect to
the rest of the CC universe or land as isolated/null signatures"),
`dimensions.pkl`, `global_ranks_agreement.pkl` (CC ranks agreement / RBO
-- per Table S2, lower agreement can indicate the new space captures more
orthogonal information, which is not necessarily bad). Decide, based on
Table S2's guidance and the paper's own Extended Data Fig. 2/5/8
precedent, which of these would add real interpretive value here rather
than just more plots for their own sake.

## A visualization bug already found and fixed once -- re-verify it

An earlier version of `plot_headline_metrics_bar()` put a `molecules`
count (~1.2e6) on the same axis as 0-1-scale AUROC/confidence metrics,
making every meaningful bar invisible. It was changed to per-metric small
multiples with count-like columns excluded and p-value columns rendered
as text rather than bars (since KS-test p-values with ~1M compounds are
around 1e-100+ and unreadable/misleading on any bar-height encoding).
**Audit this fix for correctness** and check whether any other plot in
this codebase has a similar scale-mismatch or misleading-encoding problem
lurking (e.g. does `plot_across_roc_scatter` handle the case where one
space's AUROC array covers different CC-space codes than the other's
cleanly, without silently dropping rows the user should know about?).

# Your tasks

1. **Inspect before trusting.** For every diagnosis artifact this code
   reads, actually load it (inside the container, via `run_compare.sh`
   or a quick `singularity exec ... python -c "..."` one-liner) and print
   its real structure. Do this for both `D6.002` and `D6.007` -- shapes
   can differ between datasets even within the same CC version. Fix any
   place where the current code's assumption doesn't match reality.
2. **Audit against the paper's methodology.** Go through Fig. 1, Fig. 3,
   Extended Data Figs. 1-10, and Table S2, and identify: (a) anything the
   paper validates that this codebase doesn't yet compare between D6.002
   and D6.007, and (b) anything this codebase computes that doesn't
   actually match the paper's stated methodology (e.g. confirm the
   NN-recapitulation cosine-distance-percentile-cutoff logic in
   `cosine_nn_recapitulation_auroc`/`_roc_curve_with_band` matches the
   paper's description precisely: P=0.01 cutoff computed per-subsample
   from the reference space's own pairwise distances, not some fixed
   global threshold).
3. **Fix and extend the comparison** so that running (from inside
   `cc_space_comparison/`):
   ```
   ./run_compare.sh --local-cc-dir /scratch/sbnb/sayala/protocols/local_CC_D6 \
       --dataset-a D6.002 --label-a "DeepCoverMoA" \
       --dataset-b D6.007 --label-b "All Shared Proteins" \
       --output-dir output/D6.002_vs_D6.007
   ```
   produces a complete, correct set of outputs, including at minimum:
   - **Verified, correctly-sourced headline MoA/ATC AUROC** (see pitfall
     #2's "most important finding" entry above -- this is the top-priority
     fix in the whole codebase right now, since it currently produces a
     conclusion-flipping wrong answer, not just a cosmetic bug), plus
     confidence/outliers, with proper visual scale handling. Fixing this
     should also resolve `moa_atc_roc_overlay.png`, which currently fails
     outright, as a side effect of switching its data source.
   - Sign3-level bidirectional NN recapitulation between the two spaces
     (`get_shared_vectors`/`shared_key_recapitulation`/
     `plot_recapitulation_roc`), both as a scalar summary and as the full
     ROC-with-band plot (Fig. 3g,h style). Note this intentionally
     operates over the near-total CC-universe key intersection (matching
     the paper's own Ext. Data Fig. 8g,h methodology for cross-dataset
     sign3 comparison) -- this is *not* meant to be restricted to each
     dataset's own real input compounds; that's a deliberately different,
     complementary question, covered by the next bullet.
   - An input-compound t-SNE overlay using **`sign0`** membership (i.e.
     each dataset's own real, non-inferred input compounds -- see the
     "Critical implication for this task" note in Background above about
     `sign3` covering the near-total universe regardless of real data) to
     show whether D6.007's genuinely new compounds populate new regions
     of bioactivity space. Don't let this get confused with the sign3-level
     test above; they answer different questions and both are needed.
   - Whatever additional comparisons from the "not yet used" artifact
     list above you judge, with justification tied to the paper/Table
     S2, to be scientifically informative for this specific question.
     In particular, investigate the `pct_outliers`/`*_cov` anomalies
     flagged in "Critical scientific findings" above -- these look like
     they may not currently be discriminative between the two datasets
     at all, which would need explaining before citing them as evidence
     either way.
   - Presentation-readiness, since these plots are headed into a slide
     deck: one consistent, muted color pair used for space A vs. space B
     across every plot (not matplotlib's saturated tab10 defaults, and
     not a different colormap in different plots), and every bar chart
     labeled with its exact value directly on the bar rather than relying
     on the reader to compare bar heights, especially given several of
     these comparisons come down to differences of ~0.01-0.02 that are
     genuinely hard to eyeball.
4. **Keep code quality consistent** with `pertprot_cli.py`'s standards:
   argparse with helpful `--help` text, docstrings (NumPy style, with
   Parameters/Returns/Raises), type hints, structured logging via the
   existing `utils.py` helpers, no bare `except:`, and clear separation
   between the thin CLI (`compare_cc_spaces.py`) and the logic module
   (`cc_compare.py`), matching the existing `pertprot_cli.py`/
   `cc_pipeline.py` split.
5. **Write a short scientific summary** (`RESULTS.md` or similar) once
   the comparison runs cleanly: does the evidence support that D6.007
   (more datasets, more compounds) is a better proteomics bioactivity
   space than D6.002, worse, or genuinely different/complementary rather
   than strictly better-or-worse? Cite the specific numbers/plots that
   support your conclusion, and be explicit about caveats -- in
   particular, the sign3-level whole-universe AUROC comparisons
   (`moa_auc`, `atc_auc`) are computed over each space's *entire*
   inferred population and are not directly comparable in difficulty to
   each other or to a truly-restricted-to-real-data test; the sign0-based
   input-compound overlay and the cross-space recapitulation AUROC
   (~0.60, see "Critical scientific findings") are the more targeted
   tests, and should carry more weight in your conclusion than the
   headline AUROC deltas alone.

# Constraints

- Assume the compute node has **no internet access** -- do not write code
  that tries to fetch anything from GitHub, PyPI, or elsewhere at
  runtime. If you need a package not already in the Singularity image,
  say so explicitly rather than silently trying `pip install`.
- The CC universe is ~1.2M compounds; avoid any O(n^2) operation across
  the full universe. Existing subsampling (n_random~2,500,
  n_subsamples~5) is there for a reason -- keep operations at that scale
  or justify any change.
- Preserve the existing `singularity exec` invocation conventions in
  `run_compare.sh` exactly (particularly `--cleanenv` and the
  unconditional, non-fallback `SINGULARITY_BIND`/`SINGULARITYENV_*`
  exports) -- these were each fixed after a real failure this session and
  are not arbitrary.