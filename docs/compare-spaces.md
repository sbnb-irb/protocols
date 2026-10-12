# Comparing spaces

`compare-spaces` compares Chemical Checker (CC) spaces, or a space and a plain matrix, on the
compounds they have in common. It reads fitted signatures (`sign0` to `sign3`), so it works at
any stage of a fit, and it writes tables that the plotting code reads without opening a signature.

```bash
bash scripts/chemcheck_exec.sh python -m chemcheck_protocols compare-spaces --config configs/compare_spaces/example.yaml
```

For held-out molecules and references, use `evaluate` instead (it scores sign3 on molecules left
out before sign0); `compare-spaces` asks how two spaces relate on the same compounds.

## What a comparison is

A comparison has two **sides**, `a` and `b`. A side is a **signature of a space**: a signature type
of a CC dataset (`{space: landmark-all, signature: sign2}`) or, for a matrix file, just the
space (`{space: raw-matrix}`). One definition covers every case:

| Question | Side a | Side b |
|---|---|---|
| Do two spaces agree? | space 1, sign2 | space 2, sign2 |
| Does a space keep its own structure on the way up? | space, sign0 | same space, sign2 |
| Does a fit match a published one? | ours, sign3 | published, sign3 |
| Does a space still resemble its raw data? | space, sign2 | the input matrix |

Writing every comparison out gets long, so the configuration has shorthands, expanded in this
order into one list without repeats: `pairs` (each pair at each of `signatures`), `against` (one
reference with several spaces), `within_space` (each CC space at `[lower, higher]`) and
`comparisons` (explicit entries). Use one configuration file per question, each with its own
`output`, for example `tx_vs_proteomics.yaml` and `within_space.yaml`.

## Which compounds: `compounds`

| Value | Compounds | Use |
|---|---|---|
| `measured` (default) | the compounds with real data on both sides: those present in `sign0`, `sign1` and `sign2` (the sanitizer can drop input compounds from sign1 and sign2), or the rows of a matrix; every stage pair of a space is then scored on the same compounds | the same compounds at every stage; at sign0 to sign2 this is every shared compound |
| `all_shared` | every compound present in both signatures | at sign3 this is the whole CC universe, mostly predicted signatures: do the spaces organise chemical space alike? |
| `from_file` | the keys of `compounds_file` that both sides have | a chosen set |

At most `max_compounds` (50,000) are used; a larger set is drawn with `seed`. At sign3 the two
populations answer different questions (faithfulness where there is data, against agreement
everywhere), so report both. Compounds whose vector is zero or not finite on either side are
dropped, and a comparison needs at least 50 compounds.

## Metrics

- **Neighbour recapitulation** (`recapitulation_auroc`; Comajuncosa-Creus et al. 2025, Fig. 3).
  Compound pairs closer than the background cutoff on one side are the positives; the other
  side is scored on ranking them by cosine distance. The cutoff is the lower tail
  (`p_value_cutoffs`, default 0.01 and 0.001) of the distances between random pairs. It is
  computed in both directions: `a_neighbours_by_b` (the neighbours of a, recovered by b) and
  `b_neighbours_by_a`; they differ when one space is more reproducible than the other. Values
  are averaged over `n_subsamples` draws of `n_random` compounds (a smaller pool is drawn at 80%,
  so the standard deviation means something).
- **Shuffled null** (`null_value`, `null_std`): the same AUROC after shuffling the compounds of
  the recovering side `n_permutations` times. A value that does not exceed it carries no signal.
  Chance is 0.5, but the null shows what the data actually give.
- **Distance correlation** (`distance_spearman`): Spearman correlation of the pairwise cosine
  distances of 1,000 compounds on the two sides. It needs no cutoff, so it complements the AUROC
  when few pairs qualify as neighbours.

## Outputs

With `output: results/name`:

| File | Content |
|---|---|
| `name_comparisons.csv` | one row per comparison, metric, direction and p-value: `comparison, a_space, a_signature, b_space, b_signature, metric, direction, p_value, value, std, null_value, null_std, mean_positive_rate, n_compounds` |
| `name_curves.csv` | the mean ROC curve and its spread for each recapitulation row (`comparison, direction, p_value, fpr, tpr_mean, tpr_std`, 101 FPR points) |
| `name_run.yaml` | the settings used and the list of comparisons, so a result can be read without the configuration |

## Limitations

- A matrix side has missing values set to 0 ("no change"), which suits log fold changes and
  z-scores but not every matrix.
- The comparison is on cosine distance only.
- A degenerate draw (no pair, or every pair, within the cutoff) is skipped with a warning; a
  comparison with too few compounds stops the run.
