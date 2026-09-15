# analysis/

`ci_and_corrected_pvalues.py` — reads `results/main_study/raw_predictions.csv`
and writes, next to it:

- `ci_tables.csv`: mean, SD and two 95 % confidence intervals per activation ×
  metric × region — a percentile bootstrap over the 73 patients (5 000
  resamples, `numpy.random.default_rng(42)`, activations visited in their order
  of first appearance in the CSV) and the normal approximation
  mean ± 1.96 · SD / √n. The bootstrap interval is the one in Tables 5–8.
- `corrected_pvalues.csv`: the patient-paired Wilcoxon signed-rank p-value of
  every activation against ReLU, with Holm and Benjamini–Hochberg adjustment
  applied within each metric family (Dice, HD95, Sens, Prec) across its
  11 activations × 5 regions = 55 comparisons.

```bash
python analysis/ci_and_corrected_pvalues.py            # defaults to results/main_study/
python analysis/ci_and_corrected_pvalues.py --raw path/to/raw_predictions.csv --outdir out/
```

Running it against the archived `raw_predictions.csv` reproduces the archived
`ci_tables.csv` byte for byte and the archived p-values to machine precision
(all `*`/`**` marks identical). Only numpy, pandas and scipy are needed — no
GPU, no PyTorch.
