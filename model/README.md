# Published model files

- `strict_v2_args.json`: final architecture, optimization and loss weights.
- `preprocess_stats.npz`: ordered RNA/ATAC schemas, training means and standard deviations, and fixed data splits.
- `metrics.json`: train, validation and test metrics for the selected strict-v2 run.
- `best_val_metrics.json`: validation metrics saved with the selected checkpoint.

The published `best_model.pt` is 233,266,355 bytes (222.46 MiB) and is not tracked as a normal GitHub file.

```text
SHA256  A0511044EB6A46AA22119BFC822026B4CD5CB23116548332D2A4A1E98904C23B
```

Before public release, deposit the checkpoint in a versioned archive such as Zenodo or attach it as a large-file release asset. Place the downloaded file at `model/best_model.pt` before running `pseudo_pair.py`.

The strict-v2 checkpoint was obtained by fine-tuning a parent model while freezing the RNA encoder, RNA projection head, RNA decoder and ATAC decoder. Therefore, exact full-model retraining also requires the documented parent checkpoint; the compact Fontan1 example is an executable demonstration rather than a substitute for the full paired training cohort.
