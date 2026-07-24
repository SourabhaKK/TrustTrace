# Data Card — TrustTrace Processed Dataset

Phase 1 data layer for the TrustTrace content-policy classifier. Describes the
processed `train`/`val`/`test` splits consumed by fine-tuning (Phase 3) and serving.

> **Scope & safety note.** TrustTrace is **detection-only**. This card reports
> **aggregate label distributions only — no raw comment text is shown or committed**.
> The processed split files (`data/*.parquet`) contain the original comment text and
> are **git-ignored**; they are regenerated from source by the pipeline, never stored
> in the repo. See `CLAUDE.md` for the hard scope boundary.

## Source

- **Dataset:** Jigsaw Toxic Comment Classification (6 binary labels: `toxic`,
  `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).
- **Access:** loaded via HuggingFace `datasets` from mirror
  `Arsive/toxicity_classification_jigsaw` (train split), normalized to
  `text` + the 6 binary label columns. Fallback: a locally-downloaded Kaggle
  `train.csv` (`trusttrace.data.load.load_jigsaw_csv`).
- **License / usage:** public research dataset; used here for classification/detection
  only. No new or original harmful content was sourced or generated.
- **Source pool size:** 25,960 rows.

### Source-pool raw label counts (multi-label, pre-collapse)

| toxic | severe_toxic | obscene | threat | insult | identity_hate | clean (no label) |
|---|---|---|---|---|---|---|
| 12,233 | 1,298 | 6,804 | 373 | 6,345 | 1,124 | 12,973 |

## Processing

1. **Label → schema mapping** (locked, PRD §11.1). The 6 binary labels collapse to a
   single primary `category` by priority
   `threat > identity_hate > severe_toxic > obscene > insult > toxic` (else `none`);
   `severity` is derived deterministically from the category
   (`none→none`, `toxic→low`, `insult/obscene→medium`, `identity_hate/severe_toxic/threat→high`).
   `confidence` is a **model-reported field produced at inference time**, not a
   ground-truth column — it is intentionally absent from the dataset.
2. **Stratified subsample** — class-balanced, up to **350 rows per category**,
   deterministic (`seed=42`). Chosen over proportional sampling so rare classes
   (`threat`, `identity_hate`) have usable training signal.
3. **Split** — stratified **70 / 15 / 15** on `category` (`seed=42`).

Persisted columns per split: `text`, `category`, `severity`, and the 6 raw labels
(retained for Phase 3 evaluation).

## Distribution

Subsample total: **2,450 rows**, balanced at **350 per category** across all 7 categories.

| split | rows | none | toxic | insult | obscene | identity_hate | severe_toxic | threat |
|---|---|---|---|---|---|---|---|---|
| train | 1,715 | 245 | 245 | 245 | 245 | 245 | 245 | 245 |
| val   | 367   | 52  | 52  | 53  | 52  | 53  | 52  | 53  |
| test  | 368   | 53  | 53  | 52  | 53  | 52  | 53  | 52  |

Every category is present in every split.

## Validation

`trusttrace.data.validate` enforces (and fails loudly / exits non-zero on any violation):
required columns, no null/empty `text`, `category`/`severity` within the locked enums,
`severity`↔`category` consistency, `category`↔raw-label consistency, binary label values,
and full category coverage per split.

## Reproduce

```bash
pip install -e .                          # or: pip install -r requirements.txt
python -m trusttrace.data.build_dataset   # writes data/{train,val,test}.parquet
python -m trusttrace.data.validate        # exits 0 iff all splits are valid
pytest                                    # schema + validation unit tests
```
