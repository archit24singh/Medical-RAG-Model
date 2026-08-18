# Evaluation labels

Drop ground-truth label files here to benchmark the predictive pipeline.

Format — CSV:

```
patient_key,label
620853,short
712044,long
```

or JSON:

```json
[{"patient_key": "620853", "label": "short"}]
```

Run:

```
python -m eval.evaluate --task long_length_of_stay --labels data/eval/los_labels.csv
```

`patient_key` must match the `patient_key` stored on events (from
`concept_map.yaml` → `patient_key_columns`). `label` must be one of the task's
`labels` in `prediction_tasks.yaml`.

> The Campbell demo data has **no outcome labels**, so this harness becomes
> meaningful only once labeled longitudinal data (EHRSHOT / MIMIC) is ingested.
> `example_los_labels.csv` is illustrative only.
