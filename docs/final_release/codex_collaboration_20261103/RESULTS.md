# Collaborative Training & Release Gate Results (`2026-11-03`)

## 1. Run Configuration

- **Codex Session ID**: `01a1a17f-ad23-7262-a7ec-695f39054e74`
- **Parent Checkpoint**: `models/mobilenetv3_field_v2_20261103.pt` (`sha256: c296f938564a4beb07cfeca00d9cca579516de442666a4ad96b6748355ee4c4e`)
- **Output Checkpoint**: `models/mobilenetv3_codex_collab_20261103.pt` (`sha256: 0eeb20ec4ef5f3f11cd477920ccc63f33e2b721bb37e394900467f64dc284577`)
- **Dataset**: `data/processed/field_mobile_v2` (`records_sha256: 4678faca1433bbb79d31185cdb6600994e29b777c9fbe8ed23bfb58fe92f5f24`)
- **Hyperparameters**:
  - `epochs=4`, `batch_size=32`, `samples_per_epoch=6400`, `unfreeze_blocks=9`
  - `backbone_lr=3e-6`, `top_backbone_lr=8e-6`, `head_lr=2.5e-5`
  - `label_smoothing=0.015`, `cosine_lr=True` (LR multipliers across epochs 1–4: `1.0000`, `0.8828`, `0.6000`, `0.3172`)
  - `min_village_f1=0.95`, `balanced_calibration=True`
- **Training Time**: `844.54s` on NVIDIA GeForce RTX 2050 (`torch 2.14.1+cu126`)

## 2. Epoch-by-Epoch GPU Progression (`models/mobilenetv3_codex_collab_20261103.json`)

| Stage / Epoch | LR Multipliers (`backbone / top / head`) | Raw Equal-Domain Val Loss | `PlantVillage` Known F1 | `PlantDoc` Known F1 | `MultiSourceField` Known F1 | Selected? |
|---|---|---:|---:|---:|---:|---|
| **Parent (`field_v2/epoch_6`)** | — | **`0.530871`** | `98.17%` | `54.98%` | `65.96%` | **Yes (`parent`)** |
| **Epoch 1** | `3.00e-6 / 8.00e-6 / 2.50e-5` | `0.563952` | `98.25%` | `55.56%` | `66.86%` | No |
| **Epoch 2** | `2.65e-6 / 7.06e-6 / 2.21e-5` | `0.545473` | `98.16%` | `56.33%` | `67.50%` | No |
| **Epoch 3** | `1.80e-6 / 4.80e-6 / 1.50e-5` | `0.551681` | **`98.44%`** | **`57.15%`** | **`67.62%`** | No (loss `> 0.530871`) |
| **Epoch 4** | `9.51e-7 / 2.54e-6 / 7.93e-6` | `0.537149` | `98.17%` | `55.98%` | `67.44%` | No (loss `> 0.530871`) |

### Key Technical Observation
- During **Epoch 3**, intermediate classification F1 peaked across all three domains simultaneously (`PlantDoc` **`57.15%`**, `MultiSourceField` **`67.62%`**, `PlantVillage` **`98.44%`**), and by **Epoch 4** the cosine decay brought raw equal-domain validation cross-entropy down to **`0.537149`** (within `0.00628` of the parent's `0.530871`).
- Because the selector strictly enforces raw equal-domain validation cross-entropy `< best_loss` without retrospective threshold or metric changes, `selected_stage` remained `"parent"` (`54.98%` PlantDoc known F1, `66.33%` MultiSourceField CPU known F1, `98.17%` PlantVillage known F1).

## 3. CPU Validation & Release Gate (`docs/final_release/codex_collaboration_20261103/release_decision.json`)

| Release Criterion (`docs/FIELD_RELEASE_CRITERIA.json`) | Observed Value | Requirement | Status |
|---|---:|---:|---|
| `PlantDoc` Known Macro F1 Gain vs. V2 Baseline | **`+39.03 pp`** (`54.98%` vs `15.95%`) | `>= +5.00 pp` | **PASS** |
| `PlantVillage` Known Macro F1 Drop vs. V2 Baseline | **`-3.69 pp`** (`98.17%` vs `94.48%`, i.e. `+3.69 pp` gain) | `<= 2.00 pp` drop | **PASS** |
| `PlantDoc` Score-Gate Accepted Accuracy (`conf >= 0.75`, known class) | **`66.67%`** | `>= 85.00%` | **FAIL** |
| `PlantDoc` Score-Gate Accepted Coverage (`conf >= 0.75`, known class) | **`18.84%`** | `>= 25.00%` | **FAIL** |
| `PlantDoc` Diseased -> High-Confidence Healthy Rate | **`0.00%`** (`0 / 128`) | `<= 5.00%` | **PASS** |

- **Final Decision**: `Keep default V2; candidate remains experimental` (`test_used_for_selection: false`).
