# Antigravity ↔ Codex Collaborative Training (`2026-11-03`)

This directory records the end-to-end collaborative training and technical review between **Antigravity** and **OpenAI Codex CLI** (`v0.166.0`, model `gpt-6.1-sol`, session ID `01a1a17f-ad23-7262-a7ec-695f39054e74`).

## Collaboration Workflow

1. **Protocol & Safeguard Review (`artifacts/codex_collaboration_20261103/plan_response.txt`)**:
   - Connected to Codex CLI via `codex exec` in `read-only` sandbox mode.
   - Codex audited `ml/continue_field.py`, `tests/test_field_warmstart.py`, `tests/test_field_recovery.py`, and `models/mobilenetv3_antigravity_warmstart_20261103.json`.
   - Approved a controlled 4-epoch warm-start experiment from `models/mobilenetv3_field_v2_20261103.pt` on `data/processed/field_mobile_v2` (`train` split only) with `--label-smoothing 0.015` and `--cosine-lr`, subject to preserving parameter-group LR ratios, storing scheduler state in recovery checkpoints, and keeping raw equal-domain validation cross-entropy selection (`PlantVillage` known F1 >= `0.95`).
2. **Implementation & Pre-Run Code Review (`artifacts/codex_collaboration_20261103/code_review.txt`)**:
   - Updated `ml/continue_field.py` with `build_cosine_scheduler`, `--label-smoothing`, `--cosine-lr`, and scheduler-aware `save_progress` / `restore_progress`.
   - Added unit tests in `tests/test_field_warmstart.py` and `tests/test_field_recovery.py`.
   - Resumed Codex session `01a1a17f-ad23-7262-a7ec-695f39054e74`; Codex reviewed the `git diff` and issued: **"Verdict: ready for the four-epoch GPU experiment."**
3. **4-Epoch GPU Training & CPU Validation Evaluation**:
   - Executed 4 epochs on NVIDIA GeForce RTX 2050 (`.venv-gpu`), saving `models/mobilenetv3_codex_collab_20261103.pt` and `models/mobilenetv3_codex_collab_20261103.json`.
   - Evaluated on CPU (`.venv`) across `field_mobile_v2` validation (`validation_field_v2_evaluation.json`), `field_mobile_v1` validation (`validation_field_v1_evaluation.json`), and `docs/FIELD_RELEASE_CRITERIA.json` (`release_decision.json`).
4. **Post-Training Results Review (`artifacts/codex_collaboration_20261103/results_review.txt`)**:
   - Resumed Codex session `01a1a17f-ad23-7262-a7ec-695f39054e74` to audit the training report, CPU validation reports, SHA-256 digests, and release gate decision.
   - Codex confirmed all hashes and holdout constraints and recommended: **"Release recommendation: keep default V2; retain this candidate as experimental."**
