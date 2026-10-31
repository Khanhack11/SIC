# Capstone Final Report

The Word report follows the supplied Samsung SIC Final Report template. It preserves its cover, chapter order, embedded fonts, headers, footers, blue section banners and review tables. Narrative was filled using repository evidence available on 31 October 2026. Instructor scores remain unfilled. Team-table notes describe assigned scope and recorded work; they are not signed personal testimonials.

Report: [SIC AI Capstone Project Final Report](<SIC_AI_Capstone Project_Final Report.docx>).

The original seven-page template grows naturally as text is filled. Page size, margins, section count and all package parts except document content and the field-refresh setting remain unchanged. Subsection headings are kept with their following paragraph to avoid orphan headings. The title uses deliberate line breaks within its existing box. Field refresh is requested on open; QA uses native Word PDF export. The original template is backed up locally under `artifacts/final_report_work/template_original.docx`.

## Evidence

| Reported result | Record |
|---|---|
| Data source, class counts, splits and audit limitations | [audit.json](../data_optimization/results/audit.json) |
| Frozen-backbone training and calibration | [training.json](../data_optimization/results/training.json) |
| Baseline test accuracy, F1, per-class metrics and calibration | [evaluation.json](../data_optimization/results/evaluation.json) |
| Confusion matrix | [evaluation.csv](../data_optimization/results/evaluation.csv) |
| Existing in-process benchmark | [benchmark.json](../data_optimization/results/benchmark.json) |
| Historical ONNX HTTP measurements | [api_benchmark_confirmed.json](../data_optimization/results/optimization_20261017/api_benchmark_confirmed.json) |
| Runtime quality comparison | [runtime_validation.json](../data_optimization/results/optimization_20261017/runtime_validation.json) |
| Model candidate decisions | [24 October decision](../data_optimization/results/optimization_20261024/decision.json), [shuffle decision](../data_optimization/results/optimization_20261024_shuffle/decision.json) |
| Fresh test and checksum verification | [verification_20261031.json](verification_20261031.json), [test logs](test_logs/) |

Fresh verification: 18 unit tests discovered, 16 passed and two optional ONNX tests skipped; backend 10/10 and stress/edge checks 17/17 passed. The checkpoint and test-manifest hashes match the saved baseline evaluation. No new classification evaluation or formal benchmark run was performed for this report; the stress suite includes its own short latency check.

The report does not claim completed GPU retraining, a custom CNN comparison, field validation or record-level agricultural review. Raw data and weights remain local under the repository's existing `.gitignore` policy.
