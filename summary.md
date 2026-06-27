# MMR-Bench Summary

Generated: 2026-06-27 14:51:57 UTC

Source CSV: `data/MMR-Bench.csv`

Scores are percentages computed from each model's `*_correct` column. `Overall` is row-weighted over all samples; `Macro Avg` is the unweighted average over the seven benchmark scores. `OCRBench` is shown as normalized accuracy percentage, equivalent to final score / 1000 * 100.

## Benchmark Sizes

| Benchmark | CSV prefix | Samples |
| --- | --- | --- |
| MMStar | MMStar | 1500 |
| RealWorldQA | RealWorldQA | 765 |
| SEEDBench2_Plus | SEEDBench2_Plus | 2277 |
| OCRBench | OCRBench | 1000 |
| MathVista_MINI | MathVista | 1000 |
| MathVision | MathVision | 3040 |
| MathVerse_MINI_Vision_Only | MathVerse | 788 |

## Planned MMR-Bench v2 Benchmark Set

This planned v2 set is not yet reflected in the leaderboard below. Current CSV coverage is the original seven-benchmark set above; newly selected benchmarks still need inference, evaluation, and merge support.

| Category | Benchmark | VLMEvalKit name | Samples | Judge / evaluation | Current coverage |
| --- | --- | --- | --- | --- | --- |
| General VQA | MMStar | MMStar | 1500 | MCQ option matching | In current CSV |
| General VQA | RealWorldQA | RealWorldQA | 765 | MCQ option matching | In current CSV |
| General VQA | SEEDBench2_Plus | SEEDBench2_Plus | 2277 | MCQ option matching | In current CSV |
| General VQA | SimpleVQA | SimpleVQA | 2025 | LLM-as-judge | Not run / not merged |
| OCR / Text-rich | OCRBench | OCRBench | 1000 | Substring rule matching | In current CSV |
| Document / Infographic | InfoVQA | InfoVQA_VAL | 2801 | ANLS rule matching | Not run / not merged |
| Chart / Scientific Diagram | ChartQAPro | ChartQAPro | 1948 | Rule-based chart evaluation | Not run / not merged |
| Chart / Scientific Diagram | CharXiv reasoning | CharXiv_reasoning_val | 1000 | LLM-as-judge | Not run / not merged |
| Math / Symbolic | MathVista | MathVista_MINI | 1000 | LLM-assisted extraction + matching | In current CSV |
| Math / Symbolic | MathVision | MathVision | 3040 | LLM-assisted extraction + matching | In current CSV |
| Math / Symbolic | MathVerse | MathVerse_MINI_Vision_Only | 788 | LLM-as-judge extraction + scoring | In current CSV |
| Math / Symbolic | LogicVista | LogicVista | 447 | Exact matching; optional LLM assistance | Not run / not merged |
| Math / Symbolic | WeMath | WeMath | 1740 | MCQ option matching; optional LLM assistance | Not run / not merged |
| Knowledge / Exam | MMMU-Pro | MMMU_Pro_10c | 1730 | MCQ option matching | Not run / not merged |
| Spatial / Perception | BLINK | BLINK | 1901 | MCQ option matching | Not run / not merged |
| Spatial / Perception | ERQA | ERQA | 400 | Exact matching | Not run / not merged |
| Spatial / Perception | VStar | VStarBench | 191 | MCQ option matching | Not run / not merged |
| Robustness / Hallucination | HallusionBench | HallusionBench | 951 | Yes/No extraction matching | Not run / not merged |

| Category | Benchmarks | Samples |
| --- | --- | --- |
| General VQA | 4 | 6567 |
| OCR / Text-rich | 1 | 1000 |
| Document / Infographic | 1 | 2801 |
| Chart / Scientific Diagram | 2 | 2948 |
| Math / Symbolic | 5 | 7015 |
| Knowledge / Exam | 1 | 1730 |
| Spatial / Perception | 3 | 2492 |
| Robustness / Hallucination | 1 | 951 |
| Total | 18 | 25504 |

Current v2 coverage:

| Status | Benchmarks | Samples |
| --- | --- | --- |
| Already in current CSV | 7 | 10370 |
| Not run / not merged | 11 | 15134 |
| Total planned v2 | 18 | 25504 |

## Leaderboard

| Rank | Family | Model | Source | Overall | Macro Avg | MMStar | RealWorldQA | SEEDBench2+ | OCRBench | MathVista | MathVision | MathVerse | Avg Tok | Max Tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Claude | claude-opus-4-7 | local_full_run | 78.7 | 79.3 | 77.1 | 74.8 | 76.0 | 87.7 | 84.4 | 78.2 | 76.8 | 224.5 | 3902 |
| 2 | GPT | gpt-5-2025-08-07 | csv_existing | 74.7 | 76.5 | 76.7 | 82.5 | 75.2 | 81.0 | 78.6 | 68.4 | 73.2 | 813.3 | 4072 |
| 3 | Ovis2.6 | Ovis2.6-30B-A3B | local_full_run | 73.1 | 76.4 | 75.5 | 77.0 | 74.2 | 89.8 | 83.9 | 61.2 | 73.2 | 142.5 | 4096 |
| 4 | Gemini | GeminiPro2-5 | csv_existing | 71.9 | 75.1 | 77.0 | 78.7 | 74.8 | 87.1 | 80.5 | 58.4 | 69.0 | 902.4 | 2046 |
| 5 | Qwen3-VL | Qwen3-VL-32B-Instruct | local_full_run | 72.6 | 74.8 | 78.0 | 79.7 | 75.5 | 89.6 | 83.4 | 61.0 | 56.5 | 872.7 | 4102 |
| 6 | GLM | GLM-4.1V-9B-Thinking | local_full_run | 69.3 | 72.6 | 72.2 | 72.4 | 73.4 | 86.0 | 80.7 | 55.2 | 68.3 | 152.0 | 8195 |
| 7 | Ovis2.5 | Ovis2.5-9B | local_full_run | 68.0 | 72.2 | 72.3 | 74.8 | 72.7 | 87.9 | 82.2 | 50.0 | 65.9 | 310.4 | 4098 |
| 8 | Gemini | GeminiFlash2-5 | csv_existing | 67.3 | 71.8 | 75.3 | 78.6 | 74.1 | 86.7 | 76.0 | 46.4 | 65.7 | 648.9 | 2048 |
| 9 | Qwen3-VL | Qwen3-VL-30B-A3B-Instruct | local_full_run | 69.0 | 71.6 | 72.9 | 73.7 | 71.8 | 91.2 | 79.3 | 56.4 | 55.7 | 1119.1 | 4096 |
| 10 | Gemma | Gemma4-12B-it | local_full_run | 69.2 | 70.5 | 71.9 | 69.0 | 71.0 | 74.3 | 75.9 | 62.7 | 68.9 | 602.1 | 4656 |
| 11 | Qwen3-VL | Qwen3-VL-8B-Instruct | local_full_run | 65.8 | 67.5 | 70.1 | 72.2 | 73.3 | 91.0 | 77.0 | 51.9 | 37.3 | 998.7 | 4098 |
| 12 | Ovis2.5 | Ovis2.5-2B | local_full_run | 61.5 | 67.1 | 67.5 | 68.5 | 71.5 | 88.0 | 78.9 | 35.0 | 60.5 | 312.0 | 4102 |
| 13 | Qwen2.5-VL | Qwen2.5-VL-72B-Instruct | csv_existing | 62.8 | 67.0 | 70.9 | 75.3 | 73.8 | 88.2 | 74.2 | 39.3 | 47.3 | 192.5 | 2048 |
| 14 | InternVL3.5 | InternVL3_5-30B-A3B | local_full_run | 62.4 | 66.9 | 71.1 | 71.1 | 70.4 | 88.4 | 77.6 | 39.6 | 49.9 | 251.3 | 4104 |
| 15 | Ovis2 | Ovis2-34B | local_full_run | 60.9 | 66.5 | 69.8 | 75.2 | 71.8 | 89.1 | 76.7 | 33.3 | 49.4 | 12.6 | 4096 |
| 16 | Qwen2.5-VL | Qwen2.5-VL-32B-Instruct | local_full_run | 62.1 | 66.3 | 69.3 | 71.2 | 72.9 | 85.8 | 73.2 | 39.4 | 51.9 | 362.3 | 4096 |
| 17 | InternVL3 | InternVL3-78B | csv_existing | 60.7 | 65.5 | 73.1 | 78.4 | 71.8 | 91.0 | 72.6 | 34.1 | 37.3 | 113.2 | 1025 |
| 18 | Qwen3-VL | Qwen3-VL-4B-Instruct | local_full_run | 63.2 | 64.4 | 68.5 | 70.6 | 72.0 | 87.8 | 72.1 | 49.8 | 30.3 | 1107.3 | 4106 |
| 19 | Ovis2 | Ovis2-16B | local_full_run | 59.0 | 64.4 | 67.0 | 74.1 | 71.7 | 88.0 | 74.1 | 30.8 | 44.9 | 13.0 | 4096 |
| 20 | InternVL3.5 | InternVL3_5-GPT-OSS-20B-A4B-Preview | local_full_run | 58.6 | 62.7 | 67.6 | 69.7 | 69.0 | 86.3 | 71.6 | 34.9 | 40.1 | 302.8 | 5566 |
| 21 | Other | MiniCPM-V-4.6 | local_full_run | 57.2 | 62.6 | 67.3 | 65.4 | 65.0 | 82.5 | 74.3 | 31.6 | 52.4 | 238.6 | 4097 |
| 22 | InternVL3.5 | InternVL3_5-14B-Instruct | local_full_run | 58.6 | 62.5 | 65.1 | 68.5 | 67.7 | 83.7 | 71.5 | 37.4 | 43.8 | 320.0 | 4096 |
| 23 | Ovis2 | Ovis2-8B | local_full_run | 56.3 | 62.2 | 64.1 | 72.7 | 70.1 | 89.4 | 71.2 | 25.9 | 42.3 | 17.5 | 4096 |
| 24 | InternVL3 | InternVL3-14B-Instruct | local_full_run | 56.9 | 61.6 | 65.3 | 68.9 | 69.8 | 87.0 | 70.0 | 30.5 | 39.5 | 304.7 | 4116 |
| 25 | Qwen2.5-VL | Qwen2.5-VL-7B-Instruct | local_full_run | 56.0 | 61.4 | 64.6 | 69.3 | 70.8 | 88.4 | 68.1 | 26.3 | 42.1 | 236.3 | 4096 |
| 26 | InternVL3.5 | InternVL3_5-8B-Instruct | local_full_run | 57.2 | 61.3 | 63.8 | 64.7 | 68.6 | 83.3 | 72.2 | 33.8 | 42.9 | 299.9 | 4096 |
| 27 | Ovis2 | Ovis2-4B | local_full_run | 54.0 | 60.3 | 61.9 | 71.9 | 69.2 | 91.3 | 69.2 | 21.4 | 37.4 | 20.9 | 4096 |
| 28 | InternVL3.5 | InternVL3_5-4B-Instruct | local_full_run | 55.7 | 59.7 | 65.2 | 65.6 | 67.5 | 81.6 | 66.1 | 31.7 | 40.0 | 301.7 | 4096 |
| 29 | Claude | Claude3-7V_Sonnet | csv_existing | 56.8 | 58.6 | 62.6 | 55.4 | 67.2 | 70.1 | 66.8 | 41.3 | 46.7 | 202.4 | 1776 |
| 30 | Qwen3-VL | Qwen3-VL-2B-Instruct | local_full_run | 54.6 | 57.4 | 57.5 | 64.8 | 67.6 | 86.3 | 59.4 | 34.8 | 31.1 | 1109.3 | 4096 |
| 31 | Qwen2.5-VL | Qwen2.5-VL-3B-Instruct | local_full_run | 51.9 | 56.4 | 57.3 | 66.0 | 69.2 | 82.3 | 63.4 | 24.2 | 32.4 | 237.4 | 4096 |
| 32 | Ovis2 | Ovis2-2B | local_full_run | 50.4 | 55.9 | 57.9 | 66.5 | 67.3 | 87.4 | 64.3 | 18.6 | 29.4 | 24.4 | 4097 |
| 33 | InternVL3.5 | InternVL3_5-2B-Instruct | local_full_run | 50.9 | 54.9 | 55.2 | 60.8 | 64.7 | 82.7 | 60.8 | 26.9 | 33.2 | 343.2 | 4151 |
| 34 | InternVL3 | InternVL3-8B-Instruct | local_full_run | 49.6 | 54.5 | 59.8 | 65.4 | 64.5 | 84.8 | 56.4 | 20.7 | 29.9 | 429.9 | 4098 |
| 35 | Ovis2 | Ovis2-1B | local_full_run | 46.7 | 52.3 | 51.9 | 63.7 | 61.6 | 89.1 | 59.6 | 16.4 | 23.9 | 18.7 | 4096 |
| 36 | InternVL3 | InternVL3-2B-Instruct | local_full_run | 46.5 | 51.0 | 56.8 | 64.3 | 63.1 | 81.6 | 52.4 | 17.6 | 21.1 | 293.5 | 4098 |
| 37 | Gemma | Gemma3-4B | csv_existing | 43.7 | 46.0 | 47.1 | 55.7 | 60.8 | 66.0 | 46.2 | 23.4 | 22.7 | 358.9 | 2048 |
| 38 | InternVL3 | InternVL3-1B-Instruct | local_full_run | 12.2 | 11.2 | 2.0 | 0.5 | 1.7 | 15.0 | 36.9 | 22.0 | 0.0 | 1966.8 | 2866 |

## Model Families

Families are inferred from model names. `Local Runs` counts models with a local seven-benchmark VLMEvalKit `status.json`.

| Family | Models | Local Runs | Best Model | Best Overall | Best Macro Avg | All Models |
| --- | --- | --- | --- | --- | --- | --- |
| GPT | 1 | 0 | gpt-5-2025-08-07 | 74.7 | 76.5 | gpt-5-2025-08-07 |
| Gemini | 2 | 0 | GeminiPro2-5 | 71.9 | 75.1 | GeminiPro2-5<br>GeminiFlash2-5 |
| Claude | 2 | 1 | claude-opus-4-7 | 78.7 | 79.3 | claude-opus-4-7<br>Claude3-7V_Sonnet |
| Qwen3-VL | 5 | 5 | Qwen3-VL-32B-Instruct | 72.6 | 74.8 | Qwen3-VL-32B-Instruct<br>Qwen3-VL-30B-A3B-Instruct<br>Qwen3-VL-8B-Instruct<br>Qwen3-VL-4B-Instruct<br>Qwen3-VL-2B-Instruct |
| Qwen2.5-VL | 4 | 3 | Qwen2.5-VL-72B-Instruct | 62.8 | 67.0 | Qwen2.5-VL-72B-Instruct<br>Qwen2.5-VL-32B-Instruct<br>Qwen2.5-VL-7B-Instruct<br>Qwen2.5-VL-3B-Instruct |
| InternVL3.5 | 6 | 6 | InternVL3_5-30B-A3B | 62.4 | 66.9 | InternVL3_5-30B-A3B<br>InternVL3_5-GPT-OSS-20B-A4B-Preview<br>InternVL3_5-14B-Instruct<br>InternVL3_5-8B-Instruct<br>InternVL3_5-4B-Instruct<br>InternVL3_5-2B-Instruct |
| InternVL3 | 5 | 4 | InternVL3-78B | 60.7 | 65.5 | InternVL3-78B<br>InternVL3-14B-Instruct<br>InternVL3-8B-Instruct<br>InternVL3-2B-Instruct<br>InternVL3-1B-Instruct |
| Gemma | 2 | 1 | Gemma4-12B-it | 69.2 | 70.5 | Gemma4-12B-it<br>Gemma3-4B |
| GLM | 1 | 1 | GLM-4.1V-9B-Thinking | 69.3 | 72.6 | GLM-4.1V-9B-Thinking |
| Ovis2.6 | 1 | 1 | Ovis2.6-30B-A3B | 73.1 | 76.4 | Ovis2.6-30B-A3B |
| Ovis2.5 | 2 | 2 | Ovis2.5-9B | 68.0 | 72.2 | Ovis2.5-9B<br>Ovis2.5-2B |
| Ovis2 | 6 | 6 | Ovis2-34B | 60.9 | 66.5 | Ovis2-34B<br>Ovis2-16B<br>Ovis2-8B<br>Ovis2-4B<br>Ovis2-2B<br>Ovis2-1B |
| Other | 1 | 1 | MiniCPM-V-4.6 | 57.2 | 62.6 | MiniCPM-V-4.6 |

## Leaderboard By Family

Rows are grouped by family, then sorted by `Macro Avg` within each family.

| Family | Family Rank | Model | Source | Overall | Macro Avg | MMStar | RealWorldQA | SEEDBench2+ | OCRBench | MathVista | MathVision | MathVerse | Avg Tok | Max Tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT | 1 | gpt-5-2025-08-07 | csv_existing | 74.7 | 76.5 | 76.7 | 82.5 | 75.2 | 81.0 | 78.6 | 68.4 | 73.2 | 813.3 | 4072 |
| Gemini | 1 | GeminiPro2-5 | csv_existing | 71.9 | 75.1 | 77.0 | 78.7 | 74.8 | 87.1 | 80.5 | 58.4 | 69.0 | 902.4 | 2046 |
| Gemini | 2 | GeminiFlash2-5 | csv_existing | 67.3 | 71.8 | 75.3 | 78.6 | 74.1 | 86.7 | 76.0 | 46.4 | 65.7 | 648.9 | 2048 |
| Claude | 1 | claude-opus-4-7 | local_full_run | 78.7 | 79.3 | 77.1 | 74.8 | 76.0 | 87.7 | 84.4 | 78.2 | 76.8 | 224.5 | 3902 |
| Claude | 2 | Claude3-7V_Sonnet | csv_existing | 56.8 | 58.6 | 62.6 | 55.4 | 67.2 | 70.1 | 66.8 | 41.3 | 46.7 | 202.4 | 1776 |
| Qwen3-VL | 1 | Qwen3-VL-32B-Instruct | local_full_run | 72.6 | 74.8 | 78.0 | 79.7 | 75.5 | 89.6 | 83.4 | 61.0 | 56.5 | 872.7 | 4102 |
| Qwen3-VL | 2 | Qwen3-VL-30B-A3B-Instruct | local_full_run | 69.0 | 71.6 | 72.9 | 73.7 | 71.8 | 91.2 | 79.3 | 56.4 | 55.7 | 1119.1 | 4096 |
| Qwen3-VL | 3 | Qwen3-VL-8B-Instruct | local_full_run | 65.8 | 67.5 | 70.1 | 72.2 | 73.3 | 91.0 | 77.0 | 51.9 | 37.3 | 998.7 | 4098 |
| Qwen3-VL | 4 | Qwen3-VL-4B-Instruct | local_full_run | 63.2 | 64.4 | 68.5 | 70.6 | 72.0 | 87.8 | 72.1 | 49.8 | 30.3 | 1107.3 | 4106 |
| Qwen3-VL | 5 | Qwen3-VL-2B-Instruct | local_full_run | 54.6 | 57.4 | 57.5 | 64.8 | 67.6 | 86.3 | 59.4 | 34.8 | 31.1 | 1109.3 | 4096 |
| Qwen2.5-VL | 1 | Qwen2.5-VL-72B-Instruct | csv_existing | 62.8 | 67.0 | 70.9 | 75.3 | 73.8 | 88.2 | 74.2 | 39.3 | 47.3 | 192.5 | 2048 |
| Qwen2.5-VL | 2 | Qwen2.5-VL-32B-Instruct | local_full_run | 62.1 | 66.3 | 69.3 | 71.2 | 72.9 | 85.8 | 73.2 | 39.4 | 51.9 | 362.3 | 4096 |
| Qwen2.5-VL | 3 | Qwen2.5-VL-7B-Instruct | local_full_run | 56.0 | 61.4 | 64.6 | 69.3 | 70.8 | 88.4 | 68.1 | 26.3 | 42.1 | 236.3 | 4096 |
| Qwen2.5-VL | 4 | Qwen2.5-VL-3B-Instruct | local_full_run | 51.9 | 56.4 | 57.3 | 66.0 | 69.2 | 82.3 | 63.4 | 24.2 | 32.4 | 237.4 | 4096 |
| InternVL3.5 | 1 | InternVL3_5-30B-A3B | local_full_run | 62.4 | 66.9 | 71.1 | 71.1 | 70.4 | 88.4 | 77.6 | 39.6 | 49.9 | 251.3 | 4104 |
| InternVL3.5 | 2 | InternVL3_5-GPT-OSS-20B-A4B-Preview | local_full_run | 58.6 | 62.7 | 67.6 | 69.7 | 69.0 | 86.3 | 71.6 | 34.9 | 40.1 | 302.8 | 5566 |
| InternVL3.5 | 3 | InternVL3_5-14B-Instruct | local_full_run | 58.6 | 62.5 | 65.1 | 68.5 | 67.7 | 83.7 | 71.5 | 37.4 | 43.8 | 320.0 | 4096 |
| InternVL3.5 | 4 | InternVL3_5-8B-Instruct | local_full_run | 57.2 | 61.3 | 63.8 | 64.7 | 68.6 | 83.3 | 72.2 | 33.8 | 42.9 | 299.9 | 4096 |
| InternVL3.5 | 5 | InternVL3_5-4B-Instruct | local_full_run | 55.7 | 59.7 | 65.2 | 65.6 | 67.5 | 81.6 | 66.1 | 31.7 | 40.0 | 301.7 | 4096 |
| InternVL3.5 | 6 | InternVL3_5-2B-Instruct | local_full_run | 50.9 | 54.9 | 55.2 | 60.8 | 64.7 | 82.7 | 60.8 | 26.9 | 33.2 | 343.2 | 4151 |
| InternVL3 | 1 | InternVL3-78B | csv_existing | 60.7 | 65.5 | 73.1 | 78.4 | 71.8 | 91.0 | 72.6 | 34.1 | 37.3 | 113.2 | 1025 |
| InternVL3 | 2 | InternVL3-14B-Instruct | local_full_run | 56.9 | 61.6 | 65.3 | 68.9 | 69.8 | 87.0 | 70.0 | 30.5 | 39.5 | 304.7 | 4116 |
| InternVL3 | 3 | InternVL3-8B-Instruct | local_full_run | 49.6 | 54.5 | 59.8 | 65.4 | 64.5 | 84.8 | 56.4 | 20.7 | 29.9 | 429.9 | 4098 |
| InternVL3 | 4 | InternVL3-2B-Instruct | local_full_run | 46.5 | 51.0 | 56.8 | 64.3 | 63.1 | 81.6 | 52.4 | 17.6 | 21.1 | 293.5 | 4098 |
| InternVL3 | 5 | InternVL3-1B-Instruct | local_full_run | 12.2 | 11.2 | 2.0 | 0.5 | 1.7 | 15.0 | 36.9 | 22.0 | 0.0 | 1966.8 | 2866 |
| Gemma | 1 | Gemma4-12B-it | local_full_run | 69.2 | 70.5 | 71.9 | 69.0 | 71.0 | 74.3 | 75.9 | 62.7 | 68.9 | 602.1 | 4656 |
| Gemma | 2 | Gemma3-4B | csv_existing | 43.7 | 46.0 | 47.1 | 55.7 | 60.8 | 66.0 | 46.2 | 23.4 | 22.7 | 358.9 | 2048 |
| GLM | 1 | GLM-4.1V-9B-Thinking | local_full_run | 69.3 | 72.6 | 72.2 | 72.4 | 73.4 | 86.0 | 80.7 | 55.2 | 68.3 | 152.0 | 8195 |
| Ovis2.6 | 1 | Ovis2.6-30B-A3B | local_full_run | 73.1 | 76.4 | 75.5 | 77.0 | 74.2 | 89.8 | 83.9 | 61.2 | 73.2 | 142.5 | 4096 |
| Ovis2.5 | 1 | Ovis2.5-9B | local_full_run | 68.0 | 72.2 | 72.3 | 74.8 | 72.7 | 87.9 | 82.2 | 50.0 | 65.9 | 310.4 | 4098 |
| Ovis2.5 | 2 | Ovis2.5-2B | local_full_run | 61.5 | 67.1 | 67.5 | 68.5 | 71.5 | 88.0 | 78.9 | 35.0 | 60.5 | 312.0 | 4102 |
| Ovis2 | 1 | Ovis2-34B | local_full_run | 60.9 | 66.5 | 69.8 | 75.2 | 71.8 | 89.1 | 76.7 | 33.3 | 49.4 | 12.6 | 4096 |
| Ovis2 | 2 | Ovis2-16B | local_full_run | 59.0 | 64.4 | 67.0 | 74.1 | 71.7 | 88.0 | 74.1 | 30.8 | 44.9 | 13.0 | 4096 |
| Ovis2 | 3 | Ovis2-8B | local_full_run | 56.3 | 62.2 | 64.1 | 72.7 | 70.1 | 89.4 | 71.2 | 25.9 | 42.3 | 17.5 | 4096 |
| Ovis2 | 4 | Ovis2-4B | local_full_run | 54.0 | 60.3 | 61.9 | 71.9 | 69.2 | 91.3 | 69.2 | 21.4 | 37.4 | 20.9 | 4096 |
| Ovis2 | 5 | Ovis2-2B | local_full_run | 50.4 | 55.9 | 57.9 | 66.5 | 67.3 | 87.4 | 64.3 | 18.6 | 29.4 | 24.4 | 4097 |
| Ovis2 | 6 | Ovis2-1B | local_full_run | 46.7 | 52.3 | 51.9 | 63.7 | 61.6 | 89.1 | 59.6 | 16.4 | 23.9 | 18.7 | 4096 |
| Other | 1 | MiniCPM-V-4.6 | local_full_run | 57.2 | 62.6 | 67.3 | 65.4 | 65.0 | 82.5 | 74.3 | 31.6 | 52.4 | 238.6 | 4097 |

## Local Full VLMEvalKit Runs

These models have a local `status.json` with all seven benchmark statuses marked `done`.

| Model | Completed at | Overall | Macro Avg | Status file |
| --- | --- | --- | --- | --- |
| claude-opus-4-7 | 2026-06-27T12:41:21.844738+00:00 | 78.7 | 79.3 | third_party/VLMEvalKit/outputs/mmrbench_one_by_one_claude-opus-4-7_20260626_094335/claude-opus-4-7/status.json |
| GLM-4.1V-9B-Thinking | 2026-06-22T18:29:04.421106+00:00 | 69.3 | 72.6 | third_party/VLMEvalKit/outputs/mmrbench_controlled_GLM-4.1V-9B-Thinking_20260620_102435/GLM-4.1V-9B-Thinking/status.json |
| Ovis2.6-30B-A3B | 2026-06-22T16:16:08.123451+00:00 | 73.1 | 76.4 | third_party/VLMEvalKit/outputs/mmrbench_controlled_ovis26_30b_a3b_cachefix_20260621_144248/Ovis2.6-30B-A3B/status.json |
| MiniCPM-V-4.6 | 2026-06-22T02:14:02.096499+00:00 | 57.2 | 62.6 | third_party/VLMEvalKit/outputs/mmrbench_controlled_MiniCPM-V-4.6_20260621_170746/MiniCPM-V-4.6/status.json |
| Ovis2-34B | 2026-06-21T10:06:44.990982+00:00 | 60.9 | 66.5 | third_party/VLMEvalKit/outputs/mmrbench_controlled_ovis2_34b_20260620_190812/Ovis2-34B/status.json |
| Ovis2-2B | 2026-06-20T19:12:53.512601+00:00 | 50.4 | 55.9 | third_party/VLMEvalKit/outputs/mmrbench_controlled_ovis2_1b_then_2b_20260620_164301_Ovis2-2B/Ovis2-2B/status.json |
| Ovis2-16B | 2026-06-20T18:44:40.156331+00:00 | 59.0 | 64.4 | third_party/VLMEvalKit/outputs/mmrbench_controlled_ovis2_16b_20260620_162007/Ovis2-16B/status.json |
| Ovis2-1B | 2026-06-20T17:47:13.227972+00:00 | 46.7 | 52.3 | third_party/VLMEvalKit/outputs/mmrbench_controlled_ovis2_1b_then_2b_20260620_164301_Ovis2-1B/Ovis2-1B/status.json |
| Ovis2.5-2B | 2026-06-20T15:35:22.365511+00:00 | 61.5 | 67.1 | third_party/VLMEvalKit/outputs/mmrbench_controlled_ovis25_sequence_20260620_115442_Ovis2.5-2B/Ovis2.5-2B/status.json |
| Ovis2-8B | 2026-06-20T15:08:11.588612+00:00 | 56.3 | 62.2 | third_party/VLMEvalKit/outputs/mmrbench_controlled_ovis2_sequence_20260620_115226_Ovis2-8B/Ovis2-8B/status.json |
| Ovis2.5-9B | 2026-06-20T14:25:29.098160+00:00 | 68.0 | 72.2 | third_party/VLMEvalKit/outputs/mmrbench_controlled_ovis25_sequence_20260620_115442_Ovis2.5-9B/Ovis2.5-9B/status.json |
| Ovis2-4B | 2026-06-20T13:27:07.323506+00:00 | 54.0 | 60.3 | third_party/VLMEvalKit/outputs/mmrbench_controlled_ovis2_sequence_20260620_115226_Ovis2-4B/Ovis2-4B/status.json |
| InternVL3_5-GPT-OSS-20B-A4B-Preview | 2026-06-18T06:38:56.581855+00:00 | 58.6 | 62.7 | third_party/VLMEvalKit/outputs/internvl35_gpt_oss_20b_a4b_preview_mmrbench_20260618_015136/InternVL3_5-GPT-OSS-20B-A4B-Preview/status.json |
| InternVL3_5-2B-Instruct | 2026-06-17T14:19:12.177705+00:00 | 50.9 | 54.9 | third_party/VLMEvalKit/outputs/mmrbench_controlled_InternVL3_5-2B-Instruct_20260617_094137/InternVL3_5-2B-Instruct/status.json |
| InternVL3-1B-Instruct | 2026-06-17T14:13:23.071639+00:00 | 12.2 | 11.2 | third_party/VLMEvalKit/outputs/mmrbench_controlled_InternVL3-1B-Instruct_max2048_parallel8gpu_20260615_131809/InternVL3-1B-Instruct/status.json |
| InternVL3-14B-Instruct | 2026-06-17T08:50:27.838466+00:00 | 56.9 | 61.6 | third_party/VLMEvalKit/outputs/mmrbench_controlled_InternVL3-14B-Instruct_8gpu_20260616_175703/InternVL3-14B-Instruct/status.json |
| InternVL3_5-30B-A3B | 2026-06-17T05:14:54.197144+00:00 | 62.4 | 66.9 | third_party/VLMEvalKit/outputs/mmrbench_controlled_InternVL3_5-30B-A3B_20260616_185605/InternVL3_5-30B-A3B/status.json |
| InternVL3_5-14B-Instruct | 2026-06-17T00:51:12.188826+00:00 | 58.6 | 62.5 | third_party/VLMEvalKit/outputs/mmrbench_controlled_InternVL3_5-14B-Instruct_20260616_190429/InternVL3_5-14B-Instruct/status.json |
| Qwen3-VL-8B-Instruct | 2026-06-16T16:40:19.274073+00:00 | 65.8 | 67.5 | third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-8B-Instruct_20260613_115237/Qwen3-VL-8B-Instruct/status.json |
| InternVL3-8B-Instruct | 2026-06-16T13:01:17.167425+00:00 | 49.6 | 54.5 | third_party/VLMEvalKit/outputs/mmrbench_controlled_InternVL3-8B-Instruct_20260615_114352/InternVL3-8B-Instruct/status.json |
| InternVL3-2B-Instruct | 2026-06-15T15:05:35.181174+00:00 | 46.5 | 51.0 | third_party/VLMEvalKit/outputs/mmrbench_controlled_InternVL3-2B-Instruct_parallel8gpu_20260615_105915/InternVL3-2B-Instruct/status.json |
| Qwen3-VL-4B-Instruct | 2026-06-14T19:30:15.577218+00:00 | 63.2 | 64.4 | third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-4B-Instruct_20260614_011106/Qwen3-VL-4B-Instruct/status.json |
| Qwen3-VL-30B-A3B-Instruct | 2026-06-14T16:54:33.054969+00:00 | 69.0 | 71.6 | third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-30B-A3B-Instruct_20260613_122245/Qwen3-VL-30B-A3B-Instruct/status.json |
| Qwen3-VL-32B-Instruct | 2026-06-14T10:50:14.617318+00:00 | 72.6 | 74.8 | third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-32B-Instruct_20260613_124838/Qwen3-VL-32B-Instruct/status.json |
| Gemma4-12B-it | 2026-06-14T10:47:17.126538+00:00 | 69.2 | 70.5 | third_party/VLMEvalKit/outputs/mmrbench_controlled_Gemma4-12B-it_20260613_210712/Gemma4-12B-it/status.json |
| Qwen3-VL-2B-Instruct | 2026-06-14T01:05:35.972912+00:00 | 54.6 | 57.4 | third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen3-VL-2B-Instruct_20260613_121944/Qwen3-VL-2B-Instruct/status.json |
| Qwen2.5-VL-32B-Instruct | 2026-06-13T11:18:43.472940+00:00 | 62.1 | 66.3 | third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen2.5-VL-32B-Instruct_20260612_081510/Qwen2.5-VL-32B-Instruct/status.json |
| Qwen2.5-VL-7B-Instruct | 2026-06-11T13:14:49.941447+00:00 | 56.0 | 61.4 | third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen2.5-VL-7B-Instruct_20260611_100314/Qwen2.5-VL-7B-Instruct/status.json |
| InternVL3_5-8B-Instruct | 2026-06-08T19:04:55.097342+00:00 | 57.2 | 61.3 | third_party/VLMEvalKit/outputs/mmrbench_controlled_InternVL3_5-8B-Instruct_20260608_125928/InternVL3_5-8B-Instruct/status.json |
| Qwen2.5-VL-3B-Instruct | 2026-06-08T18:15:31.983782+00:00 | 51.9 | 56.4 | third_party/VLMEvalKit/outputs/mmrbench_controlled_Qwen2.5-VL-3B-Instruct_20260608_124036/Qwen2.5-VL-3B-Instruct/status.json |
| InternVL3_5-4B-Instruct | 2026-06-08T12:40:49.448715+00:00 | 55.7 | 59.7 | third_party/VLMEvalKit/outputs/mmrbench_controlled_InternVL3_5-4B-Instruct_20260608_080532/InternVL3_5-4B-Instruct/status.json |

## CSV-Existing Complete Models

These models are complete in the CSV, but no matching local seven-benchmark `status.json` was found under `third_party/VLMEvalKit/outputs`.

| Model | Overall | Macro Avg |
| --- | --- | --- |
| gpt-5-2025-08-07 | 74.7 | 76.5 |
| GeminiPro2-5 | 71.9 | 75.1 |
| GeminiFlash2-5 | 67.3 | 71.8 |
| Qwen2.5-VL-72B-Instruct | 62.8 | 67.0 |
| InternVL3-78B | 60.7 | 65.5 |
| Claude3-7V_Sonnet | 56.8 | 58.6 |
| Gemma3-4B | 43.7 | 46.0 |

## Partial Local Outputs Not Counted As Full Runs

| Model | Done | Updated at | Status file |
| --- | --- | --- | --- |
| gpt-5.5-0424-global | 0/1 | 2026-06-27T09:56:46.220994+00:00 | third_party/VLMEvalKit/outputs/mmrbench_one_by_one_gpt-5.5-0424-global_20260626_111448/gpt-5.5-0424-global/T20260627-095644/status.json |
| LLaVA-OneVision-1.5-8B-Instruct | 0/1 | 2026-06-16T19:04:34.082571+00:00 | third_party/VLMEvalKit/outputs/mmrbench_controlled_LLaVA-OneVision-1.5-8B-Instruct_20260616_190347/LLaVA-OneVision-1.5-8B-Instruct/T20260616-190419/status.json |
| Qwen2.5-VL-72B-Instruct | 0/1 | 2026-06-12T14:37:52.982765+00:00 | third_party/VLMEvalKit/outputs/mmrbench_one_by_one_Qwen2.5-VL-72B-Instruct_20260612_130156/Qwen2.5-VL-72B-Instruct/T20260612-143522/status.json |

## Update Command

```bash
python scripts/update_summary.py
```
