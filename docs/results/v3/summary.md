# Dense GPA retrieval experiments

Synthetic structured retrieval, not official RULER/NIAH or BABILong. All runs start from random weights.
UUID4 uses actual free-generation exact match; MQAR uses per-query teacher-forced EM (three queries/example).
V1 overwrite usually queries an unchanged key. V2 always queries the single last overwrite and admits a recency shortcut.
V3 overwrites four different keys and randomly queries one of them; use V3 for the main comparison.
T256 is the training length; T512 is extrapolation. Do not pool versions, curricula or test seeds.

## Data version 1

| Model | Steps | Curriculum span | Loss average | Seed | T | Single | UUID4 | MQAR | Overwrite |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| counted4 | 6000 | 200 | mean_over_answer_tokens | 900000 | 128 | 29.69% | 11.72% | 33.07% | 25.78% |
| counted4 | 6000 | 200 | mean_over_answer_tokens | 900000 | 256 | 21.09% | 0.78% | 27.86% | 20.31% |
| counted4 | 6000 | 200 | mean_over_answer_tokens | 900000 | 512 | 26.56% | 1.56% | 22.14% | 18.75% |
| counted2 | 6000 | 200 | mean_over_answer_tokens | 900000 | 128 | 29.69% | 3.91% | 31.51% | 23.44% |
| counted2 | 6000 | 200 | mean_over_answer_tokens | 900000 | 256 | 21.09% | 0.78% | 26.04% | 17.97% |
| counted2 | 6000 | 200 | mean_over_answer_tokens | 900000 | 512 | 21.09% | 0.00% | 19.27% | 21.88% |
| counted_raw | 6000 | 200 | mean_over_answer_tokens | 900000 | 128 | 28.12% | 0.00% | 32.03% | 22.66% |
| counted_raw | 6000 | 200 | mean_over_answer_tokens | 900000 | 256 | 20.31% | 0.00% | 27.34% | 21.09% |
| counted_raw | 6000 | 200 | mean_over_answer_tokens | 900000 | 512 | 23.44% | 0.00% | 23.44% | 22.66% |
| counted_deep | 6000 | 200 | mean_over_answer_tokens | 900000 | 128 | 32.03% | 7.03% | 38.54% | 23.44% |
| counted_deep | 6000 | 200 | mean_over_answer_tokens | 900000 | 256 | 17.97% | 0.78% | 28.12% | 27.34% |
| counted_deep | 6000 | 200 | mean_over_answer_tokens | 900000 | 512 | 20.31% | 0.00% | 19.53% | 25.00% |
| transformer_deep | 6000 | 200 | mean_over_answer_tokens | 900000 | 128 | 100.00% | 100.00% | 100.00% | 87.50% |
| transformer_deep | 6000 | 200 | mean_over_answer_tokens | 900000 | 256 | 100.00% | 97.66% | 100.00% | 94.53% |
| transformer_deep | 6000 | 200 | mean_over_answer_tokens | 900000 | 512 | 100.00% | 92.97% | 100.00% | 94.53% |
| transformer | 6000 | 200 | mean_over_answer_tokens | 900000 | 128 | 100.00% | 96.88% | 99.48% | 85.94% |
| transformer | 6000 | 200 | mean_over_answer_tokens | 900000 | 256 | 100.00% | 94.53% | 99.48% | 96.88% |
| transformer | 6000 | 200 | mean_over_answer_tokens | 900000 | 512 | 100.00% | 92.19% | 99.48% | 92.97% |
| counted_transport | 6000 | 200 | mean_over_answer_tokens | 900000 | 128 | 29.69% | 13.28% | 36.46% | 28.12% |
| counted_transport | 6000 | 200 | mean_over_answer_tokens | 900000 | 256 | 21.88% | 2.34% | 25.78% | 21.88% |
| counted_transport | 6000 | 200 | mean_over_answer_tokens | 900000 | 512 | 21.09% | 0.78% | 19.01% | 14.84% |
| paired_gdn4 | 6000 | 200 | mean_over_answer_tokens | 900000 | 128 | 97.66% | 57.81% | 99.22% | 95.31% |
| paired_gdn4 | 6000 | 200 | mean_over_answer_tokens | 900000 | 256 | 97.66% | 20.31% | 94.27% | 92.97% |
| paired_gdn4 | 6000 | 200 | mean_over_answer_tokens | 900000 | 512 | 88.28% | 7.81% | 91.67% | 85.94% |
| counted_full_controls | 6000 | 200 | mean_over_answer_tokens | 900000 | 128 | 28.12% | 25.00% | 34.38% | 24.22% |
| counted_full_controls | 6000 | 200 | mean_over_answer_tokens | 900000 | 256 | 21.88% | 5.47% | 29.95% | 27.34% |
| counted_full_controls | 6000 | 200 | mean_over_answer_tokens | 900000 | 512 | 19.53% | 3.91% | 19.27% | 21.09% |
| transformer_slow | 6000 | 200 | mean_over_answer_tokens | 900000 | 128 | 100.00% | 99.22% | 99.74% | 88.28% |
| transformer_slow | 6000 | 200 | mean_over_answer_tokens | 900000 | 256 | 100.00% | 96.09% | 99.22% | 96.09% |
| transformer_slow | 6000 | 200 | mean_over_answer_tokens | 900000 | 512 | 96.88% | 92.97% | 99.22% | 94.53% |
| counted_slot_decay | 6000 | 1000 | mean_over_answer_tokens | 900000 | 128 | 39.84% | 22.66% | 58.33% | 43.75% |
| counted_slot_decay | 6000 | 1000 | mean_over_answer_tokens | 900000 | 256 | 33.59% | 6.25% | 40.36% | 34.38% |
| counted_slot_decay | 6000 | 1000 | mean_over_answer_tokens | 900000 | 512 | 28.91% | 1.56% | 26.82% | 26.56% |
| paired_gdn2_orthogonal | 6000 | 200 | mean_over_answer_tokens | 900000 | 128 | 97.66% | 62.50% | 98.70% | 89.06% |
| paired_gdn2_orthogonal | 6000 | 200 | mean_over_answer_tokens | 900000 | 256 | 96.88% | 32.81% | 95.05% | 89.84% |
| paired_gdn2_orthogonal | 6000 | 200 | mean_over_answer_tokens | 900000 | 512 | 86.72% | 15.62% | 89.84% | 82.81% |

## Data version 2

| Model | Steps | Curriculum span | Loss average | Seed | T | Single | UUID4 | MQAR | Overwrite |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| counted_strong_v2 | 6000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 32.03% | 21.09% | 36.20% | 30.47% |
| counted_strong_v2 | 6000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 21.09% | 3.12% | 29.69% | 99.22% |
| counted_strong_v2 | 6000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 10.16% | 0.00% | 13.28% | 92.97% |
| transformer_v2 | 6000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 99.22% | 99.22% | 98.96% | 61.72% |
| transformer_v2 | 6000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 96.88% | 88.28% | 97.66% | 55.47% |
| transformer_v2 | 6000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 93.75% | 77.34% | 95.31% | 49.22% |
| counted_strong_v2_holdout | 6000 | multi-record from start | mean_over_answer_tokens | 910000 | 128 | 34.38% | 16.80% | 39.84% | 33.98% |
| counted_strong_v2_holdout | 6000 | multi-record from start | mean_over_answer_tokens | 910000 | 256 | 22.66% | 3.52% | 26.56% | 96.88% |
| counted_strong_v2_holdout | 6000 | multi-record from start | mean_over_answer_tokens | 910000 | 512 | 14.06% | 1.17% | 11.98% | 96.09% |
| transformer_v2_holdout | 6000 | multi-record from start | mean_over_answer_tokens | 910000 | 128 | 97.66% | 92.58% | 98.70% | 53.52% |
| transformer_v2_holdout | 6000 | multi-record from start | mean_over_answer_tokens | 910000 | 256 | 96.48% | 80.47% | 96.88% | 52.73% |
| transformer_v2_holdout | 6000 | multi-record from start | mean_over_answer_tokens | 910000 | 512 | 94.92% | 73.05% | 95.70% | 52.34% |

## Data version 3

| Model | Steps | Curriculum span | Loss average | Seed | T | Single | UUID4 | MQAR | Overwrite |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| paired_product_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 78.12% | 98.96% | 93.75% |
| paired_product_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 96.88% | 56.25% | 97.40% | 95.31% |
| paired_product_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 90.62% | 26.56% | 87.76% | 94.53% |
| transformer_rope_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 100.00% | 100.00% | 98.44% |
| transformer_rope_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 98.44% | 100.00% | 99.22% |
| transformer_rope_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 82.03% | 78.12% | 85.42% | 100.00% |
| paired_positive_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 24.22% | 1.56% | 31.77% | 23.44% |
| paired_positive_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 17.97% | 0.00% | 24.48% | 31.25% |
| paired_positive_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 21.09% | 0.00% | 23.96% | 35.94% |
| paired_calibrated_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 27.34% | 0.00% | 33.07% | 25.00% |
| paired_calibrated_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 14.84% | 0.00% | 29.17% | 34.38% |
| paired_calibrated_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 20.31% | 0.00% | 24.22% | 37.50% |
| paired_balanced_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 82.81% | 100.00% | 100.00% |
| paired_balanced_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 98.44% | 58.59% | 99.22% | 98.44% |
| paired_balanced_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 94.53% | 35.94% | 94.01% | 98.44% |
| paired_multirecord_v3 | 20000 | multi-record from start | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 80.47% | 100.00% | 98.44% |
| paired_multirecord_v3 | 20000 | multi-record from start | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 50.78% | 99.48% | 100.00% |
| paired_multirecord_v3 | 20000 | multi-record from start | mean_sequence_answer_CE | 900000 | 512 | 99.22% | 23.44% | 97.92% | 97.66% |
| paired_context_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 92.97% | 100.00% | 100.00% |
| paired_context_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 66.41% | 99.74% | 99.22% |
| paired_context_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 98.44% | 35.94% | 97.40% | 99.22% |
| paired_marginal_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 91.41% | 99.48% | 98.44% |
| paired_marginal_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 96.09% | 62.50% | 97.14% | 100.00% |
| paired_marginal_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 96.09% | 35.16% | 95.31% | 94.53% |
| paired_quadratic_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 94.53% | 100.00% | 98.44% |
| paired_quadratic_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 74.22% | 99.22% | 98.44% |
| paired_quadratic_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 98.44% | 57.81% | 98.18% | 98.44% |
| paired_readgate_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 93.75% | 100.00% | 100.00% |
| paired_readgate_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 64.06% | 99.74% | 99.22% |
| paired_readgate_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 96.88% | 52.34% | 98.96% | 100.00% |
| paired_temporal_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 97.66% | 100.00% | 100.00% |
| paired_temporal_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 90.62% | 100.00% | 100.00% |
| paired_temporal_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 99.22% | 68.75% | 99.48% | 100.00% |
| paired_widekey_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 96.88% | 100.00% | 100.00% |
| paired_widekey_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 82.81% | 100.00% | 100.00% |
| paired_widekey_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 99.22% | 53.91% | 100.00% | 100.00% |
| transformer_widekey_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 99.22% | 100.00% | 99.22% |
| transformer_widekey_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 100.00% | 100.00% | 100.00% |
| transformer_widekey_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 86.72% | 75.00% | 87.24% | 99.22% |
| paired_dense4_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 95.31% | 100.00% | 99.22% |
| paired_dense4_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 75.78% | 99.48% | 100.00% |
| paired_dense4_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 99.22% | 55.47% | 99.74% | 100.00% |
| paired_temperature_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 98.44% | 100.00% | 100.00% |
| paired_temperature_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 89.06% | 100.00% | 100.00% |
| paired_temperature_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 100.00% | 57.03% | 99.74% | 100.00% |
| transformer_temporal_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 100.00% | 100.00% | 98.44% |
| transformer_temporal_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 99.22% | 98.44% | 100.00% | 99.22% |
| transformer_temporal_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 70.31% | 71.09% | 71.88% | 100.00% |
| paired_memory64_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 97.66% | 100.00% | 100.00% |
| paired_memory64_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 82.81% | 100.00% | 100.00% |
| paired_memory64_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 100.00% | 55.47% | 100.00% | 100.00% |
| transformer_param_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 95.31% | 100.00% | 84.38% |
| transformer_param_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 95.31% | 100.00% | 94.53% |
| transformer_param_v3b | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 80.47% | 53.91% | 85.68% | 96.09% |
| selected_gpa_fast_v3_holdout | 20000 | 1000 | mean_sequence_answer_CE | 920000 | 128 | 100.00% | 97.66% | 100.00% | 100.00% |
| selected_gpa_fast_v3_holdout | 20000 | 1000 | mean_sequence_answer_CE | 920000 | 256 | 100.00% | 93.36% | 100.00% | 100.00% |
| selected_gpa_fast_v3_holdout | 20000 | 1000 | mean_sequence_answer_CE | 920000 | 512 | 100.00% | 72.66% | 99.61% | 100.00% |
| transformer_multirecord_v3 | 20000 | multi-record from start | mean_sequence_answer_CE | 900000 | 128 | 28.91% | 0.00% | 26.82% | 12.50% |
| transformer_multirecord_v3 | 20000 | multi-record from start | mean_sequence_answer_CE | 900000 | 256 | 15.62% | 0.00% | 19.53% | 23.44% |
| transformer_multirecord_v3 | 20000 | multi-record from start | mean_sequence_answer_CE | 900000 | 512 | 13.28% | 0.00% | 15.62% | 17.97% |
| paired_scaled_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 99.22% | 82.81% | 99.74% | 100.00% |
| paired_scaled_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 62.50% | 98.70% | 100.00% |
| paired_scaled_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 96.88% | 35.94% | 98.96% | 99.22% |
| transformer_scaled_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 128 | 100.00% | 99.22% | 100.00% | 96.88% |
| transformer_scaled_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 256 | 100.00% | 98.44% | 100.00% | 97.66% |
| transformer_scaled_v3 | 20000 | 1000 | mean_sequence_answer_CE | 900000 | 512 | 81.25% | 69.53% | 79.17% | 100.00% |
| selected_gpa_v3_holdout | 20000 | 1000 | mean_sequence_answer_CE | 920000 | 128 | 100.00% | 97.66% | 100.00% | 100.00% |
| selected_gpa_v3_holdout | 20000 | 1000 | mean_sequence_answer_CE | 920000 | 256 | 100.00% | 93.36% | 100.00% | 100.00% |
| selected_gpa_v3_holdout | 20000 | 1000 | mean_sequence_answer_CE | 920000 | 512 | 100.00% | 72.66% | 99.61% | 100.00% |
| matched_transformer_v3_holdout | 20000 | 1000 | mean_sequence_answer_CE | 920000 | 128 | 100.00% | 97.27% | 100.00% | 98.83% |
| matched_transformer_v3_holdout | 20000 | 1000 | mean_sequence_answer_CE | 920000 | 256 | 100.00% | 96.09% | 100.00% | 100.00% |
| matched_transformer_v3_holdout | 20000 | 1000 | mean_sequence_answer_CE | 920000 | 512 | 86.33% | 79.69% | 85.16% | 99.61% |

