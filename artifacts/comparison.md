# DL vs GBDT — test-set comparison (Sparkov fraud)

| model | PR-AUC | ROC-AUC | R@FPR0.001 | R@FPR0.005 | R@FPR0.01 | R@FPR0.05 |
|---|---|---|---|---|---|---|
| CatBoost | 0.8730 | 0.9974 | 0.8182 | 0.9221 | 0.9515 | 0.9897 |
| XGBoost | 0.7908 | 0.9922 | 0.7371 | 0.8331 | 0.8793 | 0.9613 |
| FT-Transformer (linear) | 0.8083 | 0.9942 | 0.7282 | 0.8545 | 0.8979 | 0.9683 |
| FT-Transformer (periodic) | 0.8713 | 0.9969 | 0.8196 | 0.9021 | 0.9375 | 0.9860 |
| FT-Transformer (ple) | 0.8692 | 0.9964 | 0.8149 | 0.8942 | 0.9310 | 0.9790 |
