# Table 6 companion — DeLong reference-model contrasts (LOO, SVM, k=4)

z signed as z = (AUC_reference − AUC_baseline)/SE. Baselines from `workspace/baselines/`; reference = current GA/MI runs.

| scaler | BW | baseline | reference | AUC baseline | AUC reference | z | p | sig |
|---|---:|---|---|---:|---:|---:|---:|---|
| RobustScaler | 10 | fixed_triad3 | GA | 0.789 | 0.843 | 2.033 | 0.04202 | GA |
| RobustScaler | 10 | fixed_triad4 | GA | 0.833 | 0.843 | 0.411 | 0.6808 |  |
| RobustScaler | 10 | mwu_top4 | GA | 0.786 | 0.843 | 2.187 | 0.02872 | GA |
| RobustScaler | 10 | fixed_triad3 | MI | 0.789 | 0.706 | -1.613 | 0.1068 |  |
| RobustScaler | 10 | fixed_triad4 | MI | 0.833 | 0.706 | -3.062 | 0.0022 | fixed_triad4 |
| RobustScaler | 10 | mwu_top4 | MI | 0.786 | 0.706 | -1.570 | 0.1165 |  |
| StandardScaler | 20 | fixed_triad3 | GA | 0.787 | 0.757 | -1.151 | 0.2496 |  |
| StandardScaler | 20 | fixed_triad4 | GA | 0.829 | 0.757 | -2.654 | 0.007965 | fixed_triad4 |
| StandardScaler | 20 | mwu_top4 | GA | 0.727 | 0.757 | 1.074 | 0.2827 |  |
| StandardScaler | 20 | fixed_triad3 | MI | 0.787 | 0.778 | -0.234 | 0.8148 |  |
| StandardScaler | 20 | fixed_triad4 | MI | 0.829 | 0.778 | -1.801 | 0.07175 |  |
| StandardScaler | 20 | mwu_top4 | MI | 0.727 | 0.778 | 1.176 | 0.2397 |  |
| PowerTransformer | 10 | fixed_triad3 | GA | 0.779 | 0.781 | 0.062 | 0.9508 |  |
| PowerTransformer | 10 | fixed_triad4 | GA | 0.798 | 0.781 | -0.606 | 0.5446 |  |
| PowerTransformer | 10 | mwu_top4 | GA | 0.754 | 0.781 | 0.768 | 0.4423 |  |
| PowerTransformer | 10 | fixed_triad3 | MI | 0.779 | 0.737 | -0.790 | 0.4295 |  |
| PowerTransformer | 10 | fixed_triad4 | MI | 0.798 | 0.737 | -1.564 | 0.1177 |  |
| PowerTransformer | 10 | mwu_top4 | MI | 0.754 | 0.737 | -0.309 | 0.7576 |  |
