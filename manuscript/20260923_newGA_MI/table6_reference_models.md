# Table 6. Pooled AUC (95% CI) of the reference models and the GA- and MI-selected four-feature models at matched scalers

Columns at each method's representative config: RobustScaler BW10 (GA peak), StandardScaler BW20 (MI peak), PowerTransformer BW10. Fixed models are bin-width invariant; values from the LOO baseline run. CI = 1000-iteration percentile bootstrap.

| Model | RobustScaler (BW10) | StandardScaler (BW20) | PowerTransformer (BW10) |
|---|---|---|---|
| Fixed 3 features | 0.789 (0.685, 0.883) | 0.787 (0.682, 0.880) | 0.779 (0.669, 0.872) |
| Fixed 4 features | 0.833 (0.736, 0.915) | 0.829 (0.728, 0.915) | 0.798 (0.688, 0.891) |
| GA-selected (top 4) | 0.843 (0.746, 0.925) | 0.757 (0.630, 0.861) | 0.781 (0.672, 0.872) |
| MI-selected (top 4) | 0.706 (0.579, 0.809) | 0.778 (0.668, 0.872) | 0.737 (0.616, 0.845) |
