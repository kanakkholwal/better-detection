## Tuned test metrics (thresholds chosen to maximize val accuracy)

| Model | Threshold | AUC | Acc | F1 | Sens | Spec |
|---|---:|---:|---:|---:|---:|---:|
| SimpleCNN | 0.775 | 0.9769 | 0.9241 | 0.8963 | 0.8694 | 0.9573 |
| Inception | 0.640 | 0.9783 | 0.9286 | 0.9020 | 0.8718 | 0.9630 |
| ResNet | 0.730 | 0.9716 | 0.9161 | 0.8817 | 0.8282 | 0.9694 |
| XGB | 0.690 | 0.9767 | 0.9259 | 0.8968 | 0.8541 | 0.9694 |
| Ensemble(orig) | 0.660 | 0.9789 | 0.9290 | 0.9024 | 0.8706 | 0.9644 |
| Meta-LR | 0.605 | 0.9790 | 0.9286 | 0.9022 | 0.8741 | 0.9615 |

### Meta-LR weights (stack on base-model probabilities)

| Base model | Coefficient |
|---|---:|
| SimpleCNN | +2.0494 |
| Inception | +2.8445 |
| ResNet | +1.2088 |
| XGB | +1.4699 |
| _intercept_ | -4.4459 |

_Thresholds are picked on val (accuracy-maximizing grid search over [0.01, 0.99] step 0.005). Meta-LR threshold is tuned on 5-fold OOF val predictions to avoid in-sample bias; the final Meta-LR is fit on the full val set before scoring test._
