# QuickDraw Doodle Recognition — Results

## Model Comparison

| Model | Test Accuracy | Test Samples |
|---|---:|---:|
| Logistic Regression | 69.30% | 2,000 |
| Linear SVM | 67.00% | 2,000 |
| RBF SVM | 82.40% | 2,000 |
| CNN | 92.40% | 10,000 |
| MobileNetV2 | 90.40% | 2,000 |

## Key Findings

- The CNN achieved the highest measured accuracy at 92.40%.
- MobileNetV2 achieved 90.40% accuracy after recovering its saved trained weights.
- Among the classical machine-learning models, RBF SVM performed best at 82.40%.
- Logistic Regression achieved 69.30%, while Linear SVM achieved 67.00%.

## Evaluation Notes

- The classical models and MobileNetV2 were evaluated on 2,000 test samples.
- The CNN was evaluated on 10,000 test samples.
- Because test-set sizes differ, these results are not a perfectly controlled comparison.
- The MobileNetV2 training script used its test dataset for validation during training. Its score should therefore be treated as preliminary.
- Accuracy is the percentage of test drawings classified correctly.
- Training times may vary by machine and execution.

## Conclusion

The CNN achieved the highest measured accuracy in these experiments. RBF SVM was the strongest classical baseline. MobileNetV2 also performed well, but a controlled evaluation with an independent validation set and the same test samples is needed for a fair comparison.
