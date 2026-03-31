# Performance

This folder contains the evaluation results and visual outputs for the Wildfire Risk Prediction models trained on the Algerian Forest Fires dataset.

## Files

### Results.md
Summary of model performance metrics including accuracy, precision, recall, and F1-score for all three models.

### model_comparison.png
Bar chart comparing Accuracy and F1-Score across Logistic Regression, Decision Tree, and Random Forest.

### confusion_matrix.png
Confusion matrix for the best-performing model (Decision Tree), evaluated on the full dataset. Notably, the model achieved zero false negatives, meaning no actual fire cases were misclassified as safe.

## Results Summary

| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression | 91.84% | 0.92 |
| Decision Tree | 93.88% | 0.94 |
| Random Forest | 93.88% | 0.94 |

The Decision Tree was selected as the best model. Both Decision Tree and Random Forest achieved the highest accuracy, but Decision Tree was chosen as it reached that score first during evaluation.
