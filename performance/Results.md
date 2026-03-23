## Results

### Models Tested
- Logistic Regression  
- Decision Tree  
- Random Forest  

---

### Model Performance

#### Logistic Regression
- Accuracy: **0.9184**  
- Macro Average F1-score: **0.92**  
- Weighted Average F1-score: **0.92**  
- Confusion Matrix:

```text
[[18  3]
 [ 1 27]]
```

#### Decision Tree
- Accuracy: **0.9388**  
- Macro Average F1-score: **0.94**  
- Weighted Average F1-score: **0.94**  
- Confusion Matrix:

```text
[[18  3]
 [ 0 28]]
```

#### Random Forest
- Accuracy: **0.9388**  
- Macro Average F1-score: **0.94**  
- Weighted Average F1-score: **0.94**  
- Confusion Matrix:

```text
[[18  3]
 [ 0 28]]
```

---

### Best Model

The best-performing model was the **Decision Tree**, achieving an accuracy of **0.9388**.

---

### Summary

All three machine learning models performed well on the Algerian Forest Fires dataset. Logistic Regression achieved strong performance, but both the Decision Tree and Random Forest models performed slightly better overall.

The Decision Tree model was selected as the best model and saved as:

```text
wildfire_best_model.joblib
```

These results demonstrate that wildfire risk can be effectively predicted using supervised machine learning techniques based on environmental and weather-related features.
