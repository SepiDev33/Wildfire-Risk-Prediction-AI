# Wildfire Risk Prediction Using Machine Learning

## Overview
This project explores the use of machine learning techniques to predict wildfire risk using environmental and weather-related data. Wildfires are complex natural events influenced by multiple environmental factors, and predicting their occurrence can help improve prevention and response strategies.

The goal of this project is to build and evaluate classification models that can determine whether wildfire conditions correspond to **fire** or **not fire** based on input features.

---

## Problem Statement
Wildfires pose a significant threat to ecosystems, infrastructure, and human life. With increasing global temperatures and changing climate patterns, wildfire frequency and intensity have risen.

This project aims to answer the question:

> Can machine learning models accurately predict wildfire risk using environmental data?

---

## Dataset
This project uses the **Algerian Forest Fires dataset** from the UCI Machine Learning Repository.

### Dataset Details:
- 244 instances
- No missing values
- Contains meteorological and fire-weather index data
- Binary classification: **fire** vs **not fire**

### Features include:
- Temperature  
- Relative Humidity  
- Wind Speed  
- Rain  
- FFMC (Fine Fuel Moisture Code)  
- DMC (Duff Moisture Code)  
- DC (Drought Code)  
- ISI (Initial Spread Index)  
- BUI (Build Up Index)  
- FWI (Fire Weather Index)  

---

## Approach
The project follows a standard machine learning workflow:

1. Load dataset using `ucimlrepo`
2. Clean and preprocess data
3. Split into training and testing sets
4. Train multiple classification models
5. Evaluate model performance
6. Select and save the best model

---

## Models Used
- Logistic Regression  
- Decision Tree  
- Random Forest  

---

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## Repository Structure

```
Wildfire-Risk-Prediction-AI
├── code/
│   ├── Train_Model.py
│   ├── Evaluate_Model.py
│   └── README.md
├── dataset/
│   ├── README.md
│   └── Sources.md
├── performance/
│   ├── README.md
│   └── Results.md
├── report/
│   ├── proposal.md
│   └── proposal.docx
├── requirements.txt
└── README.md
```
## How to Run

### 1. Install Dependencies
Make sure you have Python installed, then run:

pip install -r requirements.txt

---

### 2. Navigate to the Code Directory

cd code

---

### 3. Run the Training Script

python Train_Model.py

---

### 4. Output

- Model performance metrics (Accuracy, Precision, Recall, F1-score) will be displayed in the terminal.
- The best-performing model will be saved as:

wildfire_best_model.joblib

---

## Current Status

- Project topic selected  
- Proposal completed  
- Dataset selected  
- Data preprocessing implemented  
- Model training and evaluation completed  

---

## Future Work

- Improve model performance through hyperparameter tuning  
- Experiment with additional machine learning algorithms  
- Perform feature importance analysis to understand key predictors  
- Add data visualizations and graphs for better interpretation  
- Explore real-time or larger-scale wildfire datasets  

---

## Technologies Used

- Python  
- pandas  
- NumPy  
- scikit-learn  
- matplotlib  
- seaborn  
- ucimlrepo  

---

## Author

Sepehr Delavarkhan  
University of New Brunswick  
CS4795 – Introduction to Artificial Intelligence
