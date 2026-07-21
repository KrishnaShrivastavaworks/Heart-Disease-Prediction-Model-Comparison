# Heart Disease Prediction — Model Comparison

Comparing four classification models to find the best one for predicting heart disease risk from patient health data.

## Dataset

Patient-level health records with 12 features — age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, number of major vessels, and thalassemia type. Target is binary: presence or absence of heart disease.

Data was cleaned and filtered using SQL before being brought into Python for modeling.

## Models Compared

- SVM (RBF kernel)
- Logistic Regression
- Random Forest
- KNN

## Pipeline

1. Load cleaned dataset
2. Check for missing values
3. Train/test split (80/20)
4. Standardize features
5. Train all four models
6. Evaluate on accuracy, precision, recall, F1
7. Pick the best model
8. Validate with 5-fold cross-validation

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.880 | 0.891 | 0.854 | 0.872 |
| SVM | 0.830 | 0.816 | 0.833 | 0.825 |
| Random Forest | 0.830 | 0.816 | 0.833 | 0.825 |
| KNN | 0.830 | 0.804 | 0.854 | 0.828 |

**Best model: Logistic Regression**, with the highest accuracy and F1 score, and also the most stable cross-validation performance (lowest standard deviation across folds).

## How to Run

```bash
pip install pandas scikit-learn jupyter
jupyter notebook heart_model_comparison.ipynb
```

## Files

- `heart_model_comparison.ipynb` — full pipeline, notebook form
- `heart_cleaned.csv` — cleaned dataset
