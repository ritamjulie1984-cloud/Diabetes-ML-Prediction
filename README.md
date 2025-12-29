# Diabetes-ML-Prediction
End-to-end Pima Indian Diabetes prediction in Python with data cleaning, imputation, Naive Bayes, Random Forest, and regularized Logistic Regression with cross-validation.
# Pima Indian Diabetes Prediction

This repository contains a Google Colab notebook that walks through an end‑to‑end machine learning pipeline to predict diabetes for patients in the Pima Indian dataset. The project focuses on **cleaning medical data, handling missing values, dealing with class imbalance, and comparing several classification algorithms**.

## Project goals

- Explore and clean the Pima Indian diabetes dataset.
- Handle zero/invalid medical readings using mean imputation.
- Check and reduce feature multicollinearity using a correlation matrix.
- Build and compare:
  - Gaussian Naive Bayes
  - Random Forest
  - Logistic Regression
  - Regularized Logistic Regression with:
    - manual C tuning
    - `class_weight="balanced"`
    - `LogisticRegressionCV` (cross‑validation over C)
- Evaluate models using accuracy, confusion matrix, precision, recall and F1‑score, with a focus on **improving recall for the diabetic class**.

## Repository structure

- `Pima Indian Diabetes Prediction.ipynb`  
  Main Colab notebook containing:
  - Data upload from local machine (Colab `files.upload`)
  - Exploratory analysis (`df.head()`, `df.shape`, dtype checks)
  - Correlation heatmap and dropping highly correlated `skin` feature
  - Class distribution checks before and after train/test split
  - Mean imputation for zero values in medical features using `SimpleImputer`
  - Model training and evaluation for:
    - GaussianNB
    - RandomForestClassifier
    - LogisticRegression
    - LogisticRegression with tuned C and `class_weight="balanced"`
    - LogisticRegressionCV with 10‑fold cross‑validation

- (Optional) `pima-trained-model.pkl`  
  Example of a trained `LogisticRegressionCV` model saved with `joblib.dump`.

## Dataset

The notebook expects a CSV file named `pima-data.csv` with the following fields:

- `num_preg`
- `glucose_conc`
- `diastolic_bp`
- `thickness`
- `insulin`
- `bmi`
- `diab_pred`
- `age`
- `skin` (dropped after correlation analysis)
- `diabetes` (target; converted from boolean to 0/1)

The dataset is a version of the well‑known Pima Indians Diabetes dataset (originally from the UCI Machine Learning Repository). The CSV itself is **not** included here; please download it from a trusted source and place it next to the notebook as `pima-data.csv`.

## How to run the notebook

1. Open the notebook directly in Colab using the “Open in Colab” badge or by pasting the GitHub URL into Colab.
2. Upload `pima-data.csv` when prompted by the `files.upload()` cell.
3. Run all cells from top to bottom.
4. At the end, review model comparison:
   - Naive Bayes baseline
   - Random Forest (shows overfitting: very high train accuracy vs lower test accuracy)
   - Logistic Regression variants with regularization and class weighting
   - Cross‑validated `LogisticRegressionCV` model

## Model evaluation highlights

- Class distribution: about **35% diabetic** vs **65% non‑diabetic**, so the data is moderately imbalanced.
- Random Forest reaches ~98% training accuracy but ~74% test accuracy, illustrating **overfitting**.
- Regularized logistic regression with `class_weight="balanced"` and tuned C improves **recall for the diabetic class** at the cost of some overall accuracy.
- `LogisticRegressionCV` with 10‑fold cross‑validation provides a more robust estimate of performance on unseen data.

## Requirements

Main Python libraries used:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `joblib`
- `google.colab` (for file upload/download when running on Colab)

On Colab these are pre‑installed; on a local environment, install them with:
pip install pandas numpy matplotlib scikit-learn joblib

text

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.





