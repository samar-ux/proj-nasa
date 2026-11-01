# Predicting Hazardous Near-Earth Objects (NEOs)

A machine-learning project that classifies whether a near-Earth object is **potentially hazardous**, based on its brightness, size, speed and miss distance.

## Problem

Space agencies track thousands of asteroids and comets that pass close to Earth. Only a small fraction are classified as hazardous, so the data is **highly imbalanced**. The goal is to build and compare models that can flag hazardous objects reliably without being fooled by the majority class.

## Dataset

[NASA Nearest Earth Objects (1910-2024)](https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024) on Kaggle.

| Feature | Description |
|---|---|
| `absolute_magnitude` | Brightness of the object (lower means larger or brighter) |
| `estimated_diameter_min` / `estimated_diameter_max` | Estimated size range |
| `relative_velocity` | Speed relative to Earth |
| `miss_distance` | Distance by which the object misses Earth |
| `is_hazardous` | **Target**: whether the object is classified as hazardous |

## Approach

1. **Cleaning:** remove duplicates, drop identifier and constant columns, convert the target to 0/1, coerce features to numeric.
2. **EDA:** class balance, brightness distribution by class, correlation matrix, velocity vs. miss distance.
3. **Split first:** stratified 80/20 train/test split before any preprocessing, to avoid data leakage.
4. **Preprocessing:** median imputation and standard scaling, fitted on training data only.
5. **Imbalance handling:** SMOTE applied to the **training set only**.
6. **Models:** Logistic Regression (baseline) and Random Forest.
7. **Evaluation:** precision, recall, F1, ROC-AUC, confusion matrices and ROC curves on the untouched test set.
8. **Interpretation:** Random Forest feature importance.

## Results

> Fill this table with the numbers printed by the notebook after you run it.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | | | | |
| Random Forest | | | | |

**Key takeaways:** _(2 to 3 sentences: best model, most important features, one limitation.)_

## How to run

```bash
git clone https://github.com/samar-ux/proj-nasa.git
cd proj-nasa
pip install -r requirements.txt
```

1. Download the dataset from the Kaggle link above and place the file in the project folder.
2. If the file name differs, update `DATA_PATH` in the first code cell.
3. Open `hazardous_neo_prediction.ipynb` and run all cells.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, seaborn, Jupyter.

## Author

**Samar Mohamed**: Full-Stack Developer | AI & Automation | Power BI
[LinkedIn](https://www.linkedin.com/in/samar-mohamed-71a4b7178/) | samar7ux@gmail.com
