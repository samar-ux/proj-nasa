# Predicting Hazardous Near-Earth Objects (NEOs)
   ![Project cover](<img width="1200" height="627" alt="proj-nasa-cover" src="https://github.com/user-attachments/assets/19222557-b7f3-400d-b4ab-507a7ba7c98c" />

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

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.311 | 0.878 | 0.460 | 0.838 |
| Random Forest | 0.637 | 0.704 | 0.669 | 0.949 |

**Key takeaways:** Random Forest is the best overall model (F1 = 0.669, ROC-AUC = 0.949), clearly outperforming the Logistic Regression baseline (F1 = 0.460, ROC-AUC = 0.838). Logistic Regression catches more hazardous objects (recall 0.878) but produces many false alarms (precision 0.311). Results come from a single train/test split, so cross-validation and tuning are the next steps.
## How to run

```bash
git clone https://github.com/samar-ux/proj-nasa.git
cd proj-nasa
pip install -r requirements.txt
```

The dataset file is already included in this repository, so no extra download is needed.

1. Open `hazardous_neo_prediction.ipynb` (Jupyter Notebook or JupyterLab).
2. Run all cells from top to bottom.
3. To use a different copy of the data (for example the original Kaggle CSV), change `DATA_PATH` in the first code cell.

## Project structure

```
proj-nasa/
├── hazardous_neo_prediction.ipynb   # Full analysis and modelling notebook
├── nearest-earth-objects_1910-2024_-Copy-Copy-1_Worksheet_.xlsx   # Dataset
├── requirements.txt                 # Python dependencies
└── README.md
```

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, seaborn, Jupyter.

## Author

**Samar Mohamed**: Full-Stack Developer | AI & Automation | Power BI
[LinkedIn](https://www.linkedin.com/in/samar-mohamed-71a4b7178/) | samar7ux@gmail.com
