# Bank Churn Prediction

## Introduction
This project focuses on predicting **customer churn in the banking sector** using supervised machine learning techniques. By analyzing client behavior, financial transactions, and product usage, we aim to build models that help anticipate when customers are at risk of leaving the bank.  

Our workflow is structured around four main notebooks, complemented by raw datasets, a processed dataset, and the final trained model. This repository consolidates the full process, from data exploration to model deployment.

---

## Objectives
- Explore and understand the dataset, including customer demographics, product usage, and transaction patterns.  
- Compare different machine learning models to identify the most effective approach for churn prediction.  
- Refine and tune model hyperparameters to improve predictive performance.  
- Export a final model ready for production, including a decision threshold optimized for business needs.

---

## Repository Structure
BANK-CHURN-PREDICTION/
│
├── data/
│ ├── raw/
│ │     ├── Modelo_Clasificacion_Dataset.csv
│ │     ├── Modelo_Clasificacion_Diccionario_de_Datos.xlsx
│ └── processed/
│       └── churn_prediction_dataset.parquet
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_model_comparison.ipynb
│ ├── 03_model_refinement.ipynb
│ └── 04_final_model.ipynb
│
├── final_model/
│ └── final_xgb_model_with_threshold.pkl
│
└── README.md

## Notebooks

### `01_data_exploration.ipynb`
We establish the training dataset and perform foundational data quality and preprocessing steps:
- Load the raw CSV and standardize data types (including datetime and categorical fields).
- Map the target `clase_binaria` (BAJA → 1, CONTINUA → 0) and inspect its distribution to understand class imbalance.
- Diagnose and handle missing values and duplicated records.
- Apply one-hot encoding to categorical variables and consolidate the final processed DataFrame.
- Export the cleaned dataset to `data/processed/churn_prediction_dataset.parquet`.

### `02_model_comparison.ipynb`
In this stage, we evaluate multiple machine learning models.  
- Baseline models such as Logistic Regression and Decision Trees.  
- Ensemble methods like Random Forest and Gradient Boosting.  
- Comparison of models using metrics such as accuracy, precision, recall, and AUC.  
- Selection of promising candidates for refinement.  

### `03_model_refinement.ipynb`
We refine the best-performing model (XGBoost) through hyperparameter tuning.  
- Randomized search and cross-validation for parameter optimization.  
- Adjustment of class imbalance strategies (e.g., weighting, resampling).  
- Analysis of feature importance to interpret model behavior.  

### `04_final_model.ipynb`
The final notebook consolidates the modeling process into a production-ready artifact.  
- Training the best model (XGBoost) on the full dataset.  
- Defining an optimized decision threshold to balance false positives and false negatives.  
- Exporting the trained model as a `.pkl` file for deployment.  

---

## Data

### `Modelo_Clasificacion_Dataset.csv`
The original raw dataset containing customer-level information, including:  
- Demographics (age, tenure, VIP status).  
- Banking products (loans, fixed deposits, investment funds).  
- Transactional behavior (credit card usage, payments, transfers).  
- Target variable: `clase_binaria` (1 = churn, 0 = active).  

### `Modelo_Clasificacion_Diccionario_de_Datos.xlsx`
A detailed data dictionary describing the meaning of each field, its unit of measure, and a short explanation.  

### `churn_prediction_dataset.parquet`
A processed version of the original dataset optimized for model training.  

---

## Final Model
- **File**: `final_xgb_model_with_threshold.pkl`  
- **Description**: Trained XGBoost classifier with manually tuned hyperparameters. The model includes a defined decision threshold aligned with business needs.  

This model can be directly loaded into production pipelines to generate churn predictions for new customers.  

---

## Results
- Identified XGBoost as the most effective model for predicting churn.  
- Achieved strong performance metrics with an optimized threshold.  
- Produced interpretable feature importance to understand key drivers of churn.  

---

## Next Steps
- Deploy the model in a real-time scoring environment.  
- Integrate predictions into customer retention strategies.  
- Continuously monitor performance and retrain the model with updated data.  

---

## Requirements
- Python 3.9+  
- Jupyter Notebook  
- Core libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`

Install dependencies with:  
```bash
pip install -r requirements.txt