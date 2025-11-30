# Python Project – Preprocessing and Boosting Techniques

## Overview

This repository contains a Machine Learning project focused on **data preprocessing** and the implementation of **supervised boosting classification algorithms**.  
The work was developed as part of coursework in **Machine Learning / Data Analysis**.

The project implements:
- A complete preprocessing pipeline for raw data  
- Handling of missing values and constant columns  
- Data type conversion and encoding of categorical features  
- Visualization of class distributions  
- PCA (Principal Component Analysis) for dimensionality reduction  
- Multiple boosting algorithms for classification  
- Cross-validation and nested CV for model evaluation  

---

## Repository Structure

```
main_repository/
│
├── dataset.tsv
├── description.pdf
├── implementation.ipynb
├── LICENSE
└── README.md
```

---

## Introduction

Preprocessing is a critical step in Machine Learning pipelines.  
Working with unprocessed raw data often leads to:

- Reduced generalization  
- Lower accuracy  
- Longer training times  

For this reason, the project applies a detailed preprocessing stage followed by the implementation of **boosting algorithms**, which combine multiple weak learners to create a strong classifier.

The dataset used in this work originates from **eye-tracking measurements**, consisting of numerical and categorical variables.

---

## Preprocessing Pipeline

The preprocessing workflow consists of the following steps:

### 1. Import required libraries
The project uses:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn.preprocessing.LabelEncoder`
- `sklearn.decomposition.PCA`

### 2. Load and clean dataset
- Convert comma decimal values to dot decimal format  
- Remove formatting inconsistencies  

### 3. Initial dataset inspection
```python
df.shape
df.info(verbose=True)
df.head()
```
## Preprocessing Pipeline (Detailed)

### 4. Data Type Conversion
- Convert all string/object columns into numerical values  
- Remove columns that cannot be converted due to incompatible formats  

### 5. Missing Values and Zero Columns
- Drop columns that contain missing values  
- Remove columns composed entirely of zeros  

### 6. Constant-Value Feature Removal
- Eliminate columns in which all entries share the same value  

### 7. Categorical Encoding
- Encode the target variable using `LabelEncoder`  
- Encode categorical fields inside the feature matrix `X`  

### 8. Conversion to NumPy Arrays
- Convert the feature matrix to:  
  **`X` → NumPy array of features**  
- Convert the target to:  
  **`y` → NumPy array of labels**  

### 9. Class Distribution Visualization
- Generate a plot showing the frequency of each class  
- Useful for detecting class imbalance  

### 10. Dimensionality Reduction (PCA)
- Apply **Principal Component Analysis**  
- Highlights principal directions of variance  
- Helps with visualization and potential model performance  

---

## Boosting Algorithms

The project evaluates the following **supervised ensemble models**:

- **AdaBoostClassifier**  
- **GradientBoostingClassifier**  
- **LGBMClassifier**  
- **CatBoostClassifier**

---

## Advantages of Boosting
- Sequential models iteratively correct previous errors  
- Very strong predictive performance  
- Works well even with minimal parameter tuning  

## Disadvantages
- Prone to overfitting  
- Can require long training times  
- Sensitive to noise and mislabeled data  

---

## Application Domains

Boosting methods are widely used in:

- Healthcare  
- Information technology  
- Image retrieval  
- Financial modeling  
- Web classification  

---

## Hyperparameters Tested

The following values were evaluated:

- **learning_rate:**  
  `0.001`, `0.01`, `0.1`, `1.0`  
- **n_estimators:**  
  `50`, `100`, `500`, `1000`  

Each model was tested using:

- **Train-test split**  
- **Cross-validation (CV)**  
- **Nested cross-validation (Nested CV)**  

---

## Summary of Results

- **GradientBoostingClassifier** achieved the best accuracy overall  
- Train-test split produced the most stable and slightly higher performance  
- Accuracy values typically ranged between **60% and 65%**  

---

## Implementation

### Main Script (`main.py`)

Responsible for:

- Loading the dataset  
- Executing the preprocessing pipeline  
- Training all boosting models  
- Evaluating performance  
- Saving plots and results  

### Support Modules

- **`preprocessing.py`** — performs all cleaning and feature processing  
- **`boosting_models.py`** — defines, configures, and trains ML models  
- **`evaluation.py`** — metrics, cross-validation, nested CV logic  
- **`visualization.py`** — utilities for plotting PCA, class distribution, etc.  

---

## Usage

### Install required dependencies:
```bash
pip install -r requirements.txt
```
### Run the main script:
```bash
python main.py
```
### Output:
Plots and results will be saved inside the `images/` directory.

---

## Notes

- The dataset originates from an **eye-tracking measurement system**.  
- Proper preprocessing is essential due to the presence of:  
  - inconsistent formatting  
  - redundant or uninformative variables  
  - mixed data types  
- Boosting algorithms were chosen for their **high predictive power** and **low bias characteristics**.
