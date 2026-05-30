# DevTown-Project

A machine learning project that loads a customer retail dataset, preprocesses the data, trains three classifiers, evaluates their performance, and compares the results with visualization.

## Project Overview

- Features used: `Quantity`, `UnitPrice`, `Country`
- Models compared:
  - Logistic Regression
  - Decision Tree Classifier
  - K-Nearest Neighbors (KNN)
- Evaluation metrics:
  - Accuracy score
  - Confusion matrix
- Visualization:
  - Customer distribution by country
  - Model accuracy comparison
  - Confusion matrices for each model

## Files

- `customer_ml_project.py` - implementation of the ML workflow
- `requirements.txt` - Python dependencies
- `customer_retail csv file.csv` - provided dataset used by the project

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the project:

```bash
python customer_ml_project.py
```

3. View output files:

- `customer_distribution.png`
- `model_accuracy_comparison.png`
- `confusion_matrices.png`

## Notes

- If `data/customer_retail.csv` is not present, the script generates a sample dataset and saves it automatically.
- For your own dataset, ensure it contains the `Quantity`, `UnitPrice`, and `Country` columns.
