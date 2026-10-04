# Disease Detection with Machine Learning (Python)

A machine learning project that predicts whether a breast tumour is **malignant** or **benign** from measurements of cell nuclei taken from a biopsy image.

> **This is a learning project. It is not a medical device and must never be used to diagnose anyone.**

## Data

The *Breast Cancer Wisconsin (Diagnostic)* dataset that comes with scikit-learn: 569 patients (212 malignant, 357 benign) and 30 numeric measurements each (radius, texture, perimeter, area, smoothness and so on).

## Method

1. Split the patients: 80% for training, 20% (114 patients) kept aside for the final test.
2. Compare three models with 5-fold cross-validation **on the training data only**: Logistic Regression, Support Vector Machine and Random Forest (the first two use feature scaling).
3. Pick the best model by recall of the malignant class, because missing a cancer is the costly mistake.
4. Evaluate once on the held-out test patients and save the model.

## Results (held-out test set of 114 patients)

| Model chosen | Accuracy | Malignant precision | Malignant recall | Malignant F1 |
|--------------|----------|---------------------|------------------|--------------|
| Logistic Regression | 98.2% | 97.6% | 97.6% | 97.6% |

Confusion matrix: 71 benign and 41 malignant patients were classified correctly, 1 malignant patient was missed and 1 benign patient was flagged as malignant (`confusion_matrix.png`, full numbers in `metrics.json`).

Cross-validation recall on the training data: Logistic Regression 0.975, Support Vector Machine 0.964, Random Forest 0.964.

These numbers come from a small, clean, well-known dataset, so they are higher than real clinical data would give. They show that the pipeline works, not that it is ready for real patients.

## Run it

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
python train.py                  # trains, evaluates and saves model.joblib, metrics.json, confusion_matrix.png
python predict.py --demo         # classifies 5 patients from the dataset
python predict.py --csv patients.csv   # classifies rows of your own CSV (the 30 feature columns)
```

## Files

| File | Purpose |
|------|---------|
| `train.py` | Data split, model comparison, evaluation, saving |
| `predict.py` | Uses the saved model on new rows |
| `requirements.txt` | Libraries |
| `metrics.json`, `confusion_matrix.png` | Results written by `train.py` |
