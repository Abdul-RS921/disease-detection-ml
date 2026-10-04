"""Train and evaluate models that detect whether a breast tumour is malignant or benign.

Data: the Breast Cancer Wisconsin (Diagnostic) dataset that ships with scikit-learn
(569 patients, 30 measurements of cell nuclei from a biopsy image).

Run:  python train.py
Creates: model.joblib, metrics.json, confusion_matrix.png
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT = Path(__file__).parent
RANDOM_STATE = 42


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target
    # In this dataset 0 = malignant (cancer), 1 = benign
    print(f"Patients: {X.shape[0]}, features: {X.shape[1]}")
    print(f"Malignant: {(y == 0).sum()}, benign: {(y == 1).sum()}\n")

    # Keep 20% of the patients aside. The model never sees them while it is being chosen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    candidates = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        # CalibratedClassifierCV gives the SVM probabilities (SVC(probability=True) is deprecated)
        "Support Vector Machine": make_pipeline(
            StandardScaler(), CalibratedClassifierCV(SVC(), cv=3)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }

    # Choose the best model using only the training data (5-fold cross-validation).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}
    print("Cross-validation on the training data (recall of the malignant class):")
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall_macro")
        cv_scores[name] = float(scores.mean())
        print(f"  {name:<24} {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(cv_scores, key=cv_scores.get)
    best = candidates[best_name]
    best.fit(X_train, y_train)
    print(f"\nBest model: {best_name}")

    # Final, honest evaluation on patients the model has never seen.
    y_pred = best.predict(X_test)
    # "Positive" = malignant (label 0), because missing a cancer is the costly mistake.
    metrics = {
        "model": best_name,
        "test_patients": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "malignant_precision": float(precision_score(y_test, y_pred, pos_label=0)),
        "malignant_recall": float(recall_score(y_test, y_pred, pos_label=0)),
        "malignant_f1": float(f1_score(y_test, y_pred, pos_label=0)),
        "confusion_matrix [[TN,FP],[FN,TP]] for malignant": None,
        "cross_validation_recall_macro": cv_scores,
    }
    tn_fp_fn_tp = confusion_matrix(y_test, y_pred, labels=[1, 0]).tolist()
    metrics["confusion_matrix [[TN,FP],[FN,TP]] for malignant"] = tn_fp_fn_tp

    print("\nResults on the held-out test patients:")
    print(f"  Accuracy:            {metrics['accuracy']:.3f}")
    print(f"  Malignant precision: {metrics['malignant_precision']:.3f}")
    print(f"  Malignant recall:    {metrics['malignant_recall']:.3f}  (share of real cancers that were caught)")
    print(f"  Malignant F1:        {metrics['malignant_f1']:.3f}")
    print(f"  Confusion matrix (rows = real, columns = predicted; benign first): {tn_fp_fn_tp}")

    # Save the model, the numbers and a picture of the confusion matrix.
    joblib.dump({"model": best, "feature_names": list(data.feature_names),
                 "class_names": {0: "malignant", 1: "benign"}}, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["malignant", "benign"], ax=ax, cmap="Blues")
    ax.set_title(f"{best_name}\n(test patients)")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    print("\nSaved: model.joblib, metrics.json, confusion_matrix.png")


if __name__ == "__main__":
    main()
