"""Use the trained model to classify tumours.

  python predict.py --demo              classify 5 patients from the dataset and show the real answer
  python predict.py --csv patients.csv  classify rows of a CSV file with the 30 feature columns
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer

MODEL_FILE = Path(__file__).parent / "model.joblib"
DISCLAIMER = "This is a learning project. It is NOT a medical device and must not be used to diagnose anyone."


def load_model():
    if not MODEL_FILE.exists():
        sys.exit("model.joblib not found. Run:  python train.py")
    return joblib.load(MODEL_FILE)


def classify(bundle, frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in bundle["feature_names"] if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing[:3]}{' ...' if len(missing) > 3 else ''}")
    X = frame[bundle["feature_names"]].to_numpy()  # the model was trained on plain numbers
    model = bundle["model"]
    labels = model.predict(X)
    prob_malignant = model.predict_proba(X)[:, 0]
    return pd.DataFrame({
        "prediction": [bundle["class_names"][int(v)] for v in labels],
        "chance_of_malignant": [f"{p:.1%}" for p in prob_malignant],
    }, index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Breast tumour classifier (demo project)")
    parser.add_argument("--demo", action="store_true", help="classify 5 patients from the dataset")
    parser.add_argument("--csv", help="CSV file with the 30 feature columns")
    args = parser.parse_args()
    bundle = load_model()

    if args.csv:
        result = classify(bundle, pd.read_csv(args.csv))
        print(result.to_string())
    else:
        data = load_breast_cancer(as_frame=True)
        sample = data.data.sample(5, random_state=7)
        result = classify(bundle, sample)
        result.insert(0, "real_answer", [bundle["class_names"][int(data.target[i])] for i in sample.index])
        print(result.to_string())
        print("\n(These patients may have been used for training, so this only shows how the tool is used;")
        print(" see metrics.json for the honest test results.)")
    print("\n" + DISCLAIMER)


if __name__ == "__main__":
    main()
