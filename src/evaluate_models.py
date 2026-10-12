"""Train the 7 classifiers on the 75/25 split used in the notebook and report test-set metrics.

Run from the repository root:  python src/evaluate_models.py
Writes results/metrics.csv and results/metrics.md.
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import FEATURES, RANDOM_STATE, RESULTS_DIR, TARGET, get_models, load_data


def evaluate(X_train, y_train, X_test, y_test):
    """Scale on the training set only, fit every model, and return a metrics table."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {"Majority-class baseline": DummyClassifier(strategy="most_frequent")}
    models.update(get_models())

    rows = []
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        })
    return pd.DataFrame(rows)


def to_markdown(df):
    lines = ["| Model | Accuracy | Precision | Recall | F1 | TN | FP | FN | TP |", "|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        lines.append(f"| {r.Model} | {r.Accuracy:.2f} | {r.Precision:.3f} | {r.Recall:.3f} | {r.F1:.3f} | {r.TN} | {r.FP} | {r.FN} | {r.TP} |")
    return "\n".join(lines)


def main():
    df = load_data()
    X, y = df[FEATURES].values, df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    print(f"train: {len(X_train)} | test: {len(X_test)} | positives in test: {int(y_test.sum())}")

    results = evaluate(X_train, y_train, X_test, y_test)
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    (RESULTS_DIR / "metrics.md").write_text(to_markdown(results) + "\n")
    print(to_markdown(results))
    print(f"\nSaved to {RESULTS_DIR}/metrics.csv and metrics.md")


if __name__ == "__main__":
    main()
