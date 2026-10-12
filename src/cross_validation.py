"""More robust comparison: repeated stratified 5-fold cross-validation on all 400 rows.

The single 100-row test set in the notebook can only resolve differences of 1 percentage point per
sample, so this script reports mean +/- std over 5 folds x 10 repeats. Scaling is fitted inside each
training fold (no leakage). Rows are sorted first so the result does not depend on the CSV's row order.

Run from the repository root:  python src/cross_validation.py
Writes results/cv_results.csv and results/cv_results.md.
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATURES, RANDOM_STATE, RESULTS_DIR, TARGET, get_models, load_data


def run_cv(df, n_splits=5, n_repeats=10):
    df = df.sort_values(FEATURES + [TARGET]).reset_index(drop=True)  # order-independent folds
    X, y = df[FEATURES].values, df[TARGET].values
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    models = {"Majority-class baseline": DummyClassifier(strategy="most_frequent")}
    models.update(get_models())

    rows = []
    for name, model in models.items():
        pipe = make_pipeline(StandardScaler(), model)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        row = {"Model": name}
        for m in scoring:
            vals = scores[f"test_{m}"]
            row[f"{m}_mean"], row[f"{m}_std"] = vals.mean(), vals.std()
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df):
    lines = ["| Model | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        lines.append(f"| {r.Model} | {r.accuracy_mean*100:.1f}% +/- {r.accuracy_std*100:.1f} | "
                     f"{r.precision_mean:.3f} +/- {r.precision_std:.3f} | {r.recall_mean:.3f} +/- {r.recall_std:.3f} | "
                     f"{r.f1_mean:.3f} +/- {r.f1_std:.3f} |")
    return "\n".join(lines)


def main():
    df = load_data()
    res = run_cv(df)
    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    (RESULTS_DIR / "cv_results.md").write_text(to_markdown(res) + "\n")
    print(to_markdown(res))
    print(f"\nSaved to {RESULTS_DIR}/cv_results.csv and cv_results.md")


if __name__ == "__main__":
    main()
