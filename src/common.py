"""Shared helpers: data loading and the 7 classifiers with the settings used in the project."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Social_Network_Ads.csv"
RESULTS_DIR = ROOT / "results"
FEATURES = ["Age", "EstimatedSalary"]
TARGET = "Purchased"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Dataset not found at {path}. See data/README.md for where to place Social_Network_Ads.csv.")
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing columns {missing}; expected {FEATURES + [TARGET]}.")
    return df


def get_models(random_state=RANDOM_STATE):
    """The 7 classifiers, with the same hyperparameters as the notebook."""
    return {
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
        "SVM (linear)": SVC(kernel="linear", random_state=random_state),
        "Kernel SVM (RBF)": SVC(kernel="rbf", random_state=random_state),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=random_state),
        "Random Forest (10 trees)": RandomForestClassifier(n_estimators=10, criterion="entropy",
                                                           random_state=random_state),
    }
