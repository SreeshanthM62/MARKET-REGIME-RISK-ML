from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelBundle:
    name: str
    model: object


def build_models(random_state: int = 42) -> list[ModelBundle]:
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=350,
        max_depth=6,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    return [
        ModelBundle("Logistic Regression", logistic),
        ModelBundle("Random Forest", forest),
    ]


def fit_model(bundle: ModelBundle, X_train: pd.DataFrame, y_train: pd.Series) -> ModelBundle:
    bundle.model.fit(X_train, y_train)
    return bundle
