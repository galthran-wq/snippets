# %%
from __future__ import annotations
from typing import Literal
import os

import pandas as pd
import datasets
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from fire import Fire

from sequence_classification import compute_metrics

boost_params = {
    "mod": [GradientBoostingRegressor()],
    "mod__loss": ["squared_error"],
    "mod__max_depth": [6],
    "mod__learning_rate": [0.01],
    "mod__n_estimators": [100, 200, 500, 1000],
    "mod__subsample": [0, 0.5],
    'enc__max_features': [100, 500, 1000, 2000, 5000, 10000],
    'enc__stop_words': [None, "english"],
    'enc__ngram_range': [(1, 1), (1, 2), (2, 2)],
}

forest_params ={
    "mod": [RandomForestRegressor()],  
    "mod__max_depth": [10, 15],
    "mod__min_samples_split": [10, 20, 30],
    "mod__n_estimators": [100, 200, 500, 1000],
    "mod__max_features": ["sqrt"],
    "mod__random_state": [42],
    'enc__max_features': [100, 500, 1000, 2000, 5000, 10000],
    'enc__stop_words': [None, "english"],
    'enc__ngram_range': [(1, 1), (1, 2), (2, 2)],
}

logreg_params = {
    "mod": [LogisticRegression()],  
    "mod__C": [0.1, 0.3, 0.5, 0.7, 1.0, 3],
    "mod__l1_ratio": [0, 0.3, 0.5, 0.7, 1],
    "mod__penalty": ['l1', 'l2', 'elasticnet'],
    "mod__max_iter": [1000],
    "mod__solver": ["saga"],
    'enc__max_features': [100, 500, 1000, 2000, 5000, 10000],
    'enc__stop_words': [None, "english"],
    'enc__ngram_range': [(1, 1), (1, 2), (2, 2)],
}


def main(
    *,
    dataset_checkpoint: str,
    clf_head: Literal["logreg", "gb", "rf"] = "logreg",
    text_col: str = "text",
    label_col: str = "label",
    train_subset: str = "train",
    val_subset: str | None = None,
    test_subset: str | None = None,
    report_to=None,
    n_iter=100,
):
    params = locals()
    run_name = f"{clf_head}_{dataset_checkpoint}".replace("/", "-")
    if report_to == "wandb":
        import wandb
        wandb.init(name=run_name)
    if os.path.exists(dataset_checkpoint):
        dataset= datasets.load_from_disk(dataset_checkpoint)
    else:
        dataset = load_dataset(dataset_checkpoint)

    if val_subset is None:
        val_subset = "val"
        splitted_train = dataset[train_subset].train_test_split(test_size=0.2, shuffle=True, seed=42)
        dataset[val_subset] = splitted_train['test']
        dataset[train_subset] = splitted_train['train']

    train_df = dataset[train_subset].to_pandas()
    val_df = dataset[val_subset].to_pandas()
    df = pd.concat([train_df, val_df], axis=0)


    clf = Pipeline([
        ("enc", TfidfVectorizer(max_features=1000, stop_words="english", dtype="float32")),
        ("mod", LogisticRegression())
    ])

    ps = PredefinedSplit(test_fold=[-1] * len(train_df) + [0] * len(val_df))

    if clf_head == "logreg":
        params = logreg_params
    elif clf_head == "gb":
        params = boost_params
    elif clf_head == "rf":
        params = forest_params

    grid = RandomizedSearchCV(
        estimator=clf,
        cv=ps,
        param_distributions=params,
        scoring="accuracy",
        refit=True,
        verbose=3,
        n_iter=n_iter,
    )
    X = df[text_col]
    y = df[label_col]
    grid.fit(X=X, y=y)
    if report_to == "wandb":
        import wandb
        params_to_log = grid.best_params_
        params_to_log.pop("mod")
        wandb.log(params_to_log)


    if test_subset is not None:
        test_df = dataset[test_subset].to_pandas()
        X_test = test_df[text_col]
        y_test = test_df[label_col]
        test_metrics = compute_metrics((grid.predict_proba(X_test), y_test))
        print("Test metrics: ", test_metrics)
        if report_to == "wandb":
            import wandb
            for k, v in params.items():
                wandb.config[k] = v
            wandb.log(test_metrics)


if __name__ == "__main__":
    Fire(main)
