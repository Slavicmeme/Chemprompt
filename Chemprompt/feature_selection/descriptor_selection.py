import pandas as pd
import numpy as np
import inspect
from collections import OrderedDict
from typing import Sequence, Callable

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, Crippen, Lipinski, rdMolDescriptors,
    Fragments, QED,
)

from sklearn.feature_selection import SelectKBest, f_regression, r_regression, mutual_info_regression


class descriptor_selection:
    """
    Perform feature selection on pre-computed descriptor matrices.
    Designed to:
    - Receive already-filtered descriptor DataFrame (e.g., from LLMModel)
    - Fit feature selector on training data only (to prevent leakage)
    - Transform any dataset (train/test) to same selected feature subset
    """
    def __init__(self, y=None):
        self.y = y
        self.selector = None
        self.selected_columns = None
        self.selected_indices = None
        self.selected_mask = None

    def fit_feature_selector(self, X_train: pd.DataFrame, y_train, k=50, method="f_regression"):
        """Fit feature selector using only training data"""
        method_func = self._get_method(method)
        selector = SelectKBest(score_func=method_func, k=min(k, X_train.shape[1]))
        selector.fit(X_train, y_train)

        self.selector = selector
        self.selected_columns = X_train.columns[selector.get_support()]
        self.selected_mask = selector.get_support()  # True/False mask
        self.selected_indices = np.where(selector.get_support())[0].tolist()  # index list

        print(f"[Feature Selection] {len(self.selected_columns)} features selected using {method}.")
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted selector to another dataset (train/test)"""
        if self.selected_columns is None:
            raise RuntimeError("Feature selector not fitted. Run fit_feature_selector() first.")
        missing = [c for c in self.selected_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Input missing {len(missing)} columns: {missing[:5]}")
        return X[self.selected_columns]

    @staticmethod
    def _get_method(name: str):
        methods = {
            "f_regression": f_regression,
            "r_regression": r_regression,
            "mutual_info": mutual_info_regression,
        }
        if name not in methods:
            raise ValueError(f"Unknown feature selection method: {name}")
        return methods[name]