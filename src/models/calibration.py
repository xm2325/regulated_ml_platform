from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


class PlattCalibratedClassifier(ClassifierMixin, BaseEstimator):
    """Serializable scikit-learn classifier that applies Platt scaling."""

    def __init__(self, base_model: Any, calibrator: LogisticRegression) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def fit(self, features: Any, target: np.ndarray) -> PlattCalibratedClassifier:
        self.base_model.fit(features, target)
        raw_probability = self.base_model.predict_proba(features)[:, 1]
        self.calibrator.fit(_logit(raw_probability), target)
        self.classes_ = np.array([0, 1])
        return self

    @classmethod
    def fit_calibrator(cls, base_model: Any, features: Any, target: np.ndarray) -> PlattCalibratedClassifier:
        raw_probability = base_model.predict_proba(features)[:, 1]
        calibrator = LogisticRegression(solver="lbfgs", random_state=42)
        calibrator.fit(_logit(raw_probability), target)
        return cls(base_model=base_model, calibrator=calibrator)

    def predict_proba(self, features: Any) -> np.ndarray:
        raw_probability = self.base_model.predict_proba(features)[:, 1]
        calibrated = self.calibrator.predict_proba(_logit(raw_probability))[:, 1]
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, features: Any) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)

    def raw_predict_proba(self, features: Any) -> np.ndarray:
        return self.base_model.predict_proba(features)

    @property
    def calibration_intercept(self) -> float:
        return float(self.calibrator.intercept_[0])

    @property
    def calibration_slope(self) -> float:
        return float(self.calibrator.coef_[0][0])
