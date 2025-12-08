import os
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             balanced_accuracy_score, roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_sample_weight

import shap

from joblib import dump

class RandomForestPredictor:
    def __init__(self):
        # Parameters (RandomForest)
        self.rf_n_estimators = None
        self.rf_max_depth = None
        self.rf_random_seed = None

        # Parameters (Cross validation)
        self.cv_n_splits = None
        self.cv_n_repeats = None
        self.cv_random_seed = None

        # Input data
        self.feature_table = None
        self.labels = None
        self.classes = None

        # Transformed data
        self.X = None
        self.y = None
        self.label_map = None

        # results (cross validation)
        self.cv_models = []
        self.cv_scores = []
        self.cv_scores_mean = None
        self.cv_scores_std = None
        self.cv_roc_data = []

        # results (final model)
        self.model = None
        self.scores = None
        self.roc_data = None

    def load_data(self, feature_table, labels):
        """
        Load feature tables and labels.

        """

        # check index maping
        ci = set(feature_table.index).intersection(set(labels.index))
        ci = list(ci)
        self.feature_table = feature_table.loc[ci, :]
        self.labels = labels[ci]
        print(f'Number of matched samples: {len(ci)}')

        # feature conversion
        self.X = self.feature_table.values

        encoder = LabelEncoder()
        self.y = encoder.fit_transform(self.labels.values)
        self.classes = encoder.classes_
        print(f'Classes: {self.classes}')
        self.sample_weight = compute_sample_weight(
            class_weight='balanced', y=self.y)

    def set_random_forest_parameters(self, n_estimators=1000, max_depth=None, random_seed=2025):
        self.rf_n_estimators = n_estimators
        self.rf_max_depth = max_depth
        self.rf_random_seed = random_seed

    def set_cross_validation_parameters(self, n_splits=5, n_repeats=5, random_seed=2025):
        self.cv_n_splits = n_splits
        self.cv_n_repeats = n_repeats
        self.cv_random_seed = random_seed

    def tune_hyperparameters(self, n_estimators=[], max_depth=[]):
        """
        Seek for the best hyperparameters for n_estimateros and max_depth of RandomForest
        """
        pass

    def compute_metrics(self,
                        y_true,
                        y_pred,
                        y_prob,
                        average: str = "macro"  # "macro", "weighted", or "micro"
                        ):
        """
        Parameters
        ----------
        y_true : array‑like of shape (n_samples,)
            Ground‑truth class labels.
        y_pred : array‑like of shape (n_samples,)
            Predicted class labels.
        y_prob : array‑like
            • Binary case: shape (n_samples,) or (n_samples, 2) with positive‑class probs  
            • Multi‑class case: shape (n_samples, n_classes) with per‑class probs
        average : str
            Averaging method used for multi‑class precision / recall / F1.
            "macro" (default) treats all classes equally;
            "weighted" weighs by support;
            "micro" pools contributions of all classes.

        Returns
        -------
        dict
            Keys: accuracy, precision, recall, f1, auc

        <Genrated by ChatGPT - need further inspection> 
        """
        # Core classification metrics
        metrics = {
            "accuracy":  balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred,
                                         average=average,
                                         zero_division=0),
            "recall":    recall_score(y_true, y_pred,
                                      average=average,
                                      zero_division=0),  # sensitivity
            "f1":        f1_score(y_true, y_pred,
                                  average=average,
                                  zero_division=0)
        }

        # AUC needs careful handling
        n_classes = len(np.unique(y_true))
        roc_data = {}
        try:
            if n_classes == 2:
                # y_prob can be shape (n_samples,) or (n_samples, 2)
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]            # positive‑class column
                metrics["auc"] = roc_auc_score(y_true, y_prob)
                fpr, tpr, thr = roc_curve(y_true, y_prob[:, 1])
                roc_data["binary"] = (fpr, tpr, thr)
            else:
                # multi‑class AUC (one‑vs‑one is statistically robust)
                metrics["auc"] = roc_auc_score(
                    y_true, y_prob,
                    multi_class="ovo",
                    average=average
                )

                for c in range(n_classes):
                    fpr, tpr, thr = roc_curve(
                        (y_true == c).astype(int), y_prob[:, c]
                    )
                    roc_data[c] = (fpr, tpr, thr)

        except ValueError:
            # If probabilities aren’t available or shapes mismatch
            metrics["auc"] = np.nan

        return metrics, roc_data

    def cross_validate_models(self):
        # Define Repeated Stratified K-Fold
        cv = RepeatedStratifiedKFold(
            n_splits=self.cv_n_splits, n_repeats=self.cv_n_repeats, random_state=self.cv_random_seed)

        # Perform cross-validation
        models = []
        scores = []
        roc_data = []

        for train_idx, test_idx in cv.split(self.X, self.y):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            # Apply sample weights only to training data
            train_weights = self.sample_weight[train_idx]

            clf = RandomForestClassifier(
                max_depth=self.rf_max_depth, random_state=self.rf_random_seed, n_estimators=self.rf_n_estimators, n_jobs=-1)
            # Train with weights
            clf.fit(X_train, y_train, sample_weight=train_weights)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
            models.append(clf)

            # Measure prediction performance
            score, roc_datum = self.compute_metrics(y_test, y_pred, y_prob)
            scores.append(score)
            for k in roc_datum:
                roc_data.append(roc_datum[k])

        self.cv_scores = scores
        self.cv_roc_data = roc_data
        self.models = models

        self.cv_scores_mean = pd.DataFrame(scores).mean().to_dict()
        self.cv_scores_std = pd.DataFrame(scores).std().to_dict()

    def train_final_model(self):
        """
        Train the final model with entire data.
        """
        # Random forest model
        self.model = RandomForestClassifier(
            max_depth=self.rf_max_depth, random_state=self.rf_random_seed, n_estimators=self.rf_n_estimators)
        self.model.fit(self.X, self.y, sample_weight=self.sample_weight)

        self.y_pred = self.model.predict(self.X)
        self.y_prob = self.model.predict_proba(self.X)

        # performance measrues (not important)
        self.scores, self.roc_data = self.compute_metrics(
            self.y, self.y_pred, self.y_prob)
        
        # Shapley values (SHAP value)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X)
        self.shap_values = shap_values

    def save_results(self, output_dir):
        # create output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # data
        input_data = {'feature_table': self.feature_table,
                      'labels': self.labels,
                      'classes': self.classes,
                      'X': self.X, 
                      'y': self.y}
        dump(input_data, f'{output_dir}/input_data.joblib')

        # cross validation models
        dump(self.cv_models, f'{output_dir}/cv_models.joblib')

        # cross validation scores
        dump(self.cv_scores, f'{output_dir}/cv_scores.joblib')
        dump(self.cv_scores_mean, f'{output_dir}/cv_scores_mean.joblib')
        dump(self.cv_scores_std, f'{output_dir}/cv_scores_std.joblib')
        dump(self.cv_roc_data, f'{output_dir}/cv_roc_data.joblib')

        # final model
        dump(self.model, f'{output_dir}/model.joblib')

        # final model scores
        dump(self.scores, f'{output_dir}/scores.joblib')
        dump(self.roc_data, f'{output_dir}/roc_data.joblib')
        
        # shap values 
        dump(self.shap_values, f'{output_dir}/shap_values.joblib')
        

