import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from Chemprompt.utils.data_splitter import DataSplitter
from Chemprompt.utils.metrics import get_metric_function
from Chemprompt.models.base_model import BaseModel

class ScikitLearnModel(BaseModel):
    def __init__(self, model_type="classification", fold=5, metric_names=None, save_dir=None, max_iter = 500):
        """
        Initialize Scikit-learn model with cross-validation.

        Args:
            model_type (str): 'classification', 'multilabel', or 'regression'
            fold (int): Number of folds for cross-validation (default: 5)
            metric_names (list, optional): List of metric names (e.g., ['f1_micro', 'accuracy'])
            save_dir (str, optional): Directory to save results (default: ./results)
        """
        super(ScikitLearnModel, self).__init__()
        self.model_type = model_type
        self.fold = fold
        self.max_iter = max_iter
        
        self.model = self._default_model(model_type)  # Set default model
        self.metric_names = metric_names or self._default_metrics()
        self.save_dir = save_dir or f"./results/{self.model_type}"
        os.makedirs(self.save_dir, exist_ok=True)

    def _default_model(self, model_type):
        """
        Set default model based on model type.
        """
        if model_type == "classification":
            return LogisticRegression(max_iter=self.max_iter, random_state=2025)
        elif model_type == "multilabel":
            return MultiOutputClassifier(LogisticRegression(max_iter=self.max_iter, solver="lbfgs", penalty="l2", random_state=2025))
        elif model_type == "regression":
            return Ridge(max_iter = self.max_iter, random_state=2025)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _default_metrics(self):
        """
        Define default evaluation metrics based on model type.
        """
        if self.model_type == "classification":
            return ["accuracy"]
        elif self.model_type == "multilabel":
            return ["f1_micro", "f1_macro"]
        elif self.model_type == "regression":
            #return ["rmse", "r2"]
            return ["rmse", "r2", "pearson", "spearman"]
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit_and_evaluate(self, data_list, train_x, train_y, test_x, test_y):
        """
        Train on training set and evaluate on test set.
    
        Args:
            data_list (list): SMILES strings (test set)
            train_x (array-like): Training embeddings
            train_y (array-like): Training labels
            test_x (array-like): Test embeddings
            test_y (array-like): Test labels
    
        Returns:
            dict: Dictionary containing evaluation metrics (same as before)
        """
        metric_results = {metric: [] for metric in self.metric_names}
        
        # ===== Convert to numpy arrays =====
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
    
        # ===== Detect multilabel automatically =====
        if len(train_y.shape) == 2 and train_y.shape[1] > 1 and self.model_type != "multilabel":
            self.model_type = "multilabel"
            self.model = self._default_model(self.model_type)
    
        # ===== Initialize a fresh model =====
        self.model = self._default_model(self.model_type)
    
        # ===== Train =====
        self.model.fit(train_x, train_y)
    
        # ===== Predict =====
        y_pred = self.model.predict(test_x)
    
        # ===== Z-score normalization =====
        mu, sigma = float(np.mean(test_y)), float(np.std(test_y))
        if sigma > 0:
            y_test_z = (test_y - mu) / sigma
            y_pred_z = (y_pred - mu) / sigma
        else:
            y_test_z = test_y
            y_pred_z = y_pred
    
        # Calculate metrics using Z-scored values
        for metric_name in self.metric_names:
            metric_func = get_metric_function(metric_name)
            metric_score = metric_func(y_test_z, y_pred_z)
            metric_results[metric_name].append(metric_score)
    
        # ===== Save predictions =====
        all_predictions = []
        for idx, pred in enumerate(y_pred):
            try:
                prediction = list(pred)
            except TypeError:
                prediction = pred
            all_predictions.append({
                "SMILES": data_list[idx],
                "GroundTruth": test_y[idx],
                "Prediction": prediction
            })
    
        # Save to disk
        self._save_predictions(all_predictions)
        self._save_combined_results({k: [v] for k, v in metric_results.items()})
    
        # ===== Print summary =====
        print("\n[RESULTS]")
        for k, v in metric_results.items():
            print(f"{k}: {v[0]:.3f}")
            
        return metric_results

    def fit_and_evaluate_fold(self, data_list, x, y):
        """
        Perform cross-validation training and evaluation with fold-wise model reinitialization
        and Z-score normalization for evaluation metrics.
    
        Args:
            data_list (list): SMILES strings
            x (array-like): Embedding vectors (numerical features)
            y (array-like): Target labels
        """
        x = np.array(x)
        y = np.array(y)
    
        # Automatically detect multilabel classification if y is 2D with multiple outputs
        if len(y.shape) == 2 and y.shape[1] > 1 and self.model_type != "multilabel":
            self.model_type = "multilabel"
            self.model = self._default_model(self.model_type)
    
        # Select the appropriate cross-validation strategy
        cv = StratifiedKFold(n_splits=self.fold, shuffle=True, random_state=42) if self.model_type == "classification" else KFold(n_splits=self.fold, shuffle=True, random_state=42)
    
        all_predictions = []
        metric_results = {metric: [] for metric in self.metric_names}
    
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
    
            # Reinitialize a fresh model for each fold
            self.model = self._default_model(self.model_type)
    
            # Train the model
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
    
            # Apply Z-score normalization (only for evaluation)
            y_test_mean = np.mean(y_test)
            y_test_std = np.std(y_test)
            
            if y_test_std == 0:  # Handle cases where standard deviation is zero
                y_test_z = y_test
                y_pred_z = y_pred
            else:
                y_test_z = (y_test - y_test_mean) / y_test_std
                y_pred_z = (y_pred - y_test_mean) / y_test_std
    
            # Calculate metrics using Z-scored values
            for metric_name in self.metric_names:
                metric_func = get_metric_function(metric_name)
                metric_score = metric_func(y_test_z, y_pred_z)
                metric_results[metric_name].append(metric_score)
    
            # Save predictions
            for idx, pred in zip(test_idx, y_pred):
                ground_truth = y[idx]
                try:
                    prediction = list(pred)
                except TypeError:
                    prediction = pred
                all_predictions.append({
                    "Fold": fold_idx,
                    "SMILES": data_list[idx],
                    "GroundTruth": ground_truth,
                    "Prediction": prediction
                })

        # Save all fold predictions
        self._save_predictions(all_predictions)
        # Save combined metric results
        self._save_combined_results(metric_results)
    
        # Summarize and return the final results
        summary_results = {}
        for metric_name in self.metric_names:
            scores = metric_results[metric_name]
            summary_results[f"{metric_name}_mean"] = np.mean(scores)
            summary_results[f"{metric_name}_std"] = np.std(scores)
    
        return summary_results

    def _save_predictions(self, predictions):
        if self.save_dir is None:
            save_path = os.path.join(f"./results/", f"{self.model_type}_Predictions.csv")
        else:
            save_path = os.path.join(self.save_dir, f"{self.model_type}_Predictions.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        prediction_df = pd.DataFrame(predictions, columns=["Fold", "SMILES", "GroundTruth", "Prediction"])
        prediction_df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")

    def _save_combined_results(self, metric_results):
        """
        Save mean and standard deviation of each metric to a CSV file.
        """
        combined_results = {
            "Metric": [],
            "Mean": [],
            "Standard Deviation": []
        }

        for metric_name in self.metric_names:
            scores = metric_results[metric_name]
            combined_results["Metric"].append(metric_name)
            combined_results["Mean"].append(np.mean(scores))
            combined_results["Standard Deviation"].append(np.std(scores))

        combined_df = pd.DataFrame(combined_results)
        combined_results_path = os.path.join(self.save_dir, f"{self.model_type}_CombinedResults.csv")
        combined_df.to_csv(combined_results_path, index=False)
        print(f"Combined results saved to {combined_results_path}")
        
    def predict(self, x, y_true=None, smiles_list=None):
        """
        Predict using the trained model and optionally save results if y_true is provided.
    
        Args:
            x (array-like): Input feature vectors
            y_true (array-like, optional): True target values (for saving GroundTruth)
            smiles_list (list, optional): Optional SMILES strings (to include in saved CSV)
    
        Returns:
            array-like: Model predictions
        """
        y_pred = self.model.predict(x)
    
        # Save predictions if ground truth is available
        if y_true is not None:
            save_path = os.path.join(self.save_dir, f"{self.model_type}_Predictions.csv")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
            records = []
            for i, pred in enumerate(y_pred):
                try:
                    pred_val = list(pred)
                except TypeError:
                    pred_val = pred
                record = {
                    "GroundTruth": y_true[i],
                    "Prediction": pred_val
                }
                if smiles_list:
                    record["SMILES"] = smiles_list[i]
                records.append(record)
    
            df = pd.DataFrame(records)
            df.to_csv(save_path, index=False)
            print(f"[INFO] Test predictions saved to {save_path}")
    
        return y_pred