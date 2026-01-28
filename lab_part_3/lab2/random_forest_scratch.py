import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Suppress numpy warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check termination conditions
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._variance_reduction(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _variance_reduction(self, y, X_column, threshold):
        parent_var = np.var(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = np.var(y[left_idxs]), np.var(y[right_idxs])
        child_var = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_var - child_var

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _calculate_leaf_value(self, y):
        if len(y) == 0:
            return 0 # Fallback for empty leaf
        return np.mean(y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestRegressorScratch:
    def __init__(self, n_estimators=10, min_samples_split=2, max_depth=100, n_features=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Handle default case where predictions might contain NaN due to empty leaves
        if np.isnan(predictions).any():
            predictions = np.nan_to_num(predictions)
            
        tree_avg = np.mean(predictions, axis=0)
        return tree_avg

if __name__ == "__main__":
    try:
        df = pd.read_csv('data.csv')
        df = pd.get_dummies(df)
        labels = np.array(df['actual'])
        features = df.drop('actual', axis=1)
        features = np.array(features)
        
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.25, random_state=42
        )
        
        print("Dataset loaded.")
        print(f"Train shape: {train_features.shape}")
        print(f"Test shape: {test_features.shape}")

        rf = RandomForestRegressorScratch(
            n_estimators=10, 
            max_depth=5,
            min_samples_split=2,
            n_features=int(np.sqrt(train_features.shape[1])) 
        )
        
        print("Training model...")
        rf.fit(train_features, train_labels)
        print("Done.")

        predictions = rf.predict(test_features)
        
        # Simple check for NaN in final predictions
        if np.isnan(predictions).any():
             print("Warning: Predictions contain NaN values. Filling with mean.")
             predictions = np.nan_to_num(predictions, nan=np.mean(train_labels))

        mae = mean_absolute_error(test_labels, predictions)
        print(f"MAE: {mae:.2f}")
        
        errors = abs(predictions - test_labels)
        mape = 100 * (errors / test_labels)
        accuracy = 100 - np.mean(mape)
        print(f"Accuracy: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")
