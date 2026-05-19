"""
Projekt: Implementacja drzewa decyzyjnego dla atrybutów ciągłych
Autorzy: Artur Brogowicz i Paweł Dobrosielski
Plik: my_tree.py (Implementacja logiki klasyfikatora)
"""

import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, method='ig', k=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.method = method
        self.k = k
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def get_max_depth(self):
        """Zwraca faktyczną maksymalną głębokość wytrenowanego drzewa."""

        def _depth(node):
            if node is None or node.is_leaf_node():
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))

        if self.root is None:
            return 0
        return _depth(self.root)

    def get_node_count(self):
        """Zwraca całkowitą liczbę węzłów w drzewie (w tym liście)."""

        def _count(node):
            if node is None:
                return 0
            if node.is_leaf_node():
                return 1
            return 1 + _count(node.left) + _count(node.right)

        if self.root is None:
            return 0
        return _count(self.root)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Kryteria stopu
        if (depth >= self.max_depth or n_labels <= 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = range(n_features)
        best_feat, best_thresh, best_ig = self._best_split(X, y, feat_idxs)

        # Jeżeli Zysk Informacyjny <= 0, węzeł staje się liściem
        if best_ig <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # Zabezpieczenie przed pustymi podziałami
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]

            if self.method == 'eq_width':
                thresholds = self._get_thresholds_eq_width(X_column)
            elif self.method == 'eq_freq':
                thresholds = self._get_thresholds_eq_freq(X_column)
            elif self.method == 'ig':
                thresholds = self._get_thresholds_ig(X_column, y)
            else:
                raise ValueError("Nieznana metoda dyskretyzacji.")

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh, best_gain

    def _get_thresholds_eq_width(self, X_column):
        x_min, x_max = np.min(X_column), np.max(X_column)
        if x_min == x_max or self.k is None or self.k <= 1:
            return []
        W = (x_max - x_min) / self.k
        return [x_min + i * W for i in range(1, self.k)]

    def _get_thresholds_eq_freq(self, X_column):
        if self.k is None or self.k <= 1:
            return []
        sorted_x = np.sort(X_column)
        n_total = len(sorted_x)
        if n_total < self.k:
            return []

        thresholds = set()
        n_bin = n_total / self.k

        for i in range(1, self.k):
            idx = int(i * n_bin)
            if idx < n_total and idx - 1 >= 0:
                t = (sorted_x[idx - 1] + sorted_x[idx]) / 2.0
                thresholds.add(t)
        return list(thresholds)

    def _get_thresholds_ig(self, X_column, y):
        sorted_indices = np.argsort(X_column)
        sorted_x = X_column[sorted_indices]
        sorted_y = y[sorted_indices]

        thresholds = set()
        for i in range(len(sorted_x) - 1):
            if sorted_y[i] != sorted_y[i + 1] and sorted_x[i] != sorted_x[i + 1]:
                t = (sorted_x[i] + sorted_x[i + 1]) / 2.0
                thresholds.add(t)

        return list(thresholds)

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        if len(y) == 0:
            return None
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)