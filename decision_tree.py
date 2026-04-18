import numpy as np
from collections import Counter
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time


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
        """
        Inicjalizacja drzewa decyzyjnego.
        Parametry:
        - max_depth: maksymalna głębokość drzewa
        - min_samples_split: minimalna liczba próbek w węźle wymagana do podziału
        - method: metoda dyskretyzacji ('ig' - Information Gain, 'eq_width' - Equal-Width, 'eq_freq' - Equal-Frequency)
        - k: liczba przedziałów dla metod nienadzorowanych
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.method = method
        self.k = k
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Kryteria stopu
        # 1. Osiągnięto maksymalną głębokość drzewa
        # 2. Wszystkie obserwacje należą do tej samej klasy (czystość węzła)
        # 3. Liczba obserwacji jest mniejsza niż wymagane minimum
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = range(n_features)

        # Szukanie najlepszego podziału
        best_feat, best_thresh, best_ig = self._best_split(X, y, feat_idxs)

        # Kryterium stopu: Najlepszy możliwy zysk informacyjny <= 0
        if best_ig <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Faktyczny podział węzła na dwa poddrzewa
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]

            # Generowanie progów kandydujących w zależności od wybranej metody
            if self.method == 'eq_width':
                thresholds = self._get_thresholds_eq_width(X_column)
            elif self.method == 'eq_freq':
                thresholds = self._get_thresholds_eq_freq(X_column)
            elif self.method == 'ig':
                thresholds = self._get_thresholds_ig(X_column, y)
            else:
                raise ValueError("Nieznana metoda dyskretyzacji.")

            # Wybór progu, który daje najwyższy Zysk Informacyjny
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh, best_gain

    def _get_thresholds_eq_width(self, X_column):
        """Metoda 1: Equal-Width Binning"""
        x_min, x_max = np.min(X_column), np.max(X_column)
        if x_min == x_max:
            return []

        # Obliczenie szerokości przedziału W
        W = (x_max - x_min) / self.k

        # Generowanie progów podziału
        return [x_min + i * W for i in range(1, self.k)]

    def _get_thresholds_eq_freq(self, X_column):
        """Metoda 2: Equal-Frequency Binning"""
        sorted_x = np.sort(X_column)
        n_total = len(sorted_x)
        if n_total < self.k:
            return []

        thresholds = set()
        n_bin = n_total / self.k

        # Wyznaczanie progów jako średnich między skrajnymi elementami przedziałów
        for i in range(1, self.k):
            idx = int(i * n_bin)
            if idx < n_total and idx - 1 >= 0:
                t = (sorted_x[idx - 1] + sorted_x[idx]) / 2.0
                thresholds.add(t)
        return list(thresholds)

    def _get_thresholds_ig(self, X_column, y):
        """Metoda 3: Nadzorowany optymalny podział (Information Gain)"""
        sorted_indices = np.argsort(X_column)
        sorted_x = X_column[sorted_indices]
        sorted_y = y[sorted_indices]

        thresholds = set()

        # Kandydatami są średnie z sąsiadujących wartości X, między którymi zmienia się klasa Y
        for i in range(len(sorted_x) - 1):
            if sorted_y[i] != sorted_y[i + 1] and sorted_x[i] != sorted_x[i + 1]:
                t = (sorted_x[i] + sorted_x[i + 1]) / 2.0
                thresholds.add(t)

        return list(thresholds)

    def _information_gain(self, y, X_column, threshold):
        """Obliczanie Zysku Informacyjnego (IG)"""
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Ważona entropia dzieci
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Zysk = Entropia S - (Entropia lewego + Entropia prawego)
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """Entropia zbioru S"""
        _, counts = np.unique(y, return_counts=True)
        ps = counts / len(y)
        # Obliczenie entropii tylko dla prawdopodobieństw > 0, logarytm o podstawie 2
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


def main():
    print("Pobieranie zbioru danych Iris...")
    # fetch dataset z wykorzystaniem instrukcji z UCI
    iris = fetch_ucirepo(id=53)

    # data (as pandas dataframes)
    X_df = iris.data.features
    y_df = iris.data.targets

    # Konwersja na tablice numpy (wymagane przez naszą implementację)
    X = X_df.to_numpy()
    y = y_df.to_numpy().flatten()

    print(f"Pobrano {X.shape[0]} próbek opisanych przez {X.shape[1]} atrybuty ciągłe.")

    # Podział 80/20 ze stratyfikacją klas, jak wskazano w dokumencie PDF
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Metody do przetestowania
    methods = ['ig', 'eq_width', 'eq_freq']

    for method in methods:
        print(f"\n--- Testowanie metody: {method.upper()} ---")

        # Inicjalizacja drzewa (dla metod eq_width i eq_freq parametr k=3)
        clf = DecisionTree(max_depth=5, min_samples_split=2, method=method, k=3)

        # Pomiar czasu trenowania
        start_time = time.time()
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predykcja
        predictions = clf.predict(X_test)

        # Obliczenie miar jakości
        acc = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        print(f"Czas trenowania: {training_time:.4f} sekund")
        print(f"Dokładność (Accuracy): {acc:.4f}")
        print("Macierz pomyłek:")
        print(cm)


if __name__ == "__main__":
    main()