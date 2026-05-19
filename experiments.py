"""
Projekt: Implementacja drzewa decyzyjnego dla atrybutów ciągłych
Autorzy: Artur Brogowicz i Paweł Dobrosielski
Plik: experiments.py (Skrypt przeprowadzający eksperymenty badawcze)
"""

import numpy as np
import time
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from my_tree import DecisionTree


def run_experiment(X, y, method, k=None, n_runs=25):
    """
    Uruchamia 25 prób treningu i testowania, aby spełnić warunek losowości
    i uwiarygodnienia wyników (rygor statystyczny min. 25 uruchomień).
    """
    acc_list = []
    time_list = []
    depth_list = []
    nodes_list = []
    cm_sum = None

    for i in range(n_runs):
        # Modyfikacja ziarna losowości przy każdym uruchomieniu, by uzyskać różne podziały
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i, stratify=y
        )

        clf = DecisionTree(max_depth=10, min_samples_split=2, method=method, k=k)

        start_time = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time

        preds = clf.predict(X_test)

        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        acc_list.append(acc)
        time_list.append(train_time)
        depth_list.append(clf.get_max_depth())
        nodes_list.append(clf.get_node_count())

        # Agregacja macierzy pomyłek
        if cm_sum is None:
            cm_sum = cm
        else:
            cm_sum += cm

    return {
        'acc_mean': np.mean(acc_list),
        'acc_std': np.std(acc_list),
        'acc_min': np.min(acc_list),
        'acc_max': np.max(acc_list),
        'time_mean': np.mean(time_list),
        'depth_mean': np.mean(depth_list),
        'nodes_mean': np.mean(nodes_list),
        'cm_sum': cm_sum
    }


def print_results(res):
    print(
        f"  Accuracy: Średnia={res['acc_mean']:.4f} (Odchylenie={res['acc_std']:.4f}, Min={res['acc_min']:.4f}, Max={res['acc_max']:.4f})")
    print(f"  Struktura drzewa: Śr. głębokość={res['depth_mean']:.1f}, Śr. liczba węzłów={res['nodes_mean']:.1f}")
    print(f"  Czas obliczeń: Śr. czas trenowania={res['time_mean']:.4f}s")
    print("  Zagregowana macierz pomyłek (suma z 25 podziałów):")
    for row in res['cm_sum']:
        print(f"    {row}")


def evaluate_dataset(name, uci_id):
    print(f"\n{'=' * 70}")
    print(f"ROZPOCZYNAM ANALIZĘ ZBIORU: {name} (ID UCI: {uci_id})")
    print(f"{'=' * 70}")

    # Pobieranie danych z UCI
    dataset = fetch_ucirepo(id=uci_id)
    X_df = dataset.data.features
    y_df = dataset.data.targets

    # Zabezpieczenie na wypadek danych jednowymiarowych/wielowymiarowych
    y = y_df.iloc[:, 0].to_numpy()
    X = X_df.to_numpy()

    # Imputacja ewentualnych braków danych (NaN)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    print(f"Liczba próbek: {X.shape[0]}, Liczba atrybutów: {X.shape[1]}")

    # Eksperyment 1 & 3: Metoda optymalna IG (Information Gain)
    print("\n--- Metoda: NADZOROWANA OPTYMALNA (Information Gain) ---")
    res_ig = run_experiment(X, y, method='ig', k=None, n_runs=25)
    print_results(res_ig)

    # Eksperyment 2 & 3: Metody nienadzorowane (Equal-Width, Equal-Frequency) dla zmiennego parametru K
    for method_name, method_key in [("RÓWNEJ SZEROKOŚCI (Equal-Width)", "eq_width"),
                                    ("RÓWNEJ CZĘSTOŚCI (Equal-Frequency)", "eq_freq")]:
        print(f"\n--- Metoda: NIENADZOROWANA {method_name} ---")

        # Testowanie parametru k z przedziału [2, 10] jak założono w dokumencie
        for k in range(2, 11):
            print(f"\n Parametr dyskretyzacji k = {k}:")
            res_k = run_experiment(X, y, method=method_key, k=k, n_runs=25)
            print_results(res_k)


def main():
    # Zbiory zdefiniowane w dokumencie PDF bazujące na załączonych zrzutach ekranu
    datasets_to_test = [
        ("Iris", 53),
        ("Wine", 109),
        ("Breast Cancer Wisconsin (Diagnostic)", 17)
    ]

    for name, uci_id in datasets_to_test:
        evaluate_dataset(name, uci_id)


if __name__ == "__main__":
    main()