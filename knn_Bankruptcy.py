import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time


def run_knn_b(data_path, target_column, dataset_name, k_values=[1, 2, 3, 5, 10, 25, 50, 100], seed= 1387):
    df = pd.read_csv(data_path)
    df = df.dropna()

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data with fixed random seed
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Dummy baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    print(f"Dummy Classifier Accuracy: {dummy.score(X_test, y_test):.4f}")

    test_accuracies = []
    training_times = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)

        start_time = time.time()
        accs = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
        duration = time.time() - start_time

        training_times.append(duration)
        test_accuracies.append(np.mean(accs))
        print(f"k={k}: Cross-Val Accuracy = {np.mean(accs):.4f}, Time = {duration:.4f} sec")



    # Plot validation curve
    plt.figure()
    plt.plot(k_values, test_accuracies, marker='o')
    plt.title(f'kNN Model Complexity ({dataset_name})')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.savefig(f'images/knn_model_complexity_{dataset_name.lower()}.png')
    plt.show()

    # Learning curve for best k
    best_k = k_values[np.argmax(test_accuracies)]
    knn = KNeighborsClassifier(n_neighbors=best_k)

    train_sizes, train_scores, test_scores = learning_curve(
        knn, X_scaled, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=seed
    )

    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Train")
    plt.plot(train_sizes, test_scores_mean, label="Validation")
    plt.title(f'kNN Learning Curve (k={best_k}) ({dataset_name})')
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'images/knn_learning_curve_{dataset_name}.png')
    plt.show()

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    y_pred = best_knn.predict(X_test)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix (k={best_k}) - {dataset_name}')
    plt.savefig(f'images/knn_confusion_matrix_{dataset_name.lower()}.png')
    plt.show()


if __name__ == "__main__":
    run_knn_b("data/bankruptcy_data.csv", target_column="Bankrupt?", dataset_name="Bankruptcy")
